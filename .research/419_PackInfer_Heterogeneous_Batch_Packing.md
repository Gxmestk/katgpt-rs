# Research 419: PackInfer — Heterogeneous-Batch Packing for Fixed-Shape Substrates

> **Source:** [PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference](https://arxiv.org/abs/2602.06072) — Ning, Zhang, Lai (NJU + UIUC), Feb 2026
> **Date:** 2026-07-13 (opened, after PASS-retract)
> **Status:** Active — PASS **RETRACTED** on 2026-07-13 (same failure pattern as R418 StreamDQ); revised verdict Gain (pending GOAT gate)
> **Related Research:** 418 (StreamDQ — the precedent that exposed this PASS as the same failure mode), 354 (Set Attention — heterogeneous-N hot path), 218 (Breakeven routing — covers paper's kernel-selection), 066 (TileRT — persistent-kernel substrate translation), 110 (Ciot — our tier hierarchy)
> **Related Plans:** 354 (set attention), 363 (batched GPU forward — homogeneous only), 297 (am_cross_game — variable-length prefix), 328 (CceCrowdBatchHeterogeneous)
> **Classification:** Public

---

## TL;DR

PackInfer is a **vLLM + CUDA + A100** paper (~1000 lines of CUDA/C++ for FlashAttention-2 drop-in). My initial verdict was **PASS** ("no vLLM in our stack, GPU-only, not relevant") — **wrong**, by the exact same failure mode documented in R418's postmortem: I treated the *implementation substrate* (multi-tenant GPU serving) as the *value* and skipped the substrate-translation step the skill demands for hardware/LLM-as-substrate papers.

The *technique* — **length-balanced greedy bin-packing into fixed-shape execution groups, with prefix-aware length dedup, drift-triggered regroup, and contiguous-buffer consolidation** — is substrate-independent and lands directly on **three real waste points** in our codebase. The smoking gun: `AneNpcBrainBackend::batch_evaluate` compiles a CoreML model at fixed batch=1024 and pads every sub-max batch to 1024, with a comment that literally says *"wastes some ANE cycles on padding"* — that is PackInfer's exact motivation, in our code, with an explicit waste-acknowledging comment. Plan 363 already shipped homogeneous-shape GPU batching (256× dispatch reduction, 6.97× runtime at seq=16); PackInfer extends that pattern to **heterogeneous** shapes, which Plan 363 explicitly deferred (T1.5).

**Revised verdict: Gain** (real gap, pending GOAT gate). Not Super-GOAT (Q1-Q4 all NO — bin-packing is 1970s scheduling theory, no new capability, no moat). The realistic ceiling is substrate-specific:

| Slot | Ceiling prediction | Reasoning |
|---|---|---|
| ANE NPC batch (fixed-shape CoreML) | **1.5×-3×** if we can build multi-shape model pool | Eliminate 1024-N padding for sub-max zone batches; bounded by CoreML model-loading cost |
| GPU batched forward, heterogeneous seq (Plan 363 extension) | **1.2×-2×** on mixed-length game positions | Plan 363 already proved 6.97× on homogeneous; heterogeneous adds packing overhead but unlocks mixed-length workloads |
| CPU SIMD paths | **~1.0× (no gain)** | Per-element ops with no padding; nothing to pack |
| Cross-zone `set_attention` batching | **speculative** — depends on whether we batch zones at all | Currently per-zone rayon dispatch, no padding |

The GOAT gate settles it honestly. **We do NOT pre-claim a win** — the StreamDQ precedent says the pessimistic ceiling can underpromise by 2-3×, but the ANE slot has a real hardware constraint (CoreML fixed-shape compilation) that StreamDQ's slot didn't.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is a **`BinPackedGroups` scheduler**: given N items with heterogeneous `cost_i` (length, complexity, batch size), a `group_capacity` C, and a `feasibility` predicate Φ, produce G ≤ ⌈Σcost_i / C⌉ groups minimizing `max_g L(S_g) − min_g L(S_g)`. Optional prefix-dedup pass reduces effective cost by shared-prefix length. Optional drift monitor triggers regroup when accumulated imbalance crosses C/2. The primitive is substrate-agnostic — same algorithm drives ANE batch formation, GPU heterogeneous prefill grouping, and cross-zone set-attention batching.

---

## 0. The PASS postmortem (why this note exists)

R418 (StreamDQ) documented a failure pattern: *"hardware paper → 'out of scope' reflex → missed that we simulate the substrate in software SIMD"*. I repeated it verbatim on PackInfer:

1. **Substrate-as-value fallacy.** I saw "vLLM", "CUDA", "A100", "HBM", "SMs", "FlashAttention" and treated the serving-stack substrate as the value. The value is the **packing algorithm** — substrate-independent.

2. **Skipped the substrate-translation step.** The skill's standing compute-unit translation block (LLM-as-implementation ≠ LLM-as-mechanism) applies identically to hardware papers: **serving-stack-as-substrate ≠ packing-as-mechanism**. "vLLM batches requests" is the paper's instantiation; "we batch NPC cognition / game positions / zone crowds" is ours.

3. **Didn't grep our own fixed-shape padding pattern.** `AneNpcBrainBackend` (Plan 255 Part 3) pads every sub-max batch to a fixed CoreML-compiled shape of 1024 with a comment acknowledging the waste. `batch_pairs` in `replay_to_ssd.rs` pads to max_seq_len. `SheafMaps` pads shorter selectors to `d_e_max`. A single grep for `pad.*max|padded.*batch` would have invalidated the PASS — and I ran that grep *while defending the PASS*, then dismissed the hits.

4. **My "latent reframe produces no analog" claim was anchored pessimism, not analysis.** The latent reframe is weak (the primitive operates on raw batch sizes, not latent vectors) — but **the substrate translation is strong**, and I conflated the two. Weak latent reframe ≠ no analog.

The pattern: **GPU/serving paper → "we don't run that stack" reflex → missed that the technique applies to our fixed-shape substrates (ANE, GPU prefill, batched pairs)**. This will recur for every serving-systems / GPU-kernel paper unless I run the substrate-translation step before the latent-reframe step.

---

## 1. Paper Core Findings

PackInfer addresses execution imbalance in batched LLM inference where requests have heterogeneous sequence lengths. Padded per-request tiles waste compute and memory bandwidth; GPU SMs processing short requests idle while SMs on long requests form the critical path. Results on A100 + vLLM + FlashAttention-2: 13.0–20.1% latency reduction, 9.3–24.9% throughput gain.

### 1.1 The five techniques (paper §3)

1. **Greedy length-balanced bin-packing** (§3.1) — sort requests descending by L_i, iteratively assign each to the group with minimum cumulative `L(S_g)` subject to feasibility `Φ(S_g) = (Σ L_i ≤ C) ∧ (M(S_g) ≤ M_max)`. Minimizes `max_g L(S_g) − min_g L(S_g)`. Classic LPT (Longest Processing Time) scheduling heuristic.

2. **Adaptive group capacity** (§3.1) — `C` determined by offline profiling, refined online from observed latency/throughput. Group count `G = ⌈L_total / C⌉`.

3. **Prefix-aware length dedup** (§3.2) — when requests share prefixes (system prompt, conversation history), redefine effective length `L̂_i = L_i − L_shared,i` for group-load accounting. Trie partition identifies unique prefixes and suffix sets.

4. **Drift-triggered regroup** (§3.1) — accumulate `ΔL = max_g L(S_g) − min_g L(S_g)` over t decoding steps; regroup when `t·ΔL ≥ C/2`. Reaches threshold every 20-40 steps in practice.

5. **Contiguous-buffer consolidation** (§3.2) — gather scattered KV pages into group-contiguous workspace; reserve suffix headroom δ for future tokens; only valid tokens copied (no internal fragmentation).

### 1.2 The kernel-selection insight (§6.4 appendix)

Not in the main body, but the paper compares its heuristic solver against a Z3-based optimum: the heuristic is orders of magnitude faster, with negligible solution-quality gap. Prepack (Zhao et al. 2024) uses an ILP solver that is 3× slower — PackInfer's win is partly solver overhead, partly packing quality.

---

## 2. Distillation — substrate translation (the step I skipped)

The paper's hardware framing (CUDA thread blocks, SMs, HBM, SRAM, A100) obscures that **all five techniques are substrate-independent algorithms**. The table below is the honest mapping.

| Paper technique (vLLM/CUDA substrate) | Substrate-independent technique | Our substrate analog | Current codebase state | Gap |
|---|---|---|---|---|
| Greedy bin-packing into G groups by length | LPT scheduling into capacity-C groups | ANE batch formation, GPU prefill grouping, zone-crowd batching | `NpcBrainRouter` is a thin pass-through — no sub-batching; `dispatch_layer_batched` (Plan 363) assumes homogeneous shape | **Primary gap** — no length-aware batch former |
| Adaptive group capacity C | Profile-driven threshold | Per-substrate compiled-shape / dispatch-overhead threshold | `OPTIMAL_BATCH_SIZE = 1024` hardcoded; `SetAttentionConfig::with_top_k` is per-call, not batch-aware | Capacity fixed at compile time |
| Prefix-aware length dedup `L̂_i = L_i − L_shared,i` | Shared-context dedup in cost accounting | NPCs sharing zone context / personality shard / system prompt | `CrossGamePrefix` (Plan 297) deduplicates prefix KV per-call, **not across calls in a batch** | Cross-call prefix dedup missing |
| Drift-triggered regroup | Hysteresis on imbalance signal | Per-tick batch reformation | N/A — batches aren't currently grouped | N/A until grouping ships |
| Contiguous-buffer consolidation | Cache-aligned scratch layout | Pre-allocated reusable scratch | **SHIPPED** `AneNpcBrainBackend::padded_inputs` / `scratch_*`, `set_sigmoid_attention_into` caller-owned scratch | None for the consolidation pattern |
| Heuristic solver vs Z3 optimum | LPT greedy vs bin-packing optimum | Anywhere we partition work | Breakeven routing (Plan 218) picks memory-bound vs compute-bound kernel; not the same axis | None at the algorithmic level |

### 2.1 The ANE hot path — what we do today

`riir-ai/crates/riir-engine/src/npc_ane_backend.rs:474-521`:

```rust
// Fixed-shape ANE models require the exact compiled batch size.
// Pad with default (zero) inputs up to max_batch_size, run the full
// batch, then decode only the first `batch` outputs.
// This wastes some ANE cycles on padding but avoids dynamic-shape
// pitfalls (CoreML's RangeDim support is unreliable for ANE).
if batch < self.max_batch_size {
    self.padded_inputs[..batch].clone_from_slice(inputs);
    for slot in &mut self.padded_inputs[batch..self.max_batch_size] {
        *slot = NpcBrainInput::default();
    }
} else {
    self.padded_inputs.clone_from_slice(inputs);
}
// Run prediction on the full padded batch
let prediction = self.predict_raw(self.max_batch_size, ...)?;
// Decode only the first `batch` outputs (padding rows are discarded)
decode_output(&prediction, ..., inputs, outputs)?;
```

Observations:
- `max_batch_size = 1024`, hardcoded. Model is compiled at this shape.
- For a zone with 5 active NPCs, we run **1024 evaluations** and discard 1019. That is **204× the work**.
- The comment is honest: *"wastes some ANE cycles on padding"*. The waste is more than "some" at low batch.
- CoreML's RangeDim (dynamic shape) is documented as unreliable for ANE — the constraint is real, not laziness.
- **PackInfer's contribution is exactly this problem in a different substrate**: NVIDIA GPU tiles waste cycles when L_i << T. Our ANE tile wastes cycles when batch < max_batch_size. Same waste, same fix shape.

### 2.2 The distilled primitive

```text
BinPackedGroups::form(
    items: &[(ItemId, Cost)],    // heterogeneous-cost items
    capacity: Cost,               // group capacity C (per-substrate)
    feasibility: impl Fn(&Group) -> bool,  // Φ — memory/shape constraints
    prefix_dedup: Option<&PrefixTree>,     // optional shared-context dedup
) -> Vec<Group>  // G groups, balanced, feasibility-satisfied
```

Plus a `DriftMonitor` that accumulates imbalance and triggers regroup at C/2.

The primitive is generic over what "cost" means:
- ANE slot: cost = 1 per NPC (uniform), capacity = `max_batch_size`. **Without prefix dedup, all items have equal cost → trivial round-robin**. The interesting subproblem is forming batches close to a power-of-2 shape if CoreML allows multiple compiled shapes.
- GPU batched slot: cost = seq_len per game position. Heterogeneous. **Direct PackInfer analog**.
- Set attention slot: cost = zone entity count. Heterogeneous across zones.

### 2.3 The ANE slot's hard constraint (honest caveat)

StreamDQ's slot had no hardware-imposed shape constraint — the SIMD LUT worked on any input. PackInfer's slot (GPU tiles) is hardware-flexible — you can launch any kernel shape.

**Our ANE slot is hardware-inflexible**: CoreML compiles to a fixed shape, and `RangeDim` is unreliable for ANE residency. So the naive PackInfer fix (form a group of size 5, dispatch a batch of 5) **does not work** — the model rejects batch=5.

The honest options:
1. **Multi-shape model pool**: compile the same CoreML model at several batch sizes (e.g., 8, 32, 128, 1024). At runtime, bin-pack NPCs into the smallest shape that fits. This is a real PackInfer analog — groups are formed, dispatched to the matching compiled shape. **Cost: N× model-loading memory; benefit: ~12× at batch=5 (1024 → 8).**
2. **Sort-and-defer**: collect NPCs across ticks until batch approaches a shape boundary; defer low-priority NPCs to the next tick. **Latency cost on deferred NPCs; throughput gain on the batch.**
3. **CPU fallback for small batches**: route batches below breakeven to `CpuTernaryBackend` (which already exists and has no fixed-shape constraint). **No ANE waste; depends on CPU path being fast enough.**

Option 1 is the cleanest PackInfer analog. Option 3 is the Breakeven-routing (Plan 218) analog. **Both are modelless.**

### 2.4 The GPU batched slot (Plan 363 extension)

Plan 363 already proved:
- Homogeneous-shape batched forward: 256× dispatch reduction, 6.97× runtime at seq=16.
- T1.5 (wiring batched elementwise into per-position path) **DEFERRED** with note: *"provides no speedup — matmul/attention ops still run per-position, requiring expensive buffer copies"*.

PackInfer's heterogeneous extension would let Plan 363 batch **mixed seq_len** game positions in one dispatch — currently impossible because the batched path assumes uniform shape. Game positions do have heterogeneous histories (some positions are at move 5, some at move 200), so this is a real workload, not theoretical.

**Predicted gain: 1.2×-2× on mixed-length batches**, bounded by the overhead of the packing pass and the wgpu buffer-copy cost.

### 2.5 Fusion candidates (cross-pollination)

| Fusion | What it produces | Strength |
|---|---|---|
| PackInfer bin-pack × Plan 363 batched GPU forward | Heterogeneous-seq_len batched GPU prefill — currently blocked by T1.5 defer | **Strong** — directly unblocks T1.5 with a real algorithm |
| PackInfer bin-pack × ANE multi-shape pool | Multi-shape CoreML dispatch — eliminates 1024-N padding | **Strong** — directly addresses the explicit waste comment |
| PackInfer prefix-dedup × `CrossGamePrefix` (Plan 297) | Cross-call prefix sharing across NPCs in the same zone | Medium — speculative until we batch NPC dialog |
| PackInfer bin-pack × `CceCrowdBatchHeterogeneous` (Plan 328) | Length-aware zone batching for crowd CCE | Weak — CCE matrices are fixed-shape N×A per zone, no heterogeneity |
| PackInfer drift-regroup × Collapse-aware (Plan 212) | Regroup on workload drift AND cognitive collapse | Weak — two unrelated drift signals, no synergy |
| PackInfer bin-pack × Breakeven router (R218) | Breakeven picks the kernel; PackInfer forms the batch the kernel runs on | **Strong** — complementary decision axes |

The strongest fusion is **PackInfer × Plan 363** — it directly unblocks the deferred T1.5 with a real algorithm, and Plan 363's GOAT gate already validated the homogeneous case.

### 2.6 Latent-space reframing (§1 workflow step 3 — mandatory)

PackInfer operates on raw batch sizes and sequence lengths — not latent space. The latent reframe is **weak** (weaker than StreamDQ's, which had the LUT-as-raw→latent-bridge angle):

- The "prefix tree" PackInfer builds is structurally similar to the prefix KV caches in `CrossGamePrefix`, but those are latent (compact KV). The dedup *mechanism* (trie partition, shared-prefix length subtraction) is identical; the *content* differs (PackInfer = token IDs, ours = latent KV rows).
- The "shared prefix" in our substrate is more naturally the **personality shard** or **zone context vector** — frozen latent state shared across NPCs, not a token sequence. The dedup is done at freeze/thaw time (the shard is committed once, retrieved many times), not at batch-formation time.
- The drift-triggered regroup has no latent analog — it's a control loop on raw workload imbalance.

This reframing doesn't unlock a Super-GOAT angle. The primitive stays in the "batched compute scheduling" slot (a katgpt-rs engine/systems concern), not in the HLA / functor / shard / LatCal / DEC substrate.

**The substrate translation (§2 above) is where the real value is, not the latent reframe.** Conflating the two was my PASS-verdict error.

---

## 3. Verdict

| Criterion | Assessment |
|---|---|
| **Q1 No prior art?** | NO — LPT bin-packing is 1970s scheduling theory; prefix dedup is in vLLM/SGLang already |
| **Q2 New capability class?** | NO — we already batch (Plan 363, Plan 328); heterogeneous-shape batching is an optimization |
| **Q3 Product selling point?** | NO — perf optimization of existing capability, not a new capability |
| **Q4 Force multiplier (≥2 pillars)?** | NO — touches the compute-dispatch slot, not a pillar multiplier |

**Q1-Q4 = NO → not Super-GOAT.** No private guide required (§1.5).

### 3.1 Tier verdict: **Gain** (pending GOAT gate)

One-line reasoning: real gap in our batch-formation substrate (no length-aware batch former; ANE pads to fixed 1024; Plan 363 deferred heterogeneous T1.5). The transferable primitive is a `BinPackedGroups` scheduler with optional prefix dedup and drift-triggered regroup. The win is substrate-specific — **ANE multi-shape pool slot and GPU heterogeneous-batch slot are the candidates for PROMOTE**; CPU SIMD slot is a predicted no-op (DEMOTE expected). Latency must be benchmarked, not pre-claimed.

**Honest ceiling prediction (StreamDQ lesson: pessimistic ceilings can underpromise by 2-3×):**

| Slot | Prediction | Confidence | Reasoning |
|---|---|---|---|
| ANE multi-shape pool | **1.5×-3× at small batch** | Medium | Eliminates 1024-N padding; bounded by CoreML model-load memory cost and whether multi-shape residency holds |
| GPU heterogeneous batched forward (Plan 363 extension) | **1.2×-2× on mixed seq_len** | Medium-High | Plan 363 already proved 6.97× homogeneous; heterogeneous adds packing overhead |
| CPU SIMD (per-element ops) | **~1.0× (no gain)** | High | No padding exists in per-element SIMD paths |
| Cross-zone set_attention batching | **speculative — N/A until we batch zones** | Low | Currently per-zone rayon dispatch, no padding to eliminate |

The StreamDQ precedent says the pessimistic ceiling can underpromise — but ANE has a hardware constraint (CoreML fixed-shape compilation) that StreamDQ's slot didn't. **We do NOT pre-claim a win.** The GOAT gate settles it.

### 3.2 MOAT gate per domain (§1.6)

| Domain | Bar | Verdict |
|---|---|---|
| `katgpt-rs` (public engine) | "Paper-derived fundamental / principle / base-foundation primitive that passes GOAT or Gain via fusion, with promote/demote tracked per stack" | ✅ IN SCOPE — batched-compute scheduling primitive, dispatch-slot concern. Promote/demote tracked per substrate (ANE / GPU / CPU). |
| `riir-ai` | Pillar-level or Super-GOAT | N/A — perf optimization, not pillar-level |
| `riir-chain` / `riir-neuron-db` | Pillar-level or Super-GOAT | N/A — not a chain or shard concern |

Routing: **open primitive in katgpt-rs** (generic `BinPackedGroups`), **private wiring in riir-ai** (ANE multi-shape pool, GPU heterogeneous batch).

### 3.3 Modelless-first check (§3.5)

Not a training dependency. All three ANE-slot options (multi-shape pool, sort-and-defer, CPU fallback) are modelless. No gate to unblock. No riir-train deferral.

### 3.4 Defend-wrong PoC check (§3.6)

Claims:
- **Architectural**: "the runtime analog exists (we have padding waste in `AneNpcBrainBackend` and Plan 363's homogeneous batching)" — proven by grep + read. Sufficient.
- **Latency**: "bin-packed batch dispatch is faster than padded fixed-shape dispatch" — UNPROVEN, pending GOAT gate. Explicitly marked unproven. Settles via criterion bench, not PoC.
- **Quality**: bit-exact by construction (grouping doesn't change which NPCs get evaluated, only how they're batched). No quality claim to defend.

§3.6 PoC does NOT trigger — quality is bit-exact by construction (same NPC set, same outputs, different batch shape), latency claim is explicitly unproven and deferred to the GOAT gate, no "parity" claim against the paper's GPU numbers (we explicitly disclaim that — vLLM/CUDA gains are not directly replicable on ANE/CubeCL).

### 3.5 UQ-bearing primitive check ("Report the Floor" rule)

NOT UQ-bearing. Deterministic batch scheduling. No probability distribution, no predictive interval, no coverage guarantee. Conformal-naive floor does NOT apply.

---

## 4. What ships (Plan 432, conditional on GOAT gate)

1. **Open primitive** in `katgpt-rs/crates/katgpt-core/src/`:
   - `BinPackedGroups` struct with `form(items, capacity, feasibility, prefix_dedup)` constructor
   - `DriftMonitor` with `should_regroup(accumulated_imbalance, capacity)` decision
   - Optional `PrefixTree` for shared-context dedup (thin wrapper around `CrossGamePrefix`-style trie)
   - Feature flag `bin_packed_groups` (opt-in)
   - Zero-allocation hot path — caller-owned group output buffer

2. **GOAT gate** (the benchmark that decides promote-to-default):
   - **G1 correctness**: bit-exact match — grouping doesn't change outputs, only batch shape. (Set discriminator: `{evaluated NPC set} == {original NPC set}`.)
   - **G2 latency**: criterion bench on (a) ANE multi-shape pool at batch=5 vs current padded-1024, (b) GPU heterogeneous batched forward at mixed seq_lens vs homogeneous-only, (c) CPU SIMD with grouping vs without (predicted no-op). Target: ≥ 1.2× on at least one substrate. **If G2 fails on all substrates: demote, keep feature opt-in, document.**
   - **G3 no-regression**: `--all-features` clippy + test clean
   - **G4 alloc-free**: zero allocations on the grouping hot path (sort buffer + group buffer pre-allocated)
   - **G5 substrate-isolation**: each substrate (ANE / GPU / CPU) gated independently — demote the loser per substrate (StreamDQ precedent: fused slot promoted, standalone slot demoted).
   - **G6 multi-shape ANE residency**: if implementing option 1 (multi-shape pool), validate each compiled shape holds ANE residency (< 1ms prediction) — falls back to CPU otherwise.

3. **Conditional follow-up plans** (only on G2 PASS per substrate):
   - **riir-ai/.plans/ Plan 433**: ANE multi-shape pool — compile `npc_brain.mlmodelc` at shapes {8, 32, 128, 1024}, route via `BinPackedGroups`.
   - **riir-ai/.plans/ Plan 434**: GPU heterogeneous batched forward — extend Plan 363's `dispatch_layer_batched` to accept variable seq_len per batch position (the T1.5 unblock).

### 4.1 Honest expectation-setting

The paper's 13-20% latency reduction is a **vLLM + A100** number, gained by eliminating padded CTA tiles under FlashAttention-2. Our substrates differ:

- **ANE**: the waste is larger (1024× headroom, not 128× tile), but the fix is harder (CoreML fixed-shape compilation). Multi-shape pool is the only direct analog, and it pays N× model memory.
- **GPU (CubeCL/wgpu)**: Plan 363 already eliminated the homogeneous-shape waste (256× dispatch reduction). The heterogeneous case is a smaller marginal gain on top of an already-optimized baseline.
- **CPU SIMD**: no padding exists; no gain expected.

**Realistic target: 1.5×-3× on the ANE slot at small batch (the smoking-gun comment's waste), 1.2×-2× on the GPU heterogeneous slot, ~1.0× on CPU.** The GOAT gate settles it honestly per substrate. We do NOT pre-claim a win.

**The StreamDQ lesson applied**: pessimistic ceilings can underpromise. But the ANE slot has a real hardware constraint that StreamDQ's slot didn't (CoreML fixed-shape vs unrestricted SIMD LUT). The gate decides per substrate — PROMOTE the winners, DEMOTE the losers.

---

## TL;DR

PackInfer is a vLLM + CUDA + A100 paper (~1000 lines CUDA/C++) for batched LLM serving. The initial verdict (PASS, "GPU-only, we don't run vLLM") was **wrong** — same failure pattern as R418 StreamDQ: I treated the *implementation substrate* (vLLM serving) as the *value*, skipped the substrate-translation step, and didn't grep our own fixed-shape padding pattern. The *technique* — length-balanced bin-packing with prefix dedup, drift-triggered regroup, contiguous consolidation — is substrate-independent. The smoking gun: `AneNpcBrainBackend::batch_evaluate` compiles at fixed batch=1024 and pads every sub-max batch, with a comment *"wastes some ANE cycles on padding"* — that is PackInfer's exact motivation, in our code, acknowledged. **Revised verdict: Gain** (real gap, pending GOAT gate). Plan 432 ships the open `BinPackedGroups` primitive behind `bin_packed_groups` feature; conditional Plans 433 (ANE multi-shape pool) and 434 (GPU heterogeneous batched forward, unblocking Plan 363 T1.5) follow per-substrate GOAT gate. Honest ceiling: 1.5×-3× ANE, 1.2×-2× GPU heterogeneous, ~1.0× CPU SIMD. Not Super-GOAT (Q1-Q4 NO). No private guide. Not UQ-bearing. Not a training dependency.
