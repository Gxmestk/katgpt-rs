# Research 423: GPU-Tile-Sim — Tile-Graph Overlap Prediction for ANE

> **Source:** [GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design](https://arxiv.org/abs/2607.11262) — Ding et al., SJTU/NUS/KAUST, MICRO 2026
> **Date:** 2026-07-14
> **Status:** Active — Gain (speculative, low-priority; revisit when ANE kernel fusion becomes a bottleneck)
> **Related Research:** 377 (ANE Architecture — the roofline substrate), 155 (ANE Compute Backend Verdict), 066 (TileRT — running tile pipelines, not simulating), 077 (ThunderKittens — tile DSL → CPU SIMD), 418 (StreamDQ — the R418 hardware-paper precedent)
> **Related Plans:** 379 (ANE-Aware Roofline Cost Model — the closest cousin and extension target)
> **Classification:** Public

---

## TL;DR

GPU-Tile-Sim (GTSim) models GPU kernel execution as a **dependency-driven warp-centric tile graph**: nodes = warp-level tile ops, edges = data/order dependencies, backend = throughput-oriented compute/memory/NoC pipelines. Achieves 1.22–8.71% MAPE on A100/H100, 3.5–4.6× faster than Accel-Sim. The paper is GPU-specific, but its core technique — **replacing steady-state `max(compute, memory)` with dependency-aware overlap simulation** — transfers to the ANE, where Plan 379's `ane_estimate` currently uses exactly that steady-state `max()` and therefore cannot model DMA↔MAC pipeline overlap, kernel fusion benefit, or multi-stage dependency.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is a **tile-graph overlap predictor** for tiled accelerators with producer/consumer structure. The ANE has DMA (producer) ↔ MAC array (consumer) with a 2 MB on-chip working set — the same structure GTSim models for GPU TMA ↔ wgmma with SMEM. The tile-graph replaces the `max()` in `ane_roofline.rs` with a dependency-driven simulation that accounts for overlap, answering questions the current roofline cannot: "does fusing op A + op B help?", "does double-buffering fit in working set?", "what's the latency of a fused multi-stage ANE kernel?".

---

## 1. Paper Core Findings

GTSim's key insight: **modern kernel performance is governed by dependency structure (overlap, coordination, ordering), not by individual instruction latency.** The framework has three layers:

| Layer | What it does | GTSim's GPU instantiation |
|---|---|---|
| **Tile-graph abstraction** | DAG of warp-level tile ops with data edges (producer→consumer) + order edges (sync, buffer reuse) | Nodes = `{warp_set, op_type, tile_descriptor}`; edges = data (SSA-versioned tiles) + order (signal/wait, barriers, ping-pong reuse) |
| **Automatic frontend** | Extract tile-graph from compiler IR (post pipeline + warp-specialization lowering) | Consumes TileLang IR; recovers warp roles, loop scope, pipeline stages, sync scope |
| **Graph-driven backend** | Execute the graph via readiness tracking + throughput-oriented resource models | Node readiness = in-degree == 0; issue to compute/memory/NoC pipelines; sub-op decomposition for concurrency |

**The accuracy driver (Table 4 ablation):** removing general order constraints → 5.90% MAPE (from 1.55%); removing cross-warp/group sync → 35.44%; data dependencies only → 42.19%. **The dependency structure IS the accuracy.** A steady-state model (data deps only) is 27× worse.

**The pipeline-organization insight (§7.1 case study):** for a fused GEMM+SiLU kernel, WS 3-stage (dedicated epilogue warp group) is near-optimal across all tile sizes because it fully decouples epilogue from mainloop. WS ping-pong wins only for large tiles (compute-bound); naive 2-stage wins for small tiles (memory-bound). **The optimal pipeline organization depends on the memory/compute regime — exactly what a steady-state roofline cannot predict.**

---

## 2. Distillation — the ANE angle

### 2.1 Why this is NOT an auto-PASS (R418 guard applied)

The paper's abstract contains hardware vocabulary (HBM, Tensor Core, TMA, TMEM, DSMEM, Hopper, Blackwell). Per the R418 hard rule, I checked:

1. **What is the technique, stripped of the hardware substrate?** → Tile-graph DAG with data/order edges + dependency-driven scheduling + throughput-oriented resource modeling. This is substrate-independent.
2. **Software SIMD analog?** → Not on CPU SIMD (single-threaded, no warp coordination). BUT the ANE has producer/consumer structure (DMA↔MAC) analogous to GPU TMA↔wgmma. **The analog exists on the ANE, not the CPU.**

R418 decision: the value IS the technique (dependency-driven overlap modeling), and the ANE is a substrate where it applies. NOT a hardware-fabrication advance. NOT an auto-PASS.

### 2.2 The gap in Plan 379

Plan 379's `ane_estimate` (shipped, `ane_roofline.rs`) computes:

```rust
let runtime_ms = peaks.dispatch_floor_ms
    .max(compute_ms)    // flops / compute_tflops
    .max(memory_ms);    // bytes / bandwidth_gbs
```

This is **steady-state `max()`** — it assumes NO overlap between compute and memory. It is GTSim's "naive sequential" pipeline organization (Figure 4c). It cannot answer:

| Question | Plan 379 (steady-state max) | Tile-graph (dependency-aware) |
|---|---|---|
| Does fusing op A + op B save DRAM traffic? | ❌ Can't model fused ops | ✅ Data edge A→B stays on-chip; no DRAM round-trip |
| Does DMA↔MAC overlap reduce latency below `max(compute, memory)`? | ❌ Returns `max()` | ✅ Models producer/consumer overlap |
| Does double-buffering fit in 2 MB working set? | ❌ Binary cliff (fits or doesn't) | ✅ Models N-buffer tile occupancy |
| What's the latency of a 3-stage fused ANE kernel? | ❌ Single-op only | ✅ Multi-node DAG with cross-stage deps |

### 2.3 The ANE ↔ GPU structural mapping

| GTSim (GPU) | ANE analog | Source |
|---|---|---|
| Warp (scheduling unit) | ANE core / processing element | Research 377 §1.1 |
| Tile (data unit) | ANE tile (MAC operates on tile-sized matrices) | Research 377 §1.1 |
| TMA (bulk async load, producer) | DMA engine (weight/activation load, producer) | Research 377 §1.2 |
| wgmma (tensor core compute, consumer) | MAC array (fp16 compute, consumer) | Research 377 §1.1 |
| SMEM (on-chip working set) | 2 MB on-chip working set (M1) / 4.72 MB (M5) | Research 377 §2.3, Plan 379 T1.5 |
| Warp specialization (producer/consumer roles) | DMA↔MAC producer/consumer (firmware-scheduled) | Research 377 §1.2 |
| Software pipeline (WS 2-buffer / WS 3-stage) | ANE internal double-buffering (firmware-controlled) | Research 377 §1.6 |
| L2/DRAM bandwidth | ANE shared-memory / DRAM bandwidth (85–145 GB/s) | Plan 379 T1.5 |

The structural match is strong. The ANE IS a tiled accelerator with producer/consumer coordination and on-chip working set — the same abstraction GTSim's tile-graph captures.

### 2.4 The distilled primitive (modelless)

A **tile-graph overlap predictor** that extends `ane_roofline.rs`:

```rust
/// ANE tile-graph node — one tile-level operation.
/// Mirrors GTSim's node semantics, adapted for ANE's DMA↔MAC structure.
pub struct AneTileNode {
    pub op: AneTileOp,           // DmaLoad | MacCompute | Activate | DmaStore
    pub tile_bytes: u64,         // operand size (for working-set occupancy)
    pub flops: u64,              // compute work (for MacCompute)
    pub producer_for: Vec<usize>, // data-edge successors
    pub order_after: Vec<usize>, // order-edge predecessors (buffer reuse, sync)
}

/// Dependency-driven overlap simulation.
/// Replaces `max(compute, memory)` with overlap-aware prediction.
pub fn ane_tile_graph_estimate(
    graph: &[AneTileNode],
    peaks: &AnePeaks,
) -> AneTileGraphCost {
    // 1. Topological readiness tracking (in-degree == 0 → ready)
    // 2. Issue ready nodes to DMA pipeline (producer) or MAC pipeline (consumer)
    // 3. Model overlap: DMA load[i+1] overlaps MAC compute[i] if working-set fits
    // 4. Sub-op decomposition: split MAC node into pipeline-width sub-ops
    // 5. Return predicted cycles accounting for overlap
}
```

The output is a predicted cycle count that is **≤ the steady-state `max()`** when overlap is possible, and **== `max()`** when it isn't (no overlap → sequential). This is strictly more accurate for fused/pipelined kernels and degrades gracefully to the current roofline for single-op steady-state.

### 2.5 Latent-space reframing (mandatory per workflow §1 step 3)

This paper is hardware/performance modeling, not latent-space. The latent reframing is thin:

- **The tile-graph DAG is structurally a cell complex** — nodes are cells, edges are face relationships. The dependency-driven scheduling (in-degree == 0 → ready) is topological sorting. This connects to our DEC substrate (`katgpt-core/src/dec/`) at the **graph-structure** level, but the semantics are different (scheduling dependencies vs geometric cochains). No productive fusion identified.
- **The overlap prediction connects to the Plasma→Hot tier boundary.** If the tile-graph predicts an ANE fused kernel finishes in 0.5 ms (vs the roofline's 0.8 ms `max()`), the tier-dispatch threshold shifts. This is a refinement of Plan 379's routing verdict, not a latent-space operation.

No Super-GOAT angle. The value is in ANE performance prediction accuracy, not in latent-space operations.

---

## 3. Verdict

**Gain** (speculative, low-priority; GOAT upgrade possible after validation).

**One-line reasoning:** GTSim's tile-graph technique extends Plan 379's steady-state ANE roofline to model DMA↔MAC pipeline overlap, kernel fusion benefit, and multi-stage dependency — questions the current `max(compute, memory)` cannot answer — but the ANE's firmware-controlled scheduling (opaque fusion/scheduling decisions) makes the model speculative, and the payoff is marginal for our secondary-tier ANE usage.

### Why Gain, not GOAT

- **G1 (correctness) unproven.** The tile-graph model's accuracy on ANE is unvalidated. Plan 379's G1 was already "~2× off on conv" — a more complex model could be worse if the firmware's actual scheduling diverges from our assumptions. Needs real ANE measurements on fused kernels to prove.
- **Firmware opacity.** Unlike GPU (where TileLang/CUTLASS expose the pipeline organization the programmer chose), the ANE's MIL compiler decides fusion and scheduling internally. We'd be modeling a black box. `MLComputePlan` (public API) exposes which device runs each op, but NOT the internal pipeline structure.
- **Marginal payoff.** We use the ANE as a secondary Warm tier (batched NPC inference). We don't develop custom ANE kernels. The question "does this fusion help?" is answered empirically by the ANE compiler when we compile the CoreML model — we don't need to predict it pre-deployment.
- **Engineering cost.** ANE MIL IR frontend + firmware behavior reverse-engineering + validation against real ANE measurements. Plan 379 shipped a simple `max()` roofline in one plan; a tile-graph model is 3–5× the effort for a secondary-tier accuracy improvement.

### Why not PASS (the ANE angle is real)

- The structural mapping (§2.3) is strong — the ANE has genuine producer/consumer + on-chip working-set structure.
- Plan 379's `max()` IS a known limitation (it overestimates latency for any op where DMA↔MAC overlap is possible).
- The R418 guard was correctly triggered: this is a technique (dependency-driven overlap), not a hardware-fabrication advance.
- The initial PASS verdict in conversation was too hasty — it dismissed the ANE angle without checking Plan 379's gap.

### MOAT gate (per domain)

- **katgpt-rs (public engine):** IN SCOPE as a potential extension of `ane_roofline.rs`. Generic tile-graph overlap prediction is a fundamental cost-model primitive. BUT: speculative, low-priority. Ship behind feature flag IF built; do NOT block on it.
- **riir-ai (private runtime):** NOT directly in scope. The ANE tile-graph would refine `NpcBrainRouter`'s threshold, but that's a katgpt-rs primitive consumed by riir-ai, not a riir-ai contribution.

### Per-stack promote/demote tracking

| Stack slot | Current primitive | This paper's contribution | Outcome |
|---|---|---|---|
| ANE cost model | `ane_roofline.rs` (Plan 379, steady-state `max()`) | Tile-graph overlap prediction (dependency-aware) | **Do NOT promote.** Plan 379 stays default. Tile-graph is a speculative extension — investigate only if ANE fused-kernel latency prediction becomes a real dispatch bottleneck. |

---

## 4. When to revisit (no plan opened now)

This note captures the angle but does NOT open a plan. Revisit when ANY of these triggers fire:

1. **ANE kernel fusion becomes a dispatch bottleneck** — if `NpcBrainRouter` starts making wrong ANE-vs-GPU decisions because Plan 379's `max()` overestimates fused-kernel latency.
2. **Apple exposes ANE pipeline structure** — if a future CoreML/macOS API exposes the MIL compiler's fusion/scheduling decisions (making the tile-graph model prescriptive, not predictive).
3. **`riir-gpu` starts developing custom warp-specialized kernels** — at that point GTSim's technique applies directly to the GPU kernel-development workflow (not ANE), and the verdict reopens as a potential GOAT for a GPU kernel profiling toolchain.

Until then: **noted, not planned.** Plan 379's roofline is sufficient for current ANE dispatch needs.

### 4.1 Metal (Apple GPU) — not in scope

A parallel question arose during Plan 439 Phase 1 review: should we build a Metal-specific roofline alongside the ANE one? **Decision: no.** The generic `roofline_cost` (Plan 159, `katgpt-core/src/roofline.rs`) already covers Apple GPU compute — it ships `HardwarePeaks::apple_m1()` through `apple_m4_pro()` with calibrated GFLOP/s + bandwidth + launch overhead.

The ANE got its own model (Plan 379) because it has three structural cost axes that make the generic GPU model produce **wrong routing decisions**: a 2 MB working-set cliff, a 0.23 ms dispatch floor (4.6× the GPU's), and a family-floor capability gate. Apple's Metal GPU has none of these for compute workloads — TBDR tile memory is a rendering concern, unified memory simplifies (no transfer cost), and Metal shaders run on all supported GPUs (no family gate). The GTSim tile-graph technique doesn't change this: it would add accuracy for fused Metal compute kernels, but we don't control kernel fusion (Burn/CubeCL does), and the `NpcBrainRouter` doesn't route to GPU today.

**Reopen trigger:** same bar as the ANE — we'd need (a) a structural cost axis the generic model gets wrong, (b) calibration data, and (c) an active GPU routing decision that's being misrouted. None of these hold today.

---

## 5. What this paper is genuinely valuable for (not us)

GTSim is a strong MICRO 2026 paper for its actual audience:
- **GPU kernel developers** writing FlashAttention / CUTLASS / TileLang variants
- **Compiler engineers** (Triton, TileLang, CuBridge) doing warp-specialization lowering
- **Hardware architects** exploring new GPU designs (the Blackwell case study is exactly this)
- **Data-center operators** doing pre-deployment capacity planning on hardware they can't yet measure

The R418 contrast: StreamDQ transferred because its technique (LUT dequant) is an operation we **perform**. GTSim's technique (kernel perf simulation) is an operation we **don't perform** on GPU (we consume kernels via Burn/CubeCL). The partial transfer is to ANE — where we also don't develop custom kernels, but where the structural analogy (DMA↔MAC) is strong enough to note as a speculative Gain.

---

## TL;DR

GTSim models GPU kernels as dependency-driven tile graphs and achieves 1.22–8.71% MAPE by capturing pipeline overlap that steady-state models miss. The technique transfers partially to ANE: Plan 379's `ane_estimate` uses `max(compute, memory)` (GTSim's "naive sequential" pipeline), which overestimates latency for any fused kernel where DMA↔MAC overlap is possible. A tile-graph overlap predictor would close this gap. **Verdict: Gain (speculative)** — the ANE's firmware-controlled scheduling makes the model predictive-of-a-black-box rather than prescriptive, the engineering cost is high (MIL IR frontend + firmware reverse-engineering), and the payoff is marginal for our secondary-tier ANE usage. Plan 379's roofline stays default. Revisit when ANE fusion prediction becomes a real dispatch bottleneck, or when Apple exposes ANE pipeline structure.
