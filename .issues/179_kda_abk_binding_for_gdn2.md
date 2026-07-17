# Issue 179 — KDA `a=b=k` DPLR binding for GDN2 SIMD kernel

**Date:** 2026-07-17
**Research:** [katgpt-rs/.research/447_Kimi_K3_KDA_AttnRes_LatentMoE.md](../.research/447_Kimi_K3_KDA_AttnRes_LatentMoE.md) §2.3 + §2.6 Fusion B
**Source paper:** [arxiv 2510.26692](https://arxiv.org/abs/2510.26692) — "Kimi Linear: An Expressive, Efficient Attention Architecture" (Nov 2025), §6.2 + Figure 2
**Target:** `katgpt-rs/crates/katgpt-attn/src/gdn2/types.rs` (new `Gdn2GateConfig::KdaBound` variant) + `crates/katgpt-attn/src/gdn2/kernel.rs` (the bound recurrent step)
**Sibling plan:** [`.plans/455_quantile_balancing_router_primitive.md`](../.plans/455_quantile_balancing_router_primitive.md) (the other actionable item from Research 447)
**Status:** Open — tracking issue. Not yet started; awaiting bandwidth for a Plan 105 Phase N extension. **Issue, not plan** because the algorithm-transfer question is the gate, not the implementation (which is mechanical once the GOAT gate answers the transfer question).

> **Numbering note.** The research note (447 §6) proposed this as `Issue 165`. That number is already in use (`.issues/165_dd_tree_file_split_c2.md`), and the katgpt-rs AGENTS.md rule — *"monotonic and never reused — even after a file is removed per the noise-reduction rule"* — forbids recycling it. Re-issued as **Issue 179** = `.issues/.highwater` (178) + 1. The matching plan was re-issued as **Plan 455** (not 447 — `.plans/447_freq_bandit_phase1.md` exists).

---

## Context

Plan 105 ships Gated DeltaNet-2 (GDN2) recurrent attention for CPU SIMD inference with three gate configurations (`crates/katgpt-attn/src/gdn2/types.rs`):

| `Gdn2GateConfig` | erase `b_t` | write `w_t` | decay `α_t` | Purpose |
|---|---|---|---|---|
| `EraseOnly` (default) | channel `σ(W_b x)` | scalar `β = mean(b)` | channel | ~90% of full gain, fewer params |
| `Full` | channel `σ(W_b x)` | channel `σ(W_w x)` | channel | full GDN2 expressiveness |
| `Kda` | scalar `β` | scalar `β` | channel | KDA baseline (tied gates) |

**The KDA paper (arxiv 2510.26692, §6.2 + Figure 2)** shows that the KDA transition matrix can be parameterized as a Diagonal-Plus-Low-Rank (DPLR) form with a specific **`a = b = k` binding** that collapses the recurrent step. The binding removes **2 secondary chunking steps + 3 matrix multiplications** vs the general DPLR formulation, yielding **~2× kernel speedup on GPU tensor cores** (paper Figure 2, batch=1, 16 heads, seq 2k–64k).

The KDA paper's `a = b = k` binding is **not** the same as our existing `Gdn2GateConfig::Kda` variant — that variant is just the tied-gates scalar-β baseline. The unique KDA trick is the **DPLR structural binding**, which is a new variant we don't ship.

## Vocabulary collision (R296-style — must be documented before implementation)

Three "KDA" senses in the codebase:

1. **Plan 097 `DeltaRoutingMode::DeltaAttnRes`** — NOT KDA. Different mechanism (delta-attention residual). Shared "delta" only.
2. **Plan 105 `Gdn2GateConfig::Kda`** — the tied-gates scalar-β baseline. Parameterization of KDA but **NOT** the `a=b=k` binding.
3. **Kimi KDA (this issue)** — DPLR `a = β·k, b = k⊙α, D = Diag(α)` with the `a=b=k` binding. Distinct from both above.

The new variant should be named **`KdaBound`** (not `Kda`, not `KDA`, not `KimKda`) to make the binding-the-trick explicit. Update `crates/katgpt-attn/src/gdn2/types.rs` doc comment table to add a 4th row + a footnote disambiguating the three senses.

## The GOAT question (the only honest unknown)

Paper Figure 2 shows ~2× speedup on **GPU tensor cores**. Our substrate is **CPU SIMD** (`katgpt-rs/crates/katgpt-core/src/simd.rs` — NEON on ARM, AVX2 on x86). The matmul cost structure differs:

- **GPU tensor cores:** matmul is the dominant cost; removing 3 matmuls is a huge win.
- **CPU SIMD:** the recurrent step's matvec cost (`Sᵀ(b⊙k)`, `Sᵀq`) dominates; the secondary chunking steps that KDA removes may already be cheaper on CPU than on GPU (where the chunking has fixed kernel-launch overhead).

**The transfer is not guaranteed.** Three possible outcomes:

- **GOAT PASS (promote):** CPU SIMD gets ≥1.5× speedup → promote `KdaBound` to default-on for the GDN2 family (or fold into `EraseOnly` if it dominates).
- **GOAT PARTIAL (keep opt-in):** speedup only on large seq lengths (≥4k tokens), regress on short → keep `KdaBound` opt-in, document the threshold.
- **GOAT FAIL (close as PASS):** the binding doesn't help on CPU SIMD (matmul cost structure differs) → close the issue as PASS, ship the channel-wise variant as the `EraseOnly` extension only. **Honest outcome — paper's GPU win does not always transfer.**

## Tasks (when bandwidth allows)

- [ ] **T1** Add `Gdn2GateConfig::KdaBound` variant to `crates/katgpt-attn/src/gdn2/types.rs`. Doc-comment table gets a 4th row + the vocabulary-collision footnote (above).
- [ ] **T2** Implement the DPLR `a=b=k` bound recurrent step in `crates/katgpt-attn/src/gdn2/kernel.rs`:
  - [ ] Bind `a_t = β·k_t`, `b_t = k_t ⊙ α_t`, `D_t = Diag(α_t)` (paper Eq. for KDA)
  - [ ] Skip the 2 secondary chunking steps + 3 matmuls that the general DPLR form requires (paper Figure 2 annotation)
  - [ ] Reuse existing `simd_dot_f32`, `simd_scale` kernels where possible
- [ ] **T3** Benchmark `KdaBound` vs `EraseOnly` vs `Full` on CPU SIMD:
  - [ ] Throughput: tok/s for recurrent decode at d_k=d_v ∈ {4, 8, 16, 32}
  - [ ] Sequence-length sweep: positions ∈ {8, 64, 256, 1024, 4096} — the binding's benefit (if any) should show up at longer seq
  - [ ] Architecture sweep: ARM NEON (Apple Silicon) + x86 AVX2 (Linux CI runner)
- [ ] **T4** GOAT gate decision (per the three outcomes above):
  - [ ] If ≥1.5× speedup at game-scale configs (d_k=d_v=8, seq ≤ 1024) → promote `KdaBound` to default-on for the GDN2 family
  - [ ] If speedup only at large seq (≥4k) → keep opt-in, document the threshold in `types.rs` doc + this issue
  - [ ] If no speedup on CPU SIMD → close as PASS, document that the paper's GPU win does not transfer to our substrate, ship channel-wise variant as `EraseOnly` extension only
- [ ] **T5** If T4 = PASS: update Plan 105 doc + `katgpt-rs/.research/070_Gated_DeltaNet_2_*.md` + `katgpt-rs/.research/447_*.md` with the verdict.

## Out of scope

- **GPU/CubeCL port of the `a=b=k` binding** → `riir-ai/crates/riir-gpu/`. The paper's Figure 2 numbers ARE on GPU tensor cores; a GPU port should reproduce them. Out of scope for this issue (CPU SIMD only).
- **Training-side KDA chunkwise WY representation** → `riir-train`. The `a=b=k` binding has a training-side analog (DPLR WY factorization) that the paper derives in §4; we don't train.
- **KDA at attention-layer granularity in a real transformer** (vs the kernel micro-bench) → needs a Plan 105 Phase N extension with end-to-end forward-pass wiring. This issue is the GOAT-gate first step.

## Dependencies

- Plan 105 (GDN2 recurrent attention) — DONE, all variants ship.
- `katgpt-core/src/simd.rs` SIMD primitives — exists, used by Plan 105 `EraseOnly`/`Full`/`Kda` variants.
- arxiv 2510.26692 (KDA paper) — public, equations in §6.2, Figure 2 has the kernel-speedup ablation.

## Why issue, not plan

The algorithm transfer question (does GPU tensor-core speedup survive on CPU SIMD?) is the gate. Implementation is mechanical (one new enum variant + one new kernel function + one bench file) once the GOAT gate answers the transfer question. If T4 verdicts FAIL, the plan was never worth writing. **Promote to a plan only after T3 benchmarks show a real CPU-SIMD win.**

---

## TL;DR

KDA's `a=b=k` DPLR binding (arxiv 2510.26692 §6.2) removes 2 secondary chunking steps + 3 matmuls → ~2× kernel speedup on GPU tensor cores. Plan 105 ships GDN2 but not this binding. **The open GOAT question: does the GPU speedup transfer to our CPU SIMD substrate (`katgpt-core/src/simd.rs`, NEON/AVX2)?** Three honest outcomes: ≥1.5× CPU speedup → promote to default; speedup only at large seq → keep opt-in; no CPU speedup → close as PASS (paper's GPU win doesn't always transfer to CPU matmul cost structure). Issue-tracked (not plan) because the algorithm-transfer question is the gate, not the implementation — promote to a plan only after T3 benchmarks show a real win.
