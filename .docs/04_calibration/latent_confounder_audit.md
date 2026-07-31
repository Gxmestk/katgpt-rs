# LatentConfounderAudit — CD-LAM §III-B Diagnostics (Issue 194, arxiv 2607.09185)

> **Status:** opt-in (`latent_confounder_audit` feature, default-off).
> G1–G4 PASS modellessly (Bench 194, 2026-07-28). **Stays opt-in** — diagnostic
> primitive, promoted only when a concrete consumer benchmarks a quality gain.

## What this is

Three modelless diagnostics distilling Wei et al. 2026 (*Causally Debiased Latent
Action Model for Embodied Action Conditioned World Models*, CD-LAM §III-B +
Appendix A) — the confounder-purity audit that any direction-vector consumer
(MAG / TILR / LatentFieldSteering / CommittedFieldBlend) can run **before
deploying** a mined or constructed direction vector:

| Diagnostic | Formula | Clean value | What it tests |
|---|---|---|---|
| Zero-transition response | `R₀ = RMS(‖E(x, x)‖) / D` | ≈ 0 | A no-op input pair should produce a near-zero latent |
| Shift-invariance response | `R_shift = RMS(‖E(x, T(x))‖) / D` | ≈ 0 | A nuisance transform should produce a near-zero latent |
| Shortcut leakage | `mean_cos(diff-action) − mean_cos(same-action)` | < 0 | Action similarity should dominate context similarity |

Where `D = RMS(‖E(x, x′)‖) + ε` over ordinary transitions. The encoder API is
`Fn(&[f32], &[f32], &mut [f32])` — output buffer as 3rd arg, sidestepping the
HRTB lifetime issues with `Fn(...) -> &[f32]`.

```rust
use katgpt_core::latent_confounder_audit::{audit_confounders, AuditScratch};

// Construct once, reuse across audits (zero steady-state alloc)
let mut scratch = AuditScratch::new(latent_dim);
let report = audit_confounders(&encoder, &transitions, &mut scratch);
// report.zero_transition_response, .shift_invariance_response, .shortcut_leakage
```

## GOAT gate (Bench 194, 2026-07-28)

| Gate | Target | Result |
|---|---|---|
| **G1** correctness | monotone in confounder coefficient `c`; 12 unit tests + 1 doctest | ✅ PASS — Clean (c=0): R₀<1e-5, R_shift<1e-5, L<0. Confounded (c=2.0): R₀>0.1, R_shift>0.1, L>-0.5. Monotone across c∈{0, 0.5, 1, 2, 5}. |
| **G2** perf | sub-µs per audit at HLA d=8 | ✅ PASS — **292 ns/call** at d=8 (3.4× under 1µs). Sweep: d=32 = 750ns, d=64 = 1.38µs. |
| **G3** no-regression | new module, feature-gated, no existing code touched | ✅ PASS — `cargo check --all-features` clean. Default 1814 → 1814; +12 with feature on. |
| **G4** alloc-free | `AuditScratch` pre-allocated; zero steady-state | ✅ PASS — 0 allocations across 100 audit calls (TrackingAllocator sentinel-verified). |

The audit is O(d) per check (norm + cosine). At HLA scale (d=8) it is essentially
free; at shard scale (d=64) it is still sub-2µs — comfortable for an offline
pre-deployment gate, which is the intended consumer pattern.

## What this does NOT prove (honest caveats)

1. **Does not prove the audit catches real bugs in production-mined direction
   vectors.** The G1 synthetic encoder has a known, injected confounder
   (`E(x,x') = A(x,x') + c·confounder(x)`); real mined directions (MAG Plan 418,
   TILR Plan 425, Latent Field Steering Plan 309, CommittedFieldBlend Plan 321)
   could have subtler confounders that the three diagnostics miss. The consumer
   adoption gate is where that gets tested.

2. **The "Report the Floor" rule (Research 322 / Plan 340) does NOT apply.** The
   three metrics are raw geometric measurements (norm ratios, cosine gaps), NOT
   probabilities / confidence scores / predictive intervals. There is no
   distributional claim. Conformal-naive is not a relevant floor here.

3. **Does not prove a quality gain in a downstream consumer.** Promotion to
   default-on requires a consumer to benchmark a real-bug-caught gain (fewer
   misconfigured directions deployed). No consumer has adopted the audit yet.

## Consumer adoption (the T7 promotion gate)

Any of the following could adopt the audit as a pre-deployment gate, re-opening
T7 for promotion to default-on:

| Consumer | What it audits | When |
|---|---|---|
| MAG (Plan 418) | Mined direction vectors | Before deploying a mined direction — reject if confounders detected |
| TILR (Plan 425) | Refined trajectory-invariant directions | After refinement pass |
| Latent Field Steering (Plan 309) | Steering direction vectors | Before injecting a steering vector |
| Committed Personality Blend (321) | Archetype direction vectors | Before committing a blend |
| HLA `evolve_hla` | Per-NPC affect direction vectors | CI test: verify hand-constructed directions are clean |
| `extract_functor` | Functor displacement vectors | CI test: verify functor has translation invariance |

## The false-PASS correction

The initial research verdict on CD-LAM was PASS; that was revised to **Gain**
after honest re-review. The diagnostic FRAMEWORK is a real gain (3 modelless
metrics + the encoder API contract), but the original PASS implied the primitives
shipped CD-LAM's debiasing capability, which they do not — CD-LAM's training
recipe (`L_emb + L_ctr + L_cal` + three-stage fine-tuning) is genuinely
gradient-descent and routes to riir-train if a video world model or analogous
training system is built (Research 460 §3.2; §3.5 Path 0 confirmed all three
objectives are genuinely training losses).

## Design decisions (for future maintainers)

- **Encoder API:** `Fn(&[f32], &[f32], &mut [f32])` — sidesteps HRTB lifetime issues.
- **Sign convention:** `shortcut_leakage < 0 = clean` (matches the issue spec).
  Formula: `mean_cos(diff-action, same-context) − mean_cos(same-action, diff-context)`
  so action-dominance makes the value negative.
- **RMS normalization:** R₀ and R_shift use `sqrt(mean(‖·‖²))` (matching D's form),
  not raw mean.
- **Scratch:** `AuditScratch::new(latent_dim)` pre-sizes two `Vec<f32>` buffers;
  `resize()` handles multi-dim audits. Zero steady-state allocation.
- **G4 sentinel logic:** the test allocates a known sentinel `Vec<u8>` first; if
  the counter didn't increase, the TrackingAllocator isn't installed — skip.
  Otherwise the audit truly is alloc-free. (Distinguishes "0 allocs = PASS" from
  "allocator not installed = unmeasurable".)
- **Test-fixture pitfall:** the clean encoder `clean(x, x')[i] = (x'[i] - x[i]) -
  mean(...)` mean-subtracts the displacement. A *constant* displacement
  `[0.5, 0.5, ...]` mean-subtracts to zero → cosine undefined → treated as 0 →
  shortcut_leakage = 0 instead of < 0. **Fix:** use non-constant displacements
  (a zero-mean ramp like `[-0.28, -0.20, ..., +0.28]`).

## See also

- [`.docs/04_calibration/faithfulness_probe.md`](faithfulness_probe.md) — sibling causal-intervention diagnostic (Plan 278)
- [`.docs/09_feature_catalog/opt_in_features.md` §27](../09_feature_catalog/opt_in_features.md) — feature-flag detail
- [`.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md`](../../.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md) — research note (Gain verdict)
