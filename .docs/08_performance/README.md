# Performance — Perf Engineering

> **What you find here.** Throughput tables, SIMD kernel measurements, and the
> benchmark harness that backs every GOAT gate's perf claim.

## Docs

| Doc | Role |
|---|---|
| [`engineering.md`](engineering.md) | Performance engineering — throughput tables, SIMD matmul/HLA kernels, benchmark methodology |
| [`variable_rank_monomorphization.md`](variable_rank_monomorphization.md) | T1 macro design — monomorphization escape hatch for `variable_rank_domain_expert` (Issue 189, Plan 558 G2 FAIL path to promotion) |
| [`ternary_group_q2_0_tier.md`](ternary_group_q2_0_tier.md) | `TernaryGroupWeights` / `Q2_0_g128` plasma tier — container spec, format verification, G1–G4 PASS, why it stays opt-in (closed Issue 578) |

## See also

- [`../09_feature_catalog/opt_in_features.md`](../09_feature_catalog/opt_in_features.md) — which perf-gated features ship opt-in
- [`../../.benchmarks/`](../../) — the GOAT gate benchmark results themselves
