# Performance — Perf Engineering

> **What you find here.** Throughput tables, SIMD kernel measurements, and the
> benchmark harness that backs every GOAT gate's perf claim.

## Docs

| Doc | Role |
|---|---|
| [`engineering.md`](engineering.md) | Performance engineering — throughput tables, SIMD matmul/HLA kernels, benchmark methodology |
| [`variable_rank_monomorphization.md`](variable_rank_monomorphization.md) | T1 macro design — monomorphization escape hatch for `variable_rank_domain_expert` (Issue 189, Plan 558 G2 FAIL path to promotion) |

## See also

- [`../09_feature_catalog/opt_in_features.md`](../09_feature_catalog/opt_in_features.md) — which perf-gated features ship opt-in
- [`../../.benchmarks/`](../../) — the GOAT gate benchmark results themselves
