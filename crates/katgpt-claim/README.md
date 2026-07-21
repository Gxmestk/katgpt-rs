# katgpt-claim

Claim-Level Reliability pair — `claim_rubric` (L1/L2/L3 evidence ladder
validator, Plan 307) + `clr` (sigmoid-projection vote over claim embeddings,
Plan 284). Extracted from `katgpt-rs` root per Proposal 003 Phase 11
(2026-07-04).

## Overview

Both modules belong to the "claim reliability" sub-domain. They are siblings
(zero internal coupling — verified by audit). The bridge between them lives in
downstream consumers (`riir-ai/npc_clr/claim_rubric_bridge.rs`), not in either
module, so they ship as a single sibling crate.

## Key types / modules

- `claim_rubric` — L1/L2/L3 evidence ladder validator (Plan 307,
  arXiv:2606.07612). Generic meta-discipline that grades probe/steering claims
  by evidence level. **DEFAULT-ON** (Plan 307 T3.3, 2026-06-23).
- `clr` — Claim-Level Reliability runtime (Plan 284, Research 255). Sigmoid
  projection vote over claim embeddings. **DEFAULT-ON**.

## Feature flags

`default = ["claim_rubric", "clr"]`.

| Feature | Default | Description |
|---|---|---|
| `claim_rubric` | yes | L1/L2/L3 evidence ladder validator (Plan 307). |
| `clr` | yes | Sigmoid-projection vote over claim embeddings (Plan 284). |

## Dependencies

- `katgpt-core` — `traits::FeatureClass` (claim_rubric tag) and
  `simd::simd_sum_f32` (clr mgpo + vote).
- `blake3` / `bytemuck` *(dev-only)* — for `clr` test fixtures that compute
  direction-vector hashes for tamper-evidence assertions. Production code
  does not hash.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
