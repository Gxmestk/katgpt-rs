# katgpt-proof-cert

Hierarchical GOAT Proof Certificates (Plan 145, Research 106). Standalone,
serializable proof certificates with dependency chains, topological
verification, and BLAKE3 checksum integrity. Extracted from `katgpt-rs/src/
proof_cert/` per Proposal 003 Phase 12 (2026-07-04).

## Overview

Proof certificates are the audit trail for a GOAT (Greatest-Of-All-Time)
benchmark verdict. They record which property was checked, which evidence was
used, which result was produced, and which upstream certificates the verdict
depends on — then serialize the bundle with a BLAKE3 checksum so it can be
filed alongside the benchmark.

The chain is verified by topological sort: a certificate is `valid` only if
every certificate it depends on is also `valid`.

## Key types / modules

- `certificate` — `ProofCertificate`, `ProofEvidence`, `ProofProperty`,
  `ProofResult` core types.
- `chain` — `verify_proof_chain` topological verifier.
- `serde_impls` — `load_certificates` / `save_certificates` /
  `verify_checksum` BLAKE3-integrity-checked file I/O (postcard format).
- `wasm_certificates` — `generate_wasm_validator_certificates` for validator
  proof bundles.
- `wasm_proof_witness` *(gated `wasm_proof_witness`)* — `WasmProofWitness` +
  `generate_wasm_witness_certificates` for witness-generation bundles.
- `macros` — `conditional_proof!` declarative macro (exported at crate root
  via `#[macro_export]`).

## Feature flags

`default = []`.

| Feature | Description |
|---|---|
| `wasm_proof_witness` | WASM proof witness generation. Adds the witness-generation code path (`WasmProofWitness` + `generate_wasm_witness_certificates`). No extra deps — `blake3` is already always-on for `serde_impls`. |

## Dependencies

- `serde` — `ProofCertificate` + chain serialization (always-on).
- `postcard` — binary persistence format for certificate bundles (always-on).
- `blake3` — checksum integrity for saved bundles (always-on — the binary
  format includes a checksum).

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
