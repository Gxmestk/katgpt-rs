# katgpt-ruliology

Wolfram ruliology — exhaustive simple-program enumeration (FSM / CA / TM) as
bandit arms (Plan 188, Research 168). Extracted from `katgpt-rs` root per
Proposal 003 Phase 11 (2026-07-04).

## Overview

Wolfram's ruliology proves that exhaustive enumeration of simple programs
finds winning strategies that hand-design misses. This crate enumerates all
`FSM(N)` strategies (plus cellular automata and Turing machines) as bandit
arms — zero training, inference-time only. Distinct strategies are identified
by a BLAKE3 behavioral fingerprint so duplicates collapse.

```text,ignore
let strategies = FsmEnumerator::enumerate(2);        // ~22 distinct FSMs
let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
println!("Winner: {:?}", matrix.rankings[0]);
```

## Key types / modules

- `types` — shared types (`SimpleProgram` trait, enumeration primitives).
- `fsm` — `FsmStrategy`, `FsmEnumerator` (exhaustive enumeration +
  round-robin tournament), `WinMatrix`, `MAX_STATES`.
- `ca` — `CaStrategy` (cellular-automaton strategy).
- `tm` — Turing-machine strategy.
- `payoff` — game payoff matrices.
- `bandit` — `RuliologyArm`, `RuliologyBandit`, `RuliologyAbsorbCompress`,
  `RuliologyPromoteConfig`.
- `mutation` — `FsmTemplateProposer::propose`, `delta_gated_co_evolve`
  (consumes `katgpt-pruners/g_zero::delta_absorb::DeltaGatedConfig`).
- `irreducibility` — Wolfram irreducibility classifier.
- `simulation_gate` — gates which simulations are worth running.

## Feature flags

`default = []`.

| Feature | Description |
|---|---|
| `ruliology` | Ruliology bandit — simple-program strategies as bandit arms (Plan 188). Gates the `delta_gated_co_evolve` function (which uses `DeltaGatedConfig`). Forwards to `katgpt-pruners/ruliology`. The crate's modules compile unconditionally; this feature exists for parity with the historical root feature name. |

## Dependencies

- `katgpt-core` — shared traits + primitives. Always-on.
- `katgpt-pruners` — provides `g_zero::delta_absorb::DeltaGatedConfig`
  consumed by `mutation::delta_gated_co_evolve`. Single struct, single field
  read (`config.delta_threshold`). Always-on; dep direction is clean
  (ruliology → pruners, never the reverse).
- `fastrand` — deterministic RNG for `mutation::FsmTemplateProposer::propose`
  and `delta_gated_co_evolve`.
- `blake3` — behavioral-fingerprint hashing for FSM/CA/TM dedup.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
