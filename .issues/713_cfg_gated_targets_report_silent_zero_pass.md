# Issue 713 — 430 cargo targets can report a green `0 passed` having run nothing

**Status:** T1 **LANDED** (instrument + measurement, 2026-09-03). T2 open
(katgpt-rs's own 43 load-bearing targets). T3–T5 are owner calls in sibling
repos, deliberately not made here.

**Instrument:** `scripts/cfg_gated_target_audit.py` — a **report, not a gate**
(exit 0 always), same discipline as `ci_gate_coverage.py` and
`staged_set_audit.py`.

```bash
scripts/cfg_gated_target_audit.py              # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai   # or a named repo
```

## The shape

A test file that opens with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off. Cargo then prints

```
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

and **exits 0**. That is byte-for-byte indistinguishable from a real pass.

`required-features` is the fix, and it is not cosmetic — it changes the outcome
to:

```
error: target `t063_tpr_structure` in package `riir-clippy` requires the
features: `tpr_structure`                                        (exit 101)
```

**The `#![cfg]` protects the COUNT. `required-features` protects the READER.**
Both are needed; neither substitutes for the other. This is the same defect
fixed one target at a time in riir-train `5821cba9` (`nora_phase1_hooks` — 11
real assertions reporting as a green suite having run none) and riir-clippy
`19beece` (`t063_tpr_structure`, the harness behind a whole benchmark). Fixing
them one at a time is how it stayed invisible; this measures the population.

## Measured, 2026-09-03 — 19 repos, derived population

| | count |
|---|---|
| targets scanned | 2,897 |
| carrying a whole-file `#![cfg]` | 1,740 |
| of those, declaring `required-features` | 956 |
| **SILENT-NOW** — zeroes on a plain `cargo test` | **430** |
| latent — zeroes only under `--no-default-features` | 332 |
| platform `cfg` — `required-features` *cannot* express | 21 |
| `any(...)` of features — cargo's `required-features` is AND-only | 1 |

The last five rows **partition** the 1,740: 956 + 430 + 332 + 21 + 1 = 1,740.
Stated because a classifier that silently drops a case reports a smaller
defect count and looks like good news.

**The severity split is the point.** Pooled at 762 the number is unreadable: a
target gated on a **default-on** feature still runs on a plain `cargo test` and
only vanishes under `--no-default-features`, which is a real but much rarer
hazard. A target gated on a **default-off** feature reports a green zero *every
time anyone runs it by name*. The split is computed from a within-manifest
`default` closure — `dep/feat` and `dep:x` entries are deliberately excluded,
because a dependency's feature cannot gate a `#![cfg(feature = ...)]` in **this**
crate, and including them would over-approximate the default set and silently
downgrade real findings into the latent class.

The two non-defect classes are reported apart for the same reason: a platform
predicate and an any-of feature set are shapes `required-features` genuinely
cannot express, and a report that cries wolf on the one shape cargo cannot fix
gets ignored on the 430 it can.

### Per repo, SILENT-NOW

| repo | targets | `#![cfg]` | w/ req-f | **SILENT-NOW** | latent |
|---|---|---|---|---|---|
| riir-ai | 930 | 560 | 239 | **125** | 190 |
| katgpt-rs | 921 | 541 | 332 | **106** | 101 |
| riir-chain | 170 | 133 | 66 | **48** | 19 |
| riir-game-sdk | 62 | 40 | 1 | **37** | 0 |
| riir-train | 487 | 302 | 260 | **38** | 3 |
| riir-clippy | 83 | 53 | 4 | **34** | 15 |
| riir-dapps | 30 | 25 | 0 | **25** | 0 |
| riir-neuron-db | 86 | 53 | 37 | **11** | 2 |
| riir-mmorpg-examples | 39 | 31 | 17 | **4** | 2 |
| riir-dao / riir-viewbridge | 7 / 8 | 1 / 1 | 0 / 0 | **1 / 1** | 0 / 0 |

### 141 of the 430 are load-bearing by name

Targets whose filename says `goat`, `gate`, `g<N>`, `drill`, `invariant`,
`guard`, `pin`, `proof`, `conservation`, `safety`, `security` or `audit` — the
ones whose green is the evidence for a promotion or a claim:

| repo | load-bearing SILENT-NOW |
|---|---|
| katgpt-rs | 43 |
| riir-ai | 43 |
| riir-clippy | 18 |
| riir-chain | 15 |
| riir-train | 15 |
| riir-game-sdk | 4 |
| riir-neuron-db | 3 |

## Verified, not inferred

The claim was checked by running the command, in three repos, not deduced from
the manifests:

- `riir-dapps` `cargo test --test content_vessel` → `running 0 tests` /
  `ok. 0 passed` / **exit 0**. That target is the Issue 005 T4 content-vessel
  **GOAT gate**, gated on `#![cfg(feature = "content_vessel")]`.
- `riir-game-sdk` `cargo test -p riir-e2e --test game_anticheat` → same, exit 0.
- Two-sided, on the fixed side: `riir-clippy` `cargo test --test
  t063_tpr_structure` → `error: ... requires the features: tpr_structure`,
  **exit 101**; and with the feature, `1 passed`.

## The instrument's own failure mode is pinned

`selftest()` runs on **every** invocation. Without it a regex regression makes
the audit recognise fewer gates and still print a confident low number — the
exact failure it exists to catch, committed by the tool that catches it. Six
shapes are pinned, and two matter specifically:

1. **Balanced-paren scan, not a regex.** `#![cfg(all(feature = "a", feature =
   "b"))]` is the common shape and a non-greedy `\)` stops at the first inner
   paren, reporting one feature where there are two. Pinned by asserting both
   come back.
2. **An empty `[features]` table yields an empty default set**, so every gated
   target in such a crate is severe rather than latent. The wrong way round
   would silently downgrade a whole crate's findings.

## Tasks

- [x] **T1** Build the instrument, derive the population, verify the claim by
  execution in ≥2 repos, and verify the fixed side too. LANDED 2026-09-03.
- [ ] **T2** Fix katgpt-rs's **43 load-bearing** SILENT-NOW targets: add the
  `[[test]]` / `[[bench]]` rows with `required-features`. Two-sided verify each
  batch — errors without the features, and the same test count with them.
  **The count must not move.** A target that starts failing once it actually
  runs is a separate finding and gets its own issue, not a silent revert.
- [ ] **T3 (owner call, sibling repos)** riir-ai 43 / riir-clippy 18 /
  riir-chain 15 / riir-train 15 / riir-game-sdk 4 / riir-neuron-db 3.
  Deliberately NOT done from here: adding `required-features` converts a silent
  green into a **loud red** wherever CI invokes those targets by name without
  the features, which is the point, but it is the owning repo's call when to
  take that. The report names every path and prints the exact row to add.
- [ ] **T4** Decide whether this becomes a `docs_gate.sh` `CHECKS` entry for
  katgpt-rs only (it has a single checkout in CI, so the cross-repo sweep would
  derive an empty population and print a confident green over zero repos — the
  same reason `docs_drift_sweep.py` is deliberately excluded). A committed
  floor file, in the `docs_drift_floors.txt` idiom, is the shape that works.
- [ ] **T5** The 21 platform-`cfg` targets are correctly gated and unfixable by
  `required-features`. Whether they need a *different* instrument (a per-target
  "did this run on any CI platform?" question) is a separate, unasked question.
  Recorded so it is not mistaken for done.

## Why this is not `feature_isolation_gate.py` or `ci_feature_guard.sh`

Both were checked first (substrate-first). They mention `required-features`
only incidentally — `feature_isolation_gate.py` in a comment about `name = [..]`
not being unique to `[features]`, and `ci_feature_guard.sh` to document one
`any(...)` example that deliberately declares none. Neither asks whether a
`#![cfg]`-gated target declares one. That question was unmeasured.

## Related

- riir-train `5821cba9` — the same defect, one target, found by hand.
- riir-clippy `19beece` — the same defect, one target, found by hand.
- `.issues/705` — the full gate's first two CI runs passed over ZERO compiled
  units. Same family: **an instrument that cannot fail is not passing.**
- `.issues/706` — three repos' whole compile surface in a workflow nothing
  started. Same family, one level up.
