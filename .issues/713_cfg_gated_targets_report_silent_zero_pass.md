# Issue 713 — 430 cargo targets can report a green `0 passed` having run nothing

**Status:** T1 **LANDED** (instrument + measurement, 2026-09-03). T2 **LANDED**
(39 katgpt-rs load-bearing targets armed, `180be9c5` + `1e4a52a`). T2b RUNNING
(the armed gates' actual verdicts). T3–T5 are owner calls in sibling repos,
deliberately not made here.

> **CORRECTED 2026-09-03, same day, by the fix itself.** The first published
> figures were **over-counted**: the auditor keyed a declared target by its
> `name` matched against the file's stem, so a row carrying an explicit `path`
> under a different name read as undeclared. katgpt-rs's four `*.goat.rs`
> targets are declared as `bench_256_kv_outer_goat` (underscored) pointing at
> `bench_256_kv_outer.goat.rs` (dotted), **with `required-features` already
> present**. All four were reported as defects, and "fixing" them added a
> SECOND target for the same file — which cargo warns about and which breaks
> `--test <name>` resolution. Found because the sweep in T2b returned
> `NO-RESULT-LINE` on exactly those four and nothing else.
>
> The corrected numbers are below and every figure in this issue is the
> corrected one. **SILENT-NOW 430 → 382; `w/ req-f` 956 → 1,016; latent
> 332 → 320.** The classifier still partitions (1,016 + 382 + 320 + 21 + 1 =
> 1,740), which is exactly why the partition assertion was worth stating: it
> held under a wrong classifier, so it proves the classes are exhaustive and
> **not** that each one is right.
>
> `selftest()` now builds a temp crate with a `path`-declared target and
> asserts it reads as covered. The bug is pinned, not just fixed.

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
| of those, declaring `required-features` | 1,016 |
| **SILENT-NOW** — zeroes on a plain `cargo test` | **382** |
| latent — zeroes only under `--no-default-features` | 320 |
| platform `cfg` — `required-features` *cannot* express | 21 |
| `any(...)` of features — cargo's `required-features` is AND-only | 1 |

The last five rows **partition** the 1,740: 1,016 + 382 + 320 + 21 + 1 = 1,740.
Stated because a classifier that silently drops a case reports a smaller
defect count and looks like good news.

**The severity split is the point.** Pooled at 702 the number is unreadable: a
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
| riir-ai | 930 | 560 | 253 | **122** | 179 |
| katgpt-rs | 921 | 541 | 376 | **63** | 100 |
| riir-chain | 170 | 133 | 66 | **48** | 19 |
| riir-game-sdk | 62 | 40 | 1 | **37** | 0 |
| riir-train | 487 | 302 | 262 | **36** | 3 |
| riir-clippy | 83 | 53 | 4 | **34** | 15 |
| riir-dapps | 30 | 25 | 0 | **25** | 0 |
| riir-neuron-db | 86 | 53 | 37 | **11** | 2 |
| riir-mmorpg-examples | 39 | 31 | 17 | **4** | 2 |
| riir-dao | 7 | 1 | 0 | **1** | 0 |
| riir-viewbridge | 8 | 1 | 0 | **1** | 0 |

### 93 of the 382 are load-bearing by name

Targets whose filename says `goat`, `gate`, `g<N>`, `drill`, `invariant`,
`guard`, `pin`, `proof`, `conservation`, `safety`, `security` or `audit` — the
ones whose green is the evidence for a promotion or a claim:

| repo | load-bearing SILENT-NOW |
|---|---|
| riir-ai | 40 |
| riir-clippy | 18 |
| riir-chain | 15 |
| riir-train | 13 |
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
- [x] **T2 LANDED 2026-09-03** — armed **39** katgpt-rs load-bearing SILENT-NOW
  targets (`180be9c5`; 43 rows added, 4 reverted as the false positives above).
  Baseline verified by running the FIXED auditor against the pre-fix commit in
  a throwaway worktree rather than inferring it from the delta: **102 → 63**.
  Original text follows.

  Fix katgpt-rs's load-bearing SILENT-NOW targets: add the
  `[[test]]` / `[[bench]]` rows with `required-features`. Two-sided verify each
  batch — errors without the features, and the same test count with them.
  **The count must not move.** A target that starts failing once it actually
  runs is a separate finding and gets its own issue, not a silent revert.
- [x] **T2b RAN 2026-09-03 — all 39 armed gates PASS. One real defect found,
  and it was not a perf one.**

  > **CORRECTION, and it retracts the first version of this row.** T2b's first
  > pass reported four reds and called two of them "real" perf findings after
  > re-measuring them three times on a quiet box. **All four pass.** The sweep
  > script ran `cargo test` without `--release`, and a latency gate in a debug
  > build measures an unoptimised binary. `fast_bpe_goat` takes **388 s in
  > debug and 15.6 s in release** — that is the size of the error.
  >
  > I had even tested and rejected the load-flakiness hypothesis, correctly, by
  > re-measuring quietly three times and getting a sub-1% spread. That made the
  > false reading *more* convincing, not less: a stable wrong number looks
  > exactly like a real finding. Ruling out the confounder I thought of did
  > nothing about the one I hadn't. **`bench_668` was one commit away from
  > being filed as a 4.3× perf regression that does not exist.**
  >
  > A related gate-hygiene note, which is why this was easy to fall into:
  > `plan414`'s G5 documents itself *"release mode only — debug builds are
  > unoptimized"*. `bench_668`'s and `fast_bpe_goat`'s latency gates carry no
  > such marker and no `debug_assertions` guard, so in debug they fail
  > confidently instead of skipping or saying why.

  Final verdict, **in release**:

  | gate | debug (wrong) | **release (correct)** |
  |---|---|---|
  | `bench_668_effective_degree_goat` | 2145 ns vs 500 ns bound | **ok, 7 passed** |
  | `fast_bpe_goat` | 2311× vs ≤1000× bound | **ok, 8 passed** |
  | `bench_145_binary_plasma_goat` | 1.05× vs ≥1.2× | **ok, 5 passed** |
  | `plan414_hla_committed_belief_probe_goat` | 6 allocs/1000 | **ok** (after `.issues/714`) |

  **`.issues/714` survives this correction intact and is the real find.**
  Re-tested at the pre-fix commit, **in release**, three runs: `8`, `3`, `3`
  allocs in 1000 calls — failing, and failing with a *varying* count, which is
  the signature of a concurrent contributor rather than a code path that
  allocates. Post-fix it is green. So the alloc-counter race is real in the
  optimised build and has nothing to do with the debug error above.

  So Issue 713's own claim lands where it should: not "the armed gates were
  broken" — they were not — but **"the verdicts were invisible."** Arming them
  surfaced one genuine harness defect that had made an alloc gate report a
  number it had not measured.

- [ ] ~~**T2c** File the two genuine perf findings~~ **WITHDRAWN** — there are
  no perf findings. See the correction above. Filing them is exactly what the
  correction prevented.

- [ ] **T3 (owner call, sibling repos)** The load-bearing table above, minus
  katgpt-rs (done in T2). Read the numbers from the table, not from here.
  Deliberately NOT done from here: adding `required-features` converts a silent
  green into a **loud red** wherever CI invokes those targets by name without
  the features, which is the point, but it is the owning repo's call when to
  take that. The report names every path and prints the exact row to add.
- [ ] **T4** Decide whether this becomes a `docs_gate.sh` `CHECKS` entry for
  katgpt-rs only (it has a single checkout in CI, so the cross-repo sweep would
  derive an empty population and print a confident green over zero repos — the
  same reason `docs_drift_sweep.py` is deliberately excluded). A committed
  floor file, in the `docs_drift_floors.txt` idiom, is the shape that works.
- [ ] **T6 (new axis, observed during T2b — NOT yet measured)** A **second**
  way a target prints a green zero, found by the sweep rather than by the
  auditor: `test_120_vpd_arena_goat` runs under its features and reports
  `ok. 0 passed; 0 failed; 3 ignored`. Every test in it is `#[ignore]`d.

  This is **not automatically a defect** — `#[ignore]` is the correct marker
  for a slow or hardware-gated test, and that is why it must not be folded into
  the SILENT-NOW count. But the reader-facing output is the same lie: a green
  `ok` over zero executed assertions, on a target named `_goat`.

  The distinction worth measuring is between a target with *some* ignored tests
  (normal) and one where **every** test is ignored, so the binary can never
  report anything but zero. The latter is the same shape as this issue one
  level in, and nothing counts it. Deliberately left unmeasured rather than
  guessed at.

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
