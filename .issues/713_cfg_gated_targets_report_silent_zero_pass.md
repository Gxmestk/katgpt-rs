# Issue 713 — 430 cargo targets can report a green `0 passed` having run nothing

**Status:** T1 **LANDED** (instrument + measurement, 2026-09-03). T2 **LANDED**
(39 katgpt-rs load-bearing targets armed, `180be9c5` + `1e4a52a`). T2b **RAN** —
all 39 pass in release; the debug reds were retracted (`83cb1d56`). T2c
WITHDRAWN. T4 + T4b **LANDED** — `cfg_gated_floor_gate.py` is now a
`docs_gate.sh` check with four two-sided pins, canaried, and the load-bearing
classifier ships in the auditor. T4c **LANDED** — the load-bearing classifier's
TOKEN SET was itself a blind spot (`max_load_bearing = 0` was a green over a
population it could not see): seven tokens added after a 2,157-target
measurement, **17 more katgpt-rs targets armed and run (45/45 pass)**,
SILENT-NOW 61 → 44. **T3 is an owner call in each sibling repo**,
deliberately not made here. T6 **MEASURED** — 244 all-`#[ignore]`d
targets across 19 repos (60 load-bearing); it stays a report, and it found the
release-build break in `.issues/715`. T5 **MEASURED** — the "21 platform"
targets were 3 pooled classes; only **2** (both riir-ai wasm32) are a real
coverage gap, and no new instrument is needed. **All tasks are now closed
except T3**, the sibling-repo owner call.

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

- [x] **T4b (fell out of T4) — the load-bearing classifier now ships in the
  auditor, and building it found SIX targets the published table's ad-hoc grep
  and my first token matcher disagreed about.**

  The load-bearing column in the table above was produced by a one-off
  substring grep. T4 needed the classification *inside* the auditor (the gate
  must not re-derive it — a second copy of a classifier is a second thing to
  keep in step, and the one that drifts is silently the more permissive one).
  Written independently, the token matcher returned **87**, not 93.

  Substring matching is wrong in the noisy direction: `gate` claims
  `aggregate`, `delegate`, `propagate`, `mitigate`, `investigate`. So the
  matcher splits on `[^a-z0-9]+` — which must include `.`, for the
  `bench_256_kv_outer.goat.rs` dialect. But **five of the six disagreements
  were real misses in the token matcher**, not false positives in the grep:

  | target | shape the token matcher dropped |
  |---|---|
  | `block_producer_g16f_cost.rs` | G\<N\> with a variant suffix (`g16f`) |
  | `kat_promotion_g2p.rs` | same (`g2p`) |
  | `kat_stake_client_g2s.rs` | same (`g2s`) |
  | `kat_vote_client_g9gov.rs` | same (`g9gov`) |
  | `prod_l3_sigkill_drills.rs` | plural (`drills`) |
  | `t40_fixer_regate_harness.rs` | compound (`regate`) — **kept as a named token** |

  Fixed by `^g\d+[a-z0-9]*$` for the ordinals, depluralising the stem rather
  than hand-listing plurals (`drills` was the miss that showed hand-listing
  fails), and adding `regate` as an **explicit compound** — trading one named
  token for the five false positives a substring rule would re-admit is the
  right way round. All six are pinned in `selftest()`, along with eight
  substring false positives asserted NOT load-bearing.

  With those, the auditor reproduces the published table **exactly**: 40 / 18 /
  15 / 13 / 4 / 3 = **93**, per repo identical. Two independently-built
  classifiers agreeing is worth more than either one's number, and the
  agreement is what licenses `max_load_bearing = 0` as a pin — a false negative
  in the classifier would have turned that pin into a permanent green.

- [x] **T4c LANDED 2026-09-03 — `max_load_bearing = 0` was a green over a
  population the CLASSIFIER COULD NOT SEE.** Found sideways: Plan 580 T5.3
  added tests to `certified_frontier_correctness`, and arming it revealed that
  neither it (31 assertions) nor `bench_688_certified_frontier_alloc_check`
  (an alloc budget) had a `[[test]]` row — both reported `ok. 0 passed` when
  named without the feature. Neither was in T2's armed 39, and the reason was
  not that T2 missed them: **`is_load_bearing` did not classify them.** The
  token set knew `goat`/`gate`/`g<N>`/`drill`/`proof`/`invariant`/`guard`/
  `conservation`/`safety`/`security`/`audit` and did not know the
  `*_correctness` / `*_alloc_check` / `*_determinism` dialect.

  **Every candidate token was measured against all 2,157 workspace test+bench
  target names before being added**, and the measurement is why three were
  rejected:

  | token | newly classified | verdict |
  |---|---|---|
  | `alloc` | 36 | ADDED — `*_alloc_check` is G4 in this repo's own convention |
  | `floor` | 15 | ADDED — `conformal_floor_*`, the Report-the-Floor mandate |
  | `grad` | 9 | ADDED — `*_backward_grad_check` |
  | `determinism` | 6 | ADDED |
  | `correctness` | 5 | ADDED |
  | `equivalence` | 3 | ADDED |
  | `soundness` | 1 | ADDED |
  | `budget` | 5 | **rejected** — admits a sweep (`bench_578_mcts_budget_sweep`) and a config (`game_budgets`) |
  | `calibration` | 1 | **rejected** — names a measurement record, not an assertion |
  | `check` | 13 | **rejected** — admits any smoke test |
  | `coverage` / `regression` / `bound` | 0 | rejected, matched nothing new |

  Consequence in katgpt-rs: **17 more load-bearing SILENT-NOW targets** — 8
  `*_alloc_check` G4 budgets, 4 `*_backward_grad_check`, a Report-the-Floor UQ
  gate, a determinism gate, a checkpoint-equivalence gate, two grad gates. All
  17 armed and then **RUN in release: 45 assertions, 45 pass, 0 fail.** Same
  finding as `.issues/714` T3 — silently **unverified**, not silently broken,
  which is precisely why nothing surfaced them. SILENT-NOW 61 → **44**, covered
  378 → **395**, `max_silent_now` re-pinned.

  `selftest()` pins the new dialect in **both** directions: seven real target
  names that read as not-load-bearing before, and four substring traps
  (`allocator_pressure_bench`, `flooring_math`, `gradient_descent_driver`,
  `determine_route`) that must stay excluded — the same discipline that keeps
  `aggregate`/`delegate` out of `gate`.

  **The transferable lesson, and it generalises past this gate:** a ceiling of
  zero over a classifier is only as wide as that classifier's vocabulary, and a
  vocabulary gap is indistinguishable from a clean repo. T4b established that
  two independently-built classifiers agreeing licenses the pin; T4c is the
  reminder that they can agree on the wrong *population*. The workspace-wide
  token table above is the population check, and it should be re-run whenever a
  new naming dialect appears.

  **Not extended to siblings**, deliberately — the widened classifier reports
  63 more load-bearing SILENT-NOW targets outside katgpt-rs, which is T3's
  scope and T3's owners' call.

- [ ] **T3 (owner call, sibling repos)** The load-bearing table above, minus
  katgpt-rs (done in T2). Read the numbers from the table, not from here.
  Deliberately NOT done from here: adding `required-features` converts a silent
  green into a **loud red** wherever CI invokes those targets by name without
  the features, which is the point, but it is the owning repo's call when to
  take that. The report names every path and prints the exact row to add.
- [x] **T4 LANDED 2026-09-03 — yes, katgpt-rs-scoped, with a committed pin
  file.** `scripts/cfg_gated_floor_gate.py` + `scripts/cfg_gated_floors.txt`,
  wired into `docs_gate.sh`'s `CHECKS` (now 5/5 green).

  **The auditor stays a report and the gate is a separate file.** The auditor
  must be runnable over the 18 siblings whose owners have not taken T3; an
  auditor that exits 1 on those is an auditor nobody runs. The verdict lives in
  a katgpt-rs-scoped consumer of its `--json`.

  **Four pins, two-sided.** A ceiling alone cannot fail once the instrument
  dies — an auditor whose regex stops recognising `#![cfg(...)]` reports
  SILENT-NOW 0 and passes every ceiling ever written. That is `.issues/705`
  (the full gate's first two CI runs passed over ZERO compiled units), and it
  is designed against here rather than discovered later:

  | pin | value | direction |
  |---|---|---|
  | `max_load_bearing` | **0** | ceiling — the sharp one |
  | `max_silent_now` | 63 | ceiling — the ratchet |
  | `min_targets` | 700 | **floor** — blindness |
  | `min_gated` | 400 | **floor** — blindness |

  The floors are generous (~75% of observed) on purpose: they exist to catch
  "the auditor stopped seeing anything", not to police churn. An exact floor
  reds on every legitimate test-file removal and is then ignored, which
  `docs_drift_floors.txt`'s header already argues at length.

  **Canaried, not assumed.** A temporary `tests/zz_713_canary_probe_goat.rs`
  gated on a nonexistent feature made the gate red on both ceilings
  (`load-bearing 1 > 0`, `SILENT-NOW 64 > 63`, exit 1) and green again on
  removal. The verdict logic is *also* driven over every boundary by
  `selftest()` with synthetic measurements — including the all-zeroes blind
  report, which satisfies both ceilings and must fail on both floors — because
  a gate whose failing direction is only reachable via the real corpus is a
  gate whose failing direction is never exercised.

  **The trigger list was the real hazard.** `docs_gate.yml`'s `paths` filter
  listed docs and manifests and **no `.rs` at all**, so the gate could not have
  fired on the one push it exists for: the push that ADDS an unarmed
  `*_goat.rs`. Six target-source globs plus the three new script/pin paths were
  added to **both** hand-duplicated lists (23 entries each, asserted identical;
  Actions does not support YAML anchors and a local `safe_load` resolves them,
  so they cannot be shared). This is `.issues/704`/`706` one level in — a
  workflow is identical on disk whether or not it can see the change it gates.
  Cost measured before widening: **0.20 / 0.20 / 0.18 s** over 921 targets.
- [x] **T6 MEASURED 2026-09-03 — `scripts/all_ignored_target_audit.py`. 244
  targets across 19 repos can never print anything but `ok. 0 passed`, 60 of
  them load-bearing by name. Still a REPORT, and here that is a stronger claim
  than usual.**

  The second axis, found by the T2b sweep rather than by the auditor:
  `test_120_vpd_arena_goat` runs under its features and prints `ok. 0 passed;
  0 failed; 3 ignored`. Every test in it is `#[ignore]`d. This is **not** the
  Issue 713 shape — its `#![cfg]` is satisfied, the binary is not empty — and
  `required-features` cannot address it. The reader-facing output is the same
  lie.

  **Deliberately NOT folded into SILENT-NOW.** That count's whole value is that
  every member is fixable by a three-line manifest row. `#[ignore]` is the
  *correct* marker for a slow or hardware-gated test, so no pin is defensible
  here; the measurement is the deliverable.

  | repo | targets | w/ tests | **ALL-IGNORED** | load-bear | partial | no-tests |
  |---|---|---|---|---|---|---|
  | katgpt-rs | 653 | 460 | **19** | 3 | 14 | 13 |
  | riir-ai | 770 | 613 | **197** | 51 | 29 | 9 |
  | riir-chain | 148 | 130 | **5** | 0 | 14 | 1 |
  | riir-clippy | 68 | 63 | **3** | 0 | 2 | 0 |
  | riir-mmorpg-examples | 35 | 35 | **2** | 0 | 2 | 0 |
  | riir-neuron-db | 77 | 38 | **1** | 1 | 1 | 0 |
  | riir-train | 274 | 245 | **17** | 5 | 25 | 8 |

  **The reasons are the diagnosis; the count is not.** 443 ignored tests over
  180 distinct reason strings:

  | n | reason |
  |---|---|
  | **65** | **(NO REASON GIVEN)** |
  | 48 | requires a GPU; run explicitly with `--release` |
  | 25 | pure measurement benchmark (no assertions), slow in debug |
  | 11 | PoC bench — requires Gemma GGUF |
  | 11 | benchmark — run with `--ignored` |
  | 8 | requires Gemma 2 2B GGUF + tokenizer.model |
  | 8 | requires the 7.1 GB model + GPU |
  | 7 | **GOAT gate — run with `--ignored`** |

  Most of the mass is legitimate and self-documenting (GPU, 7.1 GB weights,
  slow measurement). Two rows are not: **65 ignored tests whose source says
  nothing about why** — no reader can distinguish a deliberate manual-only test
  from one parked during a refactor and forgotten — and **7 that call
  themselves a GOAT gate** while running only if someone remembers `--ignored`.
  If any part of this becomes a pin later, "an empty `#[ignore]` reason on a
  load-bearing target" is the defensible one, not the count.

  **The instrument had to be built twice, and cargo settled it both times.**
  A file-wide `#[ignore]`-vs-`#[test]` count is not good enough:

  1. `bench_octopus_goat` has **9** `#[test]` attributes; cargo prints **8
     ignored**. Three tests are individually `#[cfg(feature = ...)]`-gated and
     one gating feature is default-off. So attributes are now associated
     **per test** (contiguous attribute blocks) and resolved against the
     crate's default closure.
  2. `bench_block_diagonal_goat` then read **9** where cargo said **8**. Its
     whole-file gate is `#![cfg(any(planar_quant, iso_quant, hybrid_oct_pq))]`,
     already satisfied by `planar_quant` — and unioning in *all* of a whole-file
     gate's features wrongly enabled `iso_quant`, counting a per-item
     `iso_quant` test as compiled. Fixed by resolving three cases separately
     (satisfied gate → add nothing; unsatisfied `all()`/single → add its
     features; unsatisfied `any()` → **ambiguous**, reported as its own class
     rather than guessed).

  All three load-bearing katgpt-rs rows now match cargo **by execution**:

  | target | auditor | cargo |
  |---|---|---|
  | `bench_octopus_goat` | 9 in source, 8 compiled, 8 ignored | `ok. 0 passed; 8 ignored` |
  | `bench_block_diagonal_goat` | 11 in source, 8 compiled, 8 ignored | `ok. 0 passed; 8 ignored` |
  | `test_120_vpd_arena_goat` | 3, 3, 3 | `ok. 0 passed; 3 ignored` |

  Note what the first two are: **not feature-gated at all**. A plain
  `cargo test` compiles and runs them, and they print a green zero
  unconditionally — no flag to notice, which is arguably worse than the Issue
  713 shape.

  **The bias runs BOTH ways**, and the first draft of this file claimed
  otherwise: macro-generated tests (`test_case`, `rstest`) are invisible to an
  attribute count (toward false positives), and unresolvable `cfg` predicates
  (`not(...)`, platform gates) resolve to compiled (toward false negatives).
  Every row is a hypothesis to check by running the target — the cheapest
  verification there is, since these targets execute nothing.

  **It also found a two-day release-build break: `.issues/715`.** Confirming
  `test_120`'s `3 ignored` required *executing* it under its own features, and
  it would not compile — an orphaned `#[cfg(debug_assertions)]` bound across a
  blank line to the wrong import, breaking every release build of `sdar_gate`
  since `26d055c6`. Fixed in `a08376a0`, class gated at zero by
  `scripts/orphaned_attr_gate.py`.

  **T6 now carries the one pin its axis defensibly supports.** Not the count —
  `#[ignore]` is correct for a slow or hardware-gated test — but *whether the
  source says why*. **27 reasonless `#[ignore]`s across 8 katgpt-rs targets**
  were given their own file's **documented** reason, which in every case
  already existed in a doc comment far from the attribute, where a reader of
  cargo's output never sees it:

  ```
  test go_integration::board_state_is_consistent ... ignored, requires a
  running AutoGo server (scripts/autogo_server.sh); run with --features go --ignored
  ```

  No reason was invented. Where a file documented none (`issue043`,
  `test_120`), the reason states what the file demonstrably **is** — measured,
  not guessed: `velocity_field_disagreement_uq_floor.rs` got *"prints a
  comparison table (0 assertions, 41 println)"* because that is what a count of
  its assertions returned. A wrong reason is worse than none: it stops the next
  reader checking.

  Pinned at **0** in `cfg_gated_floors.txt` (`max_reasonless_ignores`, with
  `min_ignore_targets` as its blindness floor — a ceiling of zero is satisfied
  perfectly by a parser that sees nothing). Canaried: a bare `#[ignore]` in a
  throwaway target reds the gate and its removal greens it.

  **A third silent-zero shape was measured and is NOT worth an instrument.**
  A load-bearing target with **zero verdict expressions** (`assert*!`,
  `panic!`, `unreachable!`, `.expect`, `.unwrap`) can never fail even when it
  runs. katgpt-rs: **2** — `bench_483_lt2_loop_stable_goat` and
  `bench_octopus_goat`, both self-described measurement benchmarks (the
  latter's `#[ignore]` reason literally says *"pure measurement benchmark (no
  assertions)"*). Measured rather than assumed, and the answer is small and
  benign, so it gets a sentence here instead of a gate.

  **`#![cfg(test)]` on an integration target is confirmed a NO-OP** by
  execution, not by reasoning about cargo's flags: `cargo test -p katgpt-core
  --features personality_composition --test
  personality_composition_integration_check` → `1 passed`. cargo passes
  `--test`, so `cfg(test)` holds. That closes the 8-target `cfg(test)` class
  in T5 as decorative.

  **NO-TESTS (31 workspace-wide) is reported apart** — a file under `tests/`
  with no test attribute and cargo's own harness. `harness = false` targets are
  excluded: a custom-harness target legitimately has no `#[test]` and its exit
  code is its verdict, and including them would make the report mostly noise.

- [x] **T5 MEASURED 2026-09-03 — the "21 platform-`cfg` targets" were THREE
  unrelated things pooled under one label, and only 2 of the 21 are a coverage
  question at all.**

  Enumerated, not sampled:

  | class | n | what it means |
  |---|---|---|
  | `not(target_arch = "wasm32")` | **11** | compiles **everywhere except** one arch — the *inverse* of a coverage hazard |
  | `target_arch = "wasm32"` | **2** | compiles **only** there |
  | `#![cfg(test)]` | **8** | a **no-op**: cargo passes `--test` to integration targets, so `cfg(test)` always holds |

  The negated and positive platform gates differ by the single token `not(`
  and are **opposite in severity**. Pooling them produced a number that meant
  nothing — 11 of the 21 run on every ordinary CI runner and were never at
  risk. The auditor now splits the class three ways (`plat-only` / `plat-exc` /
  `cfg(test)` columns), with each case pinned in `selftest()`; the split moves
  no target between the defect and non-defect classes, so **SILENT-NOW stays
  382** and the partition still holds.

  **The answer to the original question is 2 targets, both in riir-ai:**

  - `crates/riir-engine/tests/browser_e2e_inference.rs`
  - `crates/riir-engine/tests/wasm_simd_bench.rs`

  Neither compiles on any CI platform. Verified rather than assumed: riir-ai
  has exactly 3 workflows (`rust.yml`, `docs_drift.yml`, `lean_proofs.yml`) and
  **none builds for `wasm32`** — the single `wasm` match in them is a comment
  about a `.wasm` artifact, not a `--target`. So **no new instrument is
  needed**; the question just needed answering, and its answer is a two-row
  owner call for riir-ai (same shape as T3).

  Predicate detection was also substring-based (`"test" in body`) and is now
  token-based (`\btest\b`), so a feature named e.g. `fastest_path` can no
  longer be reported as carrying the `test` predicate.

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
