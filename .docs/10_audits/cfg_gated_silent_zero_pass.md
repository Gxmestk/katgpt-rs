# `#![cfg]`-gated targets that report a green `0 passed` — measurement record (Issue 713, closed 2026-09-03, file removed 2026-09-03)

Status: **historical record.** Every katgpt-rs task of Issue 713 landed; the
one remaining row (**T3**, arming sibling repos' load-bearing targets) is an
owner call per repo and is carried here as the table those owners should work
from. Recover the full narrative with `git log --all -- '.issues/713_*.md'`
(last revision `651f118a`).

**Fix commits:** `4509b7d8` (instrument + first measurement) · `a2c8aa71`
(over-count by 48 fixed, pinned) · `180be9c5` + `1e4a52a` (T2: 39 katgpt-rs
gates armed) · `b59b20b3` → `83cb1d56` (T2b run, then its perf reds
**retracted** — sweep was in debug) · `6406f710` (T4: `cfg_gated_floor_gate.py`
+ the trigger-list fix) · `52eef429` (T6: `all_ignored_target_audit.py`; found
Issue 715) · `f84f5c7e` (T6 pin: reasonless `#[ignore]` at 0) · `345e301f` (T5:
the 21 "platform" targets were three classes) · `2272b262` (T4c: classifier
token set widened, 17 more targets armed) · `651f118a` (T3 table re-measured).
Same defect fixed one target at a time earlier: riir-train `5821cba9`,
riir-clippy `19beece`.

**Instruments (all still live):** `scripts/cfg_gated_target_audit.py` (report),
`scripts/cfg_gated_floor_gate.py` + `scripts/cfg_gated_floors.txt` (the gate,
a `docs_gate.sh` check), `scripts/all_ignored_target_audit.py` (report).

## The shape

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off. Cargo prints `running 0 tests` / `ok. 0 passed` and
**exits 0** — byte-for-byte a real pass. `required-features` changes that to
`error: target … requires the features: x` (exit 101). **The `#![cfg]` protects
the COUNT; `required-features` protects the READER.** Both are needed.

Verified by execution, not inferred: riir-dapps `content_vessel` (a GOAT gate)
and riir-game-sdk `game_anticheat` both printed a green zero; on the fixed side
riir-clippy `t063_tpr_structure` errored without its feature and ran `1 passed`
with it.

## Measured — 19 repos, derived population (corrected figures)

| | count |
|---|---|
| targets scanned | 2,897 |
| carrying a whole-file `#![cfg]` | 1,740 |
| declaring `required-features` | 1,016 |
| **SILENT-NOW** — zeroes on a plain `cargo test` | **382** |
| latent — zeroes only under `--no-default-features` | 320 |
| platform `cfg` (cannot be `required-features`) | 21 |
| `any(...)` of features (cargo is AND-only) | 1 |

The five classes **partition** the 1,740 — and the partition held under a
*wrong* classifier (the first cut read 430 SILENT-NOW), so it proves the classes
are exhaustive, not that each is right. **Read the severity split, never the
pooled total:** default-on-gated targets still run on a plain `cargo test`;
default-off-gated ones report a green zero every time anyone names them.

## The lessons that generalise

1. **The auditor over-counted by 48** by keying declared targets on `name`
   against the filename stem, so a `[[test]]` row with an explicit `path` under a
   different name read as undeclared. "Fixing" those added a SECOND target for
   the same file. Found because the T2b sweep returned `NO-RESULT-LINE` on
   exactly those four. Pinned in `selftest()` with a temp crate.
2. **Run perf gates in `--release`.** T2b's first pass reported four reds and
   nearly filed two as perf regressions after three agreeing quiet re-runs
   (`fast_bpe_goat`: 388 s debug vs 15.6 s release). Ruling out the confounder
   you thought of says nothing about the one you didn't. All 39 pass in release.
   The one real find was Issue 714 (an alloc gate counting a sibling test),
   which reproduces in release — see
   [`alloc_gate_per_thread_counter.md`](alloc_gate_per_thread_counter.md).
3. **Two-sided pins.** `cfg_gated_floors.txt` carries ceilings
   (`max_load_bearing = 0`, `max_silent_now`, `max_reasonless_ignores = 0`) AND
   blindness floors (`min_targets`, `min_gated`, `min_ignore_targets`): a
   ceiling alone cannot fail once the instrument goes blind and reports zero.
   Canaried with a throwaway unarmed `*_goat.rs` (red on both ceilings, green
   on removal) and with the all-zeroes synthetic report (must fail both floors).
4. **The trigger list was the real hazard.** `docs_gate.yml`'s `paths` filter
   carried no `.rs` glob, so the gate could not have fired on the one push it
   exists for. Both hand-duplicated lists widened and asserted identical.
5. **A zero ceiling is only as wide as its classifier's vocabulary.** T4b built
   the load-bearing token matcher independently of the published ad-hoc grep;
   five of six disagreements were the matcher's misses (`g16f`/`g2p`/`g9gov`
   variant suffixes, the plural `drills`, the compound `regate`). Agreement
   licensed the pin. **T4c then showed they agreed on the wrong population:**
   the set did not know `*_correctness` / `*_alloc_check` / `*_determinism` /
   `*_equivalence` / `*_floor` / `*_grad_check`. Every candidate token was
   measured against all 2,157 workspace target names first:

   | token | newly classified | verdict |
   |---|---|---|
   | `alloc` | 36 | added (`*_alloc_check` is G4 here) |
   | `floor` | 15 | added (Report-the-Floor mandate) |
   | `grad` | 9 | added |
   | `determinism` | 6 | added |
   | `correctness` | 5 | added |
   | `equivalence` | 3 | added |
   | `soundness` | 1 | added |
   | `budget` | 5 | **rejected** — admits a sweep and a config |
   | `calibration` | 1 | **rejected** — names a record, not an assertion |
   | `check` | 13 | **rejected** — admits any smoke test |

   17 more katgpt-rs targets appeared, were armed, and ran in release: 45/45.
   Silently *unverified*, not broken. **Re-run this table when a new naming
   dialect appears** — a vocabulary gap is indistinguishable from a clean repo.
6. **T5:** the "21 platform-cfg targets" were 11 `not(wasm32)` (the inverse of a
   hazard), 8 `#![cfg(test)]` on integration targets (a **no-op**, confirmed by
   execution — cargo passes `--test`), and **2** riir-ai `wasm32`-only targets
   that no CI platform compiles. No new instrument needed; a two-row riir-ai
   owner call.
7. **T6, the second silent-zero shape:** a target whose every test is
   `#[ignore]`d prints `ok. 0 passed; N ignored` — and `required-features`
   cannot touch it. 244 such targets across 19 repos, 60 load-bearing by name,
   two of them (`bench_octopus_goat`, `bench_block_diagonal_goat`) not
   feature-gated at all. Deliberately a **report**: `#[ignore]` is the correct
   marker for a slow or hardware-gated test. The one defensible pin is *whether
   the source says why*: 27 reasonless `#[ignore]`s in katgpt-rs were given
   their file's already-documented reason (none invented), pinned at 0. The
   instrument had to be built twice — per-test attribute blocks resolved against
   the default closure, with an unsatisfied `any()` whole-file gate reported as
   its own ambiguous class rather than guessed. Executing `test_120_vpd_arena_goat`
   to confirm its count is what found Issue 715's two-day release break.

## T3 — the open owner call, per sibling (re-measured with the widened classifier)

| repo | targets | gated | SILENT-NOW | **load-bearing** | latent |
|---|---|---|---|---|---|
| riir-ai | 930 | 560 | 122 | **41** | 179 |
| riir-clippy | 84 | 54 | 34 | **18** | 15 |
| riir-chain | 170 | 133 | 48 | **16** | 19 |
| riir-train | 487 | 302 | 36 | **15** | 3 |
| riir-game-sdk | 62 | 40 | 37 | **5** | 0 |
| riir-neuron-db | 86 | 53 | 11 | **3** | 2 |
| katgpt-rs | 921 | 541 | 44 | **0** | 100 |
| 12 others | 158 | 58 | 31 | **0** | 4 |

T4c's widening added 17 load-bearing rows in katgpt-rs and only 5 across all
siblings — the dialect it taught the classifier is mostly this repo's own naming
convention. Owners should still **run the script** rather than trust this
table. Adding `required-features` rows does not red an existing `cargo test
--workspace` (it skips targets whose features are off); it turns naming the
target without its features into a loud exit 101, which is the point and the
owning repo's call when to take.

## Why this is not `feature_isolation_gate.py` or `ci_feature_guard.sh`

Both mention `required-features` only incidentally; neither asks whether a
`#![cfg]`-gated target declares one. Related family: `.issues/705` (a gate that
passed over zero compiled units) and `.issues/706` (a compile surface in a
workflow nothing started) — **an instrument that cannot fail is not passing.**
