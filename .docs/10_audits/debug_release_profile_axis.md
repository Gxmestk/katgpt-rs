# The profile is part of the claim — the dev/release axis (Issues 715 + 716, closed 2026-09-03, files removed 2026-09-03)

Status: **historical record, both CLOSED.** Issue 715 (an orphaned `#[cfg]`
binding across a blank line) is fixed and gated at zero; Issue 716 (the full
gate never ran `--release`) is fixed and the axis is `scripts/full_gate.sh`
**Layer 6**. The one open row — 716 **T3**, whether each of the 19 sibling repos
wants a release pass — is an owner call per repo, same shape as Issue 713 T3.
Recover the narratives with `git log --all -- '.issues/715_*.md' '.issues/716_*.md'`
(last revisions `52eef429`, `4b96e0e6`).

**Commits:** 715 — `26d055c6` (introduced: dropped an import, left its attribute
behind) · `7e34ccef` (erased the evidence by deleting the blank line) ·
`a08376a0` (fix, verified in both profiles) · `52eef429`
(`scripts/orphaned_attr_gate.py`, a `docs_gate.sh` check). 716 — `f0696986`
(two release-only breaks fixed) · `f84f5c7e` (Layer 6 liveness signal) ·
`4b96e0e6` (cost figures qualified as M3 Max numbers).

## Three findings on one axis, pointing in different directions

| record | what the profile did |
|---|---|
| Issue 713 T2b | debug **manufactured** four false perf reds (a latency gate on an unoptimised binary) |
| Issue 715 | debug **hid** a two-day release build break |
| Issue 716 | the full gate compiled `debug_assertions` code **only in the profile where it works** |

**Neither profile is the safe default.** A green result is a claim about its
literal command *including the profile*.

## Issue 715 — a `#[cfg]` separated from its item by a blank line still applies

Rust binds an attribute to the next item **across a blank line**. `26d055c6`
dropped `use std::cmp::Ordering;` from `katgpt-pruners/src/sdar/sdar_absorb.rs`
and left its `#[cfg(debug_assertions)]` behind, where it silently re-bound to
the next import. In every release build the import vanished while its five
usages stayed unconditional: 5 × `E0425`, `katgpt-pruners (lib)` uncompilable
under `sdar_gate`. The introducing commit's own validation line was a **debug**
run. Blast radius audited: `git show -U4` over all seven imports that commit
dropped — only this one had an attribute above it.

**The narrowing is the instrument.** Any attribute + blank line + item is
**2,044** sites across 19 repos and is NOT a defect class (dominated by
whole-file inner `#![cfg(...)]`, which bind to the enclosing module and are
conventionally followed by a blank). Restricting to **outer** `#[cfg]` /
`#[cfg_attr]` takes 2,044 to **0**, so the pin is 0 with no floor to negotiate.
Canaried against the real bug (pre-fix tree reconstructed; reported at
`sdar_absorb.rs:47`, exit 1; green on restore). `selftest()` also asserts the
walk **prunes** `target/` rather than filtering after `rglob` (the 556 s trap).
Found only because Issue 713 T6 insisted on *executing* a target rather than
reading it.

## Issue 716 — the full gate's fifth blind spot

`cargo clippy --workspace --all-targets --all-features --keep-going` runs in the
**dev** profile, so `debug_assertions` is always ON. Measured 2026-09-03 with
`--release`: **2 errors, both `E0432`**, 277 units compiled (not vacuous) —
`latent_confounder_audit.rs` and `orthogonal_factorization.rs` had `#[cfg(test)]`
blocks importing `crate::alloc`'s counters, which `alloc.rs` gates on
`debug_assertions` *by design* and documents at length. **Consequence:**
`cargo test --release -p katgpt-core --lib` did not compile at all — the very
command Issue 713 T2b tells everyone to run.

14 files reference those counters; the compiler proved exactly 2 were wrong. A
`grep -B6` for an enclosing `#[cfg]` missed 7 of the 12 correct ones, so the
narrow instrument would have said "7 more are broken".

**Fixed** by gating both tests `#[cfg(debug_assertions)]`, joining 12 siblings
in the crate already gated that way. **Deliberately NOT** release no-op stubs
returning 0: a zero-alloc assertion against a stubbed counter passes vacuously
(`.issues/705` and Issue 714 exactly). The cost, enumerated by name rather than
inferred from the delta: release runs **26 fewer** katgpt-core lib tests
(4,609 → 4,583) — 6 `alloc.rs` own tests, 14 `g4_*` alloc gates, 6
`*_panics_in_debug`; the reverse set is empty.

### Layer 6 — `cargo check --release`, deliberately not clippy

The axis is **compilation** with `debug_assertions` off; the lint surface is
covered by the dev pass. Not folded into `GATE_ARGS`, because Layer 5 asserts
AGENTS.md quotes that string verbatim. Cost measured before deciding: warm 27 s,
cold 65 s (394 units, 622 MB) — ~8% of a >13 min weekly gate, **both M3 Max /
16-core figures**; a GitHub macOS runner has ~1/4 the cores, so expect minutes.

**Its first in-situ run found a bug in itself.** Counting `Compiling`/`Checking`
lines reported INCONCLUSIVE on a warm tree (cargo compiled 0 units, printed
nothing) — *freshness must not decide a liveness verdict*. Fixed by counting
`--message-format=json` `compiler-artifact` records, emitted for fresh units too
(3 `Checking` lines vs 1,423 artifacts on the same tree), and immune to the
ANSI-colour trap that zeroed every anchored counter in `.issues/705`. Canaried
two-sided on the real tree: 1,423 / 0 → PASSED; one break reintroduced → 1,422
/ 1 → FAILED with the rendered `E0432`; restored → PASSED.

### 716 T3 — open owner call

None of the 19 repos runs a release pass. Whether each wants one is its owner's
call; the katgpt-rs Layer 6 is the reference shape.
