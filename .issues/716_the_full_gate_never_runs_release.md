# Issue 716 — the full gate has a fourth blind spot: it never runs `--release`

**Status:** T1 + T2 **LANDED 2026-09-03.** The two release-only compile breaks
are fixed (`cargo test --release -p katgpt-core --lib` compiles again — it did
not compile at all), and the profile axis is now `full_gate.sh` **Layer 6**
(cold cost 65 s, canaried over three real logs). T3 (siblings) is an owner call
per repo.

## The axis

`AGENTS.md` documents three independent ways the repo's build commands are
narrow, and `scripts/full_gate.sh` is the assertion that closes them:

```
cargo clippy --workspace --all-targets --all-features --keep-going
```

There is a **fourth**, and that command does not close it: it runs in the
**dev** profile, so `debug_assertions` is always **ON**. Every item behind
`#[cfg(debug_assertions)]` — and every item that *depends* on one — is compiled
by the full gate in the only configuration where it works.

| axis | blind spot | closed by |
|---|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes | `full_gate.sh` |
| default vs `--all-features` | gated code compiles to nothing | `full_gate.sh` |
| `-p <crate>` vs `--workspace` | root defaults switch on a crate's own feature | `full_gate.sh` |
| no `--all-targets` | skips every test / bench / example | `full_gate.sh` |
| **dev vs `--release`** | **`debug_assertions` code is only ever compiled ON** | **nothing, until T2** |

## Measured, 2026-09-03

```bash
cargo clippy --workspace --all-targets --all-features --keep-going --release
```

**2 errors, both `E0432`, both fatal to `katgpt-core (lib test)`** — 277 units
compiled, so this is not a vacuous run:

| file | line |
|---|---|
| `crates/katgpt-core/src/latent_confounder_audit.rs` | 455 |
| `crates/katgpt-core/src/orthogonal_factorization.rs` | 1001 |

Both are `#[cfg(test)]` blocks importing `crate::alloc::{get_alloc_stats,
reset_alloc_stats}` — which `alloc.rs` gates on `debug_assertions` **by
design**, at length, in a doc comment that even explains the trap:

> the `tests` module is therefore gated on `cfg(all(test, debug_assertions))`
> so a `--release` test build (where `cfg(test)` is on but `debug_assertions`
> is off) does not reference absent symbols.

`alloc.rs` did that for its own tests. These two consumers did not.
`orthogonal_factorization`'s doc comment says it means to *"skip with a message
if absent"* — the intent was right; an unconditional `use` is a compile error,
not a skip.

**Consequence:** `cargo test --release -p katgpt-core --lib` **did not compile
at all.** That is exactly the command `.issues/713` T2b's correction tells
everyone to run, and it is how the sibling defect in `.issues/715` (a release
build of `sdar_gate` broken for two days) went unnoticed.

### 14 consumers, and the compiler proved which 2 were wrong

14 files under `crates/katgpt-core/src/` reference those counters. The release
build failed with **exactly 2** errors, so the other 12 are correctly gated —
that is a proof by the compiler, not a grep over cfg contexts, which is worth
distinguishing because a `grep -B6` for an enclosing `#[cfg]` **missed 7 of
the 12** (their gate sits further above the window). The narrow answer from
the narrow instrument would have been "7 more are broken".

## Fixed how, and what was deliberately NOT done

Both tests are now `#[cfg(debug_assertions)]`. They join **12 siblings in the
same crate** already gated that way, so this restores the crate's own
established pattern rather than inventing one.

**Deliberately not solved with release no-op stubs returning 0.** A
zero-alloc assertion against a stubbed counter *passes vacuously* — that is
`.issues/705` and `.issues/714` exactly: an instrument that cannot fail is not
passing. Better a test that does not exist in release than a green one that
measured nothing.

### The cost, stated: release runs 26 fewer lib tests, all explicable

| profile | katgpt-core lib tests |
|---|---|
| dev | 4,609 |
| release | 4,583 |

The 26-test difference was **enumerated by name**, not inferred from the delta:
6 are `alloc.rs`'s own tests, 14 are `g4_*` alloc gates, 6 are
`*_panics_in_debug` (a `debug_assert!` does not fire in release, so the test
is meaningless there). The reverse set — in release but not dev — is **empty**,
as it must be. Two of the 26 are the ones fixed here.

## Tasks

- [x] **T1** Fix both sites; verify BOTH profiles compile; enumerate the
  release/dev test-set difference by name rather than by count. LANDED.
- [x] **T2 LANDED** — `full_gate.sh` Layer 6.
  Deliberately `cargo check`, not `clippy`: the axis is **compilation** under
  `debug_assertions = off`, and the lint surface is already covered by the dev
  pass. Do NOT fold `--release` into `GATE_ARGS` — Layer 5 asserts AGENTS.md
  quotes that string verbatim, and the release pass is a different question.
  **Cost measured before deciding, not after:** warm **27 s**; **cold 65 s**
  on a fresh `CARGO_TARGET_DIR` (394 units, 622 MB) — roughly **8%** on a
  >13 min weekly gate. Cheap because `check` does no codegen; a release
  *clippy* pass costs multiples of this and buys almost nothing extra here.

  **Both numbers are M3 Max / 16-core, and the CI runner is not.** The 8%
  figure is a ratio of two local measurements, which is the honest way to
  read it — the gate's own >13 min is also a local figure, so the ratio
  survives a slower box better than either absolute does. A GitHub macOS
  runner has ~1/4 the cores, so expect minutes rather than 65 s in absolute
  terms. Stated because a local number quoted as a CI cost is a claim about a
  machine that was never measured. (The registry cache was warm locally; CI
  pays download time it already pays for Layer 3.)

  **Its first in-situ run found a bug in itself, which is why it was run.**
  The first cut counted `Compiling`/`Checking` lines, the way Layer 3 does, and
  reported **INCONCLUSIVE** on a tree whose release artifacts were already
  warm: cargo compiled 0 units and printed nothing. *Freshness must not decide
  a liveness verdict.* Log-replay canaries had all passed — the flaw was only
  visible when the layer ran against a real warm tree.

  Fixed by measuring `--message-format=json` `compiler-artifact` records, which
  cargo emits for **fresh** units too. Measured on the same warm tree:
  **3 `Checking` lines vs 1,423 artifacts.** The JSON census is also immune to
  the ANSI-colour trap that zeroed every `^`-anchored counter in this gate's
  first two CI runs (`.issues/705`) — the keys carry no colour codes, so no
  strip step is needed rather than one being carefully maintained.

  **Canaried two-sided on the real warm tree**, not on synthetic logs:

  | arm | artifacts | errors | verdict |
  |---|---|---|---|
  | fixed tree, warm | 1,423 | 0 | **PASSED** |
  | one of the two breaks reintroduced | 1,422 | 1 | **FAILED**, with the rendered `E0432` |
  | restored | 1,423 | 0 | **PASSED**, tree clean |
- [ ] **T3 (owner call, sibling repos)** 19 repos, none of which run a release
  pass either. Whether each wants one is its owner's call, the same shape as
  `.issues/713` T3.

## Related

- `.issues/715` — the same axis, opposite direction: an orphaned
  `#[cfg(debug_assertions)]` broke release while debug stayed green.
- `.issues/713` T2b — the same axis, opposite *sign*: a debug run manufactured
  four false perf reds. Neither profile is the safe default. **The profile is
  part of the claim.**
- `.issues/705` — a gate that passed over zero compiled units. Why the release
  pass needs its own liveness assertion, not just an error count.
