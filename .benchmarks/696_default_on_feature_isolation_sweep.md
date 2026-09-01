# Bench 696 — the default-on feature-isolation sweep: 4.1 min, not 2.2 h, and it found two broken flags

Status: **MEASURED + FIXED 2026-09-01.** All 228 default-on (package, flag)
pairs isolated. 2 failures, both real, both fixed in `b50db0ef`. Re-verified
green. Issue 701 R1b's blocking estimate is superseded — the sweep is **32x
cheaper** than recorded, and the reason is a category error in the original
measurement, not noise.

## Headline

| | recorded in Issue 701 R1 | measured here |
|---|---|---|
| scope | "197 default-on flags" | **228** (package, flag) builds |
| per-flag mean | 39.5s (n=6) | **1.08s** (n=228, complete) |
| per-flag median | — | **0.50s** |
| whole sweep | ~2.2 h (extrapolated) | **4.1 min** (measured, not extrapolated) |
| failures | never run | **2** |

Nothing here is an extrapolation: the scope is enumerated from the manifests
and every member was built.

## The two flags that never built alone

Both `katgpt-core`, both **default-on** — i.e. on the shipped surface:

```
[164/228] ✗ katgpt-core/hebbian_kernel_memory    error[E0433]: could not find `linalg` in the crate root
[213/228] ✗ katgpt-core/velocity_field_ensemble  error[E0433]: could not find `linalg` in the crate root
```

One cause. `pub mod linalg` in `lib.rs` is gated by a `cfg(any(...))` listing
its consumers, and the comment directly above it **already states the rule**:
"each must also gate this `pub mod` so the crate compiles when only that
feature is on." Two later consumers of `linalg::ridge_solve` did not join the
list. That is what an unenforced rule does — the rule was written down, cited a
prior instance (Issue 684, `svd_cca`), and drifted twice more anyway.

**Why no existing gate could see it.** Not an oversight in any of them:

- `--all-features` compiles the **union**, where some other consumer always
  brings `linalg` in. Green, and correctly so.
- `cargo clippy --workspace --all-targets --all-features` — same union. The
  full gate (`.issues/705`) is blind to this by construction.
- The per-PR isolation gate is **diff-bounded**: it checks flags whose
  DEFINITION a diff touches. Both flags were defined correctly; they were
  broken by an omission in a *different* file. Nothing in their own definition
  ever changed.

Only a whole-scope isolation sweep finds it, and this was its first run.

## Why the 2.2 h estimate was 32x high — a category error, not variance

R1's figure came from n=6 with a 26x range and was labelled a point estimate,
so the temptation is to call this sampling luck. It is not. Re-running three
flags **from the original n=6**, same command, in the sweep's target dir:

| flag | Issue 701 recorded | re-measured here |
|---|---|---|
| `monopoly` | 110.2s | **14.0s** (29 units compiled) |
| `thicket_variance_probe` | 45.8s | **4.0s** (0 units) |
| `vocab_coreset` | 4.2s | **0.6s** |

Sample composition was also checked and is not the explanation: the 40-flag
pilot was 62% `katgpt-rs` against 59% in the full scope.

The actual mechanism: the original measured **isolated checks against a
default-featured target dir**, where each `--no-default-features --features X`
invocation changes feature unification across the whole dependency graph and
forces a broad rebuild. In a **sweep**, every check shares the same
`--no-default-features` graph, so the dependencies stay warm and only the top
crate rebuilds.

Both numbers are correct for their own question:

- **39.5s/flag = the per-PR cost.** One isolated check interleaved with normal
  default-feature builds. This is the right number for the diff-bounded gate it
  was measured to justify, and it stands.
- **1.08s/flag = the sweep cost.** Only reachable when the previous check left
  the same graph warm.

The error was extrapolating the first to the second. Worth stating because the
reverse mistake is equally available: quoting 1.08s to size a per-PR gate would
under-estimate it by ~36x.

## Per package (n, mean, total)

| package | n | mean | median | max | total |
|---|---|---|---|---|---|
| `katgpt-rs` | 135 | 1.43s | 0.70s | 32.1s | 3.22 min |
| `katgpt-core` | 76 | 0.48s | 0.45s | 0.9s | 0.61 min |
| `katgpt-device-verify` | 2 | 6.80s | 6.80s | 13.4s | 0.23 min |
| `katgpt-dec` | 4 | 0.30s | 0.35s | 0.4s | 0.02 min |
| `katgpt-transformer` | 5 | 0.20s | 0.20s | 0.3s | 0.02 min |
| `katgpt-sparse` | 2 | 0.25s | 0.25s | 0.3s | 0.01 min |
| `katgpt-band` | 2 | 0.15s | 0.15s | 0.2s | 0.01 min |
| `katgpt-claim` | 2 | 0.10s | 0.10s | 0.1s | 0.00 min |

The distribution is long-tailed, not centred: median 0.50s against mean 1.08s,
and three flags carry a disproportionate share (`katgpt-rs/plot` 32.1s,
`katgpt-device-verify/fair_roll` 13.4s, `katgpt-rs/async_qdq_overlap` 10.5s).
Quote the mean for planning a sweep and the median for a typical flag; neither
alone is the answer.

## Two scope corrections

- **228, not 197.** 197 is the count of unique flag NAMES. 31 names are defined
  in more than one manifest, and each definition is its own build that can pass
  in one package and fail in the other. Sizing the work by unique names
  undercounts it by 16%.
- **`--scope all` is 1130 pairs across 568 names**, ~2x what R1's "568 flags"
  assumed, for the same reason.

## Method / reproduction

```bash
CARGO_TARGET_DIR=/tmp/r1b_isolation \
  python3 scripts/feature_isolation_gate.py --scope default-on
```

Fresh target dir, so flag #1 pays the cold dependency build (the structure a CI
runner sees). M3 Max, 16 cores, concurrent load from sibling agents. Peak disk
6.1 GiB for the 228-pair scope. `--sample N --seed S` gives a reproducible
subset; the pilot above was `--sample 40 --seed 701`.

**Instrument trust.** 228 consecutive passes would be equally consistent with a
gate that cannot fail, so both were canaried:

- `check_flags` failure path — a nonexistent flag is correctly reported,
  collected, and returned as a failure.
- `selftest()` (runs on every invocation) pins `all ⊇ default-on` and non-empty
  scopes; breaking `all` makes it exit 1 naming 76 missing pairs.

The sweep then found 2 genuine failures on its first run, which is the
strongest evidence available that it can fail.

## What this changes

R1b's closing condition was "needs one real COLD runner timing first". The
blocker as stated dissolves: at 4.1 min measured, the cost question stops
gating the decision on any plausible runner multiplier. The remaining open
question is **cadence and where it runs** — a billing call, not a measurement
(the gate has no `cfg(target_os)` surface, so unlike the full gate it can run
on ubuntu). Recommendation and cost are in `.issues/701` R1b.

Refs: `b50db0ef` (fix + harness), `.issues/701` R1 (the estimate this
supersedes), `.issues/705` (the full gate, blind to this class by
construction), Issue 684 (the prior instance of the same `linalg` gate drift).
