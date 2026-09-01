#!/usr/bin/env bash
# Full gate — the whole-surface compile + lint claim, as an assertion.
#
# Every build command in AGENTS.md's "Build Commands" block is narrow on at
# least one of three INDEPENDENT axes, and a green result says nothing about
# what the invocation compiled to nothing:
#
#   1. `check` vs `clippy`      — two `cargo heal` escape classes are rejected
#                                 by clippy's typeck and accepted by `check`
#                                 (E0689 ambiguous-integer, E0631 deref
#                                 coercion in `redundant_closure`).
#   2. default vs --all-features — non-default gated code compiles to NOTHING.
#   3. `-p <crate>` vs --workspace — at the SAME default features: a crate's own
#                                 non-default feature can be switched on by the
#                                 ROOT crate's defaults once the root is in the
#                                 selected set. `cargo test -p katgpt-backend
#                                 --lib` compiled clean while the workspace run
#                                 failed, because `gpu.rs` sits behind
#                                 `katgpt-backend/gpu_inference` and the chain
#                                 katgpt-rs/default -> async_qdq_overlap ->
#                                 inference_router -> gpu_inference only fires
#                                 when the root crate is selected.
#
# Missing `--all-targets` is a fourth: it skips every test / bench / example,
# which is where gated code lives.
#
# Consequence, measured: this gate was RED on `develop` from at least 2cb97410
# until 3e58e821 — five broken targets — while every documented gate was green.
# Two of the five were `cargo heal` escapes that survived the healer's own
# compile gate, because that gate ran where the code was absent.
#
# Layers, all hard-fail:
#   1. cargo present
#   2. platform coverage   → `target_os = "macos"` code (katgpt-backend's
#                            gpu.rs / ane.rs) is invisible off macOS EVEN WITH
#                            --all-features, because the cfg is an `all(...)`
#                            over target_os AND feature. A Linux run of this
#                            gate reproduces the exact vacuous green it exists
#                            to catch, so off-macOS is a partial gate and says
#                            so loudly.
#   3. the gate itself     → clippy, workspace, all targets, all features
#   4. zero errors         → any `error` line or unbuildable target is a finding
#   5. doc/script parity   → AGENTS.md must quote the same command this script
#                            runs; a gate whose spec has drifted from its
#                            implementation is a gate nobody is running
#
# Usage:
#   scripts/full_gate.sh                          # strict
#   scripts/full_gate.sh --allow-partial-platform # off-macOS: warn, don't fail
#
# Honour CARGO_TARGET_DIR to avoid fighting a concurrent build:
#   CARGO_TARGET_DIR=/tmp/full_gate scripts/full_gate.sh
#
# Honour FULL_GATE_LOG to put the clippy log somewhere retrievable (CI artifact
# upload). Retained on failure, removed on a pass, either way.
#   FULL_GATE_LOG=/tmp/full_gate.log scripts/full_gate.sh
set -euo pipefail

ALLOW_PARTIAL=0
[ "${1:-}" = "--allow-partial-platform" ] && ALLOW_PARTIAL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The gate command, defined once. Layer 5 asserts AGENTS.md quotes this string,
# so the doc and the assertion cannot drift apart silently.
GATE_ARGS=(cargo clippy --workspace --all-targets --all-features --keep-going)
GATE_CMD="${GATE_ARGS[*]}"

# ── Layer 1: cargo present ──────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    echo "✗ cargo not installed — full gate cannot run"
    exit 1
fi

# ── Layer 2: platform coverage ──────────────────────────────────────────────
# Grep the real cfg rather than trusting this comment to stay true.
# `|| true` is load-bearing: grep exits 1 on no match, and under `set -e` with
# pipefail a failed substitution pipeline would kill the script here with no
# diagnostic at all.
APPLE_GATED=$({ grep -rl 'target_os = "macos"' --include='*.rs' crates src 2>/dev/null || true; } | wc -l | tr -d ' ')

# Verify the instrument before trusting its verdict. A zero here reads exactly
# like a working grep over a surface that no longer exists, and BOTH readings
# invalidate something: zero means either this grep drifted (paths moved) or the
# device-backend surface is gone — in which case the workflow is paying for a
# macOS runner, and billing macOS minutes at a multiple of Linux, for nothing.
# Either way a human must decide, so fail rather than narrate a vacuous pass.
if [ "$APPLE_GATED" -eq 0 ]; then
    echo "✗ platform layer found NO target_os = \"macos\" files under crates/ src/"
    echo "  Either the grep drifted from the tree, or the device-backend surface is"
    echo "  gone. If the latter, .github/workflows/full_gate.yml should stop paying"
    echo "  for macos-latest — revisit its 'Why macos-latest' preamble."
    exit 1
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "⚠ not macOS — $APPLE_GATED file(s) gated on target_os = \"macos\" will NOT compile,"
    echo "  even with --all-features (the cfg is all(target_os, feature))."
    echo "  This run is a PARTIAL gate: it cannot see the device-backend surface."
    if [ "$ALLOW_PARTIAL" -eq 0 ]; then
        echo "✗ refusing to report a partial run as a pass (--allow-partial-platform to override)"
        exit 1
    fi
else
    echo "✓ macOS — the $APPLE_GATED target_os-gated file(s) are in scope"
fi

# ── Layer 3: the gate ───────────────────────────────────────────────────────
# `--keep-going` is not optional: without it cargo stops at the first failing
# target. The run that found the five breaks reported only two without it.
# $FULL_GATE_LOG overrides the location so a CI job can upload the log as an
# artifact. Retention alone is not enough there: a GitHub runner is destroyed
# when the job ends, so a path printed into a dead runner's filesystem is
# unreachable — the weekly run would be left with only the summary, which is
# the situation the retention exists to fix.
LOG="${FULL_GATE_LOG:-$(mktemp -t full_gate)}"
mkdir -p "$(dirname "$LOG")"

# Retain the log when the gate fails. The summary below prints error CLASSES and
# the first diagnostic; everything else — every remaining site, every warning —
# lives only in this file, and re-deriving it costs another >13 min run. On a
# pass there is nothing in it worth the disk.
KEEP_LOG=0
trap '[ "$KEEP_LOG" -eq 1 ] && echo "  full log retained: $LOG" || rm -f "$LOG"' EXIT
echo "▸ $GATE_CMD"
set +e
"${GATE_ARGS[@]}" >"$LOG" 2>&1
set -e

# ── Layer 4: zero errors ────────────────────────────────────────────────────
# Count `error` lines and unbuildable targets separately: a target can fail to
# build with its diagnostics attributed to a dependency, and an error can be a
# deny-level lint that names no target.
# Every count is `|| true`-guarded for the same reason as Layer 2. The
# diagnostic count deliberately EXCLUDES cargo's own "error: could not compile"
# aggregates, which would otherwise inflate it by one per broken target and
# make the two numbers look like independent evidence when they are not.
DIAGS=$({ grep -E '^error(\[|:)' "$LOG" || true; } | { grep -v 'could not compile' || true; } | wc -l | tr -d ' ')
BROKEN=$({ grep -E 'could not compile' "$LOG" || true; } | sort -u)
BROKEN_N=$({ printf '%s' "$BROKEN" | grep -c . || true; })
WARNINGS=$({ grep -cE '^warning:' "$LOG" || true; })

if [ "$DIAGS" -ne 0 ] || [ "$BROKEN_N" -ne 0 ]; then
    KEEP_LOG=1
    echo "✗ full gate FAILED — $DIAGS error diagnostic(s), $BROKEN_N unbuildable target(s)"
    # `|| true`: bash exempts the left side of `&&` from `set -e`, but being
    # explicit here beats relying on that exemption in a failure path that must
    # print its diagnostics before exiting.
    { [ -n "$BROKEN" ] && { echo "  unbuildable:"; printf '%s\n' "$BROKEN" | sed 's/^/    /'; }; } || true
    echo "  error classes:"
    grep -E '^error(\[|:)' "$LOG" | sort | uniq -c | sed 's/^/    /'
    echo "  first diagnostic with location:"
    grep -A3 -m1 -E '^error(\[|:)' "$LOG" | sed 's/^/    /'
    exit 1
fi

# ── Layer 5: doc/script parity ──────────────────────────────────────────────
# A gate documented with a different command than the one asserted here is how
# a spec silently stops describing the code.
if ! grep -qF "$GATE_CMD" AGENTS.md; then
    echo "✗ AGENTS.md does not quote the gate command — doc and gate have drifted"
    echo "  expected to find: $GATE_CMD"
    exit 1
fi

echo "✓ full gate PASSED — 0 errors, 0 unbuildable targets ($WARNINGS warning line(s), not gated)"
