#!/usr/bin/env bash
# Docs gate — the three manifest/doc drift assertions, as one command.
#
# Why this file exists: all three checks already existed, and NOTHING ran any of
# them. Measured 2026-09-01, two of the three were RED on `develop`:
#
#   count_features.py       green, but only because it had just been fixed; it
#                           checked ONE README site out of five and two of 29
#                           manifests.
#   bench_doc_audit.py      exit 1 — a false positive on a doc that correctly
#                           recorded "opt-in ... promoted to DEFAULT-ON".
#   cargo_comment_audit.py  exit 1 — a false positive from a case-SENSITIVE
#                           "Opt-in" regex that missed the repo's 32 "OPT-IN"
#                           comment lines.
#
# An assertion nobody invokes is decoration, and a red one nobody invokes is
# worse: it trains the next reader to assume the tool is broken.
#
# Cost: ~3s total. That is what makes per-push affordable here, in deliberate
# contrast to scripts/full_gate.sh (>13 min, weekly). It was ~556s before the
# manifest walk was pruned — `rglob("Cargo.toml")` descended into target/
# (117 GB, ~1.3M entries) and filtered afterwards, four times per run.
#
# Unlike the full gate this is platform-INDEPENDENT: pure Python over manifests
# and markdown, no cfg(target_os) surface, so ubuntu is correct and macOS would
# only cost more. Don't "fix" it to macos-latest.
#
# Runs all three even after one fails — the same reason full_gate.sh passes
# --keep-going: stopping at the first failure under-reports the drift.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKS=(
    "scripts/count_features.py:flag counts in README + examples/README vs every manifest"
    "scripts/bench_doc_audit.py:(default-on|opt-in) labels in .benchmarks + .docs vs Cargo defaults"
    "scripts/cargo_comment_audit.py:inline Cargo.toml comments vs the default closure"
)

if ! command -v python3 >/dev/null 2>&1; then
    echo "✗ python3 not found — docs gate cannot run"
    exit 1
fi

failed=0
for entry in "${CHECKS[@]}"; do
    script="${entry%%:*}"
    what="${entry#*:}"
    if [ ! -f "$script" ]; then
        # A missing check is a failure, not a skip: silently dropping a check is
        # how this gate would rot back into the state that motivated it.
        echo "✗ $script — MISSING (expected: $what)"
        failed=$((failed + 1))
        continue
    fi
    echo "▸ $script — $what"
    if out="$(python3 "$script" 2>&1)"; then
        printf '%s\n' "$out" | tail -1 | sed 's/^/    /'
    else
        failed=$((failed + 1))
        printf '%s\n' "$out" | sed 's/^/    /'
        echo "  ✗ $script FAILED"
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "✗ docs gate FAILED — $failed of ${#CHECKS[@]} check(s)"
    exit 1
fi
echo "✓ docs gate PASSED — ${#CHECKS[@]}/${#CHECKS[@]} checks clean"
