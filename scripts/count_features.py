#!/usr/bin/env python3
"""Precise feature-flag count audit for katgpt-rs workspace.

Counts:
  - default-on features (entries in the `default` array, excluding `default` itself)
  - total feature flags (all keys under [features], excluding `default`)
  - net new opt-in flags (total - default)

Reads [features] from every Cargo.toml in the workspace (root + crates/*).
For workspace-level passthrough features (foo = ["katgpt-core/foo"]), counts
the workspace entry AND notes the underlying core feature.

Usage:
    python3 scripts/count_features.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib


def load_features(path: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Return (default_on, all_flags, raw_map) for a Cargo.toml [features] table."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    feats = data.get("features", {})
    if not feats:
        return set(), set(), {}
    all_flags = {k for k in feats if k != "default"}
    raw_default = feats.get("default", [])
    # default entries may be "katgpt-core/foo" (passthrough) or "foo"
    default_on = set()
    for entry in raw_default:
        # strip crate prefix for passthroughs
        name = entry.split("/", 1)[-1]
        default_on.add(name)
    default_on.discard("default")
    return default_on, all_flags, feats


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tomls = [root / "Cargo.toml", root / "crates" / "katgpt-core" / "Cargo.toml"]

    print("=" * 72)
    print("katgpt-rs feature-flag audit")
    print("=" * 72)

    grand_default: set[str] = set()
    grand_total: set[str] = set()

    for toml in tomls:
        rel = toml.relative_to(root)
        default_on, all_flags, feats = load_features(toml)
        opt_in = all_flags - default_on
        print(f"\n## {rel}")
        print(f"  default-on : {len(default_on)}")
        print(f"  total flags: {len(all_flags)}")
        print(f"  opt-in     : {len(opt_in)}")
        if feats.get("default"):
            print(f"  default[] length: {len(feats['default'])}")
        grand_default |= default_on
        grand_total |= all_flags

    # Union view (dedup across root + core)
    print("\n" + "=" * 72)
    print("WORKSPACE UNION (deduped across root + katgpt-core)")
    print("=" * 72)
    print(f"  default-on (unique) : {len(grand_default)}")
    print(f"  total flags (unique): {len(grand_total)}")
    print(f"  opt-in (unique)     : {len(grand_total - grand_default)}")

    # ── README claim check ──────────────────────────────────────────────────
    # This used to print a HARDCODED claim string ("140+ default-on, 320+ total
    # flags") next to the measured numbers and leave the comparison to whoever
    # read the output. That check could never fire: the literal drifted out of
    # step with the README independently of the code, and by 2026-09-01 it
    # matched neither the README (378/152) nor reality (537/189). A check that
    # compares a measurement against a constant it also owns is not a check.
    # Now it parses the README and asserts.
    print("\n## README claim check")
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    m = re.search(r"(\d+)\s+feature flags\s*\((\d+)\s+default-on", text)
    if not m:
        print(f"  ✗ no parseable feature-flag claim found in {readme.name}")
        print("    expected the form: '<N> feature flags (<M> default-on'")
        print(f"    measured: {len(grand_total)} total, {len(grand_default)} default-on")
        return 1

    claim_total, claim_default = int(m.group(1)), int(m.group(2))
    actual_total, actual_default = len(grand_total), len(grand_default)
    print(f"  README claims : {claim_total} total, {claim_default} default-on")
    print(f"  measured      : {actual_total} total, {actual_default} default-on")
    if (claim_total, claim_default) != (actual_total, actual_default):
        print("  ✗ README feature counts have drifted from the manifests")
        print(f"    update README.md to: {actual_total} feature flags "
              f"({actual_default} default-on, ...)")
        return 1
    print("  ✓ README matches the manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
