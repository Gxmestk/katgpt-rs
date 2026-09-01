#!/usr/bin/env python3
"""Per-feature isolation gate, bounded by the diff instead of the manifest.

`--all-features` proves the UNION compiles. It says nothing about a feature
compiling ON ITS OWN, and every primitive here ships behind an opt-in flag, so
"feature X alone is broken" is a live failure mode the full gate is blind to by
construction (Issue 701 R1).

The standard tool, `cargo hack --each-feature`, is not affordable here. Measured
2026-09-01 on an M3 Max with a warm target dir: mean 39.5s per flag over a
seeded random sample of 6 (range 4.2s-110.2s), so 568 flags is ~6.2 h and even
the 197 default-on ones are ~2.2 h. Marginal disk was ~0.09 GiB per flag against
60 GiB free.

So: check only the flags whose DEFINITION the diff touches. A typical 1-3 flag
change costs ~40-120s, which is affordable per-PR, and it catches the case that
actually regresses — someone adding or editing a flag without ever building it
alone.

    scripts/feature_isolation_gate.py [base_ref]

base_ref defaults to $GITHUB_BASE_REF (set on GitHub PRs), then origin/develop.
Exit 0 = every touched flag builds alone, or none were touched. Exit 1 = a flag
does not build in isolation, or the gate could not determine what to check.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `foo = [...]` at the start of a line inside a [features] table. Cargo feature
# names are [a-zA-Z0-9_-].
FEATURE_DEF_RE = re.compile(r"^\+([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*\[")
DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.*)$")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def resolve_base(argv: list[str]) -> str | None:
    for cand in (argv[1] if len(argv) > 1 else None,
                 os.environ.get("GITHUB_BASE_REF"),
                 "origin/develop"):
        if not cand:
            continue
        ref = f"origin/{cand}" if cand and "/" not in cand and \
            run(["git", "rev-parse", "--verify", cand]).returncode != 0 else cand
        if run(["git", "rev-parse", "--verify", ref]).returncode == 0:
            return ref
    return None


def manifest_facts(cargo_toml: Path) -> tuple[str | None, set[str]]:
    """(package name, declared feature names) for a manifest at HEAD."""
    try:
        with cargo_toml.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None, set()
    feats = data.get("features", {})
    return (data.get("package", {}).get("name"),
            {k for k in feats if k != "default"})


def changed_flags(base: str) -> list[tuple[str, str]]:
    """[(package, flag)] for every feature DEFINITION added/edited vs base."""
    diff = run(["git", "diff", "--unified=0", f"{base}...HEAD", "--",
                "*Cargo.toml"])
    if diff.returncode != 0:
        print(f"  ✗ git diff against {base} failed: {diff.stderr.strip()}")
        sys.exit(1)
    out: list[tuple[str, str]] = []
    cur_pkg: str | None = None
    cur_feats: set[str] = set()
    for line in diff.stdout.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            cur_pkg, cur_feats = manifest_facts(ROOT / m.group(1))
            continue
        m = FEATURE_DEF_RE.match(line)
        if not (m and cur_pkg):
            continue
        flag = m.group(1)
        # `name = [...]` is not unique to [features]. `required-features = [..]`
        # inside [[bench]] / [[test]] has the identical shape, and with
        # --unified=0 the diff never shows the enclosing table header, so the
        # section CANNOT be inferred from the diff text. The gate's first canary
        # duly reported `katgpt-core/required-features` as a broken feature.
        #
        # So candidates are confirmed against the manifest's real [features]
        # table at HEAD. This also correctly drops a flag whose definition the
        # diff DELETED: absent from HEAD's features, nothing to isolate.
        if flag == "default" or flag not in cur_feats:
            continue
        if (cur_pkg, flag) not in out:
            out.append((cur_pkg, flag))
    return out


def main(argv: list[str]) -> int:
    base = resolve_base(argv)
    if base is None:
        # Not a skip: without a base we do not know what changed, and reporting
        # a pass would be a claim we cannot support.
        print("✗ no usable base ref (tried argv, $GITHUB_BASE_REF, "
              "origin/develop) — cannot determine which flags changed")
        return 1
    print(f"▸ base: {base}")

    targets = changed_flags(base)
    if not targets:
        # Say so explicitly. A silent pass here reads identical to a real one.
        print("✓ no feature DEFINITION touched vs base — nothing to isolate")
        return 0

    print(f"▸ {len(targets)} touched flag(s): "
          + ", ".join(f"{p}/{f}" for p, f in targets))
    failed = []
    for pkg, flag in targets:
        cmd = ["cargo", "check", "-p", pkg, "--no-default-features",
               "--features", flag]
        print(f"▸ {' '.join(cmd)}")
        r = run(cmd)
        if r.returncode == 0:
            print(f"    ✓ {pkg}/{flag} builds alone")
        else:
            failed.append((pkg, flag))
            tail = [l for l in r.stderr.splitlines()
                    if l.startswith("error")][:5]
            print(f"    ✗ {pkg}/{flag} does NOT build alone")
            for l in tail:
                print(f"      {l}")

    if failed:
        print(f"✗ feature isolation FAILED — {len(failed)}/{len(targets)}: "
              + ", ".join(f"{p}/{f}" for p, f in failed))
        return 1
    print(f"✓ feature isolation PASSED — {len(targets)}/{len(targets)} "
          f"flag(s) build alone")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
