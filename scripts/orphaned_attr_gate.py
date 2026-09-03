#!/usr/bin/env python3
"""Gate on a `#[cfg]` separated from its item by a blank line.

Rust binds an attribute to the next item **across a blank line**. So this

    #[cfg(debug_assertions)]

    use crate::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};

compiles, and makes the import debug-only. It is not a lint clippy has, and it
is invisible in review because the blank line reads as separation.

## The bug this is built from, with dates

`crates/katgpt-pruners/src/sdar/sdar_absorb.rs` was in exactly that state for
two days (fixed in `a08376a0`):

- `26d055c6` dropped `use std::cmp::Ordering;`, which is what the
  `#[cfg(debug_assertions)]` above it had correctly applied to, and left the
  attribute behind. It silently re-bound to the NEXT import.
- Every release build of `katgpt-pruners --features sdar_gate` then failed with
  5 errors: `debug_assertions` is off in release, so the import vanished while
  its five usages stayed unconditional.
- `26d055c6`'s own validation was a DEBUG run (`lib 597/0`), where
  `debug_assertions` is on and the import exists. That is `.issues/713` T2b's
  lesson with the sign flipped — there, debug manufactured four false perf
  reds; here, debug hid a real build break.
- `7e34ccef` then deleted the blank line, which made the wrong binding look
  deliberate and would have erased the evidence.

## Why this can be a GATE and not a report

Measured across all 19 contract repos at the fix: **zero** sites. Not "few" —
zero. So the pin is 0, there is no floor to negotiate, and any future
occurrence is the push that introduces it.

The broader shape (**any** attribute + blank line + item) is 2,044 sites and is
NOT gateable: it is dominated by whole-file INNER attributes (`#![cfg(...)]`),
which bind to the enclosing module rather than to the next item and are
conventionally followed by a blank line. Narrowing to OUTER `#[cfg]` /
`#[cfg_attr]` is what takes 2,044 to 0 — the narrowing is the instrument.

## Scope

katgpt-rs only, same reasoning as `cfg_gated_floor_gate.py`: CI has a single
checkout. Pass a repo path to audit a sibling; adopting it there is an owner
call, like `.issues/713` T3.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# OUTER `#[cfg(...)]` / `#[cfg_attr(...)]` only. An INNER `#![cfg(...)]` binds
# to the enclosing module, so a blank line after it is correct and common —
# including it is what made the naive measurement 2,044 instead of 0.
OUTER_CFG = re.compile(r"^\s*#\s*\[\s*cfg(?:_attr)?\s*\(")
ANY_ATTR = re.compile(r"^\s*#!?\s*\[")

# Pruned during the walk, never filtered afterwards. `rglob("*.rs")` followed by
# a `"target" in parts` filter still DESCENDS into target/ (117 GB, ~1.3M
# entries) — the same trap that made bench_doc_audit.py take 556s, and the
# `find -not -path` trap one level over.
PRUNE = {"target", ".git", "node_modules", ".venv", "__pycache__"}


def offenders(repo: Path) -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in PRUNE]
        for fn in files:
            if not fn.endswith(".rs"):
                continue
            p = Path(root) / fn
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i in range(len(lines) - 2):
                if not OUTER_CFG.match(lines[i]) or lines[i + 1].strip():
                    continue
                nxt = lines[i + 2]
                # A following comment or another attribute is not the item, and
                # a blank-line run means the attribute is dangling further
                # down; both are reported only when a real item follows.
                if not nxt.strip() or nxt.lstrip().startswith("//") or ANY_ATTR.match(nxt):
                    continue
                out.append(
                    (str(p.relative_to(repo)), i + 1, lines[i].strip(), nxt.strip()[:60])
                )
    return sorted(out)


def selftest() -> None:
    """Pin both directions on every invocation.

    The false-negative direction is the one that matters: a regex regression
    makes this print `0 offenders` forever, which is indistinguishable from the
    clean state it is asserting.
    """
    import tempfile

    positive = (
        "#[cfg(debug_assertions)]\n"
        "\n"
        "use crate::absorb_compress::AbsorbCompressLayer;\n"
    )
    negatives = {
        # Correctly attached — the overwhelmingly common shape.
        "attached": "#[cfg(feature = \"x\")]\nuse a::b;\n",
        # INNER attribute: binds to the module, blank line is conventional.
        # This single case is the difference between 2,044 hits and 0.
        "inner": '#![cfg(feature = "x")]\n\nuse a::b;\n',
        # A doc comment after the blank line is not the item.
        "comment": '#[cfg(feature = "x")]\n\n// note\nuse a::b;\n',
        # Another attribute after the blank line: still an attribute run.
        "attr": '#[cfg(feature = "x")]\n\n#[derive(Debug)]\nstruct S;\n',
        # A non-cfg attribute is formatting, not conditional compilation.
        "derive": "#[derive(Debug)]\n\nstruct S;\n",
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pos.rs").write_text(positive)
        got = offenders(root)
        assert len(got) == 1, f"the real bug's shape was not detected: {got}"
        assert got[0][0] == "pos.rs"

    for name, src in negatives.items():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "n.rs").write_text(src)
            got = offenders(root)
            assert got == [], f"false positive on {name}: {got}"

    # The walk must PRUNE, not filter: a file under target/ is invisible.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "target").mkdir()
        (root / "target" / "gen.rs").write_text(positive)
        assert offenders(root) == [], "target/ was walked"


def main(argv: list[str]) -> int:
    selftest()
    repo = Path(argv[1]).resolve() if len(argv) > 1 else REPO_ROOT
    found = offenders(repo)

    print(f"orphaned-attribute gate — {repo.name}: {len(found)} offender(s)")
    for path, line, attr, item in found:
        print(f"  ✗ {path}:{line}  {attr}")
        print(f"      binds ACROSS the blank line to: {item}")
    if found:
        print(
            "\n  A `#[cfg]` separated from its item by a blank line still applies to "
            "that item.\n  Either attach it or delete it — see this file's header for "
            "the two-day release\n  break it is built from (a08376a0)."
        )
        print(f"✗ orphaned-attribute gate FAILED — {len(found)} site(s)")
        return 1
    print("✓ orphaned-attribute gate PASSED — pinned at 0, measured 0 across 19 repos")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
