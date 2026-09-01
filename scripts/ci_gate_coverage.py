#!/usr/bin/env python3
"""Which workspace repos actually gate their full compile+lint surface in CI?

Issue 701 R2. That question has now been answered by hand twice and been wrong
both times, in three separate ways — which is the argument for a script:

  1. The repo list was TYPED. It covered 12 of 18, so 5 repos with no CI at all
     and 1 with CI were simply absent from the answer (the Issue 703 class).
  2. Only workflow YAML was grepped. The gate frequently lives in a SCRIPT the
     workflow calls — `katgpt-rs/.github/workflows/full_gate.yml` runs
     `./scripts/full_gate.sh`, and `riir-neuron-db/rust.yml` runs
     `./scripts/ci_feature_guard.sh`. Grepping YAML alone under-reports both.
  3. Grepping YAML *including comments* over-reports: several workflows discuss
     `--all-features --all-targets` in a preamble while running nothing of the
     kind. Signals are read from non-comment lines only.

"Full surface" here is the katgpt-rs AGENTS.md definition — the axes a green
narrow gate cannot speak for: `--all-targets` (tests/benches/examples, where
gated code lives), `--all-features` (non-default code otherwise compiles to
nothing), `--workspace` (a crate's non-default feature can be switched on by the
root crate's defaults), clippy rather than check, and `--keep-going` (without it
the run stops at the first failing target and under-reports).

    scripts/ci_gate_coverage.py [--markdown]

Exit 0 always: this reports, it does not gate. Each repo owns its own CI per
BOUNDARY.md, so this cannot be a pass/fail assertion from here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

GIT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_RE = re.compile(r"(?:\./|\b)((?:scripts|ci|\.ci)/[A-Za-z0-9_.-]+\.sh)")
# Ordered: the report reads left-to-right as increasing surface.
SIGNALS = ("clippy", "--workspace", "--all-targets", "--all-features",
           "--each-feature", "--keep-going")


def derive_repos(root: Path) -> list[str]:
    """`-d .git`, not `-e`: a git worktree has a .git FILE and would double-count."""
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / "BOUNDARY.md").is_file()
                  and (d / ".git").is_dir())


def live_lines(path: Path) -> list[str]:
    """Non-comment lines. A workflow preamble that DISCUSSES --all-features is
    not a workflow that runs it; reading comments is how the hand survey called
    two repos gated that are not."""
    try:
        raw = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [l for l in raw if not l.lstrip().startswith("#")]


def cargo_commands(lines: list[str]) -> list[str]:
    """Every `cargo ...` invocation, with backslash continuations joined."""
    joined: list[str] = []
    acc = ""
    for raw in lines:
        s = raw.strip()
        if acc:
            acc += " " + s.rstrip("\\").strip()
            if not s.endswith("\\"):
                joined.append(acc)
                acc = ""
            continue
        if "cargo " not in s:
            continue
        acc = s.rstrip("\\").strip()
        if not s.endswith("\\"):
            joined.append(acc)
            acc = ""
    if acc:
        joined.append(acc)
    return joined


def survey(root: Path, repo: str) -> dict:
    wf_dir = root / repo / ".github" / "workflows"
    workflows = sorted(p for p in wf_dir.glob("*")
                       if p.suffix in (".yml", ".yaml")) if wf_dir.is_dir() else []
    lines: list[str] = []
    scripts: list[str] = []
    for wf in workflows:
        wl = live_lines(wf)
        lines += wl
        for m in SCRIPT_RE.finditer("\n".join(wl)):
            cand = root / repo / m.group(1)
            if cand.is_file() and m.group(1) not in scripts:
                scripts.append(m.group(1))
                lines += live_lines(cand)
    blob = "\n".join(lines)
    # Score PER COMMAND, not over a blob. `--all-targets` in one script and
    # `--all-features` in another is not a full-surface gate: AGENTS.md's whole
    # point is that the axes are INDEPENDENT, so a green on each separately says
    # nothing about their combination. A blob scan cannot tell those apart and
    # rated riir-chain "full" on axes spread across 15 scripts.
    best: list[str] = []
    best_cmd = ""
    for cmd in cargo_commands(lines):
        sig = [s for s in SIGNALS if s in cmd]
        if len(sig) > len(best):
            best, best_cmd = sig, cmd
    return {"repo": repo, "workflows": len(workflows), "scripts": scripts,
            "signals": best, "cmd": best_cmd,
            "any": sorted({s for c in cargo_commands(lines) for s in SIGNALS
                           if s in c}, key=SIGNALS.index),
            # A gate whose command is BUILT FROM DATA (a CONFIGS table of
            # pkg:features:flags rows, looped) carries its flags in array
            # literals, not in a `cargo ...` line, so per-command scoring reads
            # it as absent. riir-chain's clippy_gate.sh is exactly this and is
            # arguably MORE thorough than a single command — a feature matrix.
            # Report it as a third state. Scoring it low would be the silent
            # truncation this script exists to stop.
            # A signal that appears in the FILES but in no cargo command is
            # sitting in a data structure — a CONFIGS table of
            # pkg:features:flags rows that a loop later expands. riir-chain's
            # clippy_gate.sh is exactly that, and it is arguably MORE thorough
            # than one command (a feature matrix). Per-command scoring reads it
            # as absent, so report it as a third state; scoring it low would be
            # the silent truncation this script exists to stop.
            "dynamic": sorted(
                {s for s in SIGNALS if s in blob}
                - {s for c in cargo_commands(lines) for s in SIGNALS if s in c},
                key=SIGNALS.index)}


def main(argv: list[str]) -> int:
    repos = derive_repos(GIT_ROOT)
    rows = [survey(GIT_ROOT, r) for r in repos]
    md = "--markdown" in argv

    # The full-surface bar: clippy over every target AND every feature.
    def full(r): return {"clippy", "--all-targets", "--all-features"} <= set(r["signals"])
    def dyn(r): return bool(r.get("dynamic")) and not full(r)
    def any_ci(r): return r["workflows"] > 0

    if md:
        print(f"| repo | workflows | scripts followed | signals | full surface |")
        print("|---|---|---|---|---|")
        for r in rows:
            sig = " ".join(f"`{s}`" for s in r["signals"]) or "—"
            if r["any"] != r["signals"]:
                sig += (" <br>*(scattered across commands: "
                        + " ".join(f"`{s}`" for s in r["any"]) + ")*")
            scr = ", ".join(f"`{s}`" for s in r["scripts"]) or "—"
            mark = ("**yes**" if full(r) else
                    ("**needs a human read** — " + " ".join(f"`{s}`" for s in r["dynamic"])
                     + " live in data, not in a command") if dyn(r) else
                    "partial" if r["signals"] else
                    ("no CI" if not any_ci(r) else "no"))
            print(f"| `{r['repo']}` | {r['workflows']} | {scr} | {sig} | {mark} |")
    else:
        print(f"▸ {len(repos)} contract repos derived under {GIT_ROOT}")
        for r in rows:
            mark = ("FULL " if full(r) else "DYN? " if dyn(r) else
                    "part " if r["signals"] else
                    ("NOCI " if not any_ci(r) else "none "))
            print(f"  {mark} {r['repo']:<22} wf={r['workflows']:<2} "
                  f"scripts={len(r['scripts'])}  best-cmd: "
                  f"{' '.join(r['signals']) or '—'}"
                  + (f"   | anywhere: {' '.join(r['any'])}"
                     if r["any"] != r["signals"] else ""))

    n_full = sum(map(full, rows))
    n_noci = sum(1 for r in rows if not any_ci(r))
    n_part = sum(1 for r in rows if r["signals"] and not full(r))
    n_dyn = sum(map(dyn, rows))
    print(f"\n{n_full}/{len(rows)} repos statically gate the full surface; "
          f"{n_dyn} build the command from data and CANNOT be classified here "
          f"(read them by hand); {n_part} partial; {n_noci} have no CI at all.")
    if n_dyn:
        for r in rows:
            if dyn(r):
                print(f"  DYN? {r['repo']}: {' '.join(r['dynamic'])} appear in "
                      f"the files but in no cargo command")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
