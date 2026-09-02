#!/usr/bin/env python3
"""Report whether a staged set looks like ONE editing episode (Issue 709 T3).

`git add -A` from a repo root is indistinguishable, to git, from intent. With
several agent sessions writing into one worktree it routinely picks up a
sibling's uncommitted work — `b2527521` committed three agents' WIP in six
files, one of which was a build regression nobody noticed for a day.

This is a **REPORT, not a gate** (always exit 0), for the same reason
`docs_drift_sweep.py` is: the refusing version trades a real class of loss
against friction on every legitimate multi-file commit, and that trade is an
owner call (Issue 709 T3). What is NOT an owner call is measuring the signal,
because it is cheap and it is the technique that actually caught this by hand
twice: **worktree mtimes cluster by editing episode.** A 204-file rustfmt sweep
lands in a 3-second window; a session's own edits land in the window it was
running. Two clusters in one staged set means two episodes.

Two independent signals are reported, and neither subsumes the other:

  1. **mtime clusters** — single-linkage over the staged files' mtimes. More
     than one cluster is the "you staged someone else's episode" shape.
  2. **also-dirty** — a staged path that ALSO has unstaged changes. That is a
     concurrent editor writing into the same file *right now*, which mtime
     clustering cannot see (their write may land in your window).

Usage:  scripts/staged_set_audit.py [repo_path]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Two edits more than this far apart are different episodes. Deliberately
# generous: a session that takes 10 min to write 4 files must read as one.
GAP_SECONDS = 900.0


def cluster(stamps: list[float], gap: float = GAP_SECONDS) -> list[list[float]]:
    """Single-linkage clustering over 1-D timestamps.

    Single-linkage, not fixed-width bins: a session editing continuously for an
    hour is ONE episode even though its span exceeds the gap, because no two
    consecutive edits are `gap` apart. Fixed bins would split it and report a
    false positive on exactly the sessions that do the most work.
    """
    out: list[list[float]] = []
    for t in sorted(stamps):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def selftest() -> None:
    """Runs on every invocation — a clustering regression must not be silent.

    Without this the audit would degrade to "1 cluster, always", print a
    confident single-episode verdict, and look exactly like a clean result.
    """
    assert cluster([]) == [], "empty"
    assert cluster([100.0]) == [[100.0]], "single"
    assert len(cluster([0.0, 1.0, 2.0], gap=10)) == 1, "tight group is one episode"
    assert len(cluster([0.0, 100.0], gap=10)) == 2, "a gap splits"
    # Chained: total span 20 > gap 10, but no adjacent pair exceeds it.
    assert len(cluster([0.0, 8.0, 16.0, 20.0], gap=10)) == 1, "single-linkage chains"
    # Unsorted input must not change the answer.
    assert len(cluster([100.0, 0.0, 101.0], gap=10)) == 2, "sorted internally"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    selftest()
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

    staged = [p for p in git(repo, "diff", "--cached", "--name-only").splitlines() if p]
    if not staged:
        print("staged-set audit: nothing staged")
        return 0

    also_dirty = {p for p in git(repo, "diff", "--name-only").splitlines() if p}
    # A staged deletion has no worktree file to stat; it belongs to no episode.
    timed: list[tuple[float, str]] = []
    missing: list[str] = []
    for p in staged:
        f = repo / p
        if f.is_file():
            timed.append((f.stat().st_mtime, p))
        else:
            missing.append(p)

    groups = cluster([t for t, _ in timed])
    by_stamp = {}
    for t, p in timed:
        by_stamp.setdefault(t, []).append(p)

    print(f"staged-set audit: {len(staged)} staged path(s) in {repo.name}")
    if missing:
        print(f"  {len(missing)} deleted/absent (no mtime, no episode): {', '.join(missing[:4])}")

    import datetime as _dt

    for i, g in enumerate(groups, 1):
        files = [p for t in g for p in by_stamp[t]]
        span = g[-1] - g[0]
        when = _dt.datetime.fromtimestamp(g[0]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  episode {i}: {len(files)} file(s), started {when}, span {span:.0f}s")
        for p in files[:6]:
            print(f"      {p}")
        if len(files) > 6:
            print(f"      … {len(files) - 6} more")

    overlap = sorted(set(staged) & also_dirty)
    if overlap:
        print(
            f"  ALSO-DIRTY: {len(overlap)} staged path(s) still have unstaged changes — "
            "someone is editing them concurrently, or you staged a partial blob on purpose"
        )
        for p in overlap[:6]:
            print(f"      {p}")

    match len(groups) > 1:
        case True:
            print(
                f"  REVIEW: {len(groups)} editing episodes in one staged set. If you did not "
                "write the older one(s), unstage them — see AGENTS.md on `git add -A`."
            )
        case False:
            print("  ✓ one editing episode")
    # Report, never a gate: the refusing version is Issue 709 T3's owner call.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
