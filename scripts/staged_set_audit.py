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

Three independent signals are reported, and none subsumes another:

  1. **mtime clusters** — single-linkage over the staged files' mtimes. More
     than one cluster is the "you staged someone else's episode" shape.
  2. **also-dirty** — a staged path that ALSO has unstaged changes. That is a
     concurrent editor writing into the same file *right now*, which mtime
     clustering cannot see (their write may land in your window).
  3. **stale-vs-HEAD** — a file whose mtime PREDATES the commit that last
     touched its path. The worktree copy was written against an older version,
     so committing it reverts whatever landed since. Measured live: a 20:04:39
     rustfmt sweep of `tpr/als.rs` sat dirty while `0ef7f078` landed an Issue
     712 correctness fix in the same file at 21:07:13 — committing the sweep
     would have silently reverted it. Both other signals are blind to this:
     the sweep is ONE episode and its files are not also-dirty.

Signal 3 audits the dirty set too, not just the staged set, because the hazard
exists before anything is staged.

**Do not reach for `git log --author` first.** It is the obvious instinct and it
cannot work here: every concurrent session commits as the same git user, so
authorship does not distinguish "mine" from "a sibling's" at all. mtime is the
only signal that separates sessions, which is why signal 1 exists.

Nor can a whitespace-ignoring diff separate a reformat from a revert on its own.
On the `tpr/als.rs` case above, `git diff -w` reported 9 insertions / 30
deletions — but some of those deletions are rustfmt genuinely re-wrapping tokens
ACROSS lines (an import list, a fn signature), which `-w` cannot collapse. The
unambiguous evidence is the specific identifiers the newest commit introduced,
which is exactly what signal 3 reports back (`best_ssr` 5 occurrences at HEAD,
0 in the stale copy). Grep those, don't read the diffstat.

**A fourth signal is proposed but NOT implemented here** (Issue 709 T3c): the
rustfmt round-trip. `git show HEAD:$f | rustfmt --edition 2024 --emit stdout |
diff - $f` — if identical, the worktree copy is exactly "HEAD, formatted" and
provably carries zero content, so reverting it cannot lose anyone's work. It
splits "dirty because churn" from "dirty because work" with no mtime heuristic
and no false positives from line re-wrapping, which is what an
`--ignore-all-space` diff cannot do (it called 214/215 files non-whitespace in
one measured case). It is what made the 204-file revert here *provably* safe
rather than probably safe: 188 pure churn, 16 real. It is Rust-only and needs
a toolchain, hence a separate signal rather than a replacement for these
three.

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
    # Signal 3's confirmation stage. Without these the containment check can
    # degrade to "nothing is ever missing" and print a clean pass over a copy
    # that reverts a landed fix — which is the exact bug it exists to catch.
    assert substantive("    let mut best_snap: Option<Vec<f32>> = None;"), "real line"
    assert not substantive("}"), "a brace is not evidence"
    assert not substantive("    );"), "nor is a close-paren"
    assert not substantive(""), "nor is a blank"
    # Sample ORDER is part of the report's usefulness: comments last, so the
    # printed evidence is an identifier to grep rather than the commit's prose.
    assert is_comment("  // Best-iterate guard (Issue 712)"), "rust comment"
    assert not is_comment("    let mut best_ssr = prev;"), "code is not a comment"
    assert sorted(["// why", "let x = 1;"], key=is_comment)[0] == "let x = 1;", (
        "code sorts first"
    )


TRIVIAL = {"", "}", "{", "};", ")", ");", "),", "]", "};", "*/", "/*", "//"}


def substantive(line: str) -> bool:
    """Is this line specific enough that its absence means something?

    A `}` added by one commit and absent from a stale copy proves nothing — the
    copy has plenty of other `}`. Line-set containment is only evidence on
    lines that are unlikely to recur.
    """
    t = line.strip()
    return len(t) > 8 and t not in TRIVIAL


COMMENT_PREFIXES = ("//", "#", "/*", "*", "--", "%")


def is_comment(line: str) -> bool:
    return line.strip().startswith(COMMENT_PREFIXES)


def reverted_lines(repo: Path, path: str, sha: str) -> list[str]:
    """Substantive lines `sha` ADDED to `path` that the worktree copy lacks.

    This is the exact form of "committing this reverts what landed since": if
    the newest commit on a path added lines the worktree file does not contain,
    committing that file removes them. Set containment, not a diff, because the
    stale copy may also have moved things around — position is not the claim,
    presence is.
    """
    show = git(repo, "show", "--format=", "--unified=0", sha, "--", path)
    added = [
        ln[1:]
        for ln in show.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    have = {ln.strip() for ln in (repo / path).read_text(errors="replace").splitlines()}
    lost = [ln for ln in added if substantive(ln) and ln.strip() not in have]
    # Code before comments, order otherwise preserved. A commit's first
    # added lines are usually its explanatory comment block, and prose is
    # the weakest evidence in the set — an identifier like `best_ssr`
    # settles reformat-vs-revert with one grep, a comment does not.
    return sorted(lost, key=is_comment)


def stale_vs_head(
    repo: Path, paths: list[str]
) -> list[tuple[str, float, float, list[str]]]:
    """Paths whose worktree copy would REVERT the newest commit touching them.

    Two stages, because each alone is wrong:

    - **mtime < commit time** is the cheap filter, and on its own it
      false-positives on the commonest shape there is: you edit a file at
      21:03 and commit it at 21:04, so the commit that last touched the path
      IS your own edit. Measured — this flagged two such files here.
    - **line containment** is the confirmation and is what the warning
      actually claims. Only lines the newest commit ADDED and the worktree
      copy LACKS are evidence.

    A path with no HEAD history (newly added) cannot be stale. mtime after the
    commit is the safe direction: a checkout stamps now, so a fresh file always
    looks current.
    """
    out: list[tuple[str, float, float, list[str]]] = []
    for p in paths:
        f = repo / p
        if not f.is_file():
            continue
        head = git(repo, "log", "-1", "--format=%ct %H", "HEAD", "--", p).split()
        if len(head) != 2:
            continue
        commit_t, sha = float(head[0]), head[1]
        mtime = f.stat().st_mtime
        if mtime >= commit_t:
            continue
        lost = reverted_lines(repo, p, sha)
        if lost:
            out.append((p, mtime, commit_t, lost))
    return out


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    selftest()
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

    staged = [p for p in git(repo, "diff", "--cached", "--name-only").splitlines() if p]
    also_dirty = {p for p in git(repo, "diff", "--name-only").splitlines() if p}
    # Signal 3 applies to the DIRTY set too, so "nothing staged" is not an
    # early exit — the revert hazard exists before anything is staged.
    if not staged and not also_dirty:
        print("staged-set audit: nothing staged, working tree clean")
        return 0
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

    print(
        f"staged-set audit: {len(staged)} staged + {len(also_dirty)} dirty path(s) "
        f"in {repo.name}"
    )
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

    stale = stale_vs_head(repo, sorted(set(staged) | also_dirty))
    if stale:
        print(
            f"  STALE-vs-HEAD: {len(stale)} path(s) LACK lines the newest commit on their "
            "own path added — committing these reverts what landed since"
        )
        for p, mt, ct, lost in stale[:8]:
            w = _dt.datetime.fromtimestamp(mt).strftime("%H:%M:%S")
            c = _dt.datetime.fromtimestamp(ct).strftime("%H:%M:%S")
            print(
                f"      {p}  (written {w}, HEAD touched it {c}, "
                f"{len(lost)} line(s) would be lost)"
            )
            # Print the evidence, not only its count: these lines ARE the
            # grep that settles reformat-vs-revert, and a diffstat cannot
            # (rustfmt re-wraps tokens across lines, so `-w` still shows
            # deletions). Reconstructed by hand once already; don't make the
            # next reader do it.
            for ln in lost[:2]:
                print(f"        would lose: {ln.strip()[:96]}")
        if len(stale) > 8:
            print(f"      … {len(stale) - 8} more")

    match len(groups) > 1:
        case True if staged:
            print(
                f"  REVIEW: {len(groups)} editing episodes in one staged set. If you did not "
                "write the older one(s), unstage them — see AGENTS.md on `git add -A`."
            )
        case _ if staged:
            print("  ✓ one editing episode")
        case _:
            print("  (nothing staged — episode clustering not applicable)")
    # Report, never a gate: the refusing version is Issue 709 T3's owner call.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
