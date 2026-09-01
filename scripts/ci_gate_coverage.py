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
  4. It read the WORKING TREE and never asked whether those workflows can RUN.
     A gate on a branch GitHub will never dispatch from is decoration, and this
     script scored it identically to one that runs — see below.

Reachability is a separate axis from coverage, and it dominates
--------------------------------------------------------------
A workflow that covers every axis but never executes covers nothing. This repo
already applies that rule one level down ("treat an uninvoked assertion as
unknown, not as passing"); the reachability columns apply it to the workflows
themselves. Two GitHub rules do the work, and they differ:

  * `schedule` and `workflow_dispatch` fire ONLY from the repository's DEFAULT
    branch. A weekly cron in a file that lives only on `develop` never runs.
  * `push` / `pull_request` are evaluated against the workflow file ON THE
    PUSHED REF, so a develop-only workflow with `branches: [develop]` is fine —
    but one with `branches: [main]` never fires if work lands on develop.

Measured 2026-09-01, this axis changed the verdict for FIVE repos, including
this one: katgpt-rs `full_gate.yml` carries `schedule: 17 4 * * 1` and
`workflow_dispatch` while the default branch (`main`) does not carry the file,
so the weekly run AGENTS.md advertises has never fired. Its comment calls the
schedule "the rot check"; the rot check had rotted.

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


def _git(root: Path, repo: str, *args: str) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(root / repo), *args],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def default_branch(root: Path, repo: str) -> str:
    ref = _git(root, repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    return ref.split("/", 1)[1] if "/" in ref else (ref or "?")


def _on_branch(root: Path, repo: str, branch: str, rel: str) -> bool:
    """Does this workflow file exist on origin/<branch>? Working-tree presence
    is not the question — GitHub reads the file from a ref, never from disk."""
    return _git(root, repo, "ls-tree", "--name-only", f"origin/{branch}", rel) == rel


def _tracked(root: Path, repo: str, rel: str) -> bool:
    """Is this workflow committed at all, on any branch?

    An untracked file is UNFINISHED WORK, not a dead gate — it reaches no
    branch because nobody has pushed it yet. Reporting it as "cannot fire"
    raises a false alarm against a colleague's in-flight edit, which is how a
    report earns the reputation that gets it ignored. Caught live: a sibling
    agent added riir-deployer/rust.yml mid-run and it landed in the dead
    block."""
    return bool(_git(root, repo, "ls-files", "--", rel))

def _on_block(wf: Path) -> str:
    """The `on:` block, comments stripped, or "" if the file declares none.

    Deliberately parsed from raw text rather than with a YAML load: this script
    must not acquire a PyYAML dependency to answer a question about a handful
    of keywords. Shared by both trigger readers so they cannot disagree about
    where the block ends."""
    try:
        raw = wf.read_text(errors="replace")
    except OSError:
        return ""
    body = re.split(r"^jobs:", raw, maxsplit=1, flags=re.M)[0]
    live = "\n".join(l for l in body.splitlines()
                      if not l.lstrip().startswith("#"))
    parts = re.split(r"^on:", live, maxsplit=1, flags=re.M)
    return parts[1] if len(parts) > 1 else ""


def push_gap(root: Path, repo: str, wf: Path, dflt: str) -> str:
    """Why a declared `push` cannot fire — named, because the fix depends on it.

    GitHub evaluates `push` against the workflow file ON THE PUSHED REF, so a
    filter naming a branch that carries no copy of the file is inert no matter
    how many pushes land there. Two different repairs follow from the two
    causes (promote the file to that branch, or widen the filter to a branch
    that has it), and "never fires" alone does not say which."""
    blk = _on_block(wf)
    m = re.search(r"^\s+push:(.*?)(?=^\s{2}\S|\Z)", blk, re.M | re.S)
    if not m:
        return ""
    rel = wf.relative_to(root / repo).as_posix()
    brs = re.search(r"branches:\s*\[([^\]]*)\]", m.group(1))
    names = ([b.strip().strip("'\"") for b in brs.group(1).split(",") if b.strip()]
             if brs else [dflt])
    missing = [b for b in names if not _on_branch(root, repo, b, rel)]
    if not missing:
        return ""
    return (f"push[{','.join(names)}] inert — "
            f"{', '.join(missing)} carries no copy of this file")


def reachable_triggers(root: Path, repo: str, wf: Path, dflt: str) -> set[str]:
    """Which of this workflow's triggers can actually fire.

    Deliberately parsed from the raw `on:` block rather than with a YAML load:
    this script must not acquire a PyYAML dependency to answer a question about
    two keywords, and the block is regular enough to read directly."""
    rel = wf.relative_to(root / repo).as_posix()
    on_blk = _on_block(wf)
    if not on_blk:
        return set()
    out: set[str] = set()
    declared: set[str] = set()
    for kw in ("schedule", "workflow_dispatch", "workflow_call", "push",
               "pull_request"):
        if re.search(rf"^\s+{kw}:", on_blk, re.M):
            declared.add(kw)
    on_default = _on_branch(root, repo, dflt, rel)
    # schedule / workflow_dispatch: default branch only, full stop.
    for kw in ("schedule", "workflow_dispatch"):
        if re.search(rf"^\s+{kw}:", on_blk, re.M) and on_default:
            out.add(kw)
    # workflow_call is invoked BY REF from another repo, so it is reachable
    # from any branch a caller pins. Not a default-branch question.
    if re.search(r"^\s+workflow_call:", on_blk, re.M):
        out.add("workflow_call")
    # push: evaluated against the workflow file on the PUSHED ref, so the
    # question is whether the file exists on a branch the filter names.
    m = re.search(r"^\s+push:(.*?)(?=^\s{2}\S|\Z)", on_blk, re.M | re.S)
    if m:
        brs = re.search(r"branches:\s*\[([^\]]*)\]", m.group(1))
        names = ([b.strip().strip("'\"") for b in brs.group(1).split(",") if b.strip()]
                 if brs else [dflt])
        if any(_on_branch(root, repo, b, rel) for b in names):
            out.add("push")
    # pull_request is CONDITIONAL, not default-branch-bound: the run uses the
    # file from the PR's merge commit, so a develop-only workflow does fire on
    # a PR. Whether one is ever opened is a workflow-policy question git cannot
    # answer — several repos here land work directly on develop and never open
    # one. Reported as its own state rather than folded into either verdict;
    # calling it live would over-report and calling it dead would over-claim.
    if re.search(r"^\s+pull_request:", on_blk, re.M):
        out.add("pull_request?")
    # Declared-but-dead is the finding a workflow-level verdict cannot make: a
    # file can be "reachable" on one trigger while the two its documentation
    # advertises never fire. katgpt-rs full_gate.yml is exactly that shape.
    lost = declared - {t.rstrip("?") for t in out}
    return out | {f"-{t}" for t in lost}


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


def _best(lines: list[str]) -> tuple[list[str], str]:
    """Highest-scoring single cargo command in a pool, scored PER COMMAND.

    Extracted so the repo-wide pool and the per-workflow pools cannot drift
    apart in how they score — the join below is only meaningful if both sides
    answer the same question the same way."""
    best: list[str] = []
    best_cmd = ""
    for cmd in cargo_commands(lines):
        sig = [s for s in SIGNALS if s in cmd]
        if len(sig) > len(best):
            best, best_cmd = sig, cmd
    return best, best_cmd


def _wf_pool(root: Path, repo: str, wf: Path) -> list[str]:
    """One workflow's own lines plus those of every script IT invokes.

    Deliberately NOT deduped against sibling workflows. survey()'s aggregate
    pool follows each script once, which is right for "what does this repo
    run" and wrong for "which workflow starts this command" — a script
    invoked by two workflows belongs to both, and dropping it from the second
    would report that workflow as carrying no compile surface."""
    wl = live_lines(wf)
    out, seen = list(wl), set()
    for m in SCRIPT_RE.finditer("\n".join(wl)):
        rel = m.group(1)
        cand = root / repo / rel
        if cand.is_file() and rel not in seen:
            seen.add(rel)
            out += live_lines(cand)
    return out


def WF_DIR(repo: str) -> Path:
    return GIT_ROOT / repo / ".github" / "workflows"


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
    best, best_cmd = _best(lines)
    dflt = default_branch(root, repo)
    reach = {wf.name: reachable_triggers(root, repo, wf, dflt) for wf in workflows}
    # Per-workflow attribution. The aggregate `best` above answers "does a
    # full-surface command exist in this repo"; it cannot answer "does anything
    # START it", because it pools every workflow into one bag. Keeping the
    # per-workflow score lets the join below cross coverage with reachability
    # without disturbing the columns Issue 701 R2 quotes.
    per_wf = {}
    for wf in workflows:
        wl = _wf_pool(root, repo, wf)
        sigs, cmd = _best(wl)
        in_cmds = {s for c in cargo_commands(wl) for s in SIGNALS if s in c}
        per_wf[wf.name] = {
            "signals": sigs, "cmd": cmd,
            "dynamic": sorted({s for s in SIGNALS if s in "\n".join(wl)} - in_cmds,
                              key=SIGNALS.index)}
    return {"repo": repo, "workflows": len(workflows), "scripts": scripts,
            "default_branch": dflt, "reach": reach, "per_wf": per_wf,
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



def hand_only(rows: list[dict]) -> list[tuple[dict, list[str]]]:
    """Repos whose STRONGEST compile/lint command no automatic trigger starts.

    Separated from the printing so it can be pinned by selftest(): an assertion
    nothing has ever seen fail is not known to be able to fail, which is the
    defect this whole family of scripts exists to catch."""
    # Strength is compared, not mere presence. A first cut asked only "does ANY
    # automatically-triggered workflow carry a cargo signal", and riir-chain
    # slipped through it: the scheduled toolchain_drift.yml names the full-gate
    # flags in a data table, which was enough to mask rust.yml — the repo's
    # actual compile gate, and dispatch-only. A weak automatic gate must not
    # vouch for a strong manual one.
    #
    # Lexicographic (real commands, then data-borne): a signal sitting in a
    # CONFIGS table is worth something — riir-chain's feature matrix is
    # arguably more thorough than any single command — but never enough to
    # outrank a signal in an actual `cargo` invocation.
    AUTOMATIC = {"schedule", "push"}

    def strength(v):
        return (len(v.get("signals", ())), len(v.get("dynamic", ())))

    handonly = []
    for r in rows:
        live_wf = {n for n, t in r["reach"].items()
                   if any(not x.startswith("-") for x in t)}
        # Workflows with no live trigger at all are the `dead` block's finding,
        # already printed above; counting them here would report one defect
        # twice and blur "cannot run" into "runs only when asked".
        carriers = {n for n in live_wf if strength(r["per_wf"].get(n, {})) > (0, 0)}
        if not carriers:
            continue
        auto = {n for n, t in r["reach"].items()
                if AUTOMATIC & {x for x in t if not x.startswith("-")}}
        best_all = max(strength(r["per_wf"][n]) for n in carriers)
        best_auto = max([strength(r["per_wf"][n]) for n in carriers & auto],
                        default=(0, 0))
        if best_auto >= best_all:
            continue
        handonly.append((r, sorted(n for n in carriers
                                   if strength(r["per_wf"][n]) == best_all)))
    return handonly


def selftest() -> None:
    """Pin the join's verdict on five shapes, including the regression it hit.

    Case C is the one that matters: a scheduled workflow naming the full-gate
    flags in a DATA table must not vouch for a dispatch-only workflow that runs
    them. A first cut asked only "is any automatic workflow a carrier" and let
    riir-chain through."""
    def repo(name, per_wf, reach):
        return {"repo": name, "per_wf": per_wf, "reach": reach}
    FULL = {"signals": ["clippy", "--all-targets", "--all-features"], "dynamic": []}
    WEAK = {"signals": [], "dynamic": ["clippy", "--all-targets", "--all-features"]}
    NONE = {"signals": [], "dynamic": []}
    cases = [
        # (row, must_be_flagged, why)
        (repo("A-auto-full", {"g.yml": FULL},
              {"g.yml": {"push", "schedule"}}), False,
         "a full gate on push+schedule is covered"),
        (repo("B-manual-only", {"g.yml": FULL},
              {"g.yml": {"workflow_dispatch"}}), True,
         "dispatch-only compile gate is a button, not a schedule"),
        (repo("C-weak-auto-masks", {"rust.yml": FULL, "drift.yml": WEAK},
              {"rust.yml": {"workflow_dispatch"},
               "drift.yml": {"schedule", "workflow_dispatch"}}), True,
         "a data-borne signal on a schedule must not vouch for a manual gate"),
        (repo("D-dead-carrier", {"g.yml": FULL}, {"g.yml": set()}), False,
         "a workflow with no live trigger is the dead block's finding"),
        (repo("E-equal", {"a.yml": FULL, "b.yml": FULL},
              {"a.yml": {"workflow_dispatch"}, "b.yml": {"push"}}), False,
         "the same surface also runs on push"),
    ]
    flagged = {r["repo"] for r, _ in hand_only([c[0] for c in cases])}
    for row, want, why in cases:
        if (row["repo"] in flagged) != want:
            raise SystemExit(
                f"✗ hand_only self-test FAILED on {row['repo']}\n"
                f"  expected flagged={want}, got flagged={row['repo'] in flagged}\n"
                f"  {why}")


def main(argv: list[str]) -> int:
    selftest()
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

    # ── Reachability, reported separately ────────────────────────────────────
    # Kept out of the rows above deliberately: those columns answer "what would
    # this gate cover", and Issue 701 R2 quotes them. This answers the prior
    # question — "does it run at all" — and a repo can score FULL above and
    # zero here. Merging them would silently restate 701's measured table.
    # A repo with no remote refs (never pushed, or fetch unavailable) yields no
    # default branch. That is an unmeasured repo, NOT a dead workflow — scoring
    # it as a finding would be the confident-green-over-nothing inversion this
    # whole family of scripts exists to prevent, pointed the other way.
    dead: list[str] = []
    unknown: list[str] = []
    partial: list[str] = []
    for r in rows:
        for name, trig in sorted(r["reach"].items()):
            live = {t for t in trig if not t.startswith("-")}
            lost = sorted(t[1:] for t in trig if t.startswith("-"))
            if lost and live:
                partial.append(f"{r['repo']}/{name}: declared {' '.join(lost)} — never fires")
            if r["default_branch"] == "?":
                unknown.append(f"{r['repo']}/{name}")
            elif not live and not _tracked(GIT_ROOT, r["repo"],
                                           f".github/workflows/{name}"):
                unknown.append(f"{r['repo']}/{name} (untracked — not committed yet)")
            elif not live:
                dead.append(f"{r['repo']}/{name}")
            elif live == {"pull_request?"}:
                unknown.append(f"{r['repo']}/{name} (PR-only)")
    print(f"\n▸ reachability (default branch vs where the workflow file lives)")
    for r in rows:
        if not r["reach"]:
            continue
        bits = []
        for name, trig in sorted(r["reach"].items()):
            live = sorted(t for t in trig if not t.startswith("-"))
            bits.append(f"{name}[{','.join(live) if live else 'UNREACHABLE'}]")
        print(f"  {r['repo']:<22} default={r['default_branch']:<8} " + "  ".join(bits))
    if dead:
        print(f"\n  {len(dead)} workflow(s) cannot fire from any trigger — a gate "
              f"that never runs is decoration, not coverage:")
        for d in dead:
            print(f"    - {d}")
        print("  schedule/workflow_dispatch need the file on the DEFAULT branch; "
              "push/pull_request\n  need it on a branch their filter names.")
    else:
        print("  all workflows have at least one live trigger")
    if partial:
        print(f"\n  {len(partial)} workflow(s) RUN but lost a declared trigger — the "
              f"file fires on one\n  trigger while another it declares (and its docs "
              f"may advertise) never does:")
        for w in partial:
            print(f"    ! {w}")
    if unknown:
        print(f"\n  {len(unknown)} workflow(s) NOT classified — no remote refs to "
              f"read a default branch from, or PR-only in a repo whose\n  policy "
              f"this script cannot see. Unmeasured, not clean:")
        for u in unknown:
            print(f"    ? {u}")

    # ── The join: coverage x reachability ────────────────────────────────────
    # The two blocks above are each honest on its own and, read one after the
    # other, add up to a claim neither of them makes. The top table credits
    # riir-neuron-db with `--all-targets --all-features`; the reachability
    # table says its rust.yml fires on `workflow_dispatch` alone. Nothing
    # crossed them, so the repo read as covered while the command it was
    # credited for only ever ran when a human clicked it. A dispatch-only gate
    # is a button, not a schedule — the same "decoration, not coverage" the
    # dead-workflow block above says out loud, one step less obvious because
    # the workflow genuinely can run.
    #
    # Note what this deliberately does NOT do: it leaves the columns above
    # untouched. They answer "what would this gate cover" and Issue 701 R2
    # quotes their numbers; this answers "does anything start it".
    handonly = hand_only(rows)
    if handonly:
        print(f"\n▸ {len(handonly)} repo(s) whose compile/lint gate fires ONLY by "
              f"hand — the command is\n  real and the workflow can run, but no "
              f"schedule and no push ever starts it:")
        for r, carriers in handonly:
            print(f"    ⌾ {r['repo']}")
            for n in carriers:
                trig = sorted(t for t in r["reach"].get(n, ()) if not t.startswith("-"))
                sig = (" ".join(r["per_wf"][n]["signals"])
                       or " ".join(r["per_wf"][n]["dynamic"]) + " (in data)")
                print(f"        {n}[{','.join(trig) or 'UNREACHABLE'}]  {sig}")
                gap = push_gap(GIT_ROOT, r["repo"], WF_DIR(r["repo"]) / n,
                               r["default_branch"])
                if gap:
                    print(f"          └─ {gap}")
        print("  Several of these are a DOCUMENTED owner call (main-only, to spend\n"
              "  no Actions minutes on develop pushes) — read the workflow preamble\n"
              "  before filing. The finding is not that the choice is wrong; it is\n"
              "  that a main-only push cannot fire while main carries no copy of\n"
              "  the file, so the intended promote-to-main trigger is inert too.")
    else:
        print("\n▸ every repo carrying a compile/lint command has an automatic "
              "trigger for it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
