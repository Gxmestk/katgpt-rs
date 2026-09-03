#!/usr/bin/env python3
"""Gate katgpt-rs against Issue 713: a green `0 passed` over an empty binary.

`scripts/cfg_gated_target_audit.py` is a REPORT (exit 0 always) because two of
its classes — platform predicates and `any(...)` feature sets — are shapes
`required-features` genuinely cannot express, and a report that cries wolf on
those gets ignored on the ones cargo can fix. This file is the GATE built on
top of it: it consumes the report's `--json` and asserts pins committed in
`scripts/cfg_gated_floors.txt`.

Why the split, rather than making the auditor exit 1
-----------------------------------------------------
The auditor's job is to be runnable over any repo, including the 18 siblings
whose owners have not taken Issue 713 T3 yet. An auditor that exits 1 on those
is an auditor nobody runs. Keeping the verdict in a separate, katgpt-rs-scoped
file lets the report stay honest and the gate stay sharp.

The pin that matters is `max_load_bearing = 0`
-----------------------------------------------
Issue 713 T2 armed all 39 of katgpt-rs's load-bearing SILENT-NOW targets. So
the value is 0 today, and 0 is a value a gate can hold: any NEW target whose
filename says `goat` / `gate` / `g<N>` / `drill` / `proof` / ... and which is
gated on a default-off feature without a `required-features` row reds this gate
on the push that adds it — before its green zero is ever cited as evidence.

Two-sided, because a ceiling cannot fail once the instrument dies
-----------------------------------------------------------------
An auditor whose regex stops recognising `#![cfg(...)]` reports SILENT-NOW 0
and passes every ceiling ever written. Two of the four pins are therefore
FLOORS on the population it claims to have scanned. The auditor's own
`selftest()` also runs on every invocation and is inherited here for free,
since this gate shells out to it — a blind auditor fails its selftest, the
subprocess exits non-zero, and this gate reds rather than reporting a green
zero. That inheritance is asserted in `selftest()` below, not assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDITOR = REPO_ROOT / "scripts" / "cfg_gated_target_audit.py"
IGNORE_AUDITOR = REPO_ROOT / "scripts" / "all_ignored_target_audit.py"
PINS_FILE = REPO_ROOT / "scripts" / "cfg_gated_floors.txt"

REQUIRED_PINS = (
    "max_load_bearing",
    "max_silent_now",
    "min_targets",
    "min_gated",
    "max_reasonless_ignores",
    "min_ignore_targets",
)


def read_pins() -> dict[str, int]:
    """Committed expectations. A missing pin is an error, never a skip."""
    pins: dict[str, int] = {}
    for raw in PINS_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split()
        pins[key] = int(value)
    missing = [k for k in REQUIRED_PINS if k not in pins]
    if missing:
        raise SystemExit(f"✗ {PINS_FILE.name} is missing pins: {missing}")
    return pins


def check(m: dict[str, int], pins: dict[str, int]) -> list[str]:
    """Return the failure lines. Empty means pass.

    Pure and dict-in so selftest() can drive it with synthetic measurements —
    a gate whose verdict logic can only be exercised by the real corpus is a
    gate whose verdict logic is never exercised in the failing direction.
    """
    fail: list[str] = []

    lb = m["silent_now_load_bearing"]
    if lb > pins["max_load_bearing"]:
        fail.append(
            f"load-bearing SILENT-NOW {lb} > pinned {pins['max_load_bearing']} — "
            "a target whose name says its green is evidence reports `ok. 0 passed` "
            "over an EMPTY binary. Run scripts/cfg_gated_target_audit.py . for the "
            "paths; it prints the exact [[test]] row to add."
        )

    sn = m["silent_now"]
    if sn > pins["max_silent_now"]:
        fail.append(
            f"SILENT-NOW {sn} > pinned {pins['max_silent_now']} — "
            f"{sn - pins['max_silent_now']} newly-added target(s) are gated on a "
            "default-off feature with no required-features row. Add the row (the "
            "report prints it), or raise the pin with a reason if the target is "
            "deliberately auto-discovered."
        )

    # The blindness floors. These fail in the opposite direction from the
    # ceilings above, and that is the whole point of having them.
    if m["scanned"] < pins["min_targets"]:
        fail.append(
            f"only {m['scanned']} targets scanned, floor {pins['min_targets']} — "
            "the auditor is not seeing this repo. A shrunken population makes "
            "every ceiling above pass vacuously."
        )
    if m["gated"] < pins["min_gated"]:
        fail.append(
            f"only {m['gated']} whole-file #![cfg] gates recognised, floor "
            f"{pins['min_gated']} — the gate-recognition regex has gone blind. "
            "SILENT-NOW would read 0 for the wrong reason."
        )

    # Issue 713 T6's axis. `#[ignore]` itself is NOT gated — it is the correct
    # marker for a slow or hardware-gated test — but a reasonless one is: it
    # leaves no reader able to tell a deliberate manual-only test from one
    # parked during a refactor and forgotten. Ceiling + its own blindness floor,
    # for the same reason as every other pair here.
    rl = m["reasonless_targets"]
    if rl > pins["max_reasonless_ignores"]:
        fail.append(
            f"{rl} target(s) where EVERY test is #[ignore]d carry at least one "
            f"#[ignore] with NO reason string (pinned "
            f"{pins['max_reasonless_ignores']}). Such a target prints "
            "`ok. 0 passed` forever and says nothing about why. Add "
            '`#[ignore = "..."]` — cargo prints the string in its output. Run '
            "scripts/all_ignored_target_audit.py . for the paths."
        )
    if m["ignore_scanned"] < pins["min_ignore_targets"]:
        fail.append(
            f"the ignore audit scanned only {m['ignore_scanned']} targets, floor "
            f"{pins['min_ignore_targets']} — its `#[test]` recogniser has gone "
            "blind, and a reasonless-ignore ceiling of zero is then satisfied "
            "perfectly by seeing nothing."
        )
    return fail


def run_auditor(script: Path) -> dict:
    """Shell out. The auditor's selftest() runs, and its failure is ours."""
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"✗ {script.name} failed (exit {proc.returncode}) — its selftest() or "
            f"its parse is broken, so there is no measurement to gate on:\n"
            f"{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def measure() -> dict[str, int]:
    """Both auditors, merged into one measurement dict."""
    data = run_auditor(AUDITOR)
    # Keyed by directory NAME, and the checkout directory is not guaranteed to
    # be `katgpt-rs` (a fork, a worktree, a rename). Exactly one repo was
    # requested, so exactly one row is the correct answer regardless of what it
    # is called; anything else is refused rather than averaged.
    if len(data) != 1:
        raise SystemExit(
            f"✗ auditor returned {len(data)} rows for one repo ({list(data)}) — "
            "refusing to report a verdict over an ambiguous population"
        )
    m = dict(next(iter(data.values())))

    ig = run_auditor(IGNORE_AUDITOR)
    if len(ig) != 1:
        raise SystemExit(
            f"✗ ignore auditor returned {len(ig)} rows for one repo ({list(ig)})"
        )
    igm = next(iter(ig.values()))
    m["reasonless_targets"] = igm["reasonless_targets"]
    m["ignore_scanned"] = igm["scanned"]
    m["all_ignored"] = igm["all_ignored"]
    return m


def selftest() -> None:
    """Pin the VERDICT logic, both directions. Runs on every invocation.

    The failure this guards is the one this whole issue family is about: a
    check that can only pass. Each ceiling and each floor is driven over its
    boundary here, so a `>` that becomes a `>=`, or a floor comparison written
    the wrong way round, fails at startup instead of printing a green.
    """
    pins = {
        "max_load_bearing": 0,
        "max_silent_now": 63,
        "min_targets": 700,
        "min_gated": 400,
    }
    pins["max_reasonless_ignores"] = 0
    pins["min_ignore_targets"] = 400
    ok = {
        "scanned": 921,
        "gated": 541,
        "silent_now": 63,
        "silent_now_load_bearing": 0,
        "reasonless_targets": 0,
        "ignore_scanned": 653,
    }
    assert check(ok, pins) == [], "the observed measurement must pass"

    # Ceilings: one over the pin fails, exactly at the pin passes.
    assert check({**ok, "silent_now_load_bearing": 1}, pins), "load-bearing +1 passed"
    assert check({**ok, "silent_now": 64}, pins), "SILENT-NOW +1 passed"
    assert check({**ok, "silent_now": 62}, pins) == [], "an IMPROVEMENT failed"

    # Floors: the blind-instrument case. This is the one that matters — a
    # report of all zeroes satisfies both ceilings.
    assert check({**ok, "reasonless_targets": 1}, pins), "a reasonless ignore passed"

    blind = {
        "scanned": 0,
        "gated": 0,
        "silent_now": 0,
        "silent_now_load_bearing": 0,
        "reasonless_targets": 0,
        "ignore_scanned": 0,
    }
    assert len(check(blind, pins)) == 3, "a blind auditor passed the gate"

    # A shrunken-but-nonzero population must still red, or the floor is only
    # catching total death and not degradation.
    assert check({**ok, "gated": 399}, pins), "a degraded gate-recogniser passed"


def main() -> int:
    selftest()
    pins = read_pins()
    m = measure()

    print(
        f"cfg-gated target floor gate — {REPO_ROOT.name}: "
        f"{m['scanned']} targets, {m['gated']} #![cfg]-gated, {m['covered']} covered, "
        f"SILENT-NOW {m['silent_now']} (load-bearing {m['silent_now_load_bearing']}), "
        f"{m['all_ignored']} all-#[ignore]d target(s), "
        f"{m['reasonless_targets']} of them reasonless"
    )

    fail = check(m, pins)
    if fail:
        for line in fail:
            print(f"  ✗ {line}")
        print(f"✗ cfg-gated floor gate FAILED — {len(fail)} pin(s) violated")
        return 1

    # An improvement is not a failure, but an un-re-pinned improvement is a
    # ratchet that has quietly stopped ratcheting.
    if m["silent_now"] < pins["max_silent_now"]:
        print(
            f"  ! SILENT-NOW is {m['silent_now']}, pinned at {pins['max_silent_now']} — "
            f"re-pin max_silent_now to {m['silent_now']} in {PINS_FILE.name} to keep "
            "the ratchet tight (not a failure)"
        )

    print(f"✓ cfg-gated floor gate PASSED — {len(REQUIRED_PINS)} pins held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
