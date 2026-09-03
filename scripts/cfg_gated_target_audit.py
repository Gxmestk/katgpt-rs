#!/usr/bin/env python3
"""Find cargo targets whose whole-file `#![cfg(...)]` can zero them SILENTLY.

A test file that opens with `#![cfg(feature = "x")]` compiles to an empty
binary when `x` is off. Cargo then prints

    running 0 tests
    test result: ok. 0 passed; 0 failed; ...

and exits 0. That is indistinguishable from a real pass, and it is how eleven
real assertions in riir-train's `nora_phase1_hooks` reported as a green suite
having run none (riir-train `5821cba9`), and how riir-clippy's `t063` — the
harness behind a whole benchmark — did the same (riir-clippy `19beece`).

`required-features` is the fix, and it is not cosmetic: it changes the outcome
from a silent green to

    error: target `t063_tpr_structure` in package `riir-clippy` requires the
    features: `tpr_structure`                                   (exit 101)

**The `#![cfg]` protects the COUNT. `required-features` protects the READER.**
Both are needed; neither substitutes for the other.

## What this is NOT

A **report, not a gate** — always exit 0, same discipline as
`ci_gate_coverage.py` and `staged_set_audit.py`. Two reasons it must not block:

1. A `cfg` on `target_os` / `target_arch` / `miri` **cannot** be expressed as
   `required-features`. Those files are correctly gated and are reported in a
   separate class, never as a defect.
2. `any(...)` of several features is a legitimate shape that cargo's
   AND-only `required-features` cannot express (riir-ai's `ci_feature_guard.sh`
   documents one). Reported as ITS own class rather than flattened into the
   defect list — a report that cries wolf on the one shape cargo cannot fix
   gets ignored on the shapes it can.

## Population is derived, expectations are committed

The repo set comes from the workspace walk (a root `BOUNDARY.md` **and** a
`.git` dir), never a typed list — deriving both the population and the
expectation from the same walk is what makes a cross-repo gate permanently
green.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Target dirs and the cargo table that declares them.
TARGET_KINDS = {"tests": "test", "benches": "bench", "examples": "example"}

# `cfg` predicates that `required-features` cannot express. A file gated on one
# of these is correctly gated; it is not a finding.
NON_FEATURE_PREDICATES = (
    "target_os",
    "target_arch",
    "target_family",
    "target_env",
    "target_pointer_width",
    "target_endian",
    "miri",
    "debug_assertions",
    "unix",
    "windows",
    "test",
)

# A whole-file inner attribute at the top of a file. `#![cfg(...)]` only —
# `#![allow]`, `#![doc]` etc. are not gates.
INNER_CFG = re.compile(r"^\s*#!\s*\[\s*cfg\s*\(", re.MULTILINE)

FEATURE_IN_CFG = re.compile(r'feature\s*=\s*"([^"]+)"')


@dataclass
class Finding:
    repo: str
    manifest: str
    kind: str
    name: str
    path: str
    features: list[str]
    predicates: list[str]
    declared: bool  # a [[test]]/[[bench]]/[[example]] row exists at all
    reason: str
    # Is EVERY gating feature reachable from this crate's `default`? If so the
    # target still runs on a plain `cargo test` and only vanishes under
    # `--no-default-features` — a real hazard, but a rarer one. If ANY gating
    # feature is default-off, a plain `cargo test` compiles the file to nothing
    # and reports a green 0-pass. That is the severity split, and without it
    # the headline count pools two populations an order of magnitude apart.
    default_on: bool = False


@dataclass
class RepoReport:
    repo: str
    scanned: int = 0
    gated: int = 0
    findings: list[Finding] = field(default_factory=list)
    unexpressible: list[Finding] = field(default_factory=list)
    any_of: list[Finding] = field(default_factory=list)
    covered: int = 0

    def silent_now(self) -> list[Finding]:
        """Findings that zero on a PLAIN `cargo test` — the severe class."""
        return [f for f in self.findings if not f.default_on]

    def silent_latent(self) -> list[Finding]:
        """Findings that zero only under `--no-default-features`."""
        return [f for f in self.findings if f.default_on]


def cfg_body(text: str) -> str | None:
    """The balanced body of the FIRST whole-file `#![cfg(...)]`, or None.

    Balanced-paren scan rather than a regex: `cfg(all(feature = "a",
    feature = "b"))` is the common shape and a non-greedy `\\)` stops at the
    first inner paren, silently reporting one feature where there are two.
    """
    m = INNER_CFG.search(text)
    if not m:
        return None
    i = text.index("(", m.start())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def default_closure(features: dict) -> set[str]:
    """Own-crate features reachable from `default`.

    Within-manifest only, and deliberately so: `dep/feat` and `dep:dep` entries
    enable a DEPENDENCY's feature, which cannot gate a `#![cfg(feature = ...)]`
    in THIS crate. Including them would over-approximate the default set and
    silently downgrade real findings to the latent class.
    """
    out: set[str] = set()
    stack = list(features.get("default", []) or [])
    while stack:
        f = stack.pop()
        if f in out or "/" in f or f.startswith("dep:") or f.startswith("?"):
            continue
        out.add(f)
        stack.extend(features.get(f, []) or [])
    return out


def scan_manifest(repo: Path, manifest: Path, rep: RepoReport) -> None:
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (tomllib.TOMLDecodeError, OSError):
        return

    # Declared targets, keyed by the FILE they build, not by their name.
    #
    # Keying on `(kind, name)` and matching it against the filename stem is
    # wrong, and wrong in the direction that manufactures findings: a row may
    # carry an explicit `path`, and then its `name` need not resemble the
    # file's stem at all. katgpt-rs's four `*.goat.rs` targets are declared as
    # `bench_256_kv_outer_goat` (underscored) pointing at
    # `bench_256_kv_outer.goat.rs` (dotted) with `required-features` already
    # present — a stem match reports all four as defects, and "fixing" them
    # adds a SECOND target for the same file, which cargo warns about and which
    # breaks `--test <name>` resolution.
    #
    # So resolve every row to an absolute path (explicit `path`, else the
    # conventional `<dir>/<name>.rs`) and key on that. `name` is kept only for
    # the suggested fix line.
    declared: dict[Path, bool] = {}
    declared_name: dict[Path, str] = {}
    crate_dir = manifest.parent
    for kind, dirname in (("test", "tests"), ("bench", "benches"), ("example", "examples")):
        for row in data.get(kind, []) or []:
            if not isinstance(row, dict):
                continue
            rel = row.get("path") or (
                f"{dirname}/{row['name']}.rs" if "name" in row else None
            )
            if rel is None:
                continue
            resolved = (crate_dir / rel).resolve()
            # A file may legitimately back two rows (different feature sets).
            # Covered if ANY row declares required-features — the question is
            # whether a reader can be silently fooled, and one guarded row is
            # enough to make the omission visible.
            declared[resolved] = declared.get(resolved, False) or bool(
                row.get("required-features")
            )
            declared_name.setdefault(resolved, row.get("name", ""))

    defaults = default_closure(data.get("features", {}) or {})
    crate = manifest.parent
    for dirname, kind in TARGET_KINDS.items():
        d = crate / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.rs")):
            rep.scanned += 1
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = cfg_body(text)
            if body is None:
                continue
            rep.gated += 1

            key = f.resolve()
            name = declared_name.get(key) or f.stem
            feats = sorted(set(FEATURE_IN_CFG.findall(body)))
            preds = sorted({p for p in NON_FEATURE_PREDICATES if p in body})
            has_rf = declared.get(key)

            base = Finding(
                repo=rep.repo,
                manifest=str(manifest.relative_to(repo)),
                kind=kind,
                name=name,
                path=str(f.relative_to(repo)),
                features=feats,
                predicates=preds,
                declared=key in declared,
                reason="",
            )

            if has_rf:
                rep.covered += 1
                continue
            if not feats:
                # Purely a platform/arch/miri gate — correctly gated, and
                # `required-features` could not express it.
                base.reason = "non-feature cfg (required-features cannot express)"
                rep.unexpressible.append(base)
                continue
            if "any(" in body.replace(" ", "") and len(feats) > 1:
                base.reason = "any(feature,...) — cargo's required-features is AND-only"
                rep.any_of.append(base)
                continue
            base.default_on = bool(feats) and all(x in defaults for x in feats)
            base.reason = (
                "declared without required-features"
                if base.declared
                else "auto-discovered, no [[%s]] row at all" % kind
            )
            rep.findings.append(base)


def manifests(repo: Path) -> list[Path]:
    out = []
    root = repo / "Cargo.toml"
    if root.is_file():
        out.append(root)
    for d in ("crates", "."):
        base = repo / d
        if not base.is_dir():
            continue
        for m in sorted(base.glob("*/Cargo.toml")):
            if m not in out and ".git" not in m.parts and "target" not in m.parts:
                out.append(m)
    return out


def audit(repo: Path) -> RepoReport:
    rep = RepoReport(repo=repo.name)
    for m in manifests(repo):
        scan_manifest(repo, m, rep)
    return rep


def derive_repos(workspace: Path) -> list[Path]:
    """A root BOUNDARY.md AND a `.git` DIR — never a typed list."""
    return sorted(
        d
        for d in workspace.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )


def selftest() -> None:
    """Pin the parse shapes. Runs on EVERY invocation.

    Without this the audit degrades silently: a regex regression makes it
    recognise fewer gates and still print a confident `0 findings`, which is
    the exact failure mode it exists to catch, committed by the tool that
    catches it.
    """
    cases = [
        # (source, expected cfg body or None)
        ('#![cfg(feature = "x")]\nfn a() {}', 'feature = "x"'),
        # Balanced-paren: the nested-all case a non-greedy regex truncates.
        (
            '#![cfg(all(feature = "a", feature = "b"))]\n',
            'all(feature = "a", feature = "b")',
        ),
        ("#![allow(clippy::pedantic)]\nfn a() {}", None),
        ('//! doc\n\n#![cfg(feature = "y")]\n', 'feature = "y"'),
        ("fn a() {}\n", None),
    ]
    for src, want in cases:
        got = cfg_body(src)
        assert got == want, f"cfg_body({src!r}) = {got!r}, want {want!r}"

    # The nested case must yield BOTH features — the bug a truncating scan hides.
    body = cfg_body('#![cfg(all(feature = "a", feature = "b"))]\n')
    assert sorted(set(FEATURE_IN_CFG.findall(body))) == ["a", "b"], "nested features lost"

    # A platform gate must carry no feature, so it lands in `unexpressible`
    # rather than the defect list.
    body = cfg_body('#![cfg(target_os = "macos")]\n')
    assert FEATURE_IN_CFG.findall(body) == [], "platform gate read as a feature gate"
    assert any(p in body for p in NON_FEATURE_PREDICATES), "platform predicate not recognised"

    # A mixed gate IS a finding: the feature half is expressible.
    body = cfg_body('#![cfg(all(target_os = "macos", feature = "gpu"))]\n')
    assert FEATURE_IN_CFG.findall(body) == ["gpu"], "mixed gate lost its feature"

    # The default closure is TRANSITIVE, and stops at the crate boundary.
    feats = {
        "default": ["a"],
        "a": ["b", "dep:serde", "other/x"],
        "b": [],
        "off": [],
    }
    got = default_closure(feats)
    assert got == {"a", "b"}, f"default closure = {got}, want {{a, b}}"
    assert "off" not in got, "a non-default feature entered the closure"
    assert "other/x" not in got, "a dependency feature entered the OWN-crate closure"
    # Path resolution: a row with an explicit `path` must claim THAT file, not
    # the file whose stem matches its name. This is the false positive that
    # shipped in the first cut — it reported four already-guarded katgpt-rs
    # targets as defects, and "fixing" them added a duplicate target per file.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "tests").mkdir()
        (root / "tests" / "a.goat.rs").write_text('#![cfg(feature = "x")]\n')
        (root / "Cargo.toml").write_text(
            '[package]\nname = "p"\nversion = "0.0.0"\n\n'
            "[features]\nx = []\n\n"
            '[[test]]\nname = "a_goat"\npath = "tests/a.goat.rs"\n'
            'required-features = ["x"]\n'
        )
        r = RepoReport(repo="p")
        scan_manifest(root, root / "Cargo.toml", r)
        assert r.gated == 1, f"gate not seen: {r.gated}"
        assert r.covered == 1, "a path-declared, required-features row read as UNCOVERED"
        assert not r.findings, f"false positive: {[f.path for f in r.findings]}"

    # An empty/absent [features] table means NOTHING is default-on, so every
    # gated target is severe rather than latent. The wrong way round would
    # silently downgrade every finding in a crate with no feature table.
    assert default_closure({}) == set(), "empty feature table produced defaults"


def main(argv: list[str]) -> int:
    selftest()

    if len(argv) > 1:
        repos = [Path(a).resolve() for a in argv[1:]]
        scope = "argument"
    else:
        here = Path(__file__).resolve().parent.parent
        repos = derive_repos(here.parent)
        scope = "derived (BOUNDARY.md + .git)"

    print(f"cfg-gated target audit — {len(repos)} repo(s), population {scope}\n")
    header = (
        f"{'repo':<24} {'targets':>8} {'#![cfg]':>8} {'w/ req-f':>9} "
        f"{'SILENT-NOW':>11} {'latent':>7} {'platform':>9} {'any()':>6}"
    )
    print(header)
    print("-" * len(header))

    reports = [audit(r) for r in repos]
    total = 0
    latent = 0
    for rep in reports:
        total += len(rep.silent_now())
        latent += len(rep.silent_latent())
        print(
            f"{rep.repo:<24} {rep.scanned:>8} {rep.gated:>8} {rep.covered:>9} "
            f"{len(rep.silent_now()):>11} {len(rep.silent_latent()):>7} "
            f"{len(rep.unexpressible):>9} {len(rep.any_of):>6}"
        )

    print(
        f"\nSILENT-NOW {total}: a plain `cargo test --test <name>` compiles the file to\n"
        f"nothing and prints `0 passed` with exit 0. latent {latent}: every gating\n"
        f"feature is default-on, so it only vanishes under `--no-default-features`.\n"
    )
    for rep in reports:
        if not rep.silent_now():
            continue
        print(f"  {rep.repo}")
        for f in sorted(rep.silent_now(), key=lambda x: x.path):
            feats = ", ".join(f.features)
            plat = f" +[{', '.join(f.predicates)}]" if f.predicates else ""
            print(f"    {f.path}")
            print(f"      cfg: feature = {feats}{plat}  —  {f.reason}")
            print(
                f'      fix: [[{f.kind}]] / name = "{f.name}" / '
                f"required-features = {f.features!r}"
            )
        print()

    if total == 0:
        print("  (none)\n")

    print(
        "Report, not a gate — exit 0 always. The two non-defect classes above are\n"
        "shapes `required-features` CANNOT express (platform predicates; any-of\n"
        "feature sets, since cargo's required-features is AND-only). Reporting them\n"
        "apart is what keeps the SILENT column worth reading."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
