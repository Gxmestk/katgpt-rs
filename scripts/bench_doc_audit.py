#!/usr/bin/env python3
"""
Cross-repo bench-doc label auditor.

Catches the drift class where a Cargo feature has been promoted to
`default = [...]` in a Phase N+1 commit, but the original benchmark doc
still labels it as `(opt-in)` / `(default-off)` / etc. Also catches the
inverse (doc claims default but feature is actually opt-in).

Re-runnable. Run after every feature promotion to catch label drift early.

Usage:
    python3 scripts/bench_doc_audit.py                # audit this repo
    python3 scripts/bench_doc_audit.py /git/riir-ai   # audit a specific repo
    python3 scripts/bench_doc_audit.py /git           # walk all repos under /git

Exit code: 0 if no mismatches, 1 if any mismatch found.

Strategy
--------
1. Walk every `Cargo.toml` in the repo (skipping `target/`).
2. From each `[features]` table, collect:
   - the set of all defined feature names
   - the set of names that appear in `default = [...]`
   (Names with `pkg/foo/` or `dep:foo` form are normalized to `foo`.)
3. Walk every `.benchmarks/*.md` and `.docs/*.md` looking for lines
   matching `Feature (gate|flag|s)?:`.
4. On each such line, find every `` `feature_name` (status-word) `` token.
5. If status-word parses to default/opt-in, cross-check against the
   Cargo-derived sets. Only feature names that actually exist as Cargo
   features are checked — this avoids false positives like `chain_curator`
   which is a use-case name, not a Cargo feature.

Status word vocabulary
----------------------
default-words: default, default-on, always-on, promoted, ...
opt-in-words:  opt-in, default-off, gated, optional, experimental, ...
"""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path

import tomllib

STATUS_WORDS_DEFAULT = {
    "default",
    "default-on",
    "default-on-since",
    "defaultfeature",
    "promoted",
    "promoted-to-default",
    "always-on",
}

STATUS_WORDS_OPTIN = {
    "opt-in",
    "optin",
    "off",
    "default-off",
    "gated",
    "feature-gated",
    "optional",
    "experimental",
    "disabled-by-default",
}

# Phrases that contain the substring "default" but mean OPT-IN.
# These must be checked before STATUS_WORDS_DEFAULT substrings, otherwise
# the bare "default" substring inside them would falsely trigger a default
# verdict (the session-13 parser bug).
OPTIN_DEFAULT_PHRASES = (
    "default-off",
    "off by default",
    "off-by-default",
    "not default",
    "not in default",
    "not promoted",
    "stays opt-in",
    "stays off",
)


def _parse_feature_spec(spec: str) -> str | None:
    """Normalize a Cargo feature dep spec to a bare feature name.

    Cargo feature dep specs come in several forms:
      "foo"                -> "foo"
      "dep:foo"            -> None  (dep activation, not a feature)
      "crate/foo"          -> "foo"
      "crate/foo/bar"      -> "foo"  (path-style)
      "pkg?/foo"           -> "foo"  (weak dep)
    """
    if spec.startswith("dep:"):
        return None
    # strip optional pkg prefix
    name = spec.split("/")[-1]
    return name or None


# Directories that never contain a source manifest but dominate the walk.
# `target` alone held 117 GB / ~1.3M entries in katgpt-rs when this was written.
PRUNE_DIRS = frozenset({"target", ".git", "node_modules", "__pycache__", ".venv"})


def iter_cargo_manifests(repo_root: Path):
    """Every source Cargo.toml, pruning build/VCS dirs DURING the walk.

    The previous form was `repo_root.rglob("Cargo.toml")` followed by
    `if "target" in cargo.parts: continue` — which filters AFTER pathlib has
    already descended into target/. Measured on katgpt-rs: 1,470,215 entries in
    10.7s unpruned versus 143,275 in 0.3s pruned, and four call sites each paid
    it once per run.
    """
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        if "Cargo.toml" in filenames:
            yield Path(dirpath) / "Cargo.toml"


def find_cargo_defaults(repo_root: Path) -> set[str]:
    """All feature names that appear in any Cargo.toml's `default = [...]`,
    **plus** their transitive closure across the feature-dependency graph.

    Without transitive resolution, features that are enabled indirectly
    (e.g. `sense_lod` is in default and enables `slod` via its dep list)
    would be wrongly flagged as opt-in (the session-13 false-positive
    class on `slod`).
    """
    # Collect: for each Cargo.toml, (default_set, feature_deps_map)
    # feature_deps_map[feat_name] = set of feat-names it enables
    by_manifest: list[tuple[set[str], dict[str, set[str]]]] = []
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            print(f"WARN: could not parse {cargo}: {e}", file=sys.stderr)
            continue
        feats = data.get("features", {})
        if not feats:
            continue
        default_list = feats.get("default", [])
        defaults: set[str] = set()
        deps: dict[str, set[str]] = {}
        for name, spec in feats.items():
            normalized: set[str] = set()
            if isinstance(spec, list):
                for s in spec:
                    n = _parse_feature_spec(s)
                    if n:
                        normalized.add(n)
            deps[name] = normalized
            if name == "default" and isinstance(spec, list):
                for s in spec:
                    n = _parse_feature_spec(s)
                    if n:
                        defaults.add(n)
        by_manifest.append((defaults, deps))

    # Transitive closure per manifest, then union across manifests.
    any_default: set[str] = set()
    for defaults, deps in by_manifest:
        resolved = set(defaults)
        changed = True
        # Cap iterations at len(deps)+2 to guarantee termination.
        for _ in range(len(deps) + 2):
            if not changed:
                break
            changed = False
            new_added: set[str] = set()
            for feat in list(resolved):
                for dep in deps.get(feat, ()):
                    if dep not in resolved and dep not in new_added:
                        new_added.add(dep)
            if new_added:
                resolved |= new_added
                changed = True
        any_default |= resolved
    return any_default


def find_defined_features(repo_root: Path) -> set[str]:
    """All feature names defined in any Cargo.toml under repo_root."""
    defined: set[str] = set()
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if feats:
            defined.update(feats.keys())
    return defined


FEATURE_HEADER_RE = re.compile(
    r"^\s*\**\s*Feature(?:s)?(?:\s+(?:gate|flag|flags|gates))?\**\s*[:\-]",
    re.IGNORECASE,
)

# After a Feature header, find each `feature_name` token optionally followed
# by a status indicator. Status indicators come in many forms:
#   `foo` (opt-in)                  — paren with status word
#   `foo` (**DEFAULT-ON** ...)       — paren with bolded status
#   `foo` = [...] (opt-in, NOT ...)  — Cargo-like syntax + paren
#   `foo` — OPT-IN, GOAT-gated       — em/en-dash separator
#   `foo` (opt-in, same as Phase 1) — paren with extra context
# We capture the feature name and a window of ~80 chars after it for status
# classification (instead of requiring the status to be in tight parens).
FEATURE_TOKEN_RE = re.compile(
    r"`([a-zA-Z][a-zA-Z0-9_\-]*?)`"
    r"(?:\s*=\s*\[[^\]]*\])?"  # optional Cargo-like = [...]
    r"\s*"  # optional whitespace
    r"(?:[\(\[]|\u2014|\u2013|--|:)"  # status separator: ( [ — – -- :
    r"?"  # separator optional (status may be inline)
    r"\s*\**"  # optional opening bold
    r"([a-zA-Z][a-zA-Z0-9_\-\s,]{0,80}?)"  # status phrase
    r"\**\s*"  # optional closing bold
    # Status boundary. `(` and `*` were missing, and their absence silently
    # dropped an entire sibling's convention: riir-chain writes
    #   **Feature:** `chain_block_producer` — **default-OFF** (implies ...)
    # where the status is followed by `**` then ` (`. Neither was a boundary,
    # the lazy status group could not terminate, and the line yielded NO token
    # at all — so 26 benchmark docs audited as "0 labels, 0 mismatches", which
    # reads identical to clean. Measured 2026-09-01: 9 repos / 62 docs in that
    # state (Issue 702).
    r"(?=[\)\]\n.,;:|(*]|$)",  # lookahead: status boundary
)


def parse_status_phrase(phrase: str) -> str:
    """Normalize a status phrase to 'default' | 'opt-in' | 'unknown'.

    Uses word-boundary matching (not bare substring) so that phrases like
    "default-off" or "off by default" are correctly classified as opt-in,
    not as default. Also recognizes explicit opt-in/default-on overrides.
    """
    w = phrase.lower().strip().rstrip(".")
    if not w:
        return "unknown"
    # Opt-in phrases that contain the substring "default" must be checked
    # FIRST. Without this guard, "default-off" would match the default-word
    # "default" via substring and flip to default-on (the session-13 bug).
    for p in OPTIN_DEFAULT_PHRASES:
        if p in w:
            return "opt-in"
    # Word-boundary check for each default-word.
    # Use regex with word boundaries so "default" doesn't match inside
    # "default-off" (already handled above) or "not-default-on".
    has_default = any(
        re.search(r"\b" + re.escape(s) + r"\b", w) for s in STATUS_WORDS_DEFAULT
    )
    has_optin = any(
        re.search(r"\b" + re.escape(s) + r"\b", w) for s in STATUS_WORDS_OPTIN
    )
    has_not = " not " in f" {w} " or w.startswith("not ") or "!" in w
    # "opt-in, NOT default-on" pattern → opt-in wins.
    if has_optin and has_default and has_not:
        return "opt-in"
    if has_default and not has_not:
        return "default"
    if has_optin:
        return "opt-in"
    return "unknown"


# Keep old name as alias for backward compatibility / external callers.
def parse_status_word(word: str) -> str:
    return parse_status_phrase(word)


# A label may record its own history: "(opt-in at time of writing, 2026-06-30;
# **promoted to DEFAULT-ON 2026-07-20 by Plan 468**)". Reading only the FIRST
# status word flags that as a mismatch, which is backwards — the doc is doing
# exactly what it should, and the auditor was penalising it for being thorough.
# False positives on correct docs are how a gate earns a reputation for noise
# and gets ignored.
#
# So the LAST explicit transition on the line wins. Note this OVERRIDES rather
# than SUPPRESSES: a doc claiming "promoted to DEFAULT-ON" for a feature that
# is not in any default array now fails on the other branch, which a plain
# skip-if-promotion-mentioned rule would have hidden.
TRANSITION_RES = [
    (re.compile(r"(?:re-)?promoted\s+(?:back\s+)?to\s+\**default(?:-on)?\**", re.I),
     "default"),
    (re.compile(r"\bnow\s+\**default-on\**", re.I), "default"),
    (re.compile(r"\bdemoted\s+(?:back\s+)?to\s+\**opt-in\**", re.I), "opt-in"),
    (re.compile(r"\breverted\s+to\s+\**opt-in\**", re.I), "opt-in"),
    (re.compile(r"\bnow\s+\**opt-in\**", re.I), "opt-in"),
]


# "**NOT promoted to default — D2F is opt-in research**" contains the exact
# substring "promoted to default". The first version of the transition rule read
# that as a promotion and turned one false positive into two, on two docs that
# were stating their status correctly and emphatically. Negation is checked
# immediately before the phrase; `**` and whitespace may intervene.
NEGATION_RE = re.compile(r"\b(?:not|never|no|nor|without)\b[\s*_]*$", re.I)


def parse_terminal_transition(line: str, from_idx: int) -> tuple[str, str] | None:
    """Last explicit, non-negated status transition at/after from_idx."""
    best_at, best = -1, None
    for rx, status in TRANSITION_RES:
        for m in rx.finditer(line, from_idx):
            if NEGATION_RE.search(line[max(0, m.start() - 24):m.start()]):
                continue
            if m.start() > best_at:
                best_at, best = m.start(), (status, m.group(0))
    return best


def iter_bench_doc_labels(repo_root: Path):
    """Yield (rel_path, lineno, line, feature_name, raw_status, parsed_status)."""
    for sub in (".benchmarks", ".docs"):
        d = repo_root / sub
        if not d.is_dir():
            continue
        for md in d.rglob("*.md"):
            rel = md.relative_to(repo_root).as_posix()
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                if not FEATURE_HEADER_RE.search(line):
                    continue
                for m in FEATURE_TOKEN_RE.finditer(line):
                    feat = m.group(1)
                    raw = m.group(2)
                    parsed = parse_status_phrase(raw)
                    trans = parse_terminal_transition(line, m.end())
                    if trans is not None:
                        parsed = trans[0]
                        raw = f"{raw} | transition: {trans[1]!r}"
                    if parsed == "unknown":
                        continue
                    yield (rel, ln, line.strip(), feat, raw, parsed)


def audit_repo(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    print(f"\n=== Auditing {repo_root.name} ===")
    any_default = find_cargo_defaults(repo_root)
    defined_features = find_defined_features(repo_root)

    mismatches = 0
    checked = 0
    for rel, ln, line, feat, raw, parsed in iter_bench_doc_labels(repo_root):
        # Only consider feature_name that exists as a defined Cargo feature.
        # This avoids false positives on use-case names, paper-artifact names,
        # etc. (the session-12 insight).
        if feat not in defined_features:
            continue
        checked += 1
        is_default = feat in any_default
        if parsed == "default" and not is_default:
            mismatches += 1
            print(f"  [MISMATCH] doc says DEFAULT but feature NOT in any Cargo default")
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {feat}  raw_status: {raw!r}")
            print(f"    line: {line}")
        elif parsed == "opt-in" and is_default:
            mismatches += 1
            print(f"  [MISMATCH] doc says OPT-IN but feature IS in some Cargo default")
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {feat}  raw_status: {raw!r}")
            print(f"    line: {line}")
    print(f"  -> checked {checked} labels, {mismatches} mismatches")
    return mismatches


# Real line shapes from the workspace, each with the (feature, status) the
# tokenizer must produce. This runs on EVERY invocation — the audit is wired
# per-push via scripts/docs_gate.sh, and a silent regex regression there does
# not fail anything: it just recognises fewer labels and still prints
# "0 mismatches". That is indistinguishable from clean, which is how riir-chain
# sat at "26 docs, 0 labels" unnoticed. A dropped shape must be loud.
TOKENIZER_CASES = [
    # katgpt-rs dialect
    ("**Feature:** `foo` (opt-in)", "foo", "opt-in"),
    ("**Feature:** `bar` (**DEFAULT-ON** since 2026-07-20)", "bar", "default"),
    # riir-chain dialect: status bolded, then an unbracketed ` (` follows.
    # `(` and `*` were not status boundaries, so this yielded NO token at all.
    ("**Feature:** `baz` — **default-OFF** (implies `qux`)", "baz", "opt-in"),
    # negation must still win over the bare word "default"
    ("**Feature:** `neg` (opt-in, NOT default-on)", "neg", "opt-in"),
]


def selftest() -> None:
    for line, want_feat, want_status in TOKENIZER_CASES:
        assert FEATURE_HEADER_RE.match(line), f"header regex missed: {line!r}"
        hits = FEATURE_TOKEN_RE.findall(line)
        got = [(f, parse_status_phrase(s)) for f, s in hits]
        if (want_feat, want_status) not in got:
            raise SystemExit(
                f"✗ tokenizer self-test FAILED\n  line:   {line!r}\n"
                f"  want:   {(want_feat, want_status)}\n  got:    {got}\n"
                "  A shape this script used to recognise no longer parses. It "
                "would keep printing '0 mismatches' over fewer labels.")


def main(argv: list[str]) -> int:
    selftest()
    if len(argv) < 2:
        # Default: audit the repo this script lives in.
        here = Path(__file__).resolve().parent.parent
        return 1 if audit_repo(here) else 0
    total = 0
    for arg in argv[1:]:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            print(f"skip (not a dir): {arg}", file=sys.stderr)
            continue
        if (p / "Cargo.toml").exists():
            total += audit_repo(p)
        else:
            children = [
                d for d in p.iterdir() if d.is_dir() and (d / "Cargo.toml").exists()
            ]
            if children:
                for c in children:
                    total += audit_repo(c)
            else:
                total += audit_repo(p)
    print(f"\n=== TOTAL mismatches across all repos: {total} ===")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
