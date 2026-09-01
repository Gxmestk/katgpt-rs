#!/usr/bin/env python3
"""
Cross-repo Cargo.toml inline-comment auditor.

Catches the drift class where a Cargo feature is transitively in
`default = [...]` (e.g. `micro_belief` is pulled in by `bom_sampling`,
which is in `default`), but the inline comment on the feature line
still claims "Opt-in" / "Default-OFF" / etc. Also catches the inverse
(comment claims default-on but feature is actually opt-in).

This is the Cargo.toml-comment counterpart to bench_doc_audit.py.
Run after every feature promotion to catch comment drift early.

Usage:
    python3 scripts/cargo_comment_audit.py                # audit this repo
    python3 scripts/cargo_comment_audit.py /git/riir-ai   # audit a specific repo
    python3 scripts/cargo_comment_audit.py /git           # walk all repos under /git

Exit code: 0 if no mismatches, 1 if any mismatch found.

Strategy
--------
1. Read every GIT-TRACKED `Cargo.toml` in the repo (falling back to a
   pruned filesystem walk when the target is not a git repo).
2. From each `[features]` table, collect:
   - the set of all defined feature names
   - the set of names transitively enabled by `default = [...]`
   (per-manifest closure, unioned across manifests)
3. For every feature definition line `feat = [...]  # comment`,
   parse the comment for a status phrase (opt-in / default-on / etc).
4. Cross-check: if comment says opt-in but feature is in the default
   closure → MISMATCH. If comment says default-on but feature is not
   in any default closure → MISMATCH.

Status vocabulary
-----------------
Reuses parse_status_phrase from bench_doc_audit — same word boundaries,
same opt-in-default-phrase guard (so "default-off" is not misclassified
as "default" via substring).

Caveats
-------
- Only the comment on the feature-definition line is checked. Multi-line
  comments above the feature line are NOT checked (would require a
  multi-line state machine; out of scope for v1).
- Only features defined in `[features]` tables are checked; package
  metadata comments are ignored.
- A feature is considered "default" if it appears in the default closure
  of ANY Cargo.toml in the repo (cross-crate promotions count).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

# Reuse the status parser from bench_doc_audit for consistency.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_doc_audit import (
    UNTRACKED_SKIPPED,
    find_cargo_defaults,
    find_defined_features,
    iter_cargo_manifests,
    parse_status_phrase,
)


def _parse_intra_crate_spec(spec: str) -> str | None:
    """Normalize a Cargo feature dep spec to a bare intra-crate feature name.

    Returns None for cross-crate specs (e.g. "katgpt-dec/foo") because those
    activate features in OTHER crates and don't expand THIS crate's closure.
    """
    if spec.startswith("dep:"):
        return None
    if "/" in spec:
        return None  # cross-crate spec — don't expand this crate's closure
    return spec or None


def find_cargo_defaults_per_manifest(repo_root: Path) -> dict[Path, set[str]]:
    """Per-manifest default-feature closure.

    Returns {cargo_path: set_of_features_in_default_closure}.
    Each manifest's closure is computed independently using only intra-crate
    feature specs.
    """
    result: dict[Path, set[str]] = {}
    for cargo in iter_cargo_manifests(repo_root):
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if not feats:
            continue
        defaults: set[str] = set()
        deps: dict[str, set[str]] = {}
        for name, spec in feats.items():
            normalized: set[str] = set()
            if isinstance(spec, list):
                for s in spec:
                    n = _parse_intra_crate_spec(s)
                    if n:
                        normalized.add(n)
            deps[name] = normalized
            if name == "default":
                defaults |= normalized
        resolved = set(defaults)
        changed = True
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
        result[cargo] = resolved
    return result


# Patterns indicating the comment is being precise about LOCAL scope.
# When present, the audit trusts the comment and does NOT flag a mismatch,
# even if a cross-crate union closure suggests the feature is default-on
# somewhere else in the workspace. Handles:
#   (a) "DEFAULT-ON in katgpt-dec (...); this root flag stays opt-in (...)"
#   (b) "Opt-in in katgpt-core (consumer enables transitively). Consumer
#       feature `X` PROMOTED to default-on (...)"
#   (c) "NOT in katgpt-core `default` — root's default-on forwarder activates it"
#       (explicit local-scope opt-in claim that overrides any later default-on
#       mention — the comment is being precise about which crate's default
#       does/doesn't include this feature).
LOCAL_SCOPE_OVERRIDE_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bstays\s+opt-in\b",
        r"\bstays\s+off\b",
        r"\bstays\s+OPT-IN\b",
        r"\bopt-in\s+in\s+[a-z][a-z0-9_-]+\b",
        r"\bopt-in\s+here\b",
        r"\bNOT\s+in\s+[a-z`][a-z0-9_-`]*\s+`?default`?\b",
        r"\bNOT\s+in\s+the\s+`?default`?\b",
    ]
]

# Cross-crate default-on claim patterns. When a default-on comment uses one
# of these patterns, the audit checks the UNION closure (across all crates)
# instead of the per-manifest closure. This is because the comment is making
# a claim about the feature's status in ANOTHER crate (typically the root
# crate or via a parent feature), not the current crate.
# Example: "DEFAULT-ON via rv_gated_routing" in katgpt-pruners/Cargo.toml
# is a claim about root's rv_gated_routing feature, not katgpt-pruners's own.
CROSS_CRATE_DEFAULT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bDEFAULT-ON\s+via\s+",
        r"\bdefault-ON\s+via\s+",
        r"\bdefault-on\s+via\s+",
        r"\bDEFAULT-ON\s+in\s+(?:root|katgpt)",
        r"\bdefault-on\s+in\s+(?:root|katgpt)",
    ]
]


# Match a feature definition line `name = [...]  # comment`.
# Cargo allows the RHS to span multiple lines, but inline comments on the
# same physical line as the feature name are the common case for one-liners
# like `micro_belief = ["dep:katgpt-micro-belief"]  # Opt-in until G1.1...`.
# For multi-line RHS, we capture only the comment on the first physical
# line (which is where the convention places the status phrase).
FEATURE_LINE_RE = re.compile(
    r"""^\s*
    (?P<name>[a-zA-Z][a-zA-Z0-9_-]*)    # feature name
    \s*=\s*
    \[                                  # opening bracket of feature list
    [^\n]*                              # rest of line (the spec list + maybe comment)
    $""",
    re.VERBOSE,
)


def extract_inline_comment(line: str) -> str | None:
    """Extract the `# ...` portion of a Cargo.toml line, respecting quotes.

    A naive `line.split("#", 1)[1]` would break on `#` inside quoted strings
    (rare but possible). We walk the line char-by-char tracking quote state.
    """
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(line):
        if in_quote:
            if ch == quote_char:
                in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            continue
        if ch == "#":
            return line[i + 1 :].strip()
    return None


# Demotion / negation patterns — explicit opt-in overrides.
# These beat any "default-on" mention because they represent an explicit
# decision to NOT promote (or to demote after promotion).
DEMOTE_NEGATION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdemoted\b",
        r"\bstays\s+opt-in\b",
        r"\bstays\s+off\b",
        r"\bnot\s+default-on\b",
        r"\bnot\s+promoted\b",
        r"\bpromotion\s+(?:blocked|deferred|pending|depends|waits)\b",
    ]
]

# Canonical DEFAULT-ON promotion template: "DEFAULT-ON (Plan/Issue/date...):".
# This is the strongest default-on signal — appears in fresh promotion commits
# and is unlikely to co-exist with stale opt-in language. The parens content
# MUST start with Plan/Issue/P\d+/date to avoid false matches on phrases like
# "default-on (behavior opt-in ...)" which describe a DIFFERENT feature.
CANONICAL_DEFAULT_RE = re.compile(
    r"DEFAULT-ON\s*\(\s*(?:Plan|Issue|P\d+|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

# Less-strict DEFAULT-ON mention (no parens, e.g. "DEFAULT-ON in root").
# Used after canonical and opt-in checks fall through.
DEFAULT_ON_MENTION_RE = re.compile(r"\bDEFAULT-ON\b", re.IGNORECASE)

# Explicit "Opt-in" / "OPT-IN" (capitalized emphasis) — human-written status
# claim. Deliberately NOT case-insensitive: bare lowercase "opt-in" is casual
# prose and is handled later by STRONG_OPTIN_RES, after the default-on checks.
#
# The all-caps form was missing, and it is a convention here, not a one-off:
# 32 comment lines use "OPT-IN" against 176 using "Opt-in". Any of those 32
# that also mention "default-on" incidentally — e.g. describing ANOTHER
# feature's promotion precedent — fell through rule 3 to rule 4's
# case-INSENSITIVE `\bDEFAULT-ON\b` and was classified default-on. That is how
# `signed_coupling_dynamics`, whose comment ends "OPT-IN — promotion waits on a
# production consumer", was reported as claiming default-on.
EXPLICIT_OPTIN_RE = re.compile(r"\b(?:Opt-in|OPT-IN)\b")

# Strong default-on phrases — fallback after canonical + Opt-in checks.
STRONG_DEFAULT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdefault-on\b",
        r"\bdefault on\b",
        r"\bon by default\b",
        r"\balways-on\b",
        r"\balways on\b",
        r"\bpromoted\b",
        r"\benabled by default\b",
    ]
]

# Strong opt-in phrases — fallback after explicit Opt-in check.
STRONG_OPTIN_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bopt-in\b",
        r"\bopt in\b",
        r"\bdefault-off\b",
        r"\bdefault off\b",
        r"\boff by default\b",
        r"\bdisabled by default\b",
        r"\bnot in default\b",
        r"\bnot default\b",
    ]
]

# Weak "default" — bare word, not part of a compound (default-features,
# default-off, default-on, default `value`).
# Negative lookahead: "default" NOT followed by `-`, `=`, or whitespace+`(`
# (so "default `(0.82L→0.45L)`" and "default-features" are excluded).
WEAK_DEFAULT_RE = re.compile(r"\bdefault\b(?![-=]|\s*\(`)", re.IGNORECASE)


def classify_comment(comment: str) -> tuple[str, str]:
    """Classify an inline comment to (status, raw_phrase).

    Precedence (strongest signal first):
      1. Demotion/negation patterns ("demoted", "stays opt-in", "not default-on",
         "promotion blocked/deferred/pending/depends") — opt-in wins because
         these represent an explicit decision to NOT promote.
      2. Canonical DEFAULT-ON promotion template ("DEFAULT-ON (Plan X...):") —
         strongest default-on signal, present in fresh promotion commits.
      3. Explicit "Opt-in" (capitalized emphasis) — human-written status claim.
      4. Other DEFAULT-ON mentions (no parens, e.g. "DEFAULT-ON in root").
      5. Strong default/opt-in phrases (fallback).
      6. Weak "default" — bare word, excluding compound forms.
      7. Unknown.

    Returns (status, raw_phrase) where raw_phrase is the matched substring
    for human-readable mismatch output.
    """
    if not comment:
        return "unknown", ""
    # 1. Demotion/negation overrides everything.
    for rx in DEMOTE_NEGATION_RES:
        m = rx.search(comment)
        if m:
            return "opt-in", m.group(0)
    # 2. Canonical promotion template.
    m = CANONICAL_DEFAULT_RE.search(comment)
    if m:
        return "default", "DEFAULT-ON template"
    # 3. Explicit capitalized "Opt-in".
    m = EXPLICIT_OPTIN_RE.search(comment)
    if m:
        return "opt-in", "explicit Opt-in"
    # 4. Other DEFAULT-ON mention (e.g. "DEFAULT-ON in root").
    m = DEFAULT_ON_MENTION_RE.search(comment)
    if m:
        return "default", m.group(0)
    # 5. Strong default phrases.
    for rx in STRONG_DEFAULT_RES:
        m = rx.search(comment)
        if m:
            return "default", m.group(0)
    # 5b. Strong opt-in phrases.
    for rx in STRONG_OPTIN_RES:
        m = rx.search(comment)
        if m:
            return "opt-in", m.group(0)
    # 6. Weak "default" — bare word.
    m = WEAK_DEFAULT_RE.search(comment)
    if m:
        return "default", m.group(0)
    return "unknown", ""


def iter_cargo_comment_labels(repo_root: Path):
    """Yield (cargo_path, rel_path, lineno, line, feature_name, raw_comment, parsed_status).

    cargo_path is the absolute Path to the Cargo.toml file (for per-manifest
    closure lookup); rel_path is the display path.
    raw_comment is the full inline comment text (for the caller to apply
    local-scope-override and cross-crate-claim checks).
    """
    for cargo in iter_cargo_manifests(repo_root):
        rel = cargo.relative_to(repo_root).as_posix()
        try:
            text = cargo.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            with cargo.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        feats = data.get("features", {})
        if not feats:
            continue
        defined_names = set(feats.keys())
        for ln, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("["):
                continue
            m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*\[", stripped)
            if not m:
                continue
            name = m.group(1)
            if name not in defined_names:
                continue
            if name == "default":
                continue
            comment = extract_inline_comment(line)
            if not comment:
                continue
            parsed, _raw = classify_comment(comment)
            if parsed == "unknown":
                continue
            yield (cargo, rel, ln, line.rstrip(), name, comment, parsed)


def audit_repo(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    print(f"\n=== Auditing {repo_root.name} ===")
    per_manifest_defaults = find_cargo_defaults_per_manifest(repo_root)
    union_defaults = find_cargo_defaults(repo_root)
    defined_features = find_defined_features(repo_root)

    mismatches = 0
    checked = 0
    seen: set[tuple[str, int]] = set()
    for cargo, rel, ln, line, feat, comment, parsed in iter_cargo_comment_labels(
        repo_root
    ):
        if feat not in defined_features:
            continue
        if (rel, ln) in seen:
            continue
        seen.add((rel, ln))
        checked += 1
        # Local-scope override: comment contains "stays opt-in" / "stays off" /
        # "Opt-in in <crate>" / "Opt-in here" / "NOT in <crate> default" — the
        # comment is being precise about LOCAL scope. Trust it; don't flag.
        if any(rx.search(comment) for rx in LOCAL_SCOPE_OVERRIDE_RES):
            continue
        # Choose which closure to consult:
        # - Default-on claims: check union. The comment may describe status in
        #   any crate (root forward, sub-crate default, etc.). Per-manifest is
        #   too narrow here.
        # - Opt-in claims: check per-manifest. A root-level opt-in comment is
        #   precise about root's status; the feature may well be default-on in
        #   a sub-crate without contradicting the comment.
        if parsed == "default":
            defaults = union_defaults
            scope = "union"
        else:
            defaults = per_manifest_defaults.get(cargo, set())
            scope = "per-manifest"
        is_default = feat in defaults
        if parsed == "default" and not is_default:
            mismatches += 1
            print(
                f"  [MISMATCH] comment says DEFAULT-ON but feature NOT in any default closure"
            )
            print(f"    file: {rel}:{ln}  (checked {scope} closure)")
            print(f"    feat: {feat}")
            print(f"    line: {line}")
        elif parsed == "opt-in" and is_default:
            mismatches += 1
            print(
                f"  [MISMATCH] comment says OPT-IN but feature IS in this manifest's default closure"
            )
            print(f"    file: {rel}:{ln}")
            print(f"    feat: {feat}")
            print(f"    line: {line}")
    # Untracked manifests are excluded from the closure (see
    # bench_doc_audit._tracked_manifests) — say so, rather than letting a
    # workstation run quietly differ from CI's count with no explanation.
    _untracked = UNTRACKED_SKIPPED.get(str(repo_root), 0)
    _tail = (f"  -> checked {checked} inline comments, {mismatches} mismatches"
             + (f" [skipped: {_untracked} untracked manifest(s) not in git]"
                if _untracked else ""))
    print(_tail)
    return mismatches


def main(argv: list[str]) -> int:
    if len(argv) < 2:
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
