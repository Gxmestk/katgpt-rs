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
    for cargo in repo_root.rglob("Cargo.toml"):
        if "target" in cargo.parts:
            continue
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
    for cargo in repo_root.rglob("Cargo.toml"):
        if "target" in cargo.parts:
            continue
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
    r"(?=[\)\]\n.,;:|]|$)",  # lookahead: status boundary
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


def main(argv: list[str]) -> int:
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
