#!/usr/bin/env python3
"""Audit percentile-index sites: which reported "p99" is actually the MAX.

A REPORT, not a gate (always exit 0) — for the same reason
`cfg_gated_target_audit.py` is one: a large share of sites take their sample
count from a runtime value that no static pass can resolve, and a report that
exits 1 on those is a report nobody runs. The verdict half belongs in a
per-repo gate over the sites this one resolves.

# The defect

    let p99 = sorted[(n as f64 * 0.99) as usize];   // n = 100 -> sorted[99] -> MAX
    let p99 = sorted[n * 99 / 100];                 // n = 100 -> sorted[99] -> MAX

`floor(n*p) == n - 1` whenever `n * (1 - p) <= 1`, i.e. **n <= 1/(1-p)**:
n <= 100 for p99, n <= 20 for p95, n <= 1000 for p999. Below that boundary the
site reports the maximum under a percentile's name, and the number is ONE
observation. `.min(len - 1)` clamps prevent a panic, not a wrong statistic.

Direction matters: the naive index is one rank TOO HIGH, so for a
`p99 < budget` assert the failure mode is a false RED, not a false green.
Nothing green becomes red by fixing a site.

# Two vocabularies, and why both are committed here

The first cut of this audit (riir-mmorpg-examples Issue 093) grepped only the
FLOAT forms and reported a 14-row table as an audit "over all 19 contract
repos". The INTEGER form `n * 99 / 100` is the more common one in this
workspace and was invisible to it; riir-e2e's copy was found by accident, not
by the sweep. That is the katgpt-rs "a zero ceiling is only as wide as its
classifier" failure, committed inside the issue warning about it.

So the vocabulary is DATA, listed exhaustively below and pinned by
`selftest()`, and the population is DERIVED (BOUNDARY.md + a `.git` dir) — per
the workspace rule that deriving both from the same walk is what makes a
cross-repo report permanently empty.

# Tail support is the quantity nobody prints

`support = n - idx` — the number of samples at or above the reported rank:
1 at n=100, 2 at n=200, 10 at n=1000 (naive index). A quantile with support 1
is one observation; support 2 moves if one preemption lands above it. This
report calls anything under MIN_SUPPORT weak, whether or not it is degenerate.
"""

import os
import re
import sys

MIN_SUPPORT = 10  # samples at/above the rank before a tail can decide a verdict

# ── The vocabulary (DATA — extend here, then extend selftest) ─────────────
#
# Each entry: (name, compiled regex, group name for the n-expression,
#              a callable p-extractor over the match, safe_by_construction).
#
# `safe` marks a form that CANNOT return n-1 for any n >= 2 and so needs no
# sample count to clear it — the `(len - 1) * p` shape, which is bounded by
# n - 2. That form is the one site in the workspace that got it right and it
# must not be reported as a finding.
VOCAB = [
    # (v.len() as f64 - 1.0) * 0.99   |   (len - 1) * p / 100.0
    #
    # SAFE BY CONSTRUCTION: `floor((n-1) * p)` is bounded by n-2 for every
    # n >= 2 and every p < 1, so this form can never return the max. It needs
    # no sample count to clear it, and it must NOT be reported as a finding --
    # it is the shape the workspace's one correct site uses
    # (katgpt-speculative/tests/weaver_real_checkpoint.rs). Verified over
    # n in 2..=100_000: zero violations.
    ("float_len_minus_one",
     re.compile(r"\(\s*(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*(?:as\s+f(?:32|64)\s*)?"
                r"-\s*1(?:\.0)?\s*\)\s*\*\s*(?P<p>0\.9\d+|\w+\s*/\s*100\.0)"),
     True),
    # x as f64 * 0.99   |   (x.len() as f64) * 0.999
    ("float_mul",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s+as\s+f(?:32|64)\s*\)?"
                r"\s*\*\s*(?P<p>0\.9\d+)"),
     False),
    # 0.99 * x as f64   (reversed operands)
    ("float_mul_rev",
     re.compile(r"(?P<p>0\.9\d+)\s*\*\s*(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s+as\s+f(?:32|64)"),
     False),
    # n * 99 / 100   |   (xs.len() * 999) / 1000
    #
    # The form the FIRST cut of this audit was blind to, and the more common
    # one in this workspace. `num` is constrained to 2-3 digits starting with 9
    # so that `(boards.len() * 9) / 10` -- a fraction, not a percentile -- does
    # not match.
    # n * p / 100 where p is a VARIABLE (a closure/fn parameter).
    #
    # The third instance of the classifier-narrowness failure in this audit.
    # riir-game-sdk's `percentiles` helper is
    #     let at = |p: usize| durs[(n * p / 100).min(n - 1)];  at(50), at(99)
    # and the literal-only pattern below reported that repo as having ZERO
    # percentile sites -- while it feeds the repo's wall-clock budget gates.
    # The percentile is not statically known, so these land in UNRESOLVED
    # (honest) rather than vanishing from the population (not).
    #
    # Restricted to matches INSIDE square brackets on the same line: a
    # percentile is by definition an index into a sorted sample array, and
    # without that filter `a * b / 100` in unrelated arithmetic floods the
    # report. Applied to this entry only, because real sites do split the
    # index and the indexing across two lines.
    ("int_ratio_var",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*\*\s*(?P<pv>[a-z_]\w*)"
                r"\s*\)?\s*/\s*(?P<den>100|1000)\b"),
     False),
    ("int_ratio",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*\*\s*(?P<num>9\d{1,2})"
                r"\s*\)?\s*/\s*(?P<den>100|1000)\b"),
     False),
]

DEGENERATE, WEAK, OK, UNRESOLVED, SAFE = "DEGENERATE", "WEAK", "OK", "UNRESOLVED", "SAFE"

# `.ceil()` / `.round()` applied to the product is NOT this defect: the bug is
# floor/truncation (`as usize`). `ceil(p*n) - 1` is the correct nearest-rank
# form. It also excludes the largest false-positive class -- `0.95 * n as f32`
# computing a top-p NUCLEUS SIZE (a probability mass), not an index into a
# sorted sample array. That shape was reported as the single most actionable
# row ("WEAK + ASSERTED") on the second cut of this report, from
# katgpt-rs/tests/bench_181_dmoe_vocab_coreset_goat.rs:369, which indexes
# nothing.
ROUNDED_RE = re.compile(r"\.\s*(?:ceil|round|trunc)\s*\(\s*\)")


def parse_site(m, kind):
    """(p as a fraction, index-function) for a match, or (None, None).

    The index function reproduces the site's OWN arithmetic exactly -- integer
    truncation for the `* num / den` form, f64-multiply-then-truncate for the
    float forms. Approximating one with the other moves the boundary by a
    rank, which is the whole finding.
    """
    if kind == "int_ratio_var":
        return None, None          # percentile bound at the call site
    if kind == "int_ratio":
        num, den = int(m.group("num")), int(m.group("den"))
        return num / den, (lambda n: (n * num) // den)
    raw = (m.groupdict().get("p") or "").strip()
    if not raw.startswith("0."):
        return None, None          # `p / 100.0` with a variable p
    p = float(raw)
    return p, (lambda n: int(n * p))


def classify(n, p, idx_fn, safe):
    if safe:
        return SAFE, None, None
    if n is None or p is None or idx_fn is None or n < 1:
        return UNRESOLVED, None, None
    idx = min(idx_fn(n), n - 1)
    support = n - idx
    if idx == n - 1:
        return DEGENERATE, idx, support
    return (WEAK if support < MIN_SUPPORT else OK), idx, support


# ── Sample-count resolution ───────────────────────────────────────────────
CONST_RE = re.compile(r"const\s+(\w+)\s*:\s*\w+\s*=\s*([0-9_]+)\s*;")
LET_NUM_RE = re.compile(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([0-9_]+)\s*;")
CAP_RE = re.compile(r"let\s+(?:mut\s+)?(\w+)\s*(?::[^=\n]+)?=\s*Vec::with_capacity\(\s*([\w.]+)")
LEN_RE = re.compile(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(\w+)\s*\.\s*len\(\)")


def _uniq_ints(pairs):
    """name -> value, but a name bound to two DIFFERENT literals in one file is
    dropped rather than guessed. `content_store/goat.rs` binds `N_BLOBS` to
    both 50 and 100 in different fns; picking either would be a coin flip
    reported as a measurement."""
    seen = {}
    bad = set()
    for k, v in pairs:
        if k in seen and seen[k] != v:
            bad.add(k)
        seen[k] = v
    return {k: v for k, v in seen.items() if k not in bad}


FN_START_RE = re.compile(r"^\s*(?:pub\s+(?:\([^)]*\)\s+)?)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+\w+")


def enclosing_scope(lines, i):
    """(start, end) line indices of the fn containing line `i`.

    Both `resolve_n` and `is_load_bearing` were file-scoped on the first cut
    and both were WRONG for the same reason: a `let`/`with_capacity` binding
    or an `assert!` in a DIFFERENT function is not in scope. That produced a
    false ASSERTED on riir-neuron-db bench_003 (the assert is on `mean_us`,
    the p99 line's neighbour in a returned tuple) and resolved a slice
    PARAMETER's length from an unrelated caller in riir-chain bench_012.
    Module-level `const`s stay file-scoped, because they genuinely are.
    """
    start = 0
    for j in range(i, -1, -1):
        if FN_START_RE.match(lines[j]):
            start = j
            break
    end = len(lines)
    for j in range(i + 1, len(lines)):
        if FN_START_RE.match(lines[j]):
            end = j
            break
    return start, end


def resolve_n(expr, file_text, scope_text):
    """Resolve an n-expression to an integer, or None.

    `let` / `Vec::with_capacity` / `.len()` bindings are looked up in the
    ENCLOSING FN only; `const` is looked up in the fn first and then
    file-wide (module-level consts are visible everywhere). Follows up to 4
    links: `let n = xs.len()` -> `Vec::with_capacity(N)` -> `const N = 100`.
    """
    expr = expr.strip()
    if expr.isdigit():
        return int(expr)
    consts_local = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                              for m in CONST_RE.finditer(scope_text))
    consts_file = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                             for m in CONST_RE.finditer(file_text))
    lets = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                      for m in LET_NUM_RE.finditer(scope_text))
    caps = {m.group(1): m.group(2) for m in CAP_RE.finditer(scope_text)}
    lens = {m.group(1): m.group(2) for m in LEN_RE.finditer(scope_text)}

    seen, cur = set(), expr
    for _ in range(4):
        if cur in seen:
            return None
        seen.add(cur)
        base = cur[:-6].strip() if cur.endswith(".len()") else cur
        for table in (consts_local, lets, consts_file):
            if base in table:
                return table[base]
        if base in lens:
            cur = lens[base]
            continue
        if base in caps:
            cur = caps[base]
            continue
        return None
    return None


ASSERT_RE = re.compile(r"\bassert(?:_eq|_ne)?!")


def _balanced_args(text, open_idx):
    """Content between the parens of a macro call starting at `open_idx` (the
    index of `(`), respecting nesting. A fixed-width window instead of this
    is what crossed statement boundaries and manufactured a false ASSERTED."""
    depth, i, n = 0, open_idx, len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
        i += 1
    return text[open_idx + 1 :]


def is_load_bearing(var, lines, i):
    """Does the value feed an `assert!` IN THE SAME FUNCTION?

    Whole-word match against the assert's balanced argument list only -- not a
    character window, and not the whole file. A print-only quantile is
    misleading; an asserted one decides a verdict, and conflating them makes
    the report's most actionable row untrustworthy.
    """
    if not var:
        return False
    start, end = enclosing_scope(lines, i)
    scope = "\n".join(lines[start:end])
    word = re.compile(r"\b" + re.escape(var) + r"\b")
    for m in ASSERT_RE.finditer(scope):
        paren = scope.find("(", m.end() - 1)
        if paren == -1:
            continue
        if word.search(_balanced_args(scope, paren)):
            return True
    return False


ASSIGN_RE = re.compile(r"let\s+(\w+)\s*(?::[^=]*)?=")


def audit_file(path, rel):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("///"):
            continue
        for kind, rx, safe in VOCAB:
            m = rx.search(line)
            if not m:
                continue
            if kind == "int_ratio_var":
                before, after = line[: m.start()], line[m.end() :]
                if "[" not in before or "]" not in after:
                    continue       # not an index -> not a percentile
            if ROUNDED_RE.search(line[m.end():]):
                safe = True          # explicit rounding, not truncation
            p, idx_fn = parse_site(m, kind)
            _s, _e = enclosing_scope(lines, i)
            n = resolve_n(m.group("n"), text, "\n".join(lines[_s:_e]))
            verdict, idx, support = classify(n, p, idx_fn, safe)
            am = ASSIGN_RE.search(line)
            var = am.group(1) if am else None
            out.append({
                "file": rel, "line": i + 1, "kind": kind, "p": p, "n": n,
                "idx": idx, "support": support, "verdict": verdict,
                "asserted": is_load_bearing(var, lines, i),
                "text": stripped[:100],
            })
            break
    return out


def repos(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "BOUNDARY.md"))
        and os.path.isdir(os.path.join(root, d, ".git"))
    )


def walk_rs(repo_root):
    skip = {"target", ".git", "node_modules", ".venv"}
    for dp, dns, fns in os.walk(repo_root):
        dns[:] = [d for d in dns if d not in skip]
        for f in fns:
            if f.endswith(".rs"):
                yield os.path.join(dp, f)


def selftest():
    """Pin the tokenizer AND the arithmetic. Both failure modes are SILENT: a
    regex regression makes the report find fewer sites and still print a
    confident summary, and a classifier bug turns DEGENERATE into OK.

    Canaried by the bug this file shipped with on its first run -- the
    n-expression class included `[` and `(`, so `sorted[(n as f64 * 0.99)`
    captured `sorted[(n` and EVERY site resolved to UNRESOLVED. The selftest
    refused to print; without it the report would have shown 99 sites, zero
    findings, and looked like good news."""
    cases = [
        # (source line, expected kind, expected p, context, expected verdict)
        ("    let p99 = sorted[(n as f64 * 0.99) as usize];", "float_mul", 0.99,
         "let n = xs.len();\nlet mut xs: Vec<u64> = Vec::with_capacity(N);\nconst N: usize = 100;",
         DEGENERATE),
        ("    let p99 = sorted[n * 99 / 100];", "int_ratio", 0.99,
         "let n = 100;", DEGENERATE),
        ("    let p99 = sorted[(timings.len() * 99) / 100];", "int_ratio", 0.99,
         "let mut timings: Vec<u64> = Vec::with_capacity(ITERS);\nconst ITERS: usize = 1000;",
         OK),
        ("    let p95 = data[n * 95 / 100];", "int_ratio", 0.95,
         "let n = 20;", DEGENERATE),
        ("    let p99 = sorted[(n as f64 * 0.99) as usize];", "float_mul", 0.99,
         "let n = 200;", WEAK),
        ("    let i = ((v.len() as f64 - 1.0) * 0.99) as usize;", "float_len_minus_one", 0.99,
         "", SAFE),
        ("    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;",
         "float_len_minus_one", None, "", SAFE),
        ("    let p999 = s[(s.len() as f32 * 0.999) as usize];", "float_mul", 0.999,
         "let mut s: Vec<u64> = Vec::with_capacity(N);\nconst N: usize = 100;", DEGENERATE),
        ("    let p99_us = samples[((N_ITERS as f64) * 0.99) as usize];", "float_mul", 0.99,
         "const N_ITERS: usize = 200;", WEAK),
        # a fraction, NOT a percentile -- must not match at all
        ("    let min_expected = (boards.len() * 9) / 10;", None, None, "", None),
        # ambiguous const: two different literals for one name -> UNRESOLVED,
        # never a coin flip reported as a measurement
        ("    let p99_idx = (READS as f64 * 0.99) as usize;", "float_mul", 0.99,
         "const READS: usize = 50;\nfn other() { const READS: usize = 100; }", UNRESOLVED),
    ]
    fails = []
    for src, kind, p_exp, ctx, verdict_exp in cases:
        hit = None
        for k, rx, safe in VOCAB:
            m = rx.search(src)
            if m:
                hit = (k, m, safe)
                break
        if kind is None:
            if hit is not None:
                fails.append(f"expected NO match but got {hit[0]}: {src!r}")
            continue
        if hit is None:
            fails.append(f"NO MATCH: {src!r}")
            continue
        k, m, safe = hit
        if k != kind:
            fails.append(f"kind {k} != {kind} for {src!r}")
            continue
        p, idx_fn = parse_site(m, k)
        if p_exp is not None and (p is None or abs(p - p_exp) > 1e-9):
            fails.append(f"p {p} != {p_exp} for {src!r}")
            continue
        _ctx = ctx + "\n" + src
        n = resolve_n(m.group("n"), _ctx, _ctx)
        v, _idx, _sup = classify(n, p, idx_fn, safe)
        if v != verdict_exp:
            fails.append(f"verdict {v} != {verdict_exp} (n={n}, p={p}) for {src!r}")

    # ── variable-percentile form must be in the population ──
    var_src = "        let at = |p: usize| durs[(n * p / 100).min(n - 1)];"
    if not any(rx.search(var_src) for _k, rx, _s in VOCAB):
        fails.append("variable-percentile form (n * p / 100) not in the vocabulary")
    # ...and it must be bracket-filtered: bare arithmetic is not a percentile
    bare = "        let pct = total * frac / 100;"
    for k_, rx, _s in VOCAB:
        m2 = rx.search(bare)
        if m2 and k_ == "int_ratio_var":
            if "[" in bare[: m2.start()] and "]" in bare[m2.end() :]:
                fails.append("bracket filter admitted non-indexing arithmetic")

    # ── rounding exclusion: `.ceil()` is not this defect ──
    for src in (
        "        let expected_min = (0.95 * vocab_size as f32).ceil() as usize;",
        "        let k = (0.99 * n as f64).ceil() as usize;",
        "        let idx = ((p * n as f64).round()) as usize;",
    ):
        hit = None
        for k_, rx, safe_ in VOCAB:
            m = rx.search(src)
            if m:
                hit = (k_, m, safe_)
                break
        if hit is None:
            continue                      # not matching at all is also fine
        if not ROUNDED_RE.search(src[hit[1].end():]):
            fails.append(f"rounding exclusion missed: {src!r}")

    # ── scoping canaries (the bugs the first cut shipped) ──
    # (a) a binding in ANOTHER fn must not resolve a slice parameter's length
    other_fn = [
        "fn caller() {",
        "    let mut latencies_ns: Vec<u64> = Vec::with_capacity(N);",
        "    const N: usize = 50;",
        "}",
        "fn summarize(latencies_ns: &[u64]) {",
        "    let p99_idx = ((latencies_ns.len() as f64) * 0.99) as usize;",
        "}",
    ]
    st, en = enclosing_scope(other_fn, 5)
    if resolve_n("latencies_ns.len()", "\n".join(other_fn), "\n".join(other_fn[st:en])) is not None:
        fails.append("resolve_n crossed a fn boundary to size a slice parameter")
    # (b) an assert on a SIBLING tuple element must not mark p99 as asserted
    sib = [
        "fn bench() -> (f64, f64) {",
        "    let p99_us = samples[((N as f64) * 0.99) as usize];",
        "    (mean_us, p99_us)",
        "}",
        "fn main() {",
        "    let (mean_us, p99_us) = bench();",
        "    assert!(mean_us < BUDGET, \"mean {mean_us} over budget\");",
        "}",
    ]
    if is_load_bearing("p99_us", sib, 1):
        fails.append("is_load_bearing crossed a fn boundary / matched a sibling binding")
    # (c) ...but a real assert in the SAME fn must still be found
    same = [
        "fn row() {",
        "    let p99 = sorted[(n as f64 * 0.99) as usize];",
        "    assert!(p99 < 5_000, \"tail over budget\");",
        "}",
    ]
    if not is_load_bearing("p99", same, 1):
        fails.append("is_load_bearing missed an assert in the same fn")

    # ── arithmetic pins, independent of every regex above ──
    assert int(100 * 0.99) == 99, "n=100 p99 truncates to the max index"
    assert int(101 * 0.99) == 99, "n=101 is the first count that is NOT the max"
    assert (100 * 99) // 100 == 99 and (200 * 99) // 100 == 198
    for n, exp in ((100, 1), (200, 2), (1000, 10)):
        assert n - min(int(n * 0.99), n - 1) == exp, f"naive support at n={n}"
    # the SAFE form can never reach n-1, over the whole range it claims
    for n in range(2, 20001):
        assert int((n - 1) * 0.99) <= n - 2, f"(n-1)*0.99 reached n-1 at n={n}"

    if fails:
        print("SELFTEST FAILED — the report below cannot be trusted:")
        for f in fails:
            print("  " + f)
        sys.exit(2)


def main():
    selftest()
    root = "/Users/katopz/git"
    if len(sys.argv) > 1:
        targets = [os.path.abspath(sys.argv[1])]
        root = os.path.dirname(targets[0])
    else:
        targets = [os.path.join(root, r) for r in repos(root)]

    print(f"percentile-index audit — MIN_SUPPORT={MIN_SUPPORT}, "
          f"{len(targets)} repo(s) (derived: BOUNDARY.md + .git)\n")
    grand = {}
    all_findings = []
    for t in targets:
        name = os.path.basename(t)
        found = []
        for f in walk_rs(t):
            found += audit_file(f, os.path.relpath(f, t))
        if not found:
            continue
        tally = {}
        for r in found:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
            r["repo"] = name
        grand[name] = tally
        all_findings += found

    # ── the two rows that matter, most severe first ──
    for label, pred in (
        ("DEGENERATE + ASSERTED  (a percentile that IS the max, deciding a verdict)",
         lambda r: r["verdict"] == DEGENERATE and r["asserted"]),
        ("DEGENERATE, print-only (a percentile that IS the max, misleading a reader)",
         lambda r: r["verdict"] == DEGENERATE and not r["asserted"]),
        (f"WEAK + ASSERTED       (support < {MIN_SUPPORT}, one stall can flip it)",
         lambda r: r["verdict"] == WEAK and r["asserted"]),
    ):
        rows = [r for r in all_findings if pred(r)]
        print(f"── {label}: {len(rows)}")
        for r in sorted(rows, key=lambda r: (r["repo"], r["file"], r["line"])):
            print(f"     {r['repo']}/{r['file']}:{r['line']}  p={r['p']} n={r['n']} "
                  f"idx={r['idx']} support={r['support']}")
        print()

    print("── per-repo tally (verdict x count) " + "─" * 30)
    hdr = [DEGENERATE, WEAK, OK, UNRESOLVED, SAFE]
    print(f"  {'repo':<24}" + "".join(f"{h:>12}" for h in hdr) + f"{'total':>8}")
    tot = {h: 0 for h in hdr}
    for name in sorted(grand):
        t = grand[name]
        print(f"  {name:<24}" + "".join(f"{t.get(h, 0):>12}" for h in hdr)
              + f"{sum(t.values()):>8}")
        for h in hdr:
            tot[h] += t.get(h, 0)
    print(f"  {'ALL':<24}" + "".join(f"{tot[h]:>12}" for h in hdr)
          + f"{sum(tot.values()):>8}")
    print(f"\n  UNRESOLVED is not 'clean' — it is a sample count no static pass could\n"
          f"  reach (a runtime length, a fn parameter). Those need a per-site read.\n"
          f"  Report only; exit 0 always.")


if __name__ == "__main__":
    main()
