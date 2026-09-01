# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative per-repo contract: what this
repo **owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist** of what it may depend on, links to the cross-repo
rules' one canonical home, and the **drift ledger** of known gaps. On any
conflict with prose in this file, BOUNDARY.md wins.

- **Domain test:** is this a **modelless inference primitive** with no riir dep (this repo is upstream of everything)? NO → it belongs in another repo; file there.
- **Read it before** adding any dep, crate, module, System impl, or vocabulary
  type — and before assuming a concern is yours to implement.
- **Enforcement** is not prose: `../riir-ai/scripts/ci_boundary_contract.sh`
  fails on an undeclared cross-repo dep, on a drift row without its open issue,
  and on a contract row that no longer matches the measured graph. Run boundary
  checks VIA the `boundary-guard` skill, not as ad-hoc greps.
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`), add
  the drift row, then fix. Closing the issue removes the row in the same commit.

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime are:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — applying a **deterministically constructed** (not
   trained) LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables. These update latent state, NOT base weights.

### MANDATORY: exhaust modelless paths before deferring to riir-train

Before deferring ANY gate, mechanism, or plan task to riir-train ("this needs
training"), you MUST check whether the three modelless paths above can fix it.
See the research skill §3.5 (`.agents/skills/research/SKILL.md`) for the full
decision protocol.

**Systematic, characterizable biases are modelless-correctable candidates,
NOT automatic riir-train dependencies.** If a gate fails because of a known,
named bias (e.g., "signal doubled", "position offset", "attention asymmetry"),
check whether a deterministically constructed reader-LoRA or freeze-state
correction can fix it before concluding "needs gradient descent."

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 was prematurely
deferred to riir-train without checking whether the doubled-signal bias could
be corrected modellessly via a deterministic reader-LoRA. The bias was
systematic and characterizable — exactly the case where raw/lora hot-swap
might work. The deferral was premature and has been reverted; the modelless
investigation (Issue 003, resolved-and-removed in commit `552b4632`) is
captured in `.benchmarks/313_ac_prefix_modelless.md` (Path 2: `attends_dedup`
eliminates the bias bit-identically to iterative-MLM on single-layer
micro-GPT, 0.0 diff). `ac_prefix` re-promoted to DEFAULT-ON on that
modelless pass; multi-layer equivalence remains a non-blocking riir-train
follow-up.

## Build Commands

```bash
# Default features (the GOAT-validated, promoted primitives)
cargo check
cargo test -p katgpt-core --lib

# Single feature
cargo check --features <feature_name>

# All features
cargo check --all-features

# Specific feature's tests
cargo test -p katgpt-core --features <feature_name> --lib
```

### The full gate — none of the above is a whole-repo claim

Every command listed above is narrow in at least one of three **independent**
axes, and a green result says nothing about what it compiled to nothing:

| Axis | Blind spot |
|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes are rejected by clippy's typeck and accepted by `check` (E0689 ambiguous-integer, E0631 deref-coercion in `redundant_closure`) |
| default vs `--all-features` | non-default gated code compiles to **nothing** |
| `-p <crate>` vs `--workspace` | *at the same default features*: a crate's own non-default feature can be switched on by the ROOT crate's defaults once the root is in the selected set |
| no `--all-targets` | skips every test / bench / example — which is where gated code lives |

The third axis is the least obvious. `cargo test -p katgpt-backend --lib`
compiled clean while `cargo test --workspace --lib` failed, because `gpu.rs` is
behind `katgpt-backend/gpu_inference` and the chain
`katgpt-rs/default -> async_qdq_overlap -> inference_router -> gpu_inference`
only fires when the root crate is selected. It also silently *shrinks* coverage:
four crates reporting "0 tests" per-crate contributed 704 under `--workspace`.

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going
```

`--keep-going` is not optional — without it the run stops at the first failing
target and under-reports. This gate was **red on `develop` from at least
2cb97410 until `c284dbb2`/this commit** (5 broken targets) while every gate in
the block above was green. Treat a green gate as a claim about its literal
command, not about the code.

Don't run it by hand — `scripts/full_gate.sh` is the assertion (it also refuses
to report a pass off macOS, where the `target_os = "macos"` device backends
compile to nothing even with `--all-features`, and checks that this document
still quotes the command it runs). `.github/workflows/full_gate.yml` **declares** a
weekly cron and a manual dispatch; per-push is deliberately not enabled — see
that file's preamble for the measured cost and the promotion criterion.

That was a declaration and not a schedule until 2026-09-01. `schedule` and
`workflow_dispatch` run **only from a repository's DEFAULT branch**, and this
repo's default was `main` — frozen at the v0.1.1 promote with no
`.github/workflows/` at all — so **neither trigger had ever fired**. The file's
own comment calls the schedule "the rot check"; the rot check had rotted, and
this paragraph advertised it as running. Fixed by moving the default branch to
`develop` (`.issues/704`), which is where AGENTS.md already says work lands;
both triggers are live as of that change.

Don't take that as permanently settled — a workflow file is identical on disk
whether or not it can execute, which is why this went unnoticed. The axis is
now measured: `scripts/ci_gate_coverage.py` reports, per workflow, which
declared triggers can actually fire, keeping **dead**, **unmeasured** (no remote
refs), **untracked** (committed by nobody yet — a colleague's in-flight file is
not a defect) and **PR-only** apart. It took the workspace from 7 dead workflows
to 1. Run it rather than re-reading trigger blocks by hand.

"Can fire" is still not "does fire". A workflow reachable only by
`workflow_dispatch` is a button, not a schedule, and three sibling repos
(`riir-chain`, `riir-dao`, `riir-neuron-db`) carry their whole Rust compile/lint
surface in exactly such a file — each by a *documented* main-only owner call
whose `push` is inert anyway, because `main` carries no copy of the workflow.
The report crosses the two axes rather than printing them side by side, which is
how that state stayed invisible: the coverage table credited the command and the
reachability table listed the trigger, and nothing multiplied them together
(`.issues/706`).

### The docs gate — same discipline, opposite cadence

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
`.github/workflows/docs_gate.yml` runs it **per-push** on ubuntu-latest. Both
choices are deliberately the inverse of the full gate's, and both files say why:
this gate has no `cfg(target_os)` surface so platform cannot change its verdict,
and it costs ~3s rather than >13 min.

The `CHECKS` array in that script is the list. **The count is deliberately not
written here** — this paragraph said "the three" for one commit after the fourth
was added, which is the drift the gate itself exists to catch, committed by the
paragraph describing it.

The original three existed before that wiring and **nothing invoked any of
them**; two were red on `develop`, both on false positives against docs that
were correct. Treat an uninvoked assertion as unknown, not as passing.

Two of the checks are worth knowing about specifically:

- `skill_repo_set_gate.py` (Issue 703) fails on a `SKILL.md` command block that
  types the repo set by hand instead of deriving it. It reads sibling repos,
  which CI does not have, so it separates its **vocabulary** (committed
  `scripts/repo_set.txt`, re-derived and failed-on-drift by every workstation
  run) from its **population** (12 `SKILL.md` locally, 8 in CI) and prints both.
  A gate that skipped in CI instead would be the vacuous green it exists to
  catch. Mark a deliberately narrow block `<!-- repo-set-ok: <reason> -->`.
- `bench_doc_audit.py` runs a `selftest()` on every invocation pinning the line
  shapes its tokenizer must recognise. Without it a regex regression is silent:
  the audit recognises fewer labels and still prints "0 mismatches". That is how
  26 riir-chain benchmark docs audited as clean while being unreadable
  (`.issues/702`).

### The docs gate covers ONE repo — two more tiers cover the rest

Both auditors accept a repo path and audit any repo, and for months nothing
pointed them anywhere but here. A sibling with stale labels then looks exactly
like a sibling nobody has checked. Three tiers now, deliberately different
cadences, and none of them subsumes another:

| instrument | where | cadence | scope |
|---|---|---|---|
| `docs_gate.yml` / `docs_gate.sh` | CI + workstation | per-push | katgpt-rs only |
| `sibling_docs_drift.yml` | sibling CI (reusable) | caller's choice | one caller |
| `scripts/docs_drift_sweep.py` | workstation | on demand | every contract repo |

The sweep is **deliberately not** in `docs_gate.sh`'s `CHECKS`: CI has a single
checkout, so it would derive an empty population and print a confident green
over zero repos. Its population is derived (BOUNDARY.md + a `.git` dir); its
expectations are committed (`scripts/docs_drift_floors.txt`), because deriving
both from the same walk is what makes a cross-repo gate permanently green.

`sibling_docs_drift.yml` is `workflow_call` so the three easy-to-get-wrong facts
live once rather than once per sibling — the worst being that the auditors
default to auditing **katgpt-rs**, so a sibling that omits the path argument
passes forever against the wrong repository. The workflow asserts the audited
tree is the caller's.

`scripts/ci_gate_coverage.py` is a **report, not a gate** (always exit 0): which
of the derived contract repos actually gate their full compile+lint surface in
CI, following each workflow into the scripts it calls, **and whether anything
automatically starts it**. Run it instead of re-typing the answer — that
question has been answered by hand twice and been wrong both times
(`.issues/701` R2).

Its own join is pinned by a `selftest()` that runs on every invocation, five
shapes, and the pin was canaried by reintroducing the bug it exists to catch —
a first cut asked only whether *any* scheduled workflow carried a cargo signal,
and a data-borne mention in `riir-chain`'s scheduled `toolchain_drift.yml` was
enough to vouch for a dispatch-only `rust.yml`. A weak automatic gate must not
speak for a strong manual one.

## Lint healing — `cargo heal` before manual fixes (adopted 2026-08-24)

Mechanical clippy findings (format-arg inlining, `match_bool`, `map_or`,
capacity, `needless_return`, …) are fixed by the riir-clippy healer FIRST,
manual second:

```bash
cargo heal --fix <paths>                                  # dry run (review only)
cargo heal --fix --write --verify <paths>                 # compile-gated apply
cargo heal --fix --write --verify --verify-args "--features <set>" <paths>  # gated code
```

- Global binary `cargo heal` = `~/.cargo/bin/cargo-heal` → the sibling
  `riir-clippy/target/release/cargo-heal` (built `--features
  fix_verify,clippy_verify`; rebuild after healer source changes). Missing
  sibling → fall back to manual fixes + `cargo clippy --fix`.
- `--verify` compiles baseline → applies → re-checks → auto-REVERTS breaking
  edits. Feature-gated code needs `--verify-args "--features <set>"` (a
  default-features check compiles gated files empty — a green check proves
  nothing about them).
- The healer is deliberately SILENT on documented divergence classes
  (comment-guarded matches, array-literal defaults, named-arg renames,
  nested macro args) — those stay manual; see the `cargo-heal` skill
  (`~/.agents/skills/cargo-heal/`) for the full table + discipline.
- `cargo clippy --fix` remains fine for one-off trivial fixes; the healer
  wins on batches (span-preserving, comment guards, compile gate,
  self-evolve memory) and was validated across the full katgpt-rs sweep
  (every surface, count-identical test validation, 2026-08-19).
- Observed misses / wrong suggestions → note in the session record; they feed
  riir-clippy's post-mining queue (usage-artifact improvement intake).

## Feature Flag Discipline

Every new primitive ships behind a feature flag (opt-in). Promotion to
default-on requires the GOAT gate to pass:

1. Implement behind `feature_name = []` (opt-in).
2. Write a benchmark proving the gain (latency, quality, or security).
3. Run the GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free
   or equivalent).
4. If all gates pass AND the gain is **modelless** → promote to `default`.
5. If the gain requires riir-train (training) → keep opt-in, note the
   dependency, do NOT promote to default.

**Promotion requires modelless gain.** A perf gain on a biased/incorrect answer
is NOT a modelless gain — it's a speedup of a wrong result. The quality gate
(G1 or equivalent) must pass modellessly for the GOAT to hold.

**Lossy-surface promotion rule (adopted 2026-08-28, Issue 750 T3):** a
promotion of a **lossy** surface (quantization, compression, any bit-changing
transform) gates on **deployed-path behavior — per-family, conditional
retention**, not on bit-identity or aggregate perplexity alone: bit-identity
is only available to lossless surfaces. Three independent arrivals at this
rule: Research 502 ("Behavior Before Perplexity"), Bench 696 (the KVarN
sink-guard GOAT), and Issue 750's measured bisection (gemma-2-2b Q4_K:
first behavior flip at prefix k=1 — layer 0 alone flips the sealed family;
restoring it costs 106.7 MiB, priced by the T2 override probe). Aggregate
perplexity can be flat while family-conditional behavior flips.

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322 / Plan 340).** Any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty (collectively: **UQ-bearing**) MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, plain split conformal) — on CRPS / coverage / Winkler score. If the primitive cannot beat the floor, the GOAT gate FAILS. Existing UQ-bearing primitives (BoMSampler Plan 281, Sleep-Time Anticipator Plan 334, Best-Belief Beta Selector Plan 336, KARC+overlay) are grandfathered but must include the floor at their next re-gate; future UQ primitives must include it from the initial gate. Tracked in `.issues/010`. The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now enforceable. **Issue 010 is FULLY CLOSED (T1-T7 all complete)** — see `.benchmarks/010_report_the_floor_consolidated.md` for the cross-primitive summary. **T7 (2026-07-20)** added the KARC+overlay dedicated floor test (`conformal_floor_karc_overlay.rs`) — the composite is SCOPE-LIMITED to chaotic regimes (BEATS on Lorenz-x at crps_ratio 0.0047 with K=4; LOSES on stationary seasonal at crps_ratio 5.74 with K=4), but coverage stays calibrated on both — no false-confidence signature. **T7 K-sweep (2026-07-20)** refuted the prior "K=4 too shallow" hypothesis: K=12 (matching the period) LOSES WORSE on seasonal (CRPS 5.74 → 20.26) and WINS HARDER on Lorenz (CRPS 0.0047 → 0.0018) — the scope-limit is **structural** (KARC's Chebyshev basis + ridge-fit doesn't fit periodic data regardless of K), not parametric. Production guidance: pick K by chaotic-regime memory needs; for periodic data use the floor directly.

**Plan 467 / Proposal 007 (2026-07-18):** Shipped `DualLeoOracle` as QGF's 3rd `QGradientOracle` impl — fuses a LEO teacher head + UVFA student head via `DualLeoMixer::combine_into` at the gradient level. Sibling to `LeoHeadOracle` (Plan 268) + `FlowFieldOracle`. G1–G4 PASS mechanistically; **G5 measured FAIL on synthetic data (riir-ai Bench 553, 2026-07-18): dual 0.00% vs single 0.50% on T7 Go puzzles, but the correctness invariant (QGF+LeoHeadOracle ≡ baseline) held bit-identically — mechanism correct, quality gate FAILs because synthetic data produces near-flat Q-fields.** **G5 also measured FAIL on civ real networks (riir-ai Bench 558, 2026-07-19): dual +2.69% vs single 35.68% → 36.64% on civ action-prediction, ≥3% gate — fourth-axis stop rule.** The civ dual-LEO investigation is fully closed per riir-ai Research 322 (the "alternative critic" escape hatch was category-confused — UQ primitives produce state forecasts, not per-action Q-gradients). The Plan 460 max-pool washout lesson is encoded as a design invariant (no operator between mix and consumer). Stays opt-in (`qgf_oracle + dual_leo`) with documented unproven G5 across both synthetic and civ real-network regimes; reopens only on seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough.

## Research Workflow

See `.agents/skills/research/SKILL.md` for the full research workflow:
paper classification, 7-repo routing, fusion-first distillation, novelty gate,
GOAT gate, and the mandatory modelless-unblock protocol (§3.5).

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`.agents/skills/substrate-first/SKILL.md` skill. It enforces:

1. **Vocabulary translation** — grep 3+ name variants (concepts ship under
   operator names like `GenericSpatialBelief`, not English names like "threat
   field"). A single-vocabulary grep returns ZERO hits even when substrate
   fully exists.
2. **Codebase grep** — search `*.rs` source across all 8 repos, not just
   `.plans`/`.docs`/`.issues`.
3. **Architectural rule check** — domain classification, two-brain model, sync
   boundary, bridge pattern.
4. **Consume vs. build decision** — if substrate exists, consume it; if not,
   file an issue in the right repo FIRST.

This prevents the recurring drift pattern where an agent builds a parallel
system that duplicates already-shipped substrate under a different name
(canonical failures: ThreatField Issue 047, orchard/motivation Issues 490/493).

> **Repo count:** the **product/distillation set is 8** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-armageddon`, `riir-dapps` (private). That is NOT the repo total: the
> workspace is **18 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-unity`, `riir-viewbridge`,
> `riir-auth`, `riir-burner`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `seal-game-editor`). **Re-measured 2026-09-01** by
> `../riir-ai/scripts/ci_boundary_contract.sh` — *"boundary contract clean —
> 18 repos, 211 cross-repo dep edges measured"*. It was 15 at the 2026-08-21
> run recorded here before; the count moved because contracts were added, not
> because repos were, and **this paragraph did not** — which is the failure it
> warns about, committed by the paragraph itself. Don't re-type the number:
>
> ```bash
> cd /Users/katopz/git && for d in */; do
>   [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
> done
> ```
>
> Four of the 2026-08-21 repos had no contract at all until that run, and
> `riir-armageddon` had been consuming `riir-games` + `katgpt-core` unaudited.
> Read a count in prose as a claim, not a fact.
> The historical "5-repo quintet" terminology referred to the 5 distillation
> targets (katgpt-rs + 4 riir-* siblings); `riir-game-sdk` (game vocabulary
> facade + dev-tool workspace) and `riir-armageddon` (arena/game-product domain
> types) were added later, and `riir-dapps` (the dApp layer — game outcome →
> generic chain settlement) on 2026-08-20. See Research 003 for the canonical
> boundary.
>
> **Two axes, not one.** Research 003's repo table is the *public/private*
> axis; its §"The Second Axis: Layering (game / dApp / chain)" is the
> *layering* axis — which private repo a game concern goes in. **Three tests,
> all must pass** (revised 2026-08-20; the earlier one-question form admitted
> FAME as "value" and ignored write rate):
> **(1) Product** — would a commerce customer of the chain want this in their
> dependency? An NFT is a token, so yes; a quest, no. **(2) Value** — BigInt
> fungible currency, a token, or an authority binding? FAME / XP / items /
> reputation are game scalars, not money. **(3) Rate** — does it fit a Glacial
> tier (≤0.1 Hz)? Binds hardest; `riir-neuron-db` is 1,627× cheaper per write
> and one chain tx at 10⁵ accounts eats 63% of a 20 Hz hot tick.
> Canonical failure: game rules (quest / bounty / crafting / reputation, two of
> them moving no money at all) shipped inside `riir-chain`'s
> consensus-critical program set — `riir-chain` Issues 096 + 097, closed on the
> layering side by `riir-dapps`.

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused** — even after a file is removed per the noise-reduction rule. Before
creating a new `.issues/` file, read `.issues/.highwater`, use `value + 1` as
the number, and write the new value back. This prevents the number-recycling
collision documented in `.issues/121`. The same rule applies to `.plans/`,
`.docs/`, `.benchmarks/`, and `.research/` — never recycle a number that git
history shows was already allocated.

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Models
- riir-train/data/gemma-2-2b-it-f16.gguf
- riir-train/data/MiniCPM5-1B-F16.gguf
