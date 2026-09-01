# Issue 706 — three sibling repos' entire Rust compile/lint surface runs only when a human clicks it

Status: **OPEN — measured and instrumented; the three fixes are each repo's own
owner call.** `scripts/ci_gate_coverage.py` grew the coverage x reachability
join that finds this class (this commit); the finding it produced is that
`riir-chain`, `riir-dao` and `riir-neuron-db` each carry their whole Rust
gate in a workflow reachable **only** by `workflow_dispatch`, and that the
`push` trigger each one declares is inert for a reason its own preamble did not
anticipate. No sibling repo was modified.

## The shape

`.issues/704` asked "can this workflow fire at all" and drove the workspace
from 7 dead workflows to 1. That question has a successor it does not ask:

> A workflow reachable only by `workflow_dispatch` **can** fire. Nothing
> **does** fire it.

`workflow_dispatch` is a button, not a schedule. A gate behind it runs exactly
as often as someone remembers to press it, which for a repo that is not being
actively released is zero. That is the same "decoration, not coverage" verdict
704 prints for dead workflows, one step less obvious because the workflow
genuinely can run.

## Why it was invisible

`ci_gate_coverage.py` already measured **both** halves and printed them a
screen apart:

- the coverage table credited `riir-neuron-db` with `--all-targets
  --all-features`;
- the reachability table listed `rust.yml[workflow_dispatch]`.

Neither line is wrong. Nothing multiplied them together, so the repo read as
covered. The join is now computed (`hand_only()`), and it is deliberately
**additive** — the columns above it are untouched, because `.issues/701` R2
quotes their numbers.

## Measured

| repo | the gate | live triggers | why its declared `push` is inert |
|---|---|---|---|
| `riir-chain` | `rust.yml` — `--each-feature --keep-going` (+ `--all-targets` in data) | `workflow_dispatch` | `push[main]`, and `origin/main` carries **no** `.github/workflows/` (develop is **830** commits ahead) |
| `riir-dao` | `rust.yml` — `clippy --all-targets` | `workflow_dispatch` | `push[main]`, `origin/main` carries no workflows (develop **20** ahead) |
| `riir-neuron-db` | `rust.yml` — `--all-targets --all-features` | `workflow_dispatch` | `push[main]`, `origin/main` carries no workflows (develop **174** ahead) |

`riir-clippy` declares the same main-only `push` and is **not** in this list:
its `rust.yml` also carries a `schedule`, which fires from the default branch.
That is the shape the other three are missing, and it is a one-line difference.

## The part that is NOT a criticism of the owner's call

All three preambles document main-only as a deliberate choice to spend no
Actions minutes on `develop` pushes, and that trade is the owner's to make.
The finding is narrower and the owners did not have it:

`riir-chain`'s own comment reasons "GitHub evaluates the `push` trigger against
the workflow file ON the pushed ref, and `main` currently carries no
`.github/workflows/` — so this goes live on the next promote-to-main push."
The reasoning is exactly right and the conclusion has not arrived: no promote
has happened, `main` still carries no workflows, and `develop` has moved 830
commits. The intended trigger is not merely dormant — it cannot fire until a
promote *that includes `.github/`* lands, which is the one thing a frozen
`main` will not receive.

So the choice on the table is not "main-only vs per-push". It is:

1. **Add a `schedule`** to each `rust.yml` (the `riir-clippy` shape) — fires
   from the default branch `develop`, costs one run per period, needs no
   promote and no branch-filter change. Cheapest, and the only option that
   works while `main` stays frozen.
2. **Widen the filter** to `branches: [main, develop]` — per-push cost on the
   branch all work lands on; the thing main-only was chosen to avoid.
3. **Leave it** — an explicit, recorded decision that these repos have no
   automatic Rust gate, rather than the current state, where the preamble
   describes a trigger that will not arrive.

Recommendation: (1). It preserves the owner's stated intent (no per-push spend
on `develop`) while making the gate actually run, and it is the option the
sibling that got this right already uses.

## Instrument changes (this commit, katgpt-rs only)

- `hand_only()` — lifted out of `main()` so it is testable; flags a repo when
  its **strongest** compile command is not reachable from any `schedule`/`push`.
- Strength is compared **lexicographically** (real `cargo` commands, then
  data-borne signals), not by presence. A first cut asked only whether *any*
  automatically-triggered workflow carried a signal, and `riir-chain` slipped
  through: the scheduled `toolchain_drift.yml` names the full-gate flags in a
  data table, which was enough to vouch for the dispatch-only `rust.yml`.
  **A weak automatic gate must not speak for a strong manual one.**
- `push_gap()` — names *why* a declared `push` is inert (which branch carries
  no copy of the file), because the two causes take two different repairs and
  "never fires" does not say which.
- `_tracked()` — an **untracked** workflow is unfinished work, not a dead gate.
  Caught live: a sibling agent added `riir-deployer/rust.yml` mid-run and it
  landed in the dead block. It is now reported as unmeasured.
- `selftest()` — five shapes, run on every invocation, and **canaried**: with
  the presence-only bug reintroduced the script exits 1 on case C. An assertion
  nobody has watched fail is not known to be able to fail, which is the defect
  `.issues/705` was.

Verified additive: output diffed against `HEAD`'s script, byte-identical above
the new section (the only other delta was `riir-deployer` gaining an untracked
workflow mid-session, from another agent).

## Closing conditions

- [x] Measure the axis and cross it with coverage.
- [x] Pin the join with a selftest, and canary the pin.
- [x] Separate untracked from dead so the report does not cry wolf.
- [x] Correct AGENTS.md, which described the reachability axis as the whole
      question.
- [ ] Owner decision per repo (`riir-chain`, `riir-dao`, `riir-neuron-db`) —
      recommendation (1), add a `schedule`. **Not applied: sibling repos, and
      each has an actively-working agent.**
- [ ] Remove this file in the closing commit per the noise-reduction rule.

Refs: `.issues/704` (the predecessor axis — "can it fire at all"), `.issues/705`
(a gate that passed having compiled nothing — same family: an instrument whose
green was not backed), `.issues/701` R2 (the coverage columns this join
deliberately does not disturb), `.issues/703` (derive the repo set, never type
it).
