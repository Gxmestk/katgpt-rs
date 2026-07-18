# Research 424: Lifelong LaCAM with Local Guidance (LLLG) — Modelless Multi-Agent Pathfinding Substrate

> **Source:** Arita & Okumura (AIST) — "Lifelong LaCAM with Local Guidance for Lifelong MAPF", [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) (AAAI 2026, May 2026). Code: <https://github.com/allegorywrite/lllg>.
> **Date:** 2026-07-15
> **Status:** Active — Super-GOAT candidate (see verdict §3). Open primitive scoped.
> **Related Research:** 219 (Topological Neural Operators / DEC — path-as-cochain), 296 (Stokes Calculus DEC Vocabulary — boundary flux / flow), 354 (Cross-Datapoint Set Attention — the latent-domain analog), 274 (Optimal CCE Moderator — swarm correlated policy on latent state), 242 (HLA per-NPC belief — the latent state congestion feedback targets), 144 (Functional Emotions — congestion → frustration direction)
> **Cross-ref (riir-ai private guide):** [Research 318 — Lifelong LaCAM Multi-Agent Physical Coordination Runtime Guide](../../riir-ai/.research/318_lifelong_lacam_multi_agent_physical_coordination_guide.md)
> **Cross-ref (riir-chain):** none — physical-domain motion is local-only; the executed joint action crosses sync as a raw `TxDelta` via existing action sync, not a new commitment
> **Classification:** Public (katgpt-rs/MIT). The NPC selling point (crowd-scale collision-free town-square navigation) stays private in riir-ai/318.

---

## TL;DR

LLLG is a **purely heuristic, training-free, receding-horizon multi-agent
pathfinding** algorithm that solves the lifelong MAPF (LMAPF) problem at
unprecedented scale (10,000 agents, real-time per-step planning, ~90% runtime
reduction vs RHCR with **higher throughput** in dense settings). It has
**zero** training dependency — every piece (LaCAM, PIBT, local guidance,
warm-start, hindrance) is a closed-form heuristic. That makes it a perfect
fit for the modelless mandate: it distills into `katgpt-rs` as a generic
multi-agent pathfinding **substrate** (the public primitive) and into
`riir-ai` as a **runtime** that wires the substrate to per-NPC HLA state,
zone-density routing, and the crowd-coordination stack (the private
selling-point — see riir-ai/318).

The paper's three transferable mechanisms:

1. **Local spatiotemporal guidance** `Φ` — per-agent, per-tick, finite-window
   (`w_Φ`) collision forecast. For each agent `i`, solve a short space-time A*
   that minimizes `<dist(π[w_Φ], g_i), 0>` (reach goal) **plus**
   `Σ_t ⟨1 + α·Ind[χ>0], χ⟩` where `χ` is the count of collisions with
   sibling agents' currently-stored guidance paths. The first move `Φ[i][1]`
   becomes a lexicographic preference in PIBT's ranking, ahead of pure
   `dist(v, g_i)` goal-distance.
2. **Warm-start from previous-step solution** `Π_{t-1}[2:w_Φ]` — carry the
   suffix of last tick's plan forward as this tick's guidance initialization.
   The paper proves empirically this dominates inheriting `Φ_{t-1}` alone
   (LLLG_Π > LLLG_Φ > LLLG_∅) — the previous plan is an explicit
   collision-free forecast, stronger than the soft guidance signal.
3. **Hindrance** — a one-step scalar estimate of "how much does my move block
   neighbors next tick", added as a PIBT tiebreak. Near-zero cost, ~free
   throughput gain in dense settings.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper's *training* content is zero — there is none. The entire algorithm
ships verbatim. What lands in the public engine is a **generic multi-agent
pathfinding substrate** parameterized over `MapPos`-like positions and
producing collision-free joint actions. The substrate has four pluggable
seams (one per LLLG mechanism) so a private consumer (`riir-ai/318`) can
project the guidance field onto HLA, gate moves by sigmoid(dot(HLA,
D_congestion)), and emit KG triples for socially meaningful encounters —
without leaking any of that game IP into the open primitive.

```text
// The open primitive — generic over position type P, cost function C:
pub trait LocalGuidanceSource<P> {
    /// Per-agent short-horizon collision-aware path fragment.
    /// Returns Φ[i] = the w_Φ-step preferred trajectory for agent i.
    fn guidance(&self, i: AgentId, q: &Config<P>) -> &[P];
}

pub struct LifelongLaCam<P, C: CostFn<P>, G: LocalGuidanceSource<P>> {
    w_pi: usize,              // planning horizon
    w_phi: usize,             // guidance window (≥ w_pi for clean warm-start)
    m: u8,                    // guidance refinement rounds (paper default 2)
    guidance_source: G,       // pluggable: default = space-time A* with collision count
    cost: C,                  // pluggable: terrain, slope, threat, ...
    hindrance_enabled: bool,  // default true — near-free throughput gain
    prev_plan_suffix: Vec<Vec<P>>,  // Π_{t-1}[2:w_Φ] warm-start cache
}

impl<P, C, G> LifelongLaCam<P, C, G> {
    /// One tick: warm-start Φ from prev suffix → m rounds of refinement →
    /// PIBT generation with hindrance tiebreak → execute first joint action.
    pub fn tick(&mut self, q: &Config<P>, goals: &[P]) -> JointAction<P>;
}
```

No training, no gradient descent. The default `LocalGuidanceSource` is the
paper's space-time A* on collision-count cost (Eq. 1). A private consumer
can supply a **HLA-projected guidance source** — `Φ[i]` becomes a path whose
step costs include `α · σ(dot(HLA_i, D_frustration))` so crowded cells cost
more to enter in proportion to the NPC's emotional state. That fusion is the
Super-GOAT; it lives in riir-ai/318.

---

## 1. Paper Core Findings

### 1.1 The lifelong MAPF (LMAPF) problem

Agents on an undirected graph `G = (V, E)` move synchronously. At each tick,
each agent either waits or moves to an adjacent vertex. Two agents may not
occupy the same vertex at the same tick (vertex collision) nor traverse the
same edge in opposite directions (edge collision). **Lifelong** means tasks
arrive continuously: when agent `i` reaches its current goal `g_i^t`, it is
assigned a new goal (paper: uniform random over `V`). The objective is
**throughput** — average completed tasks per tick — sustained over long
horizons, under a hard real-time budget (next action must be computed before
the previous one finishes executing).

### 1.2 The two failure modes LLLG bridges

The paper's framing (§1) is the cleanest summary of the LMAPF design space:

| Family | Examples | Strengths | Failure mode |
|---|---|---|---|
| **Optimal-ish windowed** | RHCR (Li et al. 2021) | Near-optimal solutions in sparse settings | Degrades sharply in dense settings — repeated branch-and-replan blows up |
| **Fast suboptimal** | PIBT (Okumura 2022), LaCAM (Okumura 2023b) | Stable, real-time, scales to thousands of agents | Highly suboptimal in dense settings — purely goal-directed moves induce congestion |

LLLG inherits the speed of the suboptimal family (PIBT/LaCAM-based, ~ms/step
at 10K agents) while producing better-than-RHCR throughput in dense settings.
The mechanism is **guidance** — a soft, agent-centric spatiotemporal bias
that steers PIBT away from emerging congestion without slowing it down.

### 1.3 The three mechanisms (paper §2.D, §3)

**(a) Local guidance construction (paper Eq. 1, Algorithm 1).** For each
agent `i`, solve a single-agent space-time A* over `w_Φ+1` steps with cost:

```
cost_i(π) = ⟨dist(π[w_Φ], g_i), 0⟩                       // reach the goal
          + Σ_{t=0..w_Φ-1} ⟨1 + α·Ind[χ>0], χ⟩           // avoid collisions
```

where `χ` is the count of collisions of the transition `(π[t], π[t+1])` with
sibling paths currently stored in `Φ` (the shared guidance set). The
hyperparameter `α ∈ R_≥0` controls how aggressively collisions are penalized
(paper uses α implicitly via the `1 + α·Ind[χ>0]` term). The construction is
**sequential over agents** (each agent sees the others' already-updated
guidance) and repeated `m` rounds to reduce agent-order bias. Paper default
`m=2` (vs `m=1` in one-shot LG-LaCAM, because LLLG's windowed replanning
makes the extra round affordable).

**(b) Warm-start from previous solution (paper §3 "Leveraging LG in
Receding Horizon").** Two schemes:

- **LLLG_Φ** — initialize `Φ_t` with `Φ_{t-1}` (the previous tick's guidance
  field).
- **LLLG_Π** — initialize `Φ_t` from the **suffix** `Π_{t-1}[2:w_Φ]` of the
  previous tick's *solution* (the actual executed-path forecast).

Paper §4.C Fig. 6 shows LLLG_Π > LLLG_Φ > LLLG_∅ at essentially identical
runtime. **The intuition:** `Φ` alone is a soft collision-avoidance bias;
`Π_{t-1}` is an explicit collision-free forecast over the near future, which
is a strictly stronger initialization. This is the core LLMAPF-specific
contribution beyond one-shot LG-LaCAM.

**(c) Hindrance (Okumura & Nagai 2025, integrated as PIBT tiebreak).** A
one-step scalar `hindrance(i → u)` = how many nearby agents `j ≠ i` would be
blocked at tick `t+1` if `i` moves to `u` at tick `t`. PIBT's lexicographic
cost becomes:

```
⟨Ind[Φ[i][1] ≠ u], dist(u, g_i), hindrance(i→u), ε⟩
```

So PIBT first prefers moves consistent with the guidance, then
goal-direction, then low hindrance, then random tiebreak. Hindrance is
near-zero cost (one pass over each agent's neighborhood) and yields a small
but consistent throughput gain in dense settings (paper Fig. 11).

### 1.4 Quantitative results (paper §4.B, Fig. 4)

| Map | Agent count | LLLG throughput | Best baseline | LLLG runtime | Baseline runtime |
|---|---|---|---|---|---|
| empty-48-48 | 1000 | 27.3 | 21.4 (LaCAM) | 0.26 s/step | 0.032 s/step (LaCAM) |
| random-64-64-10 | 1000 | 21.1 | 20.7 (RHCR) | 0.21 s/step | 2.13 s/step (RHCR) |
| ht_chantry (bottlenecks) | 800 | **+81%** vs RHCR | — | **−96%** vs RHCR | — |
| warehouse-20-40-10-2-2 | **10,000** | ~30% gain over PIBT | — | **<1 s/step** | RHCR fails at 2,000 |

Headline: **scales to 10,000 agents at <1 s/step with consistent throughput
gains**, with the win amplifying as density rises. The single failure mode
is **long one-cell-wide corridors** (warehouse-20-40-10-2-1) where local
finite-window guidance cannot see oncoming traffic beyond the window —
global guidance (Guided-PIBT) wins there.

### 1.5 The anytime-refinement negative result (paper §4.C Fig. 9)

A surprising and honest finding: applying LaCAM* anytime refinement to LLLG's
windowed plan **degrades** lifelong throughput, even though it improves the
finite-window plan quality. The paper's diagnosis: the LaCAM* f-value
(`g + h` on the `w_Π`-step window) is misaligned with the lifelong throughput
metric — improving the window does not improve the executed trajectory,
because only the first step is committed and the rest is invalidated by
online goal updates. **Lesson for us:** per-tick anytime refinement on the
planning window is the wrong objective in lifelong settings. The right
objective is throughput, which is not a per-tick quantity. (This is
consistent with our `gain_cost_halt` philosophy — don't spend compute on a
local improvement that doesn't move the global objective.)

---

## 2. Distillation

### 2.1 What ships in the open primitive (katgpt-rs)

```text
crates/katgpt-core/src/multi_agent_path/
├── mod.rs                  — LifelongLaCam<P, C, G> orchestrator + JointAction<P>
├── config.rs               — Config<P> = Vec<P> joint configuration; AgentId newtype
├── pibt.rs                 — PIBT one-step generator with lexicographic cost
│                             ⟨Ind[Φ≠u], dist(u,g), hindrance, ε⟩
├── local_guidance.rs       — space-time A* on Eq. 1 cost (default impl)
│                             + LocalGuidanceSource trait (pluggable)
├── warm_start.rs           — Π_{t-1}[2:w_Φ] suffix cache + LLLG_Π / LLLG_Φ / LLLG_∅
├── hindrance.rs            — one-step blocking-count estimator
└── tests.rs                — paper-fig reproducers on toy grids
```

**Cargo feature:** `multi_agent_path = []` (opt-in). No heavy deps — pure
Rust, no rayon (the per-tick work parallelizes trivially across agents but
the paper runs single-threaded at 10K agents <1s; we leave rayon as a
consumer-side concern to keep the substrate leaf-clean per the
`katgpt-core` dep profile).

### 2.2 The four pluggable seams (the Super-GOAT hook)

The whole point of distilling into the public engine is that the substrate
is **generic** over the four LLLG mechanisms, so a private consumer can swap
any of them without forking:

| Seam | Default (paper) | Pluggable alternative (private consumer) |
|---|---|---|
| `CostFn<P>` (terrain / transition cost) | Uniform (1 per move) | Heightfield slope penalty, threat cochain value, faction zone penalty, economy toll |
| `LocalGuidanceSource<P>` (the `Φ` field) | Space-time A* on collision count (Eq. 1) | HLA-projected guidance: step cost includes `α · σ(dot(HLA_i, D_frustration))` so crowded cells cost more for stressed NPCs (riir-ai/318) |
| Warm-start scheme | `LLLG_Π` (suffix of prev solution) | Per-NPC personality-weighted blend of `LLLG_Π` and `LLLG_Φ` (curious NPCs explore, conservative NPCs stick to plan) |
| Hindrance estimator | 1-step blocking count | Affect-aware hindrance: blocks from fearful NPCs count more (the social-cost extension) |

A consumer that uses all four defaults gets the paper's LLLG verbatim. A
consumer that plugs in HLA-aware guidance gets the Super-GOAT — see
riir-ai/318.

### 2.3 Latent-space reframing (mandatory per research skill §1 step 3)

The paper operates entirely in **physical space** (grid vertices). The
latent-space reframing is **not** "lift pathfinding into latent space" —
that would break the sync-boundary rule (physical motion MUST stay raw for
deterministic replay and anti-cheat). The reframing is:

> **The local guidance field `Φ` is itself a latent-space object.** Each
> `Φ[i]` is a `w_Φ`-step trajectory — a rank-1 cochain on the time axis. The
> *collision count* `χ` is a discrete approximation to the **codifferential**
> (DEC divergence) of the joint flow field at that cell-time pair: it counts
> how many sibling flows pass through the same (cell, time). High `χ` =
> positive divergence = "agents are converging here, avoid".

This connects to the existing DEC substrate (`katgpt-core/src/dec/`):

- The joint configuration `Q_t = (p_1, ..., p_n)` is a **0-cochain** over
  the spatial complex at time `t`.
- The transition `Q_t → Q_{t+1}` is a **1-cochain** (a flow on edges).
- The collision count `χ(cell, t)` is the **codifferential δ** of the flow
  cochain — it measures local density flux. `χ > 0` at a cell means the
  flow is converging there (positive divergence in agent-density terms).
- LLLG's guidance cost `Σ_t ⟨1 + α·Ind[χ>0], χ⟩` is a **discrete line
  integral of the codifferential along the candidate path** — the agent is
  penalized in proportion to the integrated density-flux it would pass
  through.

This is **exactly** the `belief_mass_divergence` pattern (Plan 314 wrapper
on the DEC substrate), lifted from belief-cochains to motion-cochains. The
fusion hook: a private consumer can replace `χ` (raw collision count) with
`δ(behavior_flow_field)` where the behavior flow is the HLA-projected
direction-vector field — same operator, richer signal.

**Adapter routing / KV compression / speculative decode are NOT the primary
framing here.** This is a motion-planning primitive, full stop. The latent
reframing is the `Φ`-as-cochain observation above, not an adapter trick.

### 2.4 Fusion — the Super-GOAT combination

| Fusion | Source A | Source B | Novel combination |
|---|---|---|---|
| **LLLG × HLA (emotion-driven congestion avoidance)** | This paper (`Φ` collision-count cost) | R144 Functional Emotions + R242 HLA (per-NPC affect direction vectors) | **NEW CAPABILITY** — NPCs avoid congested cells in proportion to their emotional state. A calm merchant takes a slightly longer route through a busy market; a desperate fleeing NPC pushes through. Same algorithm, per-NPC personality. |
| **LLLG × DEC Stokes (codifferential-guided navigation)** | This paper (`χ` discrete divergence) | R219/R296 DEC substrate (`codifferential`, `belief_mass_divergence`) | **GOAT-tier reframing** — `χ` is a special case of `δ(flow_cochain)`. Replacing it with the DEC operator lets the guidance consume any 1-cochain field (threat, curiosity, faction tension), not just collision counts. |
| **LLLG × Crowd MCGS (physical + latent coordination)** | This paper (physical-domain joint motion) | P298 Crowd MCGS (latent-domain emergent behavior) | **NEW CAPABILITY** — the crowd MCGS currently has no physical-motion substrate; NPCs coordinate on belief but bump into each other. LLLG supplies the missing physical layer; the two compose as "coordinated motion + coordinated belief". |
| **LLLG × Warm-Path Navigation (Plan 350)** | This paper (receding-horizon windowed planning) | P350 Warm-Path (single-agent stop-and-think at obstacles) | **CLOSES P350 NON-GOAL** — P350 explicitly defers "multi-agent warm-path coordination" as v2. LLLG is that v2: when an agent escalates to `Calm` (P350 trigger), LLLG's guidance can route the whole crowd around the obstacle, not just the stuck agent. |
| **LLLG × Zone Density Routing (Plan 351, default-on)** | This paper (per-agent guidance field) | P351 `zone_density_routing` (zone-level compute routing) | **REFINEMENT** — zone-density currently routes *compute* (which zones get cognition budget); LLLG routes *motion* (which cells agents avoid). The two density signals can fuse: a dense zone gets more cognition AND its agents get stronger congestion avoidance. |
| **LLLG × Sheaf Coordination (P394, default-on)** | This paper (collision-free joint action) | P394 Sheaf ADMM (three-state consensus on HLA) | **NEW CAPABILITY** — sheaf coordination currently converges on latent belief; adding LLLG lets the converged belief *drive* the joint motion plan. "We all agreed to flee → we all move in coordinated flee formation." |

The fusion that makes this Super-GOAT (not just GOAT) is **LLLG × HLA ×
Crowd MCGS × P350**: a single substrate that produces collision-free joint
motion at 10K-agent scale, where each agent's guidance field is modulated by
its emotional state, the crowd's converged belief, and the warm-path
escalation policy. No incumbent has this; the paper alone doesn't have this;
the codebase alone doesn't have this. Only the fusion does.

### 2.5 Latent vs raw boundary (per AGENTS.md — critical for game AI)

| Data | Space | Synced? | Rule |
|---|---|---|---|
| Joint configuration `Q_t` (all agent positions) | **Raw** (physical) | **YES** — existing `MapPos` sync | Bit-identical for deterministic replay / anti-cheat. Never latent. |
| Executed joint action `Π_t[1]` | **Raw** (physical) | **YES** — existing action sync as `TxDelta` | The committed move per tick. |
| Local guidance field `Φ[i]` (per-agent `w_Φ`-step forecast) | **Latent** (semantic — it's a *preference*, not a commitment) | **NO** | Per-NPC subjective forecast; recomputed each tick from local observations + fog-of-war-gated sibling positions. Discarded after the first step is executed. |
| Collision count `χ(cell, t)` | **Raw** (derived from synced positions) | **NO** (derived locally) | Computed from the synced `Q_t`; not itself synced. |
| Hindrance scalar | **Raw** (derived) | **NO** | One-step blocking estimate; pure function of synced positions. |
| HLA-projected guidance cost `α · σ(dot(HLA_i, D_frustration))` (private extension) | **Latent** | **NO** | Per-NPC affect modulating the guidance cost; never crosses sync. |
| `Π_{t-1}` warm-start cache | **Raw** (positions) + **latent** (the *interpretation* as warm-start) | **NO** (local cache) | Per-server working memory; rebuilt on cold-start from the synced `Q_t`. |

**Bridge rule (per AGENTS.md):** raw positions stay raw and synced; latent
guidance stays local. The bridge is `Φ → Π_t[1]` — the latent guidance
*selects* the raw action that gets committed. This is the same one-way gate
as every other bridge in the codebase (raw observation → latent belief →
raw action). Zero-allocation, gateable by feature flag, no sync dependency.

**Two-brain compatibility:** LLLG operates on the **info brain** (raw
`MapPos`, synced ground truth). The latent guidance `Φ` is computed in the
think brain (per-NPC subjective forecast, fog-of-war gated). The bridge is
one-way: info brain provides `Q_t` → think brain computes `Φ` → think brain
selects action → action commits to info brain via existing action sync.
Same discipline as Plan 350 warm-path, Plan 118 fog-of-war, R134 SwiR.

---

## 3. Verdict: **Super-GOAT**

### 3.1 Novelty gate (all 4 YES)

**Q1 — No prior art?** YES. Vocabulary-translated grep across all 5 repos
(`.research/` + `.plans/` + `.docs/` for intent; `src/` + `crates/` for
shipped code) with BOTH paper vocabulary (`MAPF`, `LaCAM`, `PIBT`, `LMAPF`,
`lifelong`, `receding horizon`, `RHCR`, `windowed planning`, `local
guidance`, `hindrance`, `multi-agent pathfinding`) AND codebase vocabulary
(`pathfind`, `find_path`, `navigation`, `collision`, `crowd coordination`,
`swarm`, `flocking`, `boids`, `crowd density`, `warm start`, `replan`)
returns:
- **Single-agent A* pathfinding** (shipped): `riir-engine/src/pathfinder.rs`,
  `katgpt-pruners/src/pathfinder.rs`, `riir-ai/crates/riir-games-quest/src/quest/path.rs`,
  `riir-ai/crates/riir-engine/src/fourier/dungeon.rs`, `riir-ai/crates/riir-engine/src/fourier/path_periodic.rs`,
  `riir-ai/crates/riir-games/src/dungeon/fourier_pathfind.rs`. All single-agent.
- **Latent-domain crowd coordination** (shipped): `crowd_mcgs/`,
  `crowd_attention_bridge.rs`, `sheaf_coordination_bridge.rs`, Plan 355
  (set attention on HLA belief), Plan 394 (sheaf ADMM on HLA). All operate
  on latent HLA state, NOT on physical collision-free motion.
- **Density-aware zone routing** (shipped, default-on): Plan 351
  `zone_density_routing` — routes *compute*, not motion.
- **Warm-path navigation** (shipped, partial): Plan 350 — single-agent
  stop-and-think at obstacles. **Explicitly Non-Goal: "Multi-agent
  warm-path coordination. Each agent reasons independently. Coordinated
  reasoning is a v2 concern."**
- **Seal NavMesh** (shipped): `seal-core::map::nav::NavMesh` — graph
  topology + collision grid, **no solver**.

**Zero** shipped implementation of multi-agent collision-free pathfinding
at crowd scale. The gap is precise, documented (P350 Non-Goal), and exactly
what the paper fills.

**Q2 — New class of behavior?** YES. Today's NPCs can:
- (i) pathfind independently (single-agent A*, P350 warm-path),
- (ii) coordinate latent beliefs (crowd_mcgs, set attention, sheaf ADMM),
- (iii) route compute by zone density (P351).

They **cannot** produce collision-free joint motion at crowd scale. They
fall back to either single-agent A* (collision-prone in dense settings) or
blind RVO-style reactive avoidance (congestion-prone, no throughput
guarantee). LLLG adds a fourth capability: **coordinated physical motion
at 10K-agent scale with provable throughput**. This is not an optimization
of an existing capability; it's a new one.

**Q3 — Product selling point?** YES. The sentence: "Our MMORPG town squares
host 10,000 concurrent NPCs that navigate collision-free in real time,
route around congestion without scripted events, and never deadlock at
bottlenecks — no competitor can do this at this scale." This is directly
observable to players (no clipping, no stuck NPCs, smooth crowd flow at
marketplace events), defensible (the tuning of `α`, `w_Φ`, `m`, and the
HLA-projected guidance fusion are private), and connects to the existing
"emergent social/economic behavior" pitch (a marketplace crowd that flows
well is a marketplace where trade happens).

**Q4 — Force multiplier?** YES. Connects to ≥4 existing systems:
1. **Pillar 1 — Fourier Spatial AI** (P002, P121): LLLG supplies the
   physical-motion substrate the Fourier spatial primitives currently lack.
2. **Plan 350 — Warm-Path Navigation**: closes the explicitly-deferred
   "multi-agent warm-path coordination" Non-Goal.
3. **Crowd MCGS / Set Attention / Sheaf Coordination** (P298, P355, P394):
   supplies the missing physical-motion layer that lets converged latent
   belief drive coordinated motion.
4. **Zone Density Routing** (P351, default-on): fuses motion-routing with
   compute-routing on the same density signal.
5. **HLA / Functional Emotions** (R242, R144): per-NPC affect modulates
   guidance cost — the Super-GOAT fusion hook.

### 3.2 One-line reasoning

**Super-GOAT because:** LLLG is a modelless, training-free, 10K-agent-scale
multi-agent pathfinder that fills a precise documented gap (P350 Non-Goal),
adds a new capability class (coordinated physical motion), produces a
directly-observable product selling point (collision-free town squares),
and force-multiplies ≥4 existing pillars/systems when fused with HLA,
Crowd MCGS, and the DEC substrate.

### 3.3 MOAT gate per domain

| Domain | In scope? | MOAT contribution |
|---|---|---|
| `katgpt-rs` (this note, open primitive) | ✅ YES — paper-derived fundamental primitive (multi-agent pathfinding substrate), generic over position type, no game IP | Open adoption hook — the generic substrate. Promote/demote tracked per stack (the "navigation stack" slot). |
| `riir-ai` (private runtime, riir-ai/318) | ✅ YES — pillar-level amplifier (Pillar 1 Fourier Spatial + crowd-coordination substrate) | The private selling point — HLA-projected guidance, Crowd MCGS physical layer, P350 multi-agent closure. |
| `riir-chain` | ❌ NO — physical motion is local; only the committed action crosses sync via existing `TxDelta`. No new commitment semantics. |
| `riir-neuron-db` | ❌ NO — no shard/freeze/consolidation angle. (The HLA direction vectors used in the private fusion are already BLAKE3-committed via existing `NeuronShard`; no new shard type.) |
| `riir-train` | ❌ NO — zero training content. |

**Routing:** open primitive → `katgpt-rs/crates/katgpt-core/src/multi_agent_path/`
behind feature `multi_agent_path`. Private runtime guide →
`riir-ai/.research/318`. Plans → `katgpt-rs/.plans/440` (open primitive) +
`riir-ai/.plans/489` (runtime fusion).

### 3.4 Mandatory outputs (created in this session)

1. ✅ **Open primitive** (this note) → `katgpt-rs/.research/424_*.md`.
2. ✅ **Private architectural guide** → `riir-ai/.research/318_*.md` (the
   HLA × Crowd MCGS × P350 fusion selling-point doc).
3. ✅ **Plans** → `katgpt-rs/.plans/440_*.md` (open substrate) +
   `riir-ai/.plans/489_*.md` (runtime fusion, gated on the open primitive
   landing).

### 3.5 Validation protocol (the GOAT gate — to run AFTER plan lands)

The Super-GOAT claim is conditional on the GOAT gate passing. The gate has
two tiers — the **paper-reproduction tier** (proves we shipped LLLG
correctly) and the **fusion tier** (proves the HLA × LLLG combination adds
value beyond either alone).

**Paper-reproduction tier (G1–G4):**
- **G1 (correctness)** — on the 4 paper benchmark maps
  (empty-48-48, random-64-64-10, warehouse-10-20-10-2-2, ht_chantry) at
  800 agents / 500 steps, our LLLG throughput is within 10% of the paper's
  reported numbers. (We won't bit-match the paper's C++ implementation;
  this is a "same order of magnitude, same qualitative rankings" gate.)
- **G2 (congestion mitigation)** — heatmap of stop-counts shows LLLG
  produces qualitatively smoother traffic than PIBT (paper Fig. 3 visual
  comparator). Quantitative: LLLG max-stops-per-cell < 0.5 × PIBT
  max-stops-per-cell on empty-48-48 at 1000 agents.
- **G3 (no-regression)** — `cargo check --all-features` clean;
  `cargo test -p katgpt-core --lib` passes; existing single-agent
  `find_path` tests unaffected.
- **G4 (latency)** — per-tick planning time < 50 ms at 1000 agents on
  commodity hardware (paper reports 0.21–0.26 s/step on M1 Ultra at 1000
  agents; our Rust impl should be in the same ballpark — the gate is
  generous to allow for impl maturity).

**Fusion tier (G5–G7 — the Super-GOAT gates, run in riir-ai/489):**
- **G5 (HLA-modulated guidance adds value)** — on a toy scenario where two
  NPC populations (calm vs desperate) share a corridor, the HLA-projected
  guidance produces *qualitatively different* crowd flow than uniform-α
  LLLG: desperate NPCs take shorter, more congested routes; calm NPCs take
  longer, smoother routes. Throughput is preserved (no regression) but the
  *behavioral signature* differs measurably (e.g., per-population
  stop-count distribution diverges by > 30%).
- **G6 (crowd MCGS physical layer)** — running Crowd MCGS *with* LLLG
  enabled produces emergent region/faction/social behavior that is
  *physically coherent* (NPCs actually move to the regions the MCGS
  suggests), vs the current behavior where MCGS convergence is purely
  latent and NPCs may be physically elsewhere.
- **G7 (P350 multi-agent closure)** — the railway-crossing scenario from
  P350, run with N=50 NPCs crossing simultaneously, shows LLLG routing the
  whole crowd through the clear window — vs P350's single-agent wait/reroute
  which leaves the crowd stuck individually. Quantitative: crowd crossing
  time < 2 × single-agent crossing time at N=50.

**Defend-wrong PoC (per research skill §3.6):** the G5/G6/G7 quality claims
require a head-to-head PoC in `riir-ai/crates/riir-poc/` comparing (a)
paper-LLLG (uniform α), (b) HLA-projected LLLG (the fusion), (c) PIBT-only
baseline, (d) Crowd MCGS without physical layer. If the PoC refutes the
quality claim (e.g., HLA projection doesn't change behavior measurably),
the verdict is honestly revised per §3.6 — the architectural coverage
stands (G1–G4), the quality claim becomes a tracked follow-up.

### 3.6 Modelless-first check (§3.5 protocol)

The paper is **entirely modelless** — no training, no backprop, no gradient
descent. The three modelless unblock paths are trivially satisfied (there
is nothing to unblock):
1. **Freeze/thaw** — N/A (no weights).
2. **Raw/lora hot-swap** — N/A (no weights).
3. **Latent-space correction** — N/A (no bias to correct; the algorithm is
   a closed-form heuristic).

The HLA-projected guidance fusion (riir-ai/318) is also modelless: the HLA
direction vectors are existing frozen artifacts (R144, BLAKE3-committed in
`NeuronShard`), and the projection is a dot-product + sigmoid per
AGENTS.md §2. No training is introduced by the fusion. If a future tuning
need arises (e.g., optimal `α` per NPC personality), the §3.5 protocol
applies: exhaust modelless paths (per-personality `α` table derived from
HLA projection) before deferring any tuning to riir-train.

### 3.7 Honest caveats

1. **Long-corridor failure mode.** The paper reports LLLG underperforms
   Guided-PIBT on warehouse-20-40-10-2-1 (long one-cell-wide corridors)
   because finite-window guidance cannot see oncoming traffic beyond the
   window. Our substrate inherits this limitation. **Mitigation:** the
   `w_Φ` parameter is configurable per zone; long-corridor zones can use a
   larger `w_Φ` at perf cost, or fall back to a global-guidance source
   (pluggable via `LocalGuidanceSource`). This is a known tradeoff, not a
   defect.

2. **Anytime refinement misalignment.** Per paper §4.C Fig. 9, applying
   LaCAM* anytime refinement to LLLG's windowed plan *degrades* lifelong
   throughput. We will NOT add an anytime-refinement path to the substrate
   by default — the paper's negative result is the design guidance. (A
   consumer who wants to experiment can plug one in via the seam, but the
   default `LifelongLaCam::tick` returns the first feasible windowed plan.)

3. **We are not the paper's authors.** The paper's reference C++
   implementation is at <https://github.com/allegorywrite/lllg>. Our Rust
   port will not bit-match it. The G1 gate is "within 10% of paper
   throughput on the same maps at the same agent count" — not "identical
   tick-by-tick trajectories". This is honest about what reproduction
   means.

4. **The Super-GOAT claim is conditional on G5–G7 passing.** If the HLA
   fusion doesn't produce a measurably different behavioral signature (G5
   fails), the verdict drops to **GOAT** — the substrate still ships
   (paper-faithful LLLG is a real win), but the private fusion selling
   point weakens. The PoC (§3.5) settles this before any promotion.

5. **Scale claims are paper-reported, not yet our measurements.** The 10K
   agents / <1s/step number is from the paper's M1 Ultra run. Our Rust
   impl's actual scale will be measured at G4. If we cannot reach 10K
   agents in real time, the selling point scales down accordingly (e.g.,
   "1K agents at real time" is still a win, just a smaller one). Honest
   about this.

---

## 4. Implementation sketch (open primitive — Plan 440)

### Phase 1 — Substrate skeleton (paper-faithful LLLG)

- [ ] **T1.1** `crates/katgpt-core/src/multi_agent_path/mod.rs` —
      `LifelongLaCam<P, C, G>` struct + `tick()` orchestrator. Generic over
      `P: Position` (trait with `neighbors()`, `is_passable()`),
      `C: CostFn<P>`, `G: LocalGuidanceSource<P>`.
- [ ] **T1.2** `config.rs` — `Config<P>` (joint configuration),
      `AgentId(u32)`, `JointAction<P>`.
- [ ] **T1.3** `pibt.rs` — PIBT one-step generator with lexicographic cost
      `⟨Ind[Φ≠u], dist(u,g), hindrance, ε⟩`. Priority inheritance +
      backtracking per Okumura et al. 2022.
- [ ] **T1.4** `local_guidance.rs` — default `LocalGuidanceSource` impl:
      space-time A* on Eq. 1 cost. `m`-round sequential refinement per
      Algorithm 1.
- [ ] **T1.5** `warm_start.rs` — `Π_{t-1}[2:w_Φ]` suffix cache. Three
      schemes: `LLLG_Π`, `LLLG_Φ`, `LLLG_∅` (paper §3).
- [ ] **T1.6** `hindrance.rs` — one-step blocking-count estimator per
      Okumura & Nagai 2025.
- [ ] **T1.7** Feature flag `multi_agent_path = []` in `Cargo.toml`.
- [ ] **T1.8** Unit tests on toy grids (2 agents, 3×3 map, vertex collision;
      edge collision; deadlock; throughput sanity).

### Phase 2 — GOAT gate (paper reproduction, G1–G4)

- [ ] **T2.1** Benchmark harness: 4 paper maps, 800 agents, 500 steps.
      Reproduce paper Fig. 4 throughput numbers within 10%.
- [ ] **T2.2** Heatmap reproducer (paper Fig. 3) — stop-count visualization
      for congestion-mitigation G2.
- [ ] **T2.3** Latency gate G4 — per-tick time at 1000 agents.
- [ ] **T2.4** `cargo check --all-features` clean (G3).
- [ ] **T2.5** If G1–G4 pass → promote to default-on per AGENTS.md (the
      substrate is modelless; promotion is allowed). **OR** keep opt-in if
      the riir-ai fusion (G5–G7) hasn't passed yet — the open primitive can
      ship opt-in and the private runtime can fuse with it regardless.

### Phase 3 — Fusion hooks (the pluggable seams)

- [ ] **T3.1** `LocalGuidanceSource<P>` trait — documented extension point
      for the HLA-projected guidance (riir-ai/318 supplies the impl).
- [ ] **T3.2** `CostFn<P>` trait — documented extension point for
      heightfield/threat/economy cost.
- [ ] **T3.3** Warm-start scheme enum — documented extension point for
      personality-weighted blend.
- [ ] **T3.4** Hindrance trait — documented extension point for
      affect-aware blocking.

### Phase 4 — DOCS

- [ ] **T4.1** `katgpt-rs/README.md` feature-showcase entry for LLLG.
- [ ] **T4.2** Cross-ref from `katgpt-rs/.research/219` (DEC) and
      `katgpt-rs/.research/354` (Set Attention — the latent-domain analog).

---

## 5. References

- **Paper:** Arita & Okumura, "Lifelong LaCAM with Local Guidance for Lifelong MAPF", arXiv:2605.16855, AAAI 2026.
- **Code:** <https://github.com/allegorywrite/lllg>
- **Foundational:**
  - Okumura 2023b — LaCAM (AAAI).
  - Okumura et al. 2022 — PIBT (AIJ).
  - Okumura 2023a — LaCAM* (IJCAI).
  - Arita & Okumura 2026 — LG-LaCAM (one-shot local guidance, AAAI).
  - Okumura & Nagai 2025 — Hindrance (SoCS).
  - Li et al. 2021b — RHCR (AAAI).
- **Internal cross-refs:**
  - katgpt-rs R219, R296 — DEC substrate (`codifferential` = the formal
    version of `χ` collision count).
  - katgpt-rs R354 — Cross-Datapoint Set Attention (latent-domain crowd
    coordination analog).
  - katgpt-rs R274 — Optimal CCE Moderator (swarm correlated policy).
  - riir-ai R167 — Crowd Joint Inference guide (the latent crowd substrate
    this physical layer plugs into).
  - riir-ai P298 — Crowd MCGS (host).
  - riir-ai P350 — Warm-Path Navigation (single-agent; LLLG closes its
    multi-agent Non-Goal).
  - riir-ai P355 — Crowd Joint Inference runtime.
  - riir-ai P394 — Sheaf Coordination runtime.
  - katgpt-rs P351 — Zone Density Routing (default-on).

## TL;DR

LLLG is a modelless, training-free, 10K-agent multi-agent pathfinder. It
distills into `katgpt-rs` as a generic substrate with four pluggable seams
(cost, guidance source, warm-start scheme, hindrance). The Super-GOAT is
the fusion with HLA (per-NPC affect modulates guidance cost), Crowd MCGS
(supplies the missing physical-motion layer), and Plan 350 (closes the
explicitly-deferred multi-agent warm-path Non-Goal). Verdict:
**Super-GOAT** — new capability class (coordinated physical motion at crowd
scale), force-multiplies ≥4 pillars/systems, directly observable product
selling point, zero training dependency. Open primitive in katgpt-rs/424 +
plan 440; private guide in riir-ai/318 + plan 489. GOAT gate G1–G4
(paper reproduction) + G5–G7 (fusion quality, defend-wrong PoC) must pass
before promotion.
