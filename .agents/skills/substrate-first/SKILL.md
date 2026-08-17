---
name: substrate-first
description: Pre-implementation DRY gate + existing-code drift audit for the 7-repo workspace (katgpt-rs, riir-ai, riir-chain, riir-neuron-db, riir-train, riir-game-sdk, riir-mmorpg-examples). Use BEFORE writing any new System impl, trait, perception/cognition/emotion pipeline, state management, spatial query, or vocabulary type — to verify you're consuming existing substrate, not duplicating it. Also use to AUDIT existing code for parallel-system DRY violations (code that re-implements substrate under different names). The canonical defense is vocabulary translation; concepts ship under operator names (`GenericSpatialBelief`, `decay_confidence`), not English names ("threat field", "spatial hash") — a single-vocabulary grep returns ZERO hits even when the substrate fully exists. Sibling to boundary-guard + goat-audit + feature-gate-audit + doc-sync.
---

# Substrate-First — DRY gate + drift audit

The 7-repo workspace has a recurring failure mode: an agent receives a task
("add threat perception"), jumps to implementation without checking existing
substrate, and builds a **parallel system** that duplicates functionality
already shipped under a different name. The user then has to catch it manually.

This skill prevents that. It runs in two modes:

1. **Pre-implementation gate** — run BEFORE writing code
2. **Existing-code audit** — scan for already-shipped DRY violations

## Canonical failures (the pattern this skill prevents)

### Failure 1 — ThreatField (Issue 047, riir-mmorpg-examples, 2026-08-01)

**Built:** `ThreatField` — a spatial hash grid (`HashMap<(i32,i32), u32>`)
for threat perception. Deposited monster positions into cells; NPCs sampled
the 3×3 neighborhood.

**Already existed:** `GenericSpatialBelief<T>` + `target_within_visible_radius()`
+ `decay_confidence()` — the full fog-of-war → belief → decay pipeline in
`riir-games-shared/src/game_traits/spatial.rs`.

**Why the grep missed it:** The agent searched for "threat field" /
"spatial hash". The substrate ships as `GenericSpatialBelief` /
`SpatialBelief` / `confidence_decay`. **A single-vocabulary grep returns
ZERO hits even when the substrate fully exists.**

**Resolution:** Reverted. The plain scan (`tick_swarm_emotions`) is a simpler
POC-scale simplification. The belief-based system is deferred until fog-of-war
becomes a gameplay feature.

### Failure 2 — Orchard + Motivation in SDK src/ (Issue 490 + Issue 493)

**Built:** `NpcReasonSystem`, `AppleGrowSystem`, `OrchardGoal`, `EmotionField`,
`EmotionAxis`, `tick_feeling_brain` directly in the SDK facade's `src/`.

**Already existed:** The boundary rule (riir-game-sdk/AGENTS.md) says "no game
logic in `src/`" + Proposal 019 excludes emotion from the SDK. The substrate
belongs in `riir-games` (in riir-ai).

**Why the grep missed it:** The agent didn't check whether the types violated
the domain classification rule (latent semantic emotion ≠ raw physical
vocabulary).

**Resolution:** Extracted to `riir-games` (orchard) and `riir-games::motivation`
(emotion). The SDK re-exports them.

---

## Mode 1: Pre-implementation gate (BEFORE writing code)

Run this checklist before implementing ANY of these:

- New `impl System` or tick function
- New trait + impl (perception, cognition, emotion, state management)
- New spatial query / index / hash / grid
- New vocabulary type / DTO / config struct
- New "helper" function that does math (distance, sigmoid, projection)
- New pipeline (perception → emotion → behavior, freeze → sync → thaw)

### Step 1 — Vocabulary-translate your search

The concept you're building probably already exists under a **different name**.
Before grepping, write down 3+ name variants for the concept:

| You're building... | Also search for... | Likely substrate names |
|---|---|---|
| "threat field" / "spatial hash" | belief, perception, spatial cognition, fog-of-war, visibility | `GenericSpatialBelief`, `SpatialBelief`, `confidence_decay`, `target_within_visible_radius` |
| "emotion" / "feeling" / "mood" | affect, drive, motivation, fear, desire | `EmotionField`, `EmotionAxis`, `AffectField`, `DriveSpecSet`, `tick_feeling_brain` |
| "state sync" / "delta" / "snapshot" | replication, gossip, cache, commitment | `SyncBlock`, `ZoneDelta`, `PlayerStateCache`, `GossipDelta`, `SyncRegistry` |
| "position" / "movement" / "physics" | spatial, coordinate, force, velocity | `MapPos3D`, `ForceVector`, `SpatialIndex`, `GridSpatialIndex` |
| "save" / "persist" / "freeze" | thaw, snapshot, serialize, store | `freeze_avatar_delta`, `LocalKvStore`, `ShardIndex`, `NeuronShard` |
| "validate" / "anti-cheat" / "check" | verify, guard, proof | `AvatarAntiCheatValidator`, `AdaptiveModConfig`, `make_validator_predicate` |
| "tick" / "update" / "loop" | system, schedule, game core | `System`, `TickCtx`, `World`, `GameCore`, `FrameSnapshot` |
| "knowledge" / "relationship" / "graph" | triple, semantic, KG | `KgTriple`, `KgTripleTemplate`, `DualSignalEvidence` |
| "attack" / "damage" / "combat" | hp, health, dex, fight | `Hp`, `Dex`, `CombatConfig`, `combat_tick`, `attack_interval_ticks` |
| "npc" / "swarm" / "crowd" | agent, bot, forager | `SwarmState`, `ForagerSwarmSystem`, `ForagerAi`, `BotThought` |
| "decay" / "fade" / "forget" | sigmoid, heal, baseline, confidence | `decay_confidence`, `tick_feeling_brain`, `sigmoid`, `EmotionBaseline` |
| "embedding" / "vector" / "latent" | direction, projection, HLA, shard | `NeuronShard`, `HlaCacheProxy`, `compute_animal_emotions` |

**This is the same technique as the paper→code vocabulary translation in
AGENTS.md §"Manifold Geometry".** The R296 canonical failure applies internally
too: a concept-name grep returns zero hits because the math ships under operator
names.

### Step 2 — Grep the CODEBASE (not just docs)

```bash
# Grep ALL repos for the substrate names from Step 1.
# Use multiple variants — the concept may exist under any of them.
grep -rn 'GenericSpatialBelief\|SpatialBelief\|confidence_decay' \
    --include='*.rs' \
    /Users/katopz/git/{katgpt-rs,riir-ai,riir-chain,riir-neuron-db,riir-train,riir-game-sdk,riir-mmorpg-examples}/

# Also grep .research/ and .proposals/ for design rules that apply:
grep -rn 'two-brain\|fog-of-war\|domain classification\|sync boundary' \
    /Users/katopz/git/*/.{research,proposals,docs}/ 2>/dev/null
```

If you find existing substrate → **STOP**. Consume it. Do not build a parallel
system. Document why you're consuming it (in the plan/issue).

### Step 3 — Check AGENTS.md architectural rules

Before building, verify your design doesn't violate these:

| Rule | Source | What it means |
|---|---|---|
| **Domain classification** | AGENTS.md §"Latent vs Raw Space Rules" | Physical = raw exact; Semantic = latent dot-product + sigmoid; Social = KG triples |
| **Two-brain model** | AGENTS.md §"Spatial Cognition" | Info brain (synced ground truth) ≠ think brain (per-NPC beliefs, fog-of-war gated) |
| **Sync boundary** | AGENTS.md §"Sync Boundary Rule" | Through `SyncBlock` → quorum → Cold = raw + deterministic; Local = latent |
| **Bridge pattern** | AGENTS.md §"Bridge Pattern" | raw → latent = dot+sigmoid; latent → raw = clamp; zero-alloc, gateable |
| **KG triple emission** | AGENTS.md §"KG Triple Emission" | Semantic encounters → KG triple; Physical events → TxDelta with raw values |
| **Facade constraint** | riir-game-sdk/AGENTS.md | SDK = re-export facade, no engine deps; vocabulary in riir-games-shared |
| **Boundary rule** | riir-game-sdk/AGENTS.md | No game logic in consumer `src/`; game systems in `riir-games` |

If your design violates any of these → **STOP**. File an issue. Rethink.

### Step 4 — Decide: consume vs. build

| Situation | Action |
|---|---|
| Substrate EXISTS and fits | Consume it. Wire via trait/config. Zero new substrate code. |
| Substrate EXISTS but wrong shape | Extend the substrate (in the right repo). File a plan. |
| Substrate DOESN'T exist | File an issue in the right repo FIRST. Then build. |
| You're not sure | **STOP and file an issue.** Don't guess. |

**Never build new substrate inside a consumer.** The consumer provides data +
wiring only. If you're writing loops/math/constants in a consumer's `src/` →
you're building substrate in the wrong place.

### Step 5 — Record the decision

In the plan/issue, document:

```
## Substrate check (substrate-first skill)
- Searched for: [concept names + variants]
- Found: [existing substrate or "none"]
- Decision: [consume / extend / build new]
- Architectural rules checked: [list which rules apply + verdict]
```

---

## Mode 2: Existing-code audit (scan for drift)

Run this when reviewing code, when you suspect a parallel system, or quarterly
as a DRY-hygiene gate (alongside boundary-guard).

### Audit Step 1 — Inventory substrate primitives

For each domain, identify what substrate exists:

```bash
# Perception / spatial cognition
grep -rn 'GenericSpatialBelief\|SpatialBelief\|target_within_visible_radius' \
    --include='*.rs' /Users/katopz/git/*/  | grep -v '/tests/' | grep -v '/target/'

# Emotion / affect
grep -rn 'EmotionField\|EmotionAxis\|tick_feeling_brain\|AffectField\|DriveSpecSet' \
    --include='*.rs' /Users/katopz/git/*/

# State sync
grep -rn 'SyncBlock\|ZoneDelta\|PlayerStateCache\|GossipDelta\|SyncRegistry' \
    --include='*.rs' /Users/katopz/git/*/

# Spatial
grep -rn 'SpatialIndex\|GridSpatialIndex\|OctreeSpatialIndex\|MapPos3D' \
    --include='*.rs' /Users/katopz/git/*/
```

### Audit Step 2 — Grep for parallel systems

For each substrate primitive found in Step 1, grep consumer code for
reimplemented versions:

```bash
# Look for inline distance math (should use MapPos3D methods):
grep -rn '(dx.*dx.*dy.*dy).*sqrt\|distance_2d.*fn\|fn.*distance' \
    --include='*.rs' /Users/katopz/git/{riir-mmorpg-examples,riir-game-sdk}/src/

# Look for inline sigmoid/exp (should use substrate sigmoid or tick_feeling_brain):
grep -rn '1\.0\s*/\s*(1\.0\s*\+\|sigmoid\|exp(' \
    --include='*.rs' /Users/katopz/git/{riir-mmorpg-examples,riir-game-sdk}/src/

# Look for HashMap-based spatial structures (should use SpatialIndex substrate):
grep -rn 'HashMap.*i32.*i32\|spatial.*hash\|cell.*grid' \
    --include='*.rs' /Users/katopz/git/{riir-mmorpg-examples,riir-game-sdk}/src/

# Look for parallel belief/perception types (should use GenericSpatialBelief):
grep -rn 'struct.*Belief\|struct.*Perception\|struct.*Visibility\|last_known' \
    --include='*.rs' /Users/katopz/git/{riir-mmorpg-examples,riir-game-sdk}/src/

# Look for parallel emotion types (should use EmotionField/AffectField):
grep -rn 'struct.*Fear\|struct.*Mood\|struct.*Emotion\|fear.*f32' \
    --include='*.rs' /Users/katopz/git/{riir-mmorpg-examples,riir-game-sdk}/src/
```

### Audit Step 3 — Classify findings

For each hit, classify:

| Classification | Meaning | Action |
|---|---|---|
| **False positive** | The code is legitimately consumer-specific (e.g., `MonsterThreatSource` impl) | No action — document why |
| **POC simplification** | Duplicates substrate but produces identical behavior at POC scale | Document as known debt; fix when scale changes |
| **DRY violation** | Re-implements substrate under a different name | File issue; extract to substrate |
| **Architectural violation** | Violates two-brain model / sync boundary / domain classification | File issue; redesign |

### Audit Step 4 — Report

Summarize findings:

```
## Substrate-first audit (date)
### Substrate inventory
- [domain]: [primitive] at [location]
### Findings
- [file:line] — [classification] — [description]
### Clean
- [domain] — no violations found
```

---

## The vocabulary-translation defense (why this skill exists)

The hardest failures to catch are the ones where the substrate **exists** but
ships under a name that doesn't match the concept you're searching for. This is
the R296 canonical failure (documented in AGENTS.md §"Manifold Geometry"),
applied internally:

```
You think: "I need a threat field"
You grep:   "threat field" / "spatial hash"  → 0 hits
Substrate:  GenericSpatialBelief + decay_confidence  → exists, fully functional

You think: "I need emotion decay"
You grep:   "emotion decay" / "fear fade"  → 0 hits
Substrate:  tick_feeling_brain + DecayRates + EmotionBaseline  → exists

You think: "I need state persistence"
You grep:   "save state" / "persist"  → 0 hits
Substrate:  LocalKvStore + freeze_avatar_delta + ShardIndex  → exists
```

**The defense:** always search 3+ vocabulary variants. The translation table
in Mode 1 Step 1 is the canonical reference. Extend it when you discover new
mismatches.

---

## When NOT to use this skill

- Pure refactoring that doesn't add new concepts (renaming, reorganizing)
- Bug fixes in existing code (the substrate is already consumed or not)
- Test-only code (tests can define inline helpers)
- Build/config changes (Cargo.toml, scripts)

---

## Relationship to sibling skills

| Skill | What it checks | When |
|---|---|---|
| **substrate-first** (this) | "Does the substrate already exist? Are you duplicating it?" | Before writing code + audit |
| **boundary-guard** | "Is this code in the right repo? Is the consumer too fat?" | After writing code + audit |
| **feature-gate-audit** | "Do feature-gate claims match source wiring?" | Before promoting/demoting flags |
| **goat-audit** | "Has the katgpt-rs primitive been cherry-picked to riir-*?" | Cross-repo cherry-pick tracking |
| **doc-sync** | "Do docs match git history?" | After landing plans/issues |

`substrate-first` is **upstream** of `boundary-guard`: if substrate-first
catches the drift before it ships, boundary-guard has nothing to find. They're
complementary — substrate-first is the prevention, boundary-guard is the cure.

---

## Filing violations

When the audit finds a DRY violation or parallel system:

1. **File an issue** in the repo where the violation lives
2. **Reference this skill** + the substrate it duplicates
3. **Include the vocabulary translation** (what you searched for vs. what the
   substrate is actually called)
4. **Propose the fix** (consume substrate / extract to substrate / revert)
5. **Classify** (POC simplification vs. DRY violation vs. architectural violation)

Do NOT fix in the same commit as detection — separate detection from fix so
other agents can review the violation independently.

---

## Run log

| Date | Scope | Verdict | Record |
|---|---|---|---|
| 2026-08-17 | Mode 2 drift audit — riir-mmorpg-examples (the Plan-022/539-era fresh code: party/social_signals/monster_predation/pet_pvp/pvp_karma/shared_quest_monsters/scenario_runner) + riir-game-sdk `src/` | **CLEAN overall** — 1 new minor D1 site; 2 model consumers found | The 5 audit grep families run over `src/`+`crates/` (both consumers). Findings: (1) **pet_teaching.rs:678** — `(dx*dx+dy*dy).sqrt()` where `Position.0` IS substrate `MapPos3D` → delegatable to `distance_2d_to`; appended to mmorpg Issue 069's D1 ledger (`61899d3`, detection-only). (2) **Model consumers** (the discipline is HOLDING in fresh code): `predation.rs` wraps substrate `tick_swarm_emotions_fov` (emotion.rs:353) with scratch wiring; `pet_alarm.rs` consumes `GenericSpatialBelief<PlayerTarget>` + fog-of-war gate + `distance_2d_to`. (3) False positives documented: ~12 squared-distance comparisons (`dx*dx+dy*dy <= r*r`) are the CORRECT no-sqrt idiom (the Batch 53 `distance-sq-no-sqrt` rule) — a naive D1 "fix" there would be a pessimization; `constants.rs:796` grid-space heightfield math; `authority/karma.rs:656` wire-DTO `[f32;N]` arrays (not MapPos3D); `EntityVisibility` (render-kind mask); `QuestRestockBelief` (temporal EMA, two-brain-compliant docs); game-sdk `ai/cognition.rs:42` (doc-comment example). Clean: inline sigmoid (0 hits), HashMap spatial grids (0 hits). NOTE: audit ran against origin/develop `339f4b7` (the sibling's local develop is diverged on a line without a48af77; findings greps used the local tree — issue-filed via the temp-worktree pattern). |

## References

- AGENTS.md §"Spatial Cognition (Two-Brain Model)" — the canonical perception rules
- AGENTS.md §"Manifold Geometry (Stokes Calculus)" — the R296 vocabulary-translation
  failure pattern (this skill extends it from paper→code to codebase→codebase)
- AGENTS.md §"Latent vs Raw Space Rules" — domain classification
- `riir-games-shared/src/game_traits/spatial.rs` — `GenericSpatialBelief<T>` substrate
- `riir-games-shared/src/game_traits/` — vocabulary translation table source
- Issue 047 (riir-mmorpg-examples) — the ThreatField canonical failure
- Issue 490 + Issue 493 (riir-game-sdk) — the orchard/motivation canonical failure
- `boundary-guard` skill — sibling (boundary enforcement, post-hoc)
