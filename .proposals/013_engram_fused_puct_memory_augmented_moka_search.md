# Proposal 013 — Engram-Fused PUCT: Offline-Mined Memory for the Moka Go Search

Status: **draft**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: Engram (Plan 299 / Research 278, katgpt-core) × Moka+PUCT (Plan 565, katgpt-moka-wasm) × the Issue 565 G5 corrected-forward seam
Related: [Research 278](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md), [Plan 299](../.plans/299_Engram_Hash_Addressed_Pattern_Memory.md), `.docs/03_memory/engram.md`, `.docs/06_game_arenas/go_arena.md`

## TL;DR

Fuse the shipped Engram hash-addressed pattern memory (`katgpt-core/src/engram/`, default-on at the leaf) into the Moka 9×9 Go stack (`katgpt-moka-wasm`) at the **PUCT search seam**: mine a frozen value/prior table from offline self-play (modelless statistics — no gradient), key it by multi-head hash over the position, and sigmoid-gate its read into node expansion (prior sharpening + Q initialization). This is **our transpose** of Memory-Augmented MCTS (Xiao, Mei & Müller, AAAI 2018 Outstanding Paper) onto a hash-addressed, BLAKE3-committed, O(1) substrate; the mechanism-level answer to "is it Conv?" is: **the memory primitive is conv-shaped and conv-native (it ships a literal depthwise causal conv), but the proposed search-side fusion is a memory read + sigmoid gate, not a convolution.** The conv-native variant (trunk fusion — a memory-conditioned dynamic-conv branch inside Moka's CNN) is Phase 4, deferred until the table proves it carries signal.

## The problem this solves

`PuctPlayer` (`katgpt-moka-wasm/src/puct.rs`) is a **pure** tree search:

- **No transposition handling** — nodes are keyed by parent-chain (`PuctNode { state: Board, prior, total_value, ... }`); the same position reached via two move orders is two independent subtrees, each re-searched from scratch.
- **No cross-game memory** — every `select_move` starts from a cold tree. A position the engine has already "solved" in a previous game costs the full simulation budget again.
- **The net sees only 2 plies of history** (`expand` collects the parent chain for Moka's last-2-plies feature planes) — longer temporal patterns (joseki-shaped sequences, repeated-position dynamics) are invisible.

Moka's 12-plane features already encode the board well, so the gap is not "the net is blind" — it is "the search re-pays for knowledge it already produced." An offline-mined, frozen engram table addresses exactly that: O(1) (~48 ns measured, Bench 299) recall of past search outcomes, gated so that it only speaks when it has evidence.

## The proposed design

### Primary — search-side fusion (Phases 1–3)

```text
// OFFLINE (miner example, modelless statistics collection — no backprop):
run N self-play games with the existing PuctPlayer (arena infra: go_arena.md)
for each visited position s (with ko_point + to_play — the TT-key discipline):
    h[0..K_MAX] = multi_head_hash(canonical(s))          // existing substrate, ~10 ns
    acc[h] += (final_outcome_from_perspective, 1)        // running (sum, count)
build: EngramTableBuilder → rows [v̄, n, prior-bias vec] → build_merkle_root
freeze: EngramTableId (BLAKE3) → load at arena boot via EngramHotSwap

// ONLINE (PuctPlayer::expand — the same seam as with_corrected, Issue 565 G5):
h        = multi_head_hash(canonical(position))           // O(1)
(v̄, n)   = table.lookup_into(h)                           // ~48 ns, K=16 amortized
gate     = sigmoid((n − n_min) / τ_n)                     // evidence gate; 0 when unseen
prior'   = normalize(prior · exp(gate · β · b(s)))        // memory-sharpened prior
v_init   = (1 − gate) · v_search + gate · v_mem           // Q init only; search overrides
                                                            // as visits accumulate (the
                                                            // M-MCTS combination rule)
```

House-rule compliance: **sigmoid gate, never softmax**; table is a frozen snapshot updated only by atomic `EngramHotSwap` (freeze/thaw, no in-place mutation); hot path zero-allocation (caller scratch, existing substrate contract).

Key choice — **position-keyed, not move-n-gram-keyed**: the value table must key on `(board, ko_point, to_play)` (the transposition-table key). A move-n-gram key aliases positions with different histories — the Graph-History-Interaction hazard — and Go's rules make history matter (ko/superko). The N-gram hash machinery is reused as the *hash family*, applied over a canonical position encoding.

### Deferred — trunk fusion, the "it's Conv" variant (Phase 4)

Fuse the retrieved pattern into the trunk hidden state between conv blocks via `fuse_into_hidden_state` (the designed hook), with `IDENTITY_KERNEL` zero-init semantics. Mathematically this inserts a **content-addressed dynamic depthwise-conv branch** into a static conv net (see next section). Deferred because Moka's weights are a frozen third-party checkpoint: the trained heads never saw engram residuals, so trunk fusion risks distribution shift before the table has proven it carries signal.

## Is it Conv? (the honest math)

Three different objects, three different answers:

1. **Moka's network** — yes, literally: spatial convs over the 9×9 board (stem 3×3 12→32, bottleneck blocks with 1×1/3×3 convs, 1×1 conv heads).

2. **Engram's memory op** — conv-*shaped*, not a conv. The N-gram window is a causal receptive field (kernel size N over the token axis), and per head exactly one "kernel" (the table row) is **hard-selected by the window's content** via hash. That places it in the *dynamic convolution* family (Chen et al. 2020 aggregate K kernels with softmax attention; engram hard-routes to one row per head, K_MAX=16 heads ≈ 16 parallel routed channels). It is not a fixed linear operator: content addressing + the sigmoid gate make it nonlinear and non-stationary. **And the primitive is conv-native**: `engram/conv.rs` ships the paper's §2.3 depthwise causal conv (`conv_causal_into`, kernel 4, dilation = N-gram order), the dynamic per-timestep-kernel variant (`conv_causal_dyn_into`), and a streaming temporal step conv with backward — so composing engram with convs is first-class, not a retrofit.

3. **The proposed fusion (search-side)** — no. A hash lookup + sigmoid gate on the PUCT prior is a memory read, not a convolution. The fusion becomes "a Conv" only in the Phase-4 trunk variant, where it is a memory-conditioned dynamic depthwise branch inside the CNN (spatial static conv + temporal/content dynamic conv).

## Honest caveats — READ BEFORE IMPLEMENTING  (MANDATORY)

1. **Go is (nearly) Markovian — the #1 inventor's-regret risk.** The board features already encode the state; a position-keyed engram adds *cross-game experience reuse*, not new information. At moderate budgets search quickly overwhelms a value prior (visits → search-dominant). M-MCTS's wins were in **online RL with sparse rewards**; a strong-net Go regime is not that. The arena gate decides, and a FAIL is the expected-failure mode, not a shame. Do not promote on a noisy win.

2. **Prior art is real and load-bearing.** Xiao, Mei & Müller (AAAI 2018, Outstanding Paper) already established offline-memory + online-MCTS generalization (kernel-regression value estimates blended by visit count). Our deltas are: hash-addressed O(1) hard-routed retrieval vs O(N) kernel regression; BLAKE3-committed frozen snapshot (deterministic, hot-swappable, chain-friendly); AlphaZero-style PUCT integration; and the modelless mandate. If those deltas don't matter for the consumer, this proposal is redundant — the arena G5 is also the test of the deltas.

3. **GHI/transposition correctness.** Keying on `(board, ko_point, to_play)` is the standard TT discipline but is an approximation under superko (the crate's `Board` tracks simple ko only, no positional-superko set). Two same-key positions with different full histories can have different legal-move sets. Document as approximation; assert the key includes ko + to_play; do not "fix" by widening the key without measuring the hit-rate collapse.

4. **Collision dilution + evidence gating.** Last-write-wins at build time means a collided slot carries a wrong value with full confidence. The gate must be **count-based** (`n` in the row), not just value-based — a slot seen once is a rumor; a slot seen 500× is a statistic. Multi-head (K=16) dilutes collisions but does not remove them.

5. **Table staleness is by design (frozen), and that cuts both ways.** A frozen table is modelless-legal and commitment-clean, but it is mined against a specific opponent/self-play distribution. Fine for the arena POC; a production consumer needs a re-mining cadence via `EngramHotSwap` (the substrate exists).

6. **Wasm/browser is out of scope for Phase 2.** The miner and the fused player land behind a native-gated feature (the `research` feature precedent — `with_corrected` is already PoC-only, never in the browser build). The table is ~MB-scale; shipping it to wasm is a later consumer decision.

## Fusion lineage

| Ingredient | Where | What it contributes |
|---|---|---|
| Engram substrate | `katgpt-core/src/engram/` (Plan 299, default-on at leaf) | multi-head hash, O(1) table, sigmoid fuse, BLAKE3 commitment, hot-swap, **the shipped §2.3 causal conv + dynamic/temporal variants** |
| Moka + PUCT | `katgpt-moka-wasm` (Plan 565) | the arena with real measured baselines (98% native strength; PUCT b50 = 25.8 ms int8 wasm) and the `native_puct_winrate` test infra the G5 gate reuses |
| Corrected-forward seam | `puct.rs::with_corrected` (Issue 565 G5) | the proven injection point: a state-dependent modification applied at every leaf evaluation without touching the frozen weights |
| M-MCTS | Xiao/Mei/Müller AAAI 2018 | the external mechanism this transposes (offline memory + online search, visit-count-blended) |

## GOAT gate

Feature `engram_puct` on `katgpt-moka-wasm` (opt-in, native-gated). Promotion to default-on requires ALL of:

- **G1 (correctness/identity):** with the feature on but table empty, `select_move` output is bit-identical to feature-off across the full existing PUCT test corpus (zero-init = identity, the `IDENTITY_KERNEL` discipline).
- **G2 (perf):** engram read + gate adds < 100 ns per node expansion (substrate: 48 ns/retrieval measured; the gate is a sigmoid + fma). No regression to the 25.8 ms b50 wasm number on the paths that don't use it (feature off ⇒ zero code executed).
- **G3 (no-regression):** existing moka/puct/board test counts unchanged; `--all-features` and `--no-default-features` clean.
- **G4 (alloc):** zero heap allocations on the expansion path when the table is resident (caller-owned scratch, the substrate contract).
- **G5 (quality — the real gate):** arena win-rate of fused vs unfused at EQUAL budget, ≥ 55% over ≥ 200 games (binomial CI reported), **or** equal win-rate at ≤ 50% budget (the speedup axis — the honest alternative face of a memory win). A tie or loss on both faces ⇒ keep opt-in and record the negative result; do not promote.
- **G6 (modelless):** the table is frozen statistics (counts/means), produced by a deterministic miner, committed via BLAKE3; no gradient anywhere; updates only via hot-swap.

Not UQ-bearing (no probability distribution / interval claim is made from the table; the value row is a gated prior input, not a calibrated forecast), so the conformal-naive floor rule does not bind — noted explicitly so a future reader checks before extending it into a forecasting claim.

## What ships now (katgpt-rs) vs deferred

### Ships now — search-side fusion
- `katgpt-moka-wasm`: feature `engram_puct`; `EngramPuctPlayer` (wraps `PuctPlayer`, mirrors the `with_corrected` pattern); prior-sharpen + Q-init at `expand`.
- Miner as a native example/binary: self-play → position-keyed stats → `EngramTableBuilder` → committed artifact (`EngramTableId` recorded in the bench doc for reproducibility).
- Arena gate runner + `.benchmarks/NNN_engram_puct_goat.md`.

### Deferred — trunk fusion (Phase 4)
- `fuse_into_hidden_state` inside the Moka forward (the conv-native variant). Blocked on Phase-3 evidence that the table carries signal at all, and on an honest answer to the frozen-checkpoint distribution-shift risk.

### Explicitly NOT shipped by this proposal
- Any change to `katgpt-core/src/engram/` (the substrate is sufficient as-is; a position-encoding helper lives consumer-side).
- Superko tracking in `Board` (out of scope; the approximation is documented, caveat 3).
- Wasm/browser enablement of the fused player.
- Any riir-* wiring — this is a katgpt-rs-internal arena experiment until it GOATs.

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — miner + table (no search change)
- [ ] T1.1 Position canonicalization + TT-key hash family (board + ko + to_play) consumer-side
- [ ] T1.2 Self-play miner example → stats accumulation → frozen committed table
- [ ] T1.3 Table sanity gates: hit-rate curve vs table size; count distribution; collision audit (multi-head agreement)

### Phase 2 — PUCT fusion behind the flag
- [ ] T2.1 `engram_puct` feature + `EngramPuctPlayer` (identity when table absent — G1 test)
- [ ] T2.2 Prior-sharpen + Q-init at `expand`, count-based sigmoid gate
- [ ] T2.3 G2/G4 micro-benchmarks (expansion delta, alloc counter)

### Phase 3 — arena GOAT
- [ ] T3.1 Fused-vs-unfused at equal budget, ≥ 200 games, CI (G5 face 1)
- [ ] T3.2 Budget-equivalence sweep (G5 face 2: 25%/50%/100% budget)
- [ ] T3.3 Verdict recorded either way; promotion is an owner call on a PASS

### Phase 4 — trunk fusion (conditional on Phase 3 signal)
- [ ] T4.1 `fuse_into_hidden_state` into the Moka trunk, `IDENTITY_KERNEL` zero-init
- [ ] T4.2 Distribution-shift audit vs the frozen checkpoint (the dynamic-conv branch)

## Risks

1. **Quality risk (primary):** Markovian game + strong net ⇒ memory adds ~nothing (caveat 1). Mitigation: G5 is falsifiable and cheap (the arena exists); negative result recorded, feature stays opt-in.
2. **Correctness risk:** GHI approximation under superko aliasing (caveat 3). Mitigation: TT-key discipline + documented approximation; never widen the key silently.
3. **Perf risk:** table residency (MB-scale) + cold-cache misses on wasm-class hardware. Mitigation: native-gated first; the `ZipfianCacheHierarchy` substrate exists if tiering is ever needed.
4. **Architectural risk:** none cross-repo (katgpt-rs internal; consumes an already-default-on leaf feature; no new deps — blake3/papaya already in katgpt-core).

## Out of scope  (RECOMMENDED)

- Any KatGo/other-game generalization before the Go arena verdict.
- Learned table compression (riir-train territory; forbidden here by the modelless mandate).
- Real-time table updates during play (that's online learning — a different proposal with a different mandate).

## References

1. [arXiv:2601.07372 — Engram: Conditional Memory (Cheng et al. 2026)](https://arxiv.org/abs/2601.07372) — **distilled in-house** as [Research 278](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md); the substrate source.
2. [Xiao, Mei & Müller — Memory-Augmented Monte Carlo Tree Search, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11531) (Outstanding Paper Award) — cited-only; the offline-memory + online-search prior art this proposal transposes (see caveat 2).
3. [arXiv:1912.03458 — Dynamic Convolution: Attention over Convolution Kernels (Chen et al. 2020)](https://arxiv.org/abs/1912.03458) — cited-only; the framing for "engram's lookup is a hard-routed dynamic conv" in §Is it Conv.
4. Childs, Brodeur & Kocsis — Transpositions and Move Groups in Monte Carlo Tree Search (CIG 2008) — cited-only; the TT/GHI discipline for caveat 3.
5. Silver et al. — A general reinforcement learning algorithm that masters chess, shogi and Go through self-play (AlphaZero, 2018) — cited-only; the PUCT selection formula the fused prior feeds.

## TL;DR

**Draft, file it and gate it:** fuse Engram into PUCT at the search seam (offline-mined frozen value/prior table, sigmoid evidence gate, O(1) reads) behind opt-in `engram_puct`; the honest answer to "it's Conv?" is *the primitive is conv-shaped and ships a real causal conv, but this fusion is a gated memory read — the conv variant is Phase 4*. Next action: open a plan for Phases 1–3 on the existing arena infra; the G5 win-rate/budget gate decides promotion.
