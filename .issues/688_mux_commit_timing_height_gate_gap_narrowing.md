# Issue 688: MuxBfs commit-timing — height-gated commit + inter-step gap-trend narrowing

**Filed:** 2026-08-25, from the Coconut (arXiv:2412.06769v4) adversarial research panel.
**Repo:** katgpt-rs
**Provenance:** Research skill §1.55 actionable output. **NOT a novelty claim** — both heuristics have decades of published prior art (below). Filed because two documented gaps exist in `mux/` that exactly consume these signals, and Coconut §4.4/Fig. 6 supply the latent-search measurement that motivates wiring them.

## The two documented gaps

1. `crates/katgpt-core/src/mux/bfs.rs:104` — `let _ = depth; // preserved for API compat; pruner ignores depth`. Depth is plumbed into `MuxBfs::step()` and explicitly discarded. No shipped mux consumer reads distance-to-terminal.
2. `.benchmarks/017_simpletes_scaling_crossstrategy.md` T9 — "Wide > Balanced > Deep >> Narrow" at fixed budget (0.9988 vs 0.8266) with **no data-dependent criterion for when narrowing wins**; wide ships as a static config choice.

## The paper's measurements (the motivating evidence)

- **Coconut §4.4 (height analysis):** probed value-estimate confidence is monotone in node HEIGHT (shortest distance to leaf) — low-height (near-terminal) nodes get definitive evaluations, high-height (far) nodes get ambiguous ones. Derived strategy: defer deterministic commitment until near terminal states; early exploration is cheap, early commitment is wrong.
- **Coconut Fig. 6 (parallelism narrowing):** top-1/2/3 cumulative-value gaps SHRINK from the first to the second continuous thought — progressive explore→focus narrowing driven by the gap trend itself, not by a fixed schedule.

## Prior art (explicit — this is a classic-heuristic port, not an invention)

- Height/defer-commitment: quiescence search (Beal 1980), conspiracy-number search (McAllester 1988), LRTA*/real-time heuristic search (Korf 1990), compounding bootstrapping bias (Pohlen et al., arXiv:1806.10201).
- Progressive narrowing: successive halving (arXiv:1502.07943), Hyperband (arXiv:1603.06560), ASHA (arXiv:1810.05934), Hoeffding races (Maron & Moore 1994), MCTS progressive widening (Coulom 2006); MUX (arXiv:2607.18264) operationalizes variable-width multiplexed hypotheses.

## Signal-diff vs closest shipped cousins (verified by code read, 2026-08-25)

- `mux/bandit_width.rs` `MuxBanditWidth` consumes **historical UCB1 reward per width arm** (retrospective, cross-episode).
- `mux/bfs.rs` `detect_width_with_peaks` consumes **instantaneous top-k shape only** (memoryless per leaf).
- riir-ai `latent_functor/` `ActionVerifyGate` consumes **entropy + depth-from-ROOT** — the *opposite* depth axis (progress made, not progress remaining).
- None consumes distance-to-TERMINAL or an inter-step gap derivative.

## Tasks

- [ ] `height_gate.rs` (or extend `bfs.rs`): commit-timing gate consuming distance-to-TERMINAL (subtree max remaining depth / known solution depth). Among value-ties, min-height candidates commit first; high-height candidates hold breadth.
- [ ] Gap-trend narrowing: extend `detect_width_with_peaks` with the inter-step derivative of cumulative top-k mass gaps — narrow while gaps grow (focus), hold wide while flat (explore).
- [ ] GOAT G1: on DD-tree fixtures with planted values, height-gated commit ordering ≡ oracle ordering vs ungated baseline. **Mandatory negative control:** must NOT narrow on SimpleTES-style flat-gap fixtures — Bench 017 T9's wide-dominance is the control case (narrowing there would be a regression).
- [ ] GOAT G2: ns-scale latency; G4 zero-alloc (`_into` variants).
- [ ] Feature-gated (`mux_height_gate` or folded into an existing mux feature); opt-in until GOAT passes.

## Non-goals

- No continuous-thought feedback loop. Coconut's own ablation (w/o curriculum = 14.4% ≈ no-CoT 16.5% on GSM8k) proves the untrained latent loop is inert; Plan 276 Family A/B null results corroborate. The representation stays GD-dependent — training-side rows live in riir-train Plan 352.
- No novelty claim — classic search heuristics applied to an existing shipped substrate, motivated by a latent-search measurement.

## References

- Source paper: Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (Coconut), arXiv:2412.06769v4 (v4 is cosmetic over v3 — no mechanism delta).
- Corpus: Research 325 §7.1 (routes Coconut → 192/NextLat; "do not re-distill"), Research 158 (MUX), Research 411 (formal TC^k/FPRAS comparison), Bench 017 T9.
