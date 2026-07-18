//! **Causal-ID — Algorithmic Syntactic Causal Identification**
//! (Plan 457, Research 450, arXiv:2403.09580 Cakiqi & Little 2024).
//!
//! A modelless primitive for identifying interventional signatures on
//! Acyclic Directed Mixed Graphs (ADMGs) with unobserved confounders.
//! Distilled from the Cakiqi-Little Theorem 1 syntactic identification
//! algorithm — pure graph rewriting, no gradient descent.
//!
//! ## What this ships
//!
//! - [`Admg`] / [`NodeId`] / [`AdmgSignature`] — the substrate types
//!   ([`types`] module).
//! - [`identify`] — the top-level driver implementing the recursive
//!   Shpitser-Pearl ID algorithm (six steps: marginalise, restrict,
//!   drop, district, fix, recurse).
//! - [`fixing`] — graph-rewriting primitives ([`Admg::districts`],
//!   [`Admg::ancestors`], [`Admg::fix_node`], [`fixing::try_fixseq`]).
//! - [`extract_relevant_subgraph`] — bounded-BFS subgraph extractor
//!   (caveat #2 mitigation: keep the `O(k²)`–`O(k³)` algorithm on a
//!   ≤32-node subgraph).
//!
//! ## Why this is in katgpt-core
//!
//! The identification algorithm is pure graph rewriting on structural
//! types — no game-specific vocabulary, no game-state dependency. It is
//! the modelless half of the counterfactual reasoning story; the
//! ADMG-from-KgTriple construction + GM tool wiring live in
//! `riir-ai/crates/riir-engine/src/causal_id/` (Plan 457 Phase 3+).
//!
//! ## Provenance (the soundness correction)
//!
//! The Issue 545 PoC caught a soundness bug in the original one-pass
//! formulation: it computed districts of `G[Y⋆]` instead of `G[V]`,
//! returning `NotIdentifiable` for the classic front-door case. The
//! corrected recursive formulation (districts of `G[V]`, hedge FAIL
//! condition) is what ships here. See [`identify`] module docs for the
//! six-step algorithm.
//!
//! ## Why modelless
//!
//! Every primitive here is pure graph rewriting. No backprop, no learned
//! parameters, no gradient descent. The ADMG is constructed downstream
//! (Plan 457 Phase 3, riir-ai) from a `KgTriple` corpus + a confounder
//! injection layer; this crate only owns the identification math.
//!
//! ## Status
//!
//! Opt-in via `causal_identification`. Plan 457 Phase 1 — implementation
//! complete, GOAT gate (Phase 2) pending. The Issue 545 PoC proved the
//! algorithm strictly dominates Canvas FlowGraph reachability on a
//! 13-node game KG with a `NPC1 ↔ NPC2` confounder (S2 produces a
//! 5-node interventional signature that correctly excludes NPC1; S1
//! yields only a boolean `reaches=true` and cannot see the confounder).

pub mod fixing;
pub mod identify;
pub mod subgraph;
pub mod types;

pub use fixing::try_fixseq;
pub use identify::identify;
pub use subgraph::extract_relevant_subgraph;
pub use types::{Admg, AdmgSignature, IdentificationError, NodeId, INLINE_SIGNATURE_CAP};
