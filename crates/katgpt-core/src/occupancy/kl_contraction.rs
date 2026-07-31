//! Lemma 3.1 — KL contraction of the adjoint Bellman operator.
//!
//! **Theorem** (van der Laan & Kallus 2026, Lemma 3.1). For any Markov kernel
//! `P_π`, discount `γ ∈ [0, 1)`, and any two probability densities
//! `ω, ω̃ ∈ Δ_ν` (the `ν`-absolutely-continuous probability simplex):
//!
//! ```text
//! D_ν(B^γ_π ω ∥ B^γ_π ω̃)  ≤  γ · D_ν(ω ∥ ω̃)
//! ```
//!
//! where the adjoint Bellman operator is
//!
//! ```text
//! B^γ_π ω = (1 − γ) ω_0 + γ · d((ων) P_π) / dν
//! ```
//!
//! and `D_ν` is the `ν`-weighted KL divergence.
//!
//! # Why this matters
//!
//! This is a pure information-theoretic fact (joint convexity of KL + the
//! Markov-kernel data-processing inequality). It is the substrate-independent
//! reason FORE converges under **realizability alone** — no Bellman
//! completeness of a value/critic class is required. Each fitted KL projection
//! contracts the KL gap to the target ratio by factor `γ`.
//!
//! # Candidate for Lean 4 formalization
//!
//! This theorem is a candidate for the next Lean 4 proof in the cross-repo FV
//! rollout (deferred per Research 423 §5 caveat #4). The `RiirAiProof/Runtime/`
//! directory is the likely home once the runtime-wiring PoC confirms the
//! contraction holds empirically under float precision. The DEC-codifferential
//! isomorphism (Research 423 §2.2) is a related but distinct hypothesis — not
//! assumed here.

// Doc-only module — no items. The theorem is a mathematical fact, not a
// runtime value. The FORE algorithm in the parent module is the computational
// shadow of this contraction guarantee.
