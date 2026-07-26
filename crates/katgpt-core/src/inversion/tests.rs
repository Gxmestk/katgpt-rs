//! G1 correctness tests for SipIt inversion on a toy 2-layer decoder-only transformer.
//!
//! Per Plan 561 §"Phase 1 — Skeleton + Random Policy + G1 (essential)":
//! a real (randomly-initialized) toy transformer with GELU activation,
//! `d = 16`, `|V| = 32`, `T = 8`. The three sub-tests directly exercise
//! the paper's injectivity theorem (Theorem 2.2) and its causality lemma
//! (Lemma D.2):
//!
//! 1. `g1_exact_recovery_random_init` — generate random prompts, run the
//!    transformer forward to get `H̆^(ℓ)`, run `invert_sequence`, assert
//!    exact recovery. This is the headline correctness property.
//! 2. `g1_recovers_when_two_prompts_differ_only_at_position_t` — direct
//!    test of Lemma D.2's causality argument: the prefix-conditioned
//!    verifier correctly distinguishes tokens that differ at one position.
//! 3. `g1_no_false_positive_on_mismatched_observed` — wrong `H̆` does not
//!    produce the original prompt (negative control).

use crate::inversion::{
    InversionConfig, InversionForward, InversionResult, ObservedStates, invert_sequence,
};

// ─────────────────────────────────────────────────────────────────────────
// Toy transformer: embedding lookup → 2 × (Linear d→4d, GELU, Linear 4d→d) →
// returns the per-position hidden state at the LAST layer (layer-ℓ = layer 2).
//
// This is the *minimum* that satisfies Theorem 2.2's hypotheses:
//   - Real-analytic activation (GELU).
//   - Causal (position-t output depends only on prefix π and token v at t).
//   - Vocabulary-indexed embedding lookup.
//
// We deliberately keep the architecture tiny so the G1 tests run in
// microseconds, not seconds. The G2 latency benchmark (Phase 3) will
// scale this up.
// ─────────────────────────────────────────────────────────────────────────

const D: usize = 16;
const V: u32 = 32;
const T: usize = 8;

/// Toy 2-layer GELU transformer.
struct ToyTransformer {
    /// Embedding matrix: `V × D`, row `v` is the embedding of token `v`.
    /// Plain flat row-major.
    embedding: Vec<f32>,
    /// Layer-1 + Layer-2 weights. Each layer = (W_up D×4D, W_down 4D×D).
    /// Plain flat row-major.
    w1_up: Vec<f32>,
    w1_down: Vec<f32>,
    w2_up: Vec<f32>,
    w2_down: Vec<f32>,
}

impl ToyTransformer {
    /// Random init with a deterministic seed.
    fn new(rng: &mut fastrand::Rng) -> Self {
        let mut embedding = vec![0.0_f32; (V as usize) * D];
        let mut w1_up = vec![0.0_f32; D * 4 * D];
        let mut w1_down = vec![0.0_f32; 4 * D * D];
        let mut w2_up = vec![0.0_f32; D * 4 * D];
        let mut w2_down = vec![0.0_f32; 4 * D * D];
        // Small Gaussian-ish init via uniform [-1/sqrt(D), 1/sqrt(D)].
        let scale = 1.0 / (D as f32).sqrt();
        for x in embedding.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w1_up.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w1_down.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w2_up.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w2_down.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        Self {
            embedding,
            w1_up,
            w1_down,
            w2_up,
            w2_down,
        }
    }

    /// Forward pass returning the full `T × D` layer-2 hidden state matrix
    /// for `prompt`. Allocates a fresh `Vec<f32>` per call — this is the
    /// test helper, not the hot path.
    fn forward_full(&self, prompt: &[u32]) -> Vec<f32> {
        debug_assert!(prompt.len() <= T);
        let t = prompt.len();
        let mut out = vec![0.0_f32; t * D];

        // Per-position: embed, then apply 2 layers, writing into the row.
        // We compute the full row at each position; the causal constraint
        // is satisfied trivially because each row depends only on its own
        // token (no attention between positions in this toy — we keep
        // the substrate minimal to make the injectivity theorem
        // straightforward to verify empirically).
        for (position, &v) in prompt.iter().enumerate() {
            let row_offset = position * D;
            self.embed_into(v, &mut out[row_offset..row_offset + D]);
            self.apply_layer_into(
                &self.w1_up,
                &self.w1_down,
                &mut out[row_offset..row_offset + D],
            );
            self.apply_layer_into(
                &self.w2_up,
                &self.w2_down,
                &mut out[row_offset..row_offset + D],
            );
        }
        out
    }

    #[inline]
    fn embed_into(&self, token: u32, out: &mut [f32]) {
        let base = (token as usize) * D;
        out.copy_from_slice(&self.embedding[base..base + D]);
    }

    /// Apply one transformer layer in-place: `x ← W_down · gelu(W_up · x)`.
    /// Uses a 4D intermediate buffer on the stack.
    #[inline]
    fn apply_layer_into(&self, w_up: &[f32], w_down: &[f32], x: &mut [f32]) {
        debug_assert_eq!(x.len(), D);
        let mut hidden = [0.0_f32; 4 * D];
        // hidden = W_up · x   (W_up is 4D × D row-major)
        for i in 0..4 * D {
            let row = &w_up[i * D..(i + 1) * D];
            let mut acc = 0.0_f32;
            for (j, xi) in x.iter().enumerate() {
                acc += row[j] * xi;
            }
            hidden[i] = gelu(acc);
        }
        // x ← W_down · hidden   (W_down is D × 4D row-major)
        for i in 0..D {
            let row = &w_down[i * 4 * D..(i + 1) * 4 * D];
            let mut acc = 0.0_f32;
            for (j, hi) in hidden.iter().enumerate() {
                acc += row[j] * hi;
            }
            x[i] = acc;
        }
    }
}

impl InversionForward for ToyTransformer {
    fn hidden_at_into(
        &self,
        prefix: &[u32],
        candidate: u32,
        position: usize,
        out: &mut [f32],
    ) -> Result<(), crate::inversion::InversionError> {
        debug_assert_eq!(position, prefix.len());
        debug_assert_eq!(out.len(), D);

        // Build the full prompt = prefix ⊕ [candidate] and run forward.
        // For Phase 1 we re-run the full forward per trial; Phase 2 will
        // cache the prefix's intermediate state. This is O(T·|V|·D) per
        // position, which is fine for G1 on a toy; G2 (Phase 3) measures
        // latency on the toy, G4 (Phase 3) checks alloc-free of the
        // *driver itself* (not the test forward).
        let mut full: Vec<u32> = Vec::with_capacity(prefix.len() + 1);
        full.extend_from_slice(prefix);
        full.push(candidate);
        let states = self.forward_full(&full);
        let row_offset = position * D;
        out.copy_from_slice(&states[row_offset..row_offset + D]);
        Ok(())
    }
}

/// GELU approximation (tanh-form). Real-analytic, satisfies Theorem 2.2.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x)).tanh())
}

// ─── G1 tests ─────────────────────────────────────────────────────────────

fn random_prompt(rng: &mut fastrand::Rng) -> Vec<u32> {
    (0..T).map(|_| rng.u32(0..V)).collect()
}

#[test]
fn g1_exact_recovery_random_init() {
    // Headline G1: random init → random prompts → exact recovery.
    let mut rng = fastrand::Rng::with_seed(0xC0DE);
    let transformer = ToyTransformer::new(&mut rng);

    let mut success_count = 0;
    const N_PROMPTS: usize = 8;
    for prompt_seed in 0..N_PROMPTS {
        let mut prompt_rng = fastrand::Rng::with_seed(0xA5A5 + prompt_seed as u64);
        let prompt = random_prompt(&mut prompt_rng);

        let buf = transformer.forward_full(&prompt);
        let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, V, &transformer, &cfg, prompt_seed as u64).unwrap();
        match result {
            InversionResult::Recovered(recovered) => {
                assert_eq!(recovered, prompt, "prompt {prompt:?} not recovered");
                success_count += 1;
            }
            InversionResult::Failed {
                failed_position,
                candidates_tried,
            } => {
                // Should not happen on a toy with T·|V| budget — but if it
                // does, surface the position so the failure is debuggable.
                panic!(
                    "failed at position {failed_position} after {candidates_tried} candidates \
                     (prompt {prompt:?})"
                );
            }
        }
    }
    // All 8 must succeed; if even one fails, the theorem isn't holding on
    // this toy and the primitive has a bug.
    assert_eq!(
        success_count, N_PROMPTS,
        "expected all {N_PROMPTS} random prompts to recover exactly"
    );
}

#[test]
fn g1_recovers_when_two_prompts_differ_only_at_position_t() {
    // Lemma D.2 causality: if two prompts differ only at position t, the
    // observed states at position t (under the recovered prefix π) are
    // different — the prefix-conditioned verifier correctly distinguishes.
    let mut rng = fastrand::Rng::with_seed(0xCAFE);
    let transformer = ToyTransformer::new(&mut rng);

    // Construct a base prompt, then mutate it at each position one at a
    // time, asserting each mutated prompt still recovers exactly.
    let base_prompt: Vec<u32> = vec![3, 7, 11, 15, 19, 23, 27, 31];
    for t in 0..T {
        let mut mutated = base_prompt.clone();
        let old = mutated[t];
        // Find a different token at this position.
        let new = if old + 1 < V { old + 1 } else { old - 1 };
        mutated[t] = new;

        let buf = transformer.forward_full(&mutated);
        let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, V, &transformer, &cfg, 0xBEEF).unwrap();
        match result {
            InversionResult::Recovered(recovered) => assert_eq!(
                recovered, mutated,
                "mutated-at-position-{t} prompt not recovered exactly"
            ),
            InversionResult::Failed {
                failed_position, ..
            } => panic!(
                "failed at position {failed_position} when only position {t} was mutated"
            ),
        }
    }
}

#[test]
fn g1_no_false_positive_on_mismatched_observed() {
    // Negative control: observed states from prompt A, but checked against
    // the transformer's actual forward on prompt A. We invert observed-A
    // and then RE-RUN the full forward on the recovered prompt. If
    // observed-A was actually for prompt B, the recovered prompt should
    // NOT match prompt A.
    //
    // Stronger version: take observed-A, mutate it slightly (above the
    // tolerance), and assert that the inversion either fails OR returns
    // a prompt that, when re-run forward, produces states different from
    // the (mutated) observed-A.
    let mut rng = fastrand::Rng::with_seed(0xFEED);
    let transformer = ToyTransformer::new(&mut rng);

    let prompt_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut buf_a = transformer.forward_full(&prompt_a);
    // Corrupt position 0's first coordinate by 1.0 (way above tolerance 1e-3).
    buf_a[0] += 1.0;

    let observed = ObservedStates::from_row_major(&buf_a, T, D).unwrap();
    let cfg = InversionConfig::default();
    let result = invert_sequence(&observed, V, &transformer, &cfg, 0).unwrap();
    match result {
        InversionResult::Failed { .. } => {
            // Correct: corrupted observed doesn't match any token's true state.
            // This is the cleanest negative-control result.
        }
        InversionResult::Recovered(recovered) => {
            // Allow recovery of SOME prompt (the corrupted observed might
            // happen to land in another token's acceptance region) — but
            // then re-running the forward must produce states that DO
            // match the corrupted observed, AND those states must NOT
            // match the original prompt A's states within tolerance.
            let re_buf = transformer.forward_full(&recovered);
            let re_observed = ObservedStates::from_row_major(&re_buf, T, D).unwrap();
            // The recovered prompt's forward must match the (corrupted)
            // observed within tolerance — otherwise the inversion is wrong.
            for t in 0..T {
                let diff: f32 = observed
                    .row(t)
                    .iter()
                    .zip(re_observed.row(t).iter())
                    .map(|(o, c)| (o - c).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    diff <= cfg.tolerance,
                    "recovered prompt's forward diverges from observed at position {t}: {diff}"
                );
            }
            // Sanity: the recovered prompt should not equal prompt_a
            // (because observed was corrupted away from prompt_a's state).
            assert_ne!(
                recovered, prompt_a,
                "inversion recovered prompt A from corrupted-A observed — false positive"
            );
        }
    }
}
