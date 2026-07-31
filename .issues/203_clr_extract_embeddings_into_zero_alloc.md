# Issue 203 — CLR `extract_embeddings_into` + `verify_embedding`: zero-alloc flat-embedding vote path

## Status: CLOSED — FULLY IMPLEMENTED (2026-07-29, Session 18)

All design items shipped. Retained as reference (linked from README + Bench 570 +
riir-ai Issue 568). The Layer 1 sibling (`downcast_trajectories` outer Vec —
riir-ai Issue 568 Opt B) remains open but deprioritized (1 alloc/NPC, not 33).

**Implementing commits:**
- katgpt-rs `cc841a2c` — `extract_embeddings_into` + `verify_embedding` trait methods
- katgpt-rs `4c337339` — `ClrConfig.embedding_dim` + `ClrScratch::new(k, m, embedding_dim)`
- riir-ai `3bdd43a38` — 5 production extractor overrides (Guard/Merchant/Healer/Scout/Diplomat)
- riir-ai `df2af7f88` — G4 alloc-free test (`clr_extract_g4_alloc.rs`) + Bench 570
- riir-ai `92332aba6` — fix false-positive G4 infra (Issue 569 `extern crate` link bug)

See [Bench 570](../../riir-ai/.benchmarks/570_clr_extract_zero_alloc_g4.md) for the
G4 gate results.

---

**Filed:** 2026-07-29
**Source:** Sibling of `riir-ai/.issues/568` (CLR dispatch per-NPC allocation)
**Severity:** Optimization (not a correctness bug)
**Scope:** `katgpt-claim` trait + vote path + `riir-games-civ` extractor overrides

## Context

`riir-ai/.issues/568` identified a two-layer allocation hotspot in the CLR
per-NPC decision path. Layer 2 is `ClaimExtractor::extract` returning
`Vec<Claim<T>>` per trajectory — the current trait signature forces a heap
allocation. For a 1000-NPC crowd at decision cadence, this produces
~32,000+ hidden allocations per decision tick on rayon worker threads.

The root cause: `clr_vote_minimal` calls `extractor.extract(traj)` →
`Vec<Claim<T>>` per trajectory. The `Claim<T>` values are consumed
immediately (verdicts written into `ClrScratch.verdicts`) then dropped.
The verifier (`SigmoidProjectionVerifier`) only reads `claim.embedding` —
the payload `T` is irrelevant to the vote computation.

## Design: flat embedding scratch + new trait methods

Eliminate Layer 2 by making the hot-path vote work with raw embeddings
(flat `&mut [f32]`) instead of `Vec<Claim<T>>`:

### 1. `ClaimExtractor::extract_embeddings_into` (new trait method)

```rust
pub trait ClaimExtractor<T> {
    /// Extract M claim embeddings into the flat buffer `out`.
    /// `out.len()` must be >= `M * k`. Row `m` is at `out[m*k..(m+1)*k]`.
    /// Default: calls `extract` + copies (still allocates — override for zero-alloc).
    fn extract_embeddings_into(&self, trajectory: &Trajectory<T>, out: &mut [f32], k: usize) {
        let claims = self.extract(trajectory);
        debug_assert!(claims.len() * k <= out.len());
        for (i, claim) in claims.iter().enumerate() {
            out[i * k..(i + 1) * k].copy_from_slice(&claim.embedding[..k]);
        }
    }

    /// Extract claims, returning a fresh Vec. Allocates. Kept for the audit trail path.
    fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>>;
}
```

### 2. `ClaimVerifier::verify_embedding` (new trait method)

```rust
pub trait ClaimVerifier<T> {
    /// Verify a raw embedding slice. The core computation.
    fn verify_embedding(&self, embedding: &[f32], direction_idx: usize) -> Verdict;

    /// Default: delegate to `verify_embedding` via the claim's embedding.
    fn verify(&self, claim: &Claim<T>, direction_idx: usize) -> Verdict {
        self.verify_embedding(&claim.embedding, direction_idx)
    }
}
```

### 3. `ClrScratch` gains `claim_embeddings: Vec<f32>`

Flat `M * k` buffer. Pre-allocated in `new`, cleared+resized in `reset_for`.

### 4. `clr_vote_minimal` hot path

```rust
for (k_idx, traj) in trajectories.iter().enumerate() {
    extractor.extract_embeddings_into(traj, &mut scratch.claim_embeddings, config.k);
    for m_idx in 0..m {
        let emb = &scratch.claim_embeddings[m_idx * config.k..(m_idx + 1) * config.k];
        scratch.verdicts[k_idx * m + m_idx] = verifier.verify_embedding(emb, m_idx);
    }
}
```

Zero allocation: no `Vec<Claim<T>>`, no per-claim `Vec<f32>` embedding.

### 5. Production extractor overrides (`riir-games-civ`)

The 5 extractors (Guard, Merchant, Healer, Scout, Diplomat) override
`extract_embeddings_into` to write directly into the flat buffer using
`write_binary_claim_into` / `write_scalar_claim_into` helpers (in-place
versions of `encode_binary_claim` / `encode_scalar_claim`).

### 6. `clr_vote_minimal_fused` (renoise_fusion)

`perturb_reverify_drift` works on `&mut [f32]` slices (the flat
embedding buffer rows) instead of `&mut [Claim<T>]`. The save/perturb/
restore cycle uses `copy_from_slice` into a pre-sized scratch, same as
today — just operating on raw slices instead of Claim fields.

## GOAT gate

- **G1** (correctness): `clr_vote_minimal` output bit-identical before/after
  (the embedding values are the same, just written in-place vs allocated).
- **G2** (perf): alloc count drops from ~6/trajectory to 0 (per-trajectory
  `Vec::with_capacity` + 5 `vec![0.0; DIR_DIM]` embedding allocs eliminated).
- **G3** (no-regression): all existing CLR tests pass.
- **G4** (alloc-free): a serial G4 test for the full vote path (without
  rayon, on the main thread) asserts 0 allocs after warmup.

## Sizing the gain

| Crowd | NPCs | Allocs/decision-tick (current Layer 2) | After |
|---|---|---|---|
| Town (POC) | 10 | ~600 (6/traj × ~10 traj/NPC × 10 NPC) | 0 |
| Crowd-scale | 1000 | ~60,000+ | 0 |
| Production | 10000 | ~600,000+ | 0 |

(6 allocs/trajectory = 1 `Vec::with_capacity(5)` + 5 `vec![0.0; DIR_DIM]`
embedding buffers.)
