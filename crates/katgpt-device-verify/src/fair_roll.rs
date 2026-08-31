//! Fair-roll verification — the device half of the split-key commit-reveal.
//!
//! A `Satellite` committed α, the node resolved β from its master secret, and
//! the combined seed is `BLAKE3(α ‖ β)` — a value neither party controlled
//! alone. The device re-derives that seed and re-rolls the die. If the item it
//! was handed does not match, the node rigged the outcome.
//!
//! Bit-identical to `riir-chain::split_key::{BetaResolver::combine,
//! FairRng::roll_die}` — the node delegates here, so this file **is** the
//! seam. The rejection rule is **v2**: every draw is threshold-tested (see
//! [`FairRollVerifier::roll_die`]). Do not "simplify" it back to an
//! unconditional fallback byte — that reintroduces the v1 bias and forks
//! the seam.
//!
//! Alloc-free and panic-free except for the documented `sides == 0` case,
//! which has a [`FairRollVerifier::checked_roll_die`] escape hatch.

use crate::Hash;

/// Combine the two commit-reveal halves into the seed that drives the roll.
///
/// `BLAKE3(α ‖ β)`. Bit-identical to `riir-chain`'s `BetaResolver::combine`.
///
/// The device knows α because it chose it, and receives β in the reveal. It
/// must recompute the combination itself: accepting a node-supplied combined
/// seed would let the node pick any seed it liked and the commit-reveal would
/// be decorative.
#[inline]
#[must_use]
pub fn combine_seed(alpha: &Hash, beta: &Hash) -> Hash {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(alpha);
    buf[32..].copy_from_slice(beta);
    *blake3::hash(&buf).as_bytes()
}

/// Verifier over an already-combined fair-roll seed.
///
/// Construct with [`FairRollVerifier::from_combined_seed`] after recomputing
/// the seed via [`combine_seed`]. Holds 32 bytes and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FairRollVerifier {
    combined_seed: Hash,
}

impl FairRollVerifier {
    /// Wrap a combined `BLAKE3(α ‖ β)` seed for replay/verification.
    ///
    /// Mirrors `riir-chain`'s `FairRng::from_combined_seed`, which that repo
    /// already documents as the "for replay/verification" constructor.
    #[inline]
    #[must_use]
    pub const fn from_combined_seed(combined_seed: Hash) -> Self {
        Self { combined_seed }
    }

    /// The seed this verifier will roll from.
    #[inline]
    #[must_use]
    pub const fn combined_seed(&self) -> &Hash {
        &self.combined_seed
    }

    /// Roll in `[0, 1)` — the first hash byte over 256.
    ///
    /// Bit-identical to `riir-chain`'s `FairRng::roll`. `f32` division by a
    /// power of two is exact, so this is reproducible across targets without
    /// a soft-float caveat.
    #[inline]
    #[must_use]
    pub fn roll_unit(&self) -> f32 {
        f32::from(blake3::hash(&self.combined_seed).as_bytes()[0]) / 256.0
    }

    /// Roll a single die with `sides` faces. Returns `1..=sides`.
    ///
    /// One `blake3::hash` of 32 bytes plus integer arithmetic in the
    /// overwhelmingly common case. No allocation, no entropy source, no
    /// float.
    ///
    /// # Exact uniformity (v2) — every draw is rejection-tested
    ///
    /// Each hash byte is tested against `threshold = 256 - (256 % sides)` —
    /// that is `sides * (256 / sides)`, the largest multiple of `sides` that
    /// fits in a byte. The **first byte below the threshold** decides the
    /// roll: `byte % sides + 1`. A byte at or above the threshold is
    /// rejected and the next byte is drawn; if all 32 hash bytes reject
    /// (probability ≤ `(127/256)^32 ≈ 2e-10` for any `sides`), the
    /// keystream extends deterministically by hashing the hash. An accepted
    /// byte is uniform over exactly `threshold` values — a whole number of
    /// `sides`-blocks — so `byte % sides` is exactly uniform and every face
    /// in `1..=sides` is reachable. No modulo is ever applied to a
    /// rejected value.
    ///
    /// # v1 → v2 (2026-08-24, owner call — `riir-chain` Issue 108)
    ///
    /// v1 tested only the first byte and used the second **unconditionally**
    /// when the first rejected, so a `sides` not dividing 256 was slightly
    /// non-uniform (the low `256 % sides` faces over-represented). v2
    /// rejection-tests every draw. Outcomes change **only** for seeds whose
    /// first *two* hash bytes both meet the threshold (e.g. ~0.02% of seeds
    /// at `sides = 6`); every other seed rolls the same face as v1. The
    /// fixture labels carry `v2` so the two seams cannot be confused.
    ///
    /// # Panics
    ///
    /// If `sides == 0` (`256 % 0`). Use [`Self::checked_roll_die`] on any path
    /// where `sides` is attacker-influenced — on an MCU a panic is a reset.
    #[inline]
    #[must_use]
    pub fn roll_die(&self, sides: u8) -> u8 {
        let threshold = 256u16 - (256 % u16::from(sides));
        let mut hash = blake3::hash(&self.combined_seed);
        loop {
            for &byte in hash.as_bytes() {
                if u16::from(byte) < threshold {
                    return byte % sides + 1;
                }
            }
            // All 32 bytes rejected (≤ ~2e-10 for any sides): extend the
            // keystream deterministically. Never taken in practice; exists
            // so the function is total without an allocation or a panic.
            hash = blake3::hash(hash.as_bytes());
        }
    }

    /// [`Self::roll_die`], returning `None` instead of panicking on `sides == 0`.
    ///
    /// The verify path on a device must be total: a malformed field on the
    /// wire is a thing that happens, and a reset loop is a worse answer than a
    /// refused claim.
    #[inline]
    #[must_use]
    pub fn checked_roll_die(&self, sides: u8) -> Option<u8> {
        match sides {
            0 => None,
            n => Some(self.roll_die(n)),
        }
    }

    /// Verify a node-supplied die against this seed.
    ///
    /// The whole point of the crate, in one call: `true` iff the node dealt
    /// what the commit-reveal actually determined. A malformed `sides` is a
    /// failed verification, never a panic.
    #[inline]
    #[must_use]
    pub fn verify_die(&self, sides: u8, claimed: u8) -> bool {
        self.checked_roll_die(sides) == Some(claimed)
    }

    /// Roll `count` dice via the BLAKE3 XOF keystream.
    ///
    /// Bit-identical to `riir-chain`'s `FairRng::roll_dice`. Like
    /// [`Self::roll_die`] (v2), this path rejection-tests **every** byte, so
    /// both functions are exactly uniform. They remain *different*
    /// distributions because they draw from different byte streams — this
    /// one a keyed XOF keystream, `roll_die` the plain hash bytes chained on
    /// exhaustion — so the same seed rolls different sequences under the
    /// two, on both sides of the seam.
    ///
    /// Behind `alloc` because it returns a `Vec`. A daily single-item claim
    /// needs only [`Self::roll_die`]; a device build leaves this off.
    ///
    /// # Panics
    ///
    /// If `sides == 0`, for the same reason as [`Self::roll_die`].
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn roll_dice(&self, sides: u8, count: usize) -> alloc::vec::Vec<u8> {
        let key = blake3::derive_key("riir-chain-fair-dice-v1", &self.combined_seed);
        let mut reader = blake3::Hasher::new_keyed(&key).finalize_xof();

        let threshold = 256u16 - (256 % u16::from(sides));
        let mut buf = [0u8; 64];
        let mut results = alloc::vec::Vec::with_capacity(count);

        while results.len() < count {
            reader.fill(&mut buf);
            for &byte in &buf {
                if results.len() >= count {
                    break;
                }
                if u16::from(byte) < threshold {
                    results.push(byte % sides + 1);
                }
            }
        }

        results.shrink_to_fit();
        results
    }
}
