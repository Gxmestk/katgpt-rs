//! Fair-roll verification — the device half of the split-key commit-reveal.
//!
//! A `Satellite` committed α, the node resolved β from its master secret, and
//! the combined seed is `BLAKE3(α ‖ β)` — a value neither party controlled
//! alone. The device re-derives that seed and re-rolls the die. If the item it
//! was handed does not match, the node rigged the outcome.
//!
//! Bit-identical to `riir-chain::split_key::{BetaResolver::combine,
//! FairRng::roll_die}`. **Do not "improve" the arithmetic here** — see the
//! residual-bias note on [`FairRollVerifier::roll_die`].
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
    /// One `blake3::hash` of 32 bytes plus integer arithmetic. No allocation,
    /// no entropy source, no float.
    ///
    /// # Residual bias — deliberately preserved, not fixed here
    ///
    /// The first byte is rejection-tested against
    /// `threshold = 256 - (256 % sides)`, but the **fallback byte is used
    /// unconditionally**, without its own rejection test. So for a `sides`
    /// that does not divide 256 the distribution is not exactly uniform: the
    /// low `256 % sides` faces are over-represented by the fallback's share.
    ///
    /// This is **reproduced on purpose**. The node computes the same skew, and
    /// bit-identity with the node is the property that keeps an honest claim
    /// from looking like fraud. Correcting the distribution is a
    /// consensus-visible change to the settled outcome of every historical
    /// roll, so it belongs in a versioned `_v2` seam agreed on both sides —
    /// not in a device-side "cleanup". Tracked in `riir-chain` Issue 108.
    ///
    /// # Panics
    ///
    /// If `sides == 0` (`256 % 0`). Use [`Self::checked_roll_die`] on any path
    /// where `sides` is attacker-influenced — on an MCU a panic is a reset.
    #[inline]
    #[must_use]
    pub fn roll_die(&self, sides: u8) -> u8 {
        let hash = blake3::hash(&self.combined_seed);
        let threshold = 256u16 - (256 % u16::from(sides));
        let val = u16::from(hash.as_bytes()[0]);
        let used = match val < threshold {
            true => val as u8,
            false => hash.as_bytes()[1],
        };
        used % sides + 1
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
    /// Bit-identical to `riir-chain`'s `FairRng::roll_dice`, **including** its
    /// different (and here fully-correct) rejection rule: the XOF path rejects
    /// every out-of-range byte rather than falling back to a second one, so it
    /// does *not* share [`Self::roll_die`]'s residual bias. The two functions
    /// are different distributions by construction, on both sides.
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

        results
    }
}
