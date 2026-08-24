//! Exhaustive raw-value → face mapping for `roll_die` (v2) — the uniformity
//! proof for the fix that retired `riir-chain` Issue 108's residual bias.
//!
//! For each `sides`, a seed is **searched** for every possible first hash
//! byte `b ∈ 0..=255`, so the complete raw-value space is mapped through the
//! real BLAKE3 path — not a mock of the reduction:
//!
//! - an accepted `b < threshold` must deal exactly `b % sides + 1` — the
//!   rejection-sampling reduction, verified for **every** raw value;
//! - the accepted values distribute **exactly** uniformly: each face has
//!   precisely `threshold / sides` accepted preimages, so every face is
//!   reachable from the accepted set alone (full coverage);
//! - a rejected `b ≥ threshold` re-draws: the dealt face is in range.
//!
//! Uniformity of the *total* distribution follows by induction: the re-draw
//! samples from the same process, so
//! `P(face) = per_face/256 + (1 − threshold/256)·P(face)` ⇒ `P(face) = 1/sides`
//! exactly. The sweep pins the reduction rule and the conditional uniformity
//! it rests on; rejected first bytes cannot be mapped statically (their
//! outcome depends on later bytes), which is why the sweep pins the rule,
//! not the tail.
//!
//! For a `sides` that divides 256 nothing ever rejects, so the sweep is a
//! **complete** exhaustive proof: each face hit exactly `256/sides` times.

use katgpt_device_verify::fair_roll::FairRollVerifier;

fn threshold(sides: u8) -> u16 {
    256u16 - (256 % u16::from(sides))
}

/// Search a seed whose first hash byte is exactly `target`.
fn seed_with_first_byte(sides: u8, target: u8) -> [u8; 32] {
    (0u32..)
        .map(|i| {
            let label = format!("exhaustive/fair_roll/v2/{sides}/{target}/{i}");
            *blake3::hash(label.as_bytes()).as_bytes()
        })
        .find(|seed| blake3::hash(seed).as_bytes()[0] == target)
        .expect("every first-byte value is reachable by search")
}

fn exhaustive(sides: u8) {
    let t = threshold(sides);
    let per_face = u32::from(t / u16::from(sides)); // = floor(256 / sides)
    let mut counts = [0u32; 256];
    for target in 0u16..256 {
        let seed = seed_with_first_byte(sides, target as u8);
        let die = FairRollVerifier::from_combined_seed(seed).roll_die(sides);
        assert!(
            (1..=sides).contains(&die),
            "sides={sides} target={target}: die {die} outside 1..={sides}"
        );
        if target < t {
            // The reduction rule, pinned per raw value: an accepted byte maps
            // by modulo — and only ever an ACCEPTED byte.
            assert_eq!(
                die,
                target as u8 % sides + 1,
                "sides={sides} target={target}: accepted byte must reduce by modulo"
            );
        }
        counts[die as usize] += 1;
    }
    for face in 1..=u16::from(sides) {
        let c = counts[face as usize];
        assert!(c > 0, "sides={sides}: face {face} unreachable");
        if t == 256 {
            // Nothing rejects: exact uniformity over the whole raw space.
            assert_eq!(
                c, per_face,
                "sides={sides}: face {face} hit {c}×, expected exactly {per_face}"
            );
        } else {
            // The accepted set alone gives every face exactly per_face
            // preimages; rejected bytes may only add.
            assert!(
                c >= per_face,
                "sides={sides}: face {face} hit {c}×, expected >= {per_face} \
                 accepted preimages"
            );
        }
    }
    assert_eq!(counts.iter().sum::<u32>(), 256);
}

#[test]
fn dividing_sides_are_exactly_uniform_over_the_whole_raw_space() {
    // threshold == 256: every byte accepts, so the 256-value sweep IS the
    // full distribution — uniformity and reachability proven exhaustively.
    for sides in [1u8, 2, 4, 8, 16, 32, 64, 128] {
        exhaustive(sides);
    }
}

#[test]
fn nondividing_sides_map_every_raw_value_and_reach_every_face() {
    // The v1→v2 territory: rejection is live. The sweep pins the reduction
    // rule per raw value + exact conditional uniformity + full reachability;
    // total uniformity follows by induction (module doc).
    for sides in [3u8, 5, 6, 7, 10, 12, 20, 100, 255] {
        exhaustive(sides);
    }
}
