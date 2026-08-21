//! Portable `std::simd` recipe skeleton.
//!
//! Patterns distilled from mcyoung's `vb64` writeup:
//! https://mcyoung.xyz/2023/11/27/simd-base64/
//!
//! Requires nightly: `#![feature(portable_simd)]`.
//! Demonstrates: perfect-hash swizzle lookup, sub-byte packing via widening cast,
//! slop-buffer commit, delayed failure. Apply the same shapes to any codec/parser
//! where auto-vectorization fails and branching dominates.

#![feature(portable_simd)]

use std::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount, Swizzle};

/// Branchless ASCII → 6-bit sextet lookup using a perfect hash + single shuffle.
///
/// Hash: `(byte >> 4) - (byte == b'/')` maps the 5 base64 ranges onto indices 1..=7.
/// The 8-entry offset table is the "match arm result" for each hash bucket.
/// One `swizzle_dyn` replaces 8 scalar compares.
#[inline]
fn ascii_to_sextets<const N: usize>(ascii: Simd<u8, N>) -> (Simd<u8, N>, Mask<i8, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // Perfect hash. `simd_eq(...).to_int()` yields -1 / 0 per lane; cast to u8 wraps.
    let solidus = ascii.simd_eq(Simd::splat(b'/')).to_int().cast::<u8>();
    let hashes = (ascii >> Simd::splat(4)) + solidus;

    // Lookup table indexed by hash. Same length requirement as `swizzle_dyn` indices.
    // Values = the constant C such that `sextet = byte - C` for that range.
    let offsets = Simd::<i8, 8>::from([0, 16, 19, 4, -65, -65, -71, -71])
        .cast::<u8>();

    let sextets = ascii - hashes.swizzle_dyn_dyn(offsets);

    // Validation: cheap branchless check. Replace with a bloom-filter mask for
    // tighter error detection — see vb64 source for the exact bitmask.
    let valid_upper = ascii.simd_ge(Simd::splat(b'+')) & ascii.simd_le(Simd::splat(b'z'));
    let ok = valid_upper | ascii.simd_eq(Simd::splat(b'A')); // placeholder; tighten per domain

    (sextets, ok)
}

/// Pack 4 sextets into 3 bytes by widening to u16, shifting, then OR-ing halves.
///
/// No single instruction crosses byte boundaries with sub-byte granularity, so
/// widen the lanes so the shift puts each bit in the right byte, then split +
/// recombine. `rotate_lanes_left::<1>()` aligns the high half with the low half.
#[inline]
fn pack_sextets<const N: usize>(sextets: Simd<u8, N>) -> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let shifted = sextets.cast::<u16>() << tiled_shifts::<u16, N>();
    let lo = shifted.cast::<u8>();
    let hi = (shifted >> Simd::splat(8)).cast::<u8>();
    let packed = lo | hi.rotate_lanes_left::<1>();

    // Drop every 4th lane (garbage from the 4→3 compression).
    DeleteEveryFourth.swizzle(packed)
}

/// Lane-deletion swizzle: `out[i] = in[i + i/3]`, skipping garbage at indices 3,7,11,...
struct DeleteEveryFourth;
impl<const N: usize> Swizzle<N> for DeleteEveryFourth {
    const INDEX: [usize; N] = {
        let mut idx = [0; N];
        let mut i = 0;
        while i < N {
            idx[i] = i + i / 3;
            i += 1;
        }
        idx
    };
}

/// Slop-buffer commit: pre-reserve with N/4 extra bytes, write full SIMD stores,
/// then commit the final length only on success. On error, never call `set_len`
/// — the speculative writes are invisible to safe code.
///
/// Invariant: `out`'s spare capacity ≥ `decoded_len(input) + N/4`.
pub fn decode_into<const N: usize>(
    input: &[u8],
    out: &mut Vec<u8>,
) -> Result<(), &'static str>
where
    LaneCount<N>: SupportedLaneCount,
{
    let payload = strip_padding(input);
    let final_len = decoded_len(payload.len());
    out.reserve(final_len + N / 4);

    let start = out.as_mut_ptr_range().end();
    let mut write = start;
    let mut error = false;

    // Hot path: full N-byte chunks, single SIMD load each.
    let mut chunks = payload.chunks_exact(N);
    for chunk in &mut chunks {
        let ascii = Simd::from_slice(chunk);
        let (sextets, ok) = ascii_to_sextets::<N>(ascii);
        let bytes = pack_sextets::<N>(sextets);
        error |= !ok;

        unsafe {
            write.cast::<Simd<u8, N>>().write_unaligned(bytes);
            write = write.add(decoded_len(N));
        }
    }

    // Cold tail: pad to N with 'A' (zero-sextet), one SIMD op.
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let mut buf = [b'A'; 64]; // ensure >= N at compile time for your use site
        debug_assert!(rest.len() <= buf.len());
        buf[..rest.len()].copy_from_slice(rest);
        let ascii = Simd::from_slice(&buf[..N]);
        let (sextets, ok) = ascii_to_sextets::<N>(ascii);
        let bytes = pack_sextets::<N>(sextets);
        error |= !ok;

        unsafe {
            write.cast::<Simd<u8, N>>().write_unaligned(bytes);
            write = write.add(decoded_len(rest.len()));
        }
    }

    if error {
        return Err("invalid base64 byte"); // garbage writes never committed
    }

    unsafe {
        let len = write.offset_from(start) as usize;
        debug_assert_eq!(len, final_len);
        out.set_len(out.len() + len);
    }
    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

#[inline]
fn decoded_len(input: usize) -> usize {
    // Branchless: matches `input/4*3 + [0,1,1,2][input%4]`.
    let mod4 = input % 4;
    input / 4 * 3 + (mod4 - mod4 / 2)
}

#[inline]
fn strip_padding(data: &[u8]) -> &[u8] {
    match data {
        [p @ .., b'=', b'='] | [p @ .., b'='] | p => p,
    }
}

/// Tile a small pattern across an N-lane vector. Const-eval, zero runtime cost.
const fn tiled_shifts<T: SimdElement, const N: usize>() -> Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // Specialized for the pack_sextets case: [2, 4, 6, 8] tiled.
    // In real code, generate this generically; kept inline here for readability.
    panic!("call site specialization required — see vb64 src/util.rs `tiled()`")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_len_branchless_matches_naive() {
        for n in 0..128usize {
            let naive = n / 4 * 3
                + match n % 4 {
                    1 | 2 => 1,
                    3 => 2,
                    _ => 0,
                };
            assert_eq!(decoded_len(n), naive, "mismatch at n={n}");
        }
    }
}
