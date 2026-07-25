// SPDX-License-Identifier: MIT
//
// Vendored from gigatoken https://github.com/marcelroed/gigatoken (Marcel Rød).
// Upstream file: `src/pretokenize/mod.rs` (pack_pretoken_key + pretoken_key_hash +
// pack_mask_halves — only the parts `crate::bpe` actually depends on).
// Upstream commit at vendor time: master 2026-07-25.
//
// Adaptations vs upstream:
// - Dropped the `pretokenize_par_bytes` / `PretokenSpans` / `SpanBatch` types
//   (those live in the upstream `pretokenize/` module which needs
//   `#![feature(portable_simd)]`).
// - Dropped the `crc_hash_selected` per-process bit (kept internal here).
//
// `allow(unused)]` is intentional: `pack_pretoken_key` + `pack_mask_halves`
// are vendored substrate for the day someone adds pretokenization (Issue 191
// Phase 1 §"What's NOT here"). They are bit-identical to upstream.

#![allow(unused)]

/// Both 64-bit halves of the per-length pack mask, in scalar ALU ops. A
/// u128 `MAX >> s` lowers to a multi-instruction sequence and the 16-entry
/// table this replaces put a dependent L1 load on the `n → key → store`
/// chain; per-half variable shifts are single 1-cycle ops, so the halves
/// cost two independent 3-deep chains and no load port.
///
/// `const`: the pretoken-cache build path computes masks at runtime via this
/// function, so the ALU form is the only one in this crate.
#[inline(always)]
pub(crate) const fn pack_mask_halves(n: usize) -> (u64, u64) {
    debug_assert!(n >= 1 && n <= 15);
    let s = (n * 8) as u32;
    let lo = if n < 8 {
        u64::MAX >> (64u32.wrapping_sub(s) & 63)
    } else {
        u64::MAX
    };
    let hi = if n > 8 {
        u64::MAX >> (128u32.wrapping_sub(s) & 63)
    } else {
        0
    };
    (lo, hi)
}

/// Pack a pretoken of ≤ 15 bytes into a `u128` cache key: bytes in the low
/// 15 lanes, length in the top byte (so keys of different lengths never
/// collide, and a real key is never 0). Returns `None` for longer pretokens,
/// which use the slice-keyed fallback map.
///
/// The common path is a single unaligned 16-byte load followed by a mask,
/// avoiding both a variable-length `memcpy` and per-byte branching. The
/// load is only taken when it cannot cross a page boundary, so it can never
/// touch an unmapped page; the rare near-boundary case falls back to a plain
/// copy. Both paths produce the identical key.
#[inline(always)]
pub fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    if n == 0 {
        // Empty pretokens pack to key 0, which the short table reserves as
        // its empty sentinel — the encode loop routes key 0 to the long map.
        // Also keeps the read below from touching a zero-length slice's
        // dangling pointer.
        return Some(0);
    }
    let p = bytes.as_ptr();
    let low = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: the offset within the (≥ 4096-byte) page is ≤ 4096 - 16,
        // so a 16-byte read stays inside the page holding `p`, which is
        // mapped because `p` points to at least one valid byte.
        let v = unsafe { (p as *const u128).read_unaligned() };
        let (mask_lo, mask_hi) = pack_mask_halves(n);
        ((v as u64 & mask_lo) as u128) | ((((v >> 64) as u64 & mask_hi) as u128) << 64)
    } else {
        // Rare: `p` is within 16 bytes of a page boundary. Gather with a
        // plain copy (≤ 15 bytes) — correctness over speed on this cold
        // path. Lanes past `n` stay zero, so no mask is needed.
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        u128::from_le_bytes(lanes)
    };
    Some(low | ((n as u128) << 120))
}

/// The multiply-fold arm of [`pretoken_key_hash`]: one folded multiply, the
/// cheapest mix whose low bits still see every key bit. Every target can
/// execute it; it is the process's hash wherever no hardware CRC arm
/// applies. Maps key 0 to hash 0 (0 · M = 0).
#[allow(dead_code)] // no cfg arm references it under aarch64 + crc
#[inline(always)]
fn pretoken_key_hash_fold(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut h = (lo ^ hi.rotate_right(25)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h
}

/// The hardware CRC32C (SSE4.2) arm of [`pretoken_key_hash`]: same shape
/// and rationale as the aarch64 CRC32 arm — linear over GF(2) so the low
/// bits (the table index) see every key bit, 3-cycle latency and one µop
/// per `crc32` on Zen 2 (two chained ops vs the 5-op multiply fold), and
/// `_mm_crc32_u64(0, 0) == 0` preserves the key 0 -> hash 0 property.
///
/// `sse4.2` is NOT in baseline x86-64, so this arm is selected per process
/// by [`pretoken_key_hash`] via `std::arch::is_x86_feature_detected!`.
///
/// # Safety
///
/// The CPU must support SSE4.2: callers reach this only via
/// [`pretoken_key_hash`]'s runtime detection (or from a build with
/// `sse4.2` statically enabled).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
unsafe fn pretoken_key_hash_crc32c(key: u128) -> u64 {
    use core::arch::x86_64::_mm_crc32_u64;
    // SAFETY: SSE4.2 is enabled on this function; the caller (per the
    // contract above) only reaches it on a CPU that has it.
    unsafe { _mm_crc32_u64(_mm_crc32_u64(0, key as u64), (key >> 64) as u64) }
}

/// Hash of a packed pretoken key. Quality is noncritical for correctness
/// (the table compares full keys), but every consumer — the cache's
/// `get_or_slot` probe path, its `grow` rehash — must compute the same
/// function of the key.
///
/// On aarch64 with the `crc` target feature the hardware CRC32 path is
/// compile-time-selected; on x86_64 it is runtime-selected per process via
/// `std::arch::is_x86_feature_detected!("sse4.2")`; everywhere else the
/// multiply-fold arm is the only one available.
///
/// All arms map key 0 to hash 0, which the cache reserves as its empty
/// sentinel route.
#[inline(always)]
pub fn pretoken_key_hash(key: u128) -> u64 {
    // Note: `crc` is in the default feature set for aarch64-apple-darwin
    // but NOT for aarch64-unknown-linux-gnu — generic aarch64 Linux builds
    // need `-C target-feature=+crc` (e.g. via RUSTFLAGS) to get this fast
    // hash; without it they silently take the multiply fold below.
    #[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
    {
        // Hardware CRC32: two 3-cycle ops replace the 5-op multiply fold.
        // SAFETY: gated on the `crc` target feature at compile time.
        use core::arch::aarch64::__crc32d;
        unsafe { __crc32d(__crc32d(0, key as u64), (key >> 64) as u64) as u64 }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            // SAFETY: runtime detection just above.
            unsafe { pretoken_key_hash_crc32c(key) }
        } else {
            pretoken_key_hash_fold(key)
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "crc"),
        target_arch = "x86_64"
    )))]
    {
        pretoken_key_hash_fold(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_key_round_trip() {
        for n in 1..=15usize {
            let bytes: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(7)).collect();
            let key = pack_pretoken_key(&bytes).unwrap();
            // Length in the top byte.
            assert_eq!((key >> 120) as usize, n);
            // Bytes recover from the low lanes.
            let mut lanes = [0u8; 16];
            lanes[..n].copy_from_slice(&bytes);
            let expected_low = u128::from_le_bytes(lanes);
            assert_eq!(key & ((1u128 << 120) - 1), expected_low);
        }
        assert_eq!(pack_pretoken_key(b""), Some(0));
        assert_eq!(pack_pretoken_key(&[0u8; 16]), None);
    }

    #[test]
    fn hash_is_deterministic_per_target() {
        let k = 0x0100_0000_0000_0000_0000_0000_0000_0041u128;
        let h1 = pretoken_key_hash(k);
        let h2 = pretoken_key_hash(k);
        assert_eq!(h1, h2);
        // Key 0 maps to hash 0 on every arm.
        assert_eq!(pretoken_key_hash(0), 0);
    }
}
