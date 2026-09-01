//! Issue 700 — soundness regression gate for the SIMD `len`-taking kernels.
//!
//! These are SAFE public fns whose backends do unchecked pointer loads/stores
//! up to a caller-supplied `len`. Before the fix a `len` exceeding the slices
//! produced a SILENT out-of-bounds read (garbage / NaN, no panic) and — for the
//! `dst: &mut [f32]` family — a silent out-of-bounds WRITE that corrupted
//! neighbouring heap, all reachable from 100% safe code (CWE-125 / CWE-787).
//!
//! Each test below is safe code. A test that stops panicking means the
//! entry-point reslice was removed and the UB is back.

use half::f16;
use katgpt_types::simd::{
    simd_dist_sq, simd_dot_f16_f16, simd_dot_f16_f32, simd_dot_f32, simd_fma_row,
    simd_fused_scale_acc, simd_fused_scale_acc_f16, simd_fused_sub_acc, simd_l_inf_distance_f32,
    simd_sum_sq,
};

#[test]
#[should_panic(expected = "out of range")]
fn dot_f32_rejects_len_past_end() {
    let (a, b) = (vec![1.0f32; 4], vec![1.0f32; 4]);
    simd_dot_f32(&a, &b, 4096);
}

#[test]
#[should_panic(expected = "out of range")]
fn fma_row_rejects_len_past_end() {
    let (a, b) = (vec![1.0f32; 4], vec![1.0f32; 4]);
    simd_fma_row(&a, &b, 4096);
}

#[test]
#[should_panic(expected = "out of range")]
fn sum_sq_rejects_len_past_end() {
    simd_sum_sq(&[1.0f32; 4], 4096);
}

#[test]
#[should_panic(expected = "out of range")]
fn dist_sq_rejects_len_past_end() {
    let (a, b) = (vec![1.0f32; 4], vec![1.0f32; 4]);
    simd_dist_sq(&a, &b, 4096);
}

/// The pre-fix guard here was `debug_assert_eq!(a.len(), b.len())`, which is
/// insufficient twice over: it compares the slices to EACH OTHER and never to
/// `len`, and it vanishes entirely in release. 4 == 4 passed while len was 4096.
#[test]
#[should_panic(expected = "out of range")]
fn l_inf_rejects_len_past_end_even_with_equal_slice_lens() {
    let (a, b) = (vec![1.0f32; 4], vec![1.0f32; 4]);
    simd_l_inf_distance_f32(&a, &b, 4096);
}

#[test]
#[should_panic(expected = "out of range")]
fn dot_f16_f32_rejects_len_past_end() {
    let w = vec![f16::from_f32(1.0); 4];
    let x = vec![1.0f32; 4];
    simd_dot_f16_f32(&w, &x, 4096);
}

#[test]
#[should_panic(expected = "out of range")]
fn dot_f16_f16_rejects_len_past_end() {
    let w = vec![f16::from_f32(1.0); 4];
    simd_dot_f16_f16(&w, &w, 4096);
}

// ── The write family: these corrupted neighbouring heap before the fix ──

#[test]
#[should_panic(expected = "out of range")]
fn fused_sub_acc_rejects_len_past_dst() {
    let mut dst = vec![0.0f32; 16];
    let (a, b) = (vec![5.0f32; 64], vec![1.0f32; 64]);
    simd_fused_sub_acc(&mut dst, &a, &b, 48);
}

#[test]
#[should_panic(expected = "out of range")]
fn fused_scale_acc_rejects_len_past_dst() {
    let mut dst = vec![0.0f32; 16];
    let src = vec![1.0f32; 64];
    simd_fused_scale_acc(&mut dst, &src, 2.0, 48);
}

#[test]
#[should_panic(expected = "out of range")]
fn fused_scale_acc_f16_rejects_len_past_dst() {
    let mut dst = vec![0.0f32; 16];
    let src = vec![f16::from_f32(1.0); 64];
    simd_fused_scale_acc_f16(&mut dst, &src, 2.0, 48);
}

/// A short `len` (a genuine prefix request) must still be honoured, not
/// rejected — the guard clamps nothing and changes no result.
#[test]
fn short_len_is_a_valid_prefix_request() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b = vec![1.0f32; 64];
    let full = simd_dot_f32(&a, &b, 64);
    let prefix = simd_dot_f32(&a, &b, 8);
    assert_eq!(prefix, (0..8).map(|i| i as f32).sum::<f32>());
    assert_eq!(full, (0..64).map(|i| i as f32).sum::<f32>());
}
