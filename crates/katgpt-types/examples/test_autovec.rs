use half::f16;

/// Dot product over `len` f16 elements.
///
/// # Safety
///
/// Callers must guarantee that `w` and `x` point to at least `len`
/// valid `f16` values each, and that the slices do not overlap in a way
/// that would violate Rust's aliasing rules.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn dot_f16_f16_autovec(w: *const f16, x: *const f16, len: usize) -> f32 {
    let w = unsafe { std::slice::from_raw_parts(w, len) };
    let x = unsafe { std::slice::from_raw_parts(x, len) };
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        acc0 = w[i].to_f32().mul_add(x[i].to_f32(), acc0);
        acc1 = w[i + 1].to_f32().mul_add(x[i + 1].to_f32(), acc1);
        acc2 = w[i + 2].to_f32().mul_add(x[i + 2].to_f32(), acc2);
        acc3 = w[i + 3].to_f32().mul_add(x[i + 3].to_f32(), acc3);
        i += 4;
    }
    acc0 + acc1 + acc2 + acc3
}

fn main() {
    let w: Vec<f16> = (0..1024).map(|i| f16::from_f32(i as f32 * 0.001)).collect();
    let x: Vec<f16> = (0..1024).map(|i| f16::from_f32((i as f32).sin())).collect();
    let result = unsafe { dot_f16_f16_autovec(w.as_ptr(), x.as_ptr(), 1024) };
    println!("result = {}", result);
}
