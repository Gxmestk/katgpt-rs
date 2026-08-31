//! Shared deterministic test scaffolding for the `mi` module (unit tests
//! only — `#[cfg(test)]`; the integration GOAT test carries its own copy per
//! the house standalone-binary convention).

/// SplitMix64 — the house deterministic RNG for benches/tests
/// (bench_576 convention).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Standard-normal draw (Box–Muller over two uniforms, f64 → f32).
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln())
            .sqrt()
            .mul_add((2.0 * std::f64::consts::PI * u2).cos(), 0.0) as f32
    }
}

pub fn splitmix(seed: u64) -> SplitMix64 {
    SplitMix64(seed)
}

/// Deterministic jointly-Gaussian 1-D pairs: `x ~ N(0,1)`,
/// `y = ρ·x + √(1−ρ²)·ε`. Ground truth `I = −½·log(1−ρ²)` nats.
pub fn gaussian_pairs(rho: f32, n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = splitmix(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let gx = rng.normal();
        let ge = rng.normal();
        x.push(gx);
        y.push(rho * gx + (1.0 - rho * rho).sqrt() * ge);
    }
    (x, y)
}

/// Deterministic d-dim pairs with the FIRST `dep` dims correlated at `rho`
/// and the rest independent — the structured fixture for the GOAT grid
/// (ground truth `dep · −½·log(1−ρ²)`).
pub fn gaussian_pairs_dep(
    rho: f32,
    n: usize,
    d: usize,
    dep: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    assert!(dep <= d);
    let mut rng = splitmix(seed);
    let mut x = vec![0.0f32; n * d];
    let mut y = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let gx = rng.normal();
            if j < dep {
                let ge = rng.normal();
                x[i * d + j] = gx;
                y[i * d + j] = rho * gx + (1.0 - rho * rho).sqrt() * ge;
            } else {
                let gy = rng.normal();
                x[i * d + j] = gx;
                y[i * d + j] = gy;
            }
        }
    }
    (x, y)
}
