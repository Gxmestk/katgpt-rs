//! Minimal `f32` complex scalar.
//!
//! `katgpt-core` has no complex-number dependency and this module needs only
//! add / multiply / conjugate on `d ≤ 8` amplitudes, so a local `Copy` struct is
//! cheaper than pulling in `num-complex` for a leaf primitive. Deliberately not
//! a general-purpose complex type — it carries exactly the operations the SU(d)
//! basis construction and the Hermitian power iteration need.

/// A single-precision complex number `re + i·im`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct C32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl C32 {
    /// Additive identity `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// Multiplicative identity `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// The imaginary unit `i`.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct from real and imaginary parts.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Embed a real number.
    #[inline]
    pub const fn real(re: f32) -> Self {
        Self { re, im: 0.0 }
    }

    /// Complex conjugate `re − i·im`.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus `|z|²`.
    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `|z|`.
    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Sum.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    /// Difference.
    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    /// Product.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Product with a real scalar.
    #[inline]
    pub fn scale(self, k: f32) -> Self {
        Self {
            re: self.re * k,
            im: self.im * k,
        }
    }

    /// `self + a·b`, the accumulation step of every inner product here.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        self.add(a.mul(b))
    }
}
