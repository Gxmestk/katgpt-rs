//! Shared `CountingAllocator` test/bench infrastructure for `katgpt-dec`.
//!
//! Mirrors `katgpt-core/tests/common/mod.rs` (Issue 044 T3): emits a global
//! allocator tracking alloc/dealloc counts plus an `alloc_delta` helper.
//! Eliminates ~20 lines of boilerplate per G4 alloc-check bench — the
//! inline pattern previously copy-pasted into `bench_407_sheaf_admm_goat`,
//! `bench_422_cochain_point_sampler_goat`, and `bench_454_3d_nca_goat`.
//!
//! This file is **self-contained** (no `use crate::*`, all paths fully
//! qualified) so it can be `#[path]`-included from bench binaries, which
//! are separate compilation units from the `katgpt_dec` lib crate. The
//! DEC-field helpers in sibling `mod.rs` use `crate::types::*` and are
//! therefore NOT bench-includable; this file is intentionally split out.
//!
//! # Usage from a bench file
//!
//! ```ignore
//! #[path = "../tests/common/counting_allocator.rs"]
//! mod counting_allocator;
//! counting_allocator!();
//! ```
//!
//! (The macro is `#[macro_export]`'d, so it lives at the consuming crate's
//! root regardless of the `mod counting_allocator;` path — same convention
//! as `katgpt-core/tests/common/mod.rs` Issue 044 T3.)
//!
//! After macro invocation, the following names are in scope at the crate
//! root (where the macro was invoked):
//! - `ALLOC_COUNT` — `AtomicUsize` of total allocations
//! - `DEALLOC_COUNT` — `AtomicUsize` of total deallocations
//! - `alloc_delta(f)` — runs `f()` and returns `(result, alloc_count_delta)`
//!
//! Callers that read counters directly should add their own
//! `use std::sync::atomic::Ordering;` — the macro does NOT emit `use`
//! statements, to avoid import conflicts at the call site.

/// Install a global `CountingAllocator` that counts alloc/dealloc calls.
///
/// Emits, at the call site:
/// - `struct CountingAllocator;`
/// - `static ALLOC_COUNT: AtomicUsize`
/// - `static DEALLOC_COUNT: AtomicUsize`
/// - `unsafe impl GlobalAlloc for CountingAllocator`
/// - `#[global_allocator] static A: CountingAllocator`
/// - `fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize)`
///
/// All paths are fully qualified to avoid import conflicts.
#[macro_export]
macro_rules! counting_allocator {
    () => {
        struct CountingAllocator;

        static ALLOC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static DEALLOC_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
                ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
                DEALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.dealloc(ptr, layout) }
            }
        }

        #[global_allocator]
        static A: CountingAllocator = CountingAllocator;

        #[inline]
        #[allow(dead_code)]
        fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
            let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let r = f();
            let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            (r, after - before)
        }
    };
}
