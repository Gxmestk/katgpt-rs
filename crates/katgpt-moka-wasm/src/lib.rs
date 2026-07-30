//! Minimal standalone WASM build of the Moka v1 port (Plan 565).
//!
//! Exists purely to answer one measurement question: is our native Rust
//! port of Moka's own network, compiled to WASM and run in a real browser,
//! faster than the real deployed Moka (which the same plan measured directly
//! at ~6.4 ms/move median in real Chrome — see `.docs/06_game_arenas/
//! go_arena.md`)? No `katgpt-core` dependency (so no ahash/getrandom wasm32
//! backend friction a browser deployment has no reason to carry), no game
//! engine beyond what's needed to generate realistic self-play positions.
//!
//! API shape deliberately mirrors the real Moka JS package
//! (`createGameState`/`encodeStudentFeatures`/`getLegalMoves`/`playMove`/
//! `selectHighestLegalMove`, from `github.com/millionco/moka`'s
//! `src/game.ts`) so the browser benchmark harness can drive both sides
//! with the same self-play loop shape.

mod board;
mod moka;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmGame {
    board: board::Board,
    history: Vec<Option<(usize, usize)>>,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            board: board::Board::new(),
            history: Vec::new(),
        }
    }

    /// Legal board-point move indices (0..81). Pass is always additionally
    /// legal but not listed here, matching the real Moka JS `getLegalMoves`.
    pub fn legal_moves(&self) -> Vec<u32> {
        self.board.legal_moves().into_iter().map(|i| i as u32).collect()
    }

    pub fn play(&mut self, idx: u32) {
        let idx = idx as usize;
        self.board.play(idx);
        self.history.push(Some((idx / board::SIZE, idx % board::SIZE)));
    }

    pub fn pass(&mut self) {
        self.board.pass();
        self.history.push(None);
    }

    /// Moka's 12-plane `9*9*12` HWC feature tensor for the current position.
    pub fn encode_features(&self) -> Vec<f32> {
        moka::encode_features(&self.board, &self.history)
    }
}

impl Default for WasmGame {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct WasmMoka {
    weights: moka::MokaWeights,
    scratch: moka::MokaScratch,
}

#[wasm_bindgen]
impl WasmMoka {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            weights: moka::MokaWeights::load(),
            scratch: moka::MokaScratch::new(),
        }
    }

    /// Returns 83 floats: 82 policy logits (81 board points + pass at index
    /// 81), then the value estimate as the last element.
    pub fn infer(&mut self, features: &[f32]) -> Vec<f32> {
        let (policy, value) = moka::forward_with_scratch(&self.weights, features, &mut self.scratch);
        let mut out = policy.to_vec();
        out.push(value);
        out
    }
}

impl Default for WasmMoka {
    fn default() -> Self {
        Self::new()
    }
}

/// Picks the legal move (or pass) with the highest policy logit — mirrors
/// the real Moka JS `selectHighestLegalMove`. `inference` is the 83-float
/// output of [`WasmMoka::infer`]; `legal` is [`WasmGame::legal_moves`]'s
/// output. Returns 81 for pass.
#[wasm_bindgen]
pub fn select_highest_legal_move(inference: &[f32], legal: &[u32]) -> u32 {
    let mut best_idx = 81u32;
    let mut best = inference[81];
    for &m in legal {
        let v = inference[m as usize];
        if v > best {
            best = v;
            best_idx = m;
        }
    }
    best_idx
}

// ── Raw C-ABI exports (Plan 565 Phase 4: wasmi comparison) ────────────────
//
// wasmi needs no JS interop at all, so these bypass wasm-bindgen's ABI
// entirely (which is designed for JS↔wasm marshalling, not native embedding)
// in favor of the same raw-linear-memory pattern this repo's own
// `wasm_pruner.rs` already uses with wasmi: caller allocates a buffer in
// wasm memory, writes floats into it, calls the compute function, reads the
// result back out. No JS runtime involved — this measures pure interpreted-
// wasm execution cost, isolated from any JS-boundary marshalling cost.

static mut WASMI_WEIGHTS: Option<moka::MokaWeights> = None;
static mut WASMI_SCRATCH: Option<moka::MokaScratch> = None;

/// Must be called once before `wasmi_infer`. Not thread-safe (fine — wasmi
/// benchmarking is single-threaded by construction, one `Store` per test).
#[unsafe(no_mangle)]
// `&raw mut` is intentional: it avoids forming a reference to the `static mut`
// (denied by Rust 2024). clippy::deref_addrof's suggested rewrite (`WASMI_WEIGHTS = ...`)
// would reintroduce that reference, so the lint is a false positive here.
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_init() {
    unsafe {
        *(&raw mut WASMI_WEIGHTS) = Some(moka::MokaWeights::load());
        *(&raw mut WASMI_SCRATCH) = Some(moka::MokaScratch::new());
    }
}

/// Allocates a `len`-f32 buffer in wasm linear memory and returns its
/// pointer, so a wasmi host can write input features into it directly.
#[unsafe(no_mangle)]
pub extern "C" fn wasmi_alloc(len: usize) -> *mut f32 {
    let mut v = vec![0f32; len];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// Runs one forward pass on the `POLICY_MOVES * 4` (features) input at
/// `features_ptr`, writing 83 floats (82 policy logits + value) to
/// `out_ptr`. Both pointers must come from [`wasmi_alloc`] (or otherwise be
/// valid, correctly-sized wasm-memory pointers) — this is a raw FFI
/// boundary, not a safe Rust API.
///
/// # Safety
/// `features_ptr` must point to at least `moka::INPUT_ELEMENT_COUNT` valid
/// `f32`s, and `out_ptr` to at least 83.
#[unsafe(no_mangle)]
// See `wasmi_init`: `&raw const`/`&raw mut` avoid forming references to the
// `static mut` (Rust 2024). clippy::deref_addrof is a false positive here.
#[allow(clippy::deref_addrof)]
pub unsafe extern "C" fn wasmi_infer(features_ptr: *const f32, out_ptr: *mut f32) {
    unsafe {
        let features = std::slice::from_raw_parts(features_ptr, moka::INPUT_ELEMENT_COUNT);
        // `&raw` avoids ever forming a reference to the `static mut` itself
        // (Rust 2024 denies that) — single-threaded wasm32, one Store per
        // benchmark, so the aliasing this would otherwise risk can't happen.
        let weights = (*(&raw const WASMI_WEIGHTS)).as_ref().expect("wasmi_init not called");
        let scratch = (*(&raw mut WASMI_SCRATCH)).as_mut().expect("wasmi_init not called");
        let (policy, value) = moka::forward_with_scratch(weights, features, scratch);
        let out = std::slice::from_raw_parts_mut(out_ptr, moka::POLICY_MOVES + 1);
        out[..moka::POLICY_MOVES].copy_from_slice(&policy);
        out[moka::POLICY_MOVES] = value;
    }
}
