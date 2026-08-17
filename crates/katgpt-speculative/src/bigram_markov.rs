//! Bigram Markov head — the modelless sequential drafter (Issue 659, Research 316 §3.5 path 2).
//!
//! DSpark's sequential head, constructed deterministically from corpus bigram
//! co-occurrence statistics — no training, no gradient descent. This is the
//! Metal-viable alternative drafter for the case where the Prism-ML-trained
//! DSpark 6-layer drafter is switched off (Bench 656 failure mode 2: a separate
//! multi-layer drafter pays its own forward every step, which batch-1
//! verification cannot amortise on Apple Silicon).
//!
//! A bigram head is a **table lookup, not a forward pass** — it does not incur
//! mode 2 at all. Emission cost is O(steps × top_m) sparse writes (~128 writes
//! at top_m=16, lookahead 8 ≈ sub-µs) versus a 6-layer forward per step.
//!
//! # Construction (deterministic)
//!
//! - `BigramMarkovBuilder::add_sequence(&[u32])` accumulates `(prev, next)`
//!   pairs packed as `u64` (`prev << 32 | next`).
//! - `build(vocab_size, top_m)` sorts the pairs (total order on u64 — no
//!   reliance on sort stability), run-length counts each distinct bigram, and
//!   keeps the top-`m` successors per prev token by `(count desc, next asc)`.
//!   Same corpus → bit-identical table.
//! - Out-of-vocabulary pairs (either side ≥ `vocab_size`) are discarded
//!   deterministically during the build pass; row totals count in-vocab
//!   successors only, so stored probabilities are true conditionals over the
//!   in-vocab support and the stored top-m mass is ≤ 1.0 (the truncated tail
//!   is dropped mass, never renormalised).
//!
//! # Emission (zero-alloc, mirrors `dflash_predict_with`)
//!
//! `bigram_predict` fills a caller-owned [`BigramMarginalBuffer`] with
//! `steps × vocab` dense marginals, step-major (step `i` occupies
//! `[i*vocab .. (i+1)*vocab]`) — the exact layout `dflash_predict_with`
//! produces and [`crate::dd_tree::build_dd_tree`] consumes.
//!
//! The sparse rows are scattered into the buffer, and only the touched
//! entries are reset on the next call (the scoped-init-at-consume-site
//! pattern): steady-state emission performs **zero allocation and zero
//! O(vocab) work** — O(steps × top_m) writes plus O(|previous touched|)
//! resets. A dense fill would cost `steps × vocab × 4 B` (~4 MB at Bonsai
//! scale, lookahead 8) per draft cycle, which would dwarf the target-model
//! forward.
//!
//! ## Greedy-chain conditioning
//!
//! The dd_tree seam takes static per-depth marginals (`&[&[f32]]`), not
//! path-conditioned ones. The head therefore emits the **greedy-path
//! conditioning**: marginal at depth `k` is `P(· | greedy_token_{k-1})`
//! where `greedy_token_0 = last_token` and `greedy_token_{k}` is the argmax
//! successor of `greedy_token_{k-1}` (= `successors[0]`, ties broken by
//! token id ascending). Deeper branches of the tree still explore siblings
//! of the greedy path under the bigram distribution around the greedy
//! predecessor — the same position-indexed refinement DSpark's Markov head
//! applies over the parallel backbone.
//!
//! ## Unseen prev token → zero row (not uniform)
//!
//! A prev token with no in-vocab bigram evidence emits an **all-zero row**.
//! The tree builder skips `prob <= 0.0` candidates at every expansion site,
//! so a zero row proposes nothing — the honest drafter behaviour when the
//! head has no information (a uniform row would spend verification budget
//! on arbitrary tokens). Under best-first log-prob scoring a zero row is
//! rank-equivalent to uniform for every candidate it would admit anyway;
//! it simply admits none. The greedy chain stalls on the same prev token.
//!
//! # Memory bound (G5)
//!
//! Worst case (every row full): `(V+1 + V·m·2 + V) × 4 B`. At Bonsai scale
//! (`V = 131_072`, `m = 16`) that is **17,825,796 B ≈ 17 MB** — versus a
//! dense `V × V` table at ~68 GB and a DSpark low-rank `r=256` head at
//! ~268 MB.
//!
//! # Composition (Issue 659 design sketch)
//!
//! - Bebop entropy confidence (`crate::acceptance_forecast`) — shipped.
//! - Hardware-Aware Prefix Scheduler (`crate::prefix_scheduler`, Plan 339) — shipped.
//! - The Bonsai consumer + GOAT G2/G3 wall-clock gate belongs in riir-ai
//!   (Plan 528); this crate ships the primitive.
//!
//! Feature-gated behind `bigram_markov` — off by default until the riir-ai
//! Bonsai consumer gate proves the gain (Issue 659 T4).

use katgpt_core::speculative::types::TreeNode;

// ── Table ──────────────────────────────────────────────────────

/// CSR-style top-`m` bigram transition table: `P(next | prev)`.
///
/// Built once at load from corpus counts (deterministic); read-only on the
/// hot path. Rows are sorted by `(count desc, next asc)` — `successors[0]`
/// is the argmax successor. Note: rows are NOT token-sorted, so use
/// [`BigramMarkovTable::probability`] (linear scan, `m ≤ 64`) for point
/// lookups, not `binary_search`.
#[derive(Debug, Clone, PartialEq)]
pub struct BigramMarkovTable {
    vocab_size: usize,
    top_m: usize,
    /// `vocab_size + 1` entries; successors of `prev` live at
    /// `successors[row_offsets[prev]..row_offsets[prev + 1]]`.
    row_offsets: Vec<u32>,
    /// Successor token ids, row-major.
    successors: Vec<u32>,
    /// `P(next | prev)` aligned with `successors`. True conditional
    /// probabilities (count / row_total), NOT renormalised after top-m
    /// truncation — the stored mass is ≤ 1.0.
    probs: Vec<f32>,
    /// Total in-vocab bigram count per prev token (the normaliser,
    /// including the truncated tail).
    row_totals: Vec<u32>,
}

impl BigramMarkovTable {
    /// Top-`m` successors of `prev` and their probabilities, or `None` if
    /// `prev` is out of vocab or has no in-vocab bigram evidence.
    #[inline]
    pub fn successors(&self, prev: u32) -> Option<(&[u32], &[f32])> {
        let prev = prev as usize;
        if prev >= self.vocab_size {
            return None;
        }
        let start = self.row_offsets[prev] as usize;
        let end = self.row_offsets[prev + 1] as usize;
        if start == end {
            return None;
        }
        Some((&self.successors[start..end], &self.probs[start..end]))
    }

    /// `P(next | prev)` — `0.0` when the pair is absent (out of vocab,
    /// no evidence, or truncated out of top-m). Linear scan over the row
    /// (`m ≤ 64` entries, count-sorted — binary search is not applicable).
    #[inline]
    pub fn probability(&self, prev: u32, next: u32) -> f32 {
        let Some((succs, probs)) = self.successors(prev) else {
            return 0.0;
        };
        for (&s, &p) in succs.iter().zip(probs.iter()) {
            if s == next {
                return p;
            }
        }
        0.0
    }

    /// Total in-vocab bigram count for `prev` (0 when unseen).
    #[inline]
    pub fn row_total(&self, prev: u32) -> u32 {
        let prev = prev as usize;
        if prev >= self.vocab_size {
            return 0;
        }
        self.row_totals[prev]
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    pub fn top_m(&self) -> usize {
        self.top_m
    }

    /// Exact heap footprint: CSR offsets + successors + probs + row totals.
    ///
    /// At Bonsai scale (`vocab = 131_072`, `top_m = 16`, worst case every
    /// row full): 17,825,796 B ≈ 17 MB — versus a dense `V × V` table at
    /// ~68 GB and a low-rank `r=256` head at ~268 MB.
    pub fn memory_bytes(&self) -> usize {
        (self.row_offsets.len() + self.successors.len() + self.probs.len() + self.row_totals.len())
            * std::mem::size_of::<u32>()
    }
}

// ── Builder ────────────────────────────────────────────────────

/// Deterministic bigram-count builder. Accumulates `(prev, next)` pairs
/// packed as `u64`, then [`build`](BigramMarkovBuilder::build)s the CSR table
/// via sort + two pointer passes.
///
/// Build is offline (once at load); the 8-byte-per-pair working set is the
/// only memory cost. `add_sequence` contributes nothing for sequences
/// shorter than 2 tokens.
#[derive(Debug, Default, Clone)]
pub struct BigramMarkovBuilder {
    pairs: Vec<u64>,
}

impl BigramMarkovBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-reserve for `n_pairs` bigram pairs (a corpus of `T` tokens across
    /// `S` sequences yields `T - S` pairs).
    pub fn with_capacity(n_pairs: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(n_pairs),
        }
    }

    /// Add one token sequence's bigrams: `(t[0],t[1]), (t[1],t[2]), …`.
    pub fn add_sequence(&mut self, tokens: &[u32]) {
        for w in tokens.windows(2) {
            let packed = (w[0] as u64) << 32 | w[1] as u64;
            self.pairs.push(packed);
        }
    }

    /// Number of accumulated bigram pairs (pre-dedup).
    pub fn len_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Build the CSR top-`m` table. Out-of-vocab pairs (either side ≥
    /// `vocab_size`) are discarded deterministically (sorted prev-major,
    /// everything at prev ≥ `vocab_size` is never visited).
    ///
    /// Same corpus + same `(vocab_size, top_m)` → bit-identical table:
    /// `sort_unstable` on a total order, run-length counting, and
    /// `(count desc, next asc)` top-m insertion are all deterministic.
    pub fn build(self, vocab_size: usize, top_m: usize) -> BigramMarkovTable {
        let mut pairs = self.pairs;
        pairs.sort_unstable();

        // Pass 1: per-prev in-vocab totals (the probability normaliser).
        let mut row_totals = vec![0u32; vocab_size];
        {
            let mut i = 0usize;
            while i < pairs.len() {
                let prev = (pairs[i] >> 32) as usize;
                if prev >= vocab_size {
                    break; // prev-major sorted: the rest is all OOV
                }
                while i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                    if ((pairs[i] & 0xFFFF_FFFF) as usize) < vocab_size {
                        row_totals[prev] += 1;
                    }
                    i += 1;
                }
            }
        }

        // Pass 2: per-row top-m selection by (count desc, next asc).
        // `row_offsets[prev]` is advanced only by rows that store entries;
        // empty rows (unseen / fully-OOV prevs) get equal boundaries.
        let mut row_offsets = Vec::with_capacity(vocab_size + 1);
        let mut successors: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        // Running top-m buffer: (count, next), kept sorted count desc / next asc.
        let mut best: Vec<(u32, u32)> = Vec::with_capacity(top_m);

        let mut prefix = 0u32;
        row_offsets.push(prefix);
        let mut i = 0usize;
        for (prev, total) in row_totals.iter().enumerate() {
            if i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                best.clear();
                while i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                    let next = pairs[i] & 0xFFFF_FFFF;
                    let run_start = i;
                    while i < pairs.len() && pairs[i] == pairs[run_start] {
                        i += 1;
                    }
                    if (next as usize) < vocab_size {
                        let count = (i - run_start) as u32;
                        insert_top_m(&mut best, (count, next as u32), top_m);
                    }
                }
                for &(count, next) in &best {
                    successors.push(next);
                    probs.push(count as f32 / *total as f32);
                }
                prefix += best.len() as u32;
            }
            row_offsets.push(prefix);
        }

        BigramMarkovTable {
            vocab_size,
            top_m,
            row_offsets,
            successors,
            probs,
            row_totals,
        }
    }
}

/// Insert `(count, next)` into the running top-`m` buffer, keeping it sorted
/// by `(count desc, next asc)`. Deterministic: equal counts order by next asc.
fn insert_top_m(best: &mut Vec<(u32, u32)>, cand: (u32, u32), top_m: usize) {
    if top_m == 0 {
        return;
    }
    // Insertion position: before the first held entry that is strictly worse
    // than `cand` (lower count, or equal count and higher next).
    let mut pos = best.len();
    for (k, &(c, n)) in best.iter().enumerate() {
        if (cand.0 > c) || (cand.0 == c && cand.1 < n) {
            pos = k;
            break;
        }
    }
    if pos == best.len() {
        if best.len() < top_m {
            best.push(cand);
        }
        return;
    }
    if best.len() < top_m {
        best.insert(pos, cand);
    } else {
        // Full: shift the strictly-worse tail right, dropping the old worst.
        let last = best.len() - 1;
        best.copy_within(pos..last, pos + 1);
        best[pos] = cand;
        best.truncate(top_m);
    }
}

// ── Emission buffer ────────────────────────────────────────────

/// Caller-owned persistent marginal buffer enforcing the zero-outside-touched
/// invariant: every entry not written by the last [`bigram_predict`] call is
/// `0.0`.
///
/// Allocate once per session (`steps × vocab × 4 B`, ~4 MB at Bonsai scale,
/// lookahead 8); reuse across draft cycles. The buffer starts zeroed and
/// [`bigram_predict`] resets only the entries it previously touched, so
/// steady-state emission is allocation-free and O(steps × top_m).
#[derive(Debug, Clone)]
pub struct BigramMarginalBuffer {
    flat: Vec<f32>,
    /// Flat indices written by the last predict call (to reset next call).
    touched: Vec<u32>,
    steps: usize,
    vocab_size: usize,
}

impl BigramMarginalBuffer {
    /// Zeroed buffer for `steps` draft positions × `vocab_size` tokens.
    /// Pre-reserves `steps × 16` touched slots (the typical top_m); a larger
    /// table top_m grows it once on first predict.
    pub fn new(steps: usize, vocab_size: usize) -> Self {
        Self {
            flat: vec![0.0; steps * vocab_size],
            touched: Vec::with_capacity(steps * 16),
            steps,
            vocab_size,
        }
    }

    /// The full flat marginal slab, step-major: step `i` occupies
    /// `[i*vocab .. (i+1)*vocab]`.
    #[inline]
    pub fn marginals(&self) -> &[f32] {
        &self.flat
    }

    /// One step's dense marginal row.
    #[inline]
    pub fn row(&self, step: usize) -> &[f32] {
        let v = self.vocab_size;
        &self.flat[step * v..(step + 1) * v]
    }

    #[inline]
    pub fn steps(&self) -> usize {
        self.steps
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of sparse entries the last predict call wrote (diagnostic).
    #[inline]
    pub fn touched_len(&self) -> usize {
        self.touched.len()
    }
}

// ── Emission ───────────────────────────────────────────────────

/// Emit greedy-chain bigram marginals into `buf` (zero-alloc steady state).
///
/// Fills `steps` dense rows (step `i` = `P(· | greedy_token_{i-1})`, greedy
/// chain seeded at `last_token`; see the [module docs](self) for the
/// conditioning + zero-row semantics). Returns the number of rows populated
/// (= `max_steps.min(buf.steps())`).
///
/// O(steps × top_m) writes + O(previous |touched|) resets after the first
/// call. `buf` must have been created with the same `vocab_size` as `table`
/// (debug-asserted).
pub fn bigram_predict(
    table: &BigramMarkovTable,
    last_token: u32,
    max_steps: usize,
    buf: &mut BigramMarginalBuffer,
) -> usize {
    debug_assert_eq!(
        buf.vocab_size, table.vocab_size,
        "BigramMarginalBuffer vocab must match the table"
    );
    let vocab = buf.vocab_size;
    let steps = max_steps.min(buf.steps);

    // Reset only the entries the previous call touched (the buffer's
    // zero-outside-touched invariant). First call: touched is empty.
    for &idx in buf.touched.iter() {
        buf.flat[idx as usize] = 0.0;
    }
    buf.touched.clear();

    let mut prev = last_token;
    for step in 0..steps {
        let row_start = step * vocab;
        if let Some((succs, probs)) = table.successors(prev) {
            for (&next, &p) in succs.iter().zip(probs.iter()) {
                let idx = row_start + next as usize;
                buf.flat[idx] = p;
                buf.touched.push(idx as u32);
            }
            // Rows are (count desc, next asc): successors[0] is the argmax.
            prev = succs[0];
        }
        // Unseen prev → zero row (proposes nothing; see module docs). The
        // greedy chain stalls on the same token.
    }
    steps
}

/// Build a DDTree verification tree from bigram marginals (Issue 659 T3 —
/// the `build_dd_tree` seam wiring).
///
/// Emits `config.draft_lookahead` greedy-chain marginals into `buf`, then
/// feeds them to [`crate::dd_tree::build_dd_tree`] as `&[&[f32]]` rows.
/// The row-slice `Vec` is allocated here (tree building allocates anyway —
/// this is not the per-token hot path; [`bigram_predict`] is).
///
/// Tokens proposed by the returned tree are exactly the tokens the seam's
/// best-first search admits from the bigram rows: siblings of the greedy
/// chain under `P(· | greedy predecessor)`. Zero rows (unseen prev) end the
/// draft — no candidates are proposed past that depth.
pub fn bigram_build_tree(
    table: &BigramMarkovTable,
    last_token: u32,
    config: &katgpt_types::Config,
    buf: &mut BigramMarginalBuffer,
) -> Vec<TreeNode> {
    let steps = bigram_predict(table, last_token, config.draft_lookahead, buf);
    let vocab = buf.vocab_size;
    let rows: Vec<&[f32]> = buf.marginals().chunks_exact(vocab).take(steps).collect();
    crate::dd_tree::build_dd_tree(&rows, config)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference (brute-force) construction: HashMap counting + full sort.
    fn reference_table(corpus: &[&[u32]], vocab_size: usize, top_m: usize) -> BigramMarkovTable {
        use std::collections::HashMap;
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for seq in corpus {
            for w in seq.windows(2) {
                if (w[0] as usize) < vocab_size && (w[1] as usize) < vocab_size {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
        }
        let mut row_offsets = vec![0u32; vocab_size + 1];
        let mut successors = Vec::new();
        let mut probs = Vec::new();
        let mut row_totals = vec![0u32; vocab_size];
        for prev in 0..vocab_size as u32 {
            let mut row: Vec<(u32, u32)> = counts
                .iter()
                .filter(|(k, _)| k.0 == prev)
                .map(|(k, &c)| (c, k.1))
                .collect();
            row_totals[prev as usize] = row.iter().map(|&(c, _)| c).sum();
            row.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            row.truncate(top_m);
            for &(c, n) in &row {
                successors.push(n);
                probs.push(c as f32 / row_totals[prev as usize] as f32);
            }
            row_offsets[prev as usize + 1] = successors.len() as u32;
        }
        BigramMarkovTable {
            vocab_size,
            top_m,
            row_offsets,
            successors,
            probs,
            row_totals,
        }
    }

    fn toy_corpus() -> Vec<Vec<u32>> {
        vec![
            vec![0, 1, 2, 3, 0, 1, 2, 0, 1],
            vec![1, 2, 3, 4, 1, 2, 3, 4, 0],
            vec![5, 0, 1, 2, 0, 1, 2, 3],
        ]
    }

    fn build_toy(vocab: usize, top_m: usize) -> BigramMarkovTable {
        let mut b = BigramMarkovBuilder::new();
        for seq in &toy_corpus() {
            b.add_sequence(seq);
        }
        b.build(vocab, top_m)
    }

    #[test]
    fn g1_build_deterministic_bit_identical() {
        let a = build_toy(16, 4);
        let b = build_toy(16, 4);
        assert_eq!(a, b, "same corpus must produce a bit-identical table");
    }

    #[test]
    fn g1_top_m_matches_bruteforce_reference() {
        for top_m in [1usize, 2, 4, 8] {
            let corpus_vec = toy_corpus();
            let corpus: Vec<&[u32]> = corpus_vec.iter().map(|v| v.as_slice()).collect();
            let mine = build_toy(8, top_m);
            let reference = reference_table(&corpus, 8, top_m);
            assert_eq!(mine, reference, "top_m={top_m}");
        }
    }

    #[test]
    fn g1_row_offsets_sparse_prevs() {
        // Toy corpus prevs: 0,1,2,3,4,5 all appear; rows 6,7 must be empty
        // with equal boundaries.
        let t = build_toy(8, 4);
        assert_eq!(t.row_offsets_len(), 9);
        for prev in [6usize, 7] {
            let (s, e) = t.row_bounds(prev);
            assert_eq!(s, e, "row {prev} must be empty");
            assert!(t.successors(prev as u32).is_none());
        }
        // Every non-empty row's range is exactly the stored slice.
        for prev in 0..8u32 {
            if let Some((succs, _)) = t.successors(prev) {
                let (s, e) = t.row_bounds(prev as usize);
                assert_eq!(e - s, succs.len());
            }
        }
    }

    #[test]
    fn g1_oov_pairs_discarded() {
        // vocab 4: tokens 4+ are OOV.
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 9, 1, 2, 9, 3]);
        let t = b.build(4, 4);
        // In-vocab pairs only: (0,1), (1,2). (1,9),(9,1),(2,9),(9,3) dropped.
        assert_eq!(t.row_total(0), 1);
        assert_eq!(t.row_total(1), 1);
        assert_eq!(t.row_total(2), 0, "(2,9) is OOV — no evidence");
        assert_eq!(t.row_total(9), 0, "OOV prev discards everything");
        assert_eq!(t.successors(2), None);
        assert_eq!(t.successors(0).unwrap().0, &[1]);
    }

    #[test]
    fn g1_tie_break_count_desc_then_token_asc() {
        // prev 0 → next 3 (×2), next 1 (×2), next 2 (×1): top-2 must be
        // (2, token 1), (2, token 3) — count desc, then token asc.
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 3, 0, 1, 0, 3, 0, 1, 0, 2]);
        let t = b.build(8, 2);
        let (succs, probs) = t.successors(0).unwrap();
        assert_eq!(succs, &[1, 3]);
        assert!((probs[0] - 2.0 / 5.0).abs() < 1e-7);
        assert!((probs[1] - 2.0 / 5.0).abs() < 1e-7);
    }

    #[test]
    fn g2_marginal_rows_sub_stochastic() {
        let t = build_toy(8, 2);
        let mut buf = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf);
        for step in 0..4 {
            let row = buf.row(step);
            let sum: f32 = row.iter().sum();
            assert!(
                sum <= 1.0 + 1e-6,
                "step {step}: stored mass {sum} must be ≤ 1.0 (truncated tail dropped)"
            );
        }
    }

    #[test]
    fn g2_greedy_chain_follows_argmax_successor() {
        let t = build_toy(8, 4);
        let mut buf = BigramMarginalBuffer::new(3, 8);
        bigram_predict(&t, 0, 3, &mut buf);

        let mut prev = 0u32;
        for step in 0..3 {
            let (succs, probs) = t.successors(prev).expect("toy corpus covers 0,1,2");
            let row = buf.row(step);
            // Every stored successor appears with its true probability.
            for (&next, &p) in succs.iter().zip(probs.iter()) {
                assert_eq!(row[next as usize], p, "step {step} token {next}");
            }
            // The row argmax is successors[0] (count desc, token asc; the
            // toy corpus has no count ties on the 0→1→2 chain).
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(argmax as u32, succs[0], "step {step}");
            prev = succs[0];
        }
    }

    #[test]
    fn g2_unseen_prev_emits_zero_row_and_chain_stalls() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 2, 3]);
        let t = b.build(8, 4);
        let mut buf = BigramMarginalBuffer::new(3, 8);
        // Token 6 unseen: all rows zero, chain stalls, nothing written.
        bigram_predict(&t, 6, 3, &mut buf);
        for step in 0..3 {
            assert!(buf.row(step).iter().all(|&p| p == 0.0), "step {step}");
        }
        assert_eq!(buf.touched_len(), 0, "zero rows write nothing");

        // Chain 0→1→2→3: token 3 has no successors → row 3 is zero, stall.
        let mut buf2 = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf2);
        assert!(buf2.row(0).iter().any(|&p| p > 0.0));
        assert!(buf2.row(3).iter().all(|&p| p == 0.0));
    }

    #[test]
    fn g2_probability_accessor() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 0, 1, 0, 2]);
        let t = b.build(4, 4);
        // prev 0: next 1 ×2, next 2 ×1 → total 3.
        assert!((t.probability(0, 1) - 2.0 / 3.0).abs() < 1e-7);
        assert!((t.probability(0, 2) - 1.0 / 3.0).abs() < 1e-7);
        assert_eq!(t.probability(0, 3), 0.0, "no evidence");
        assert_eq!(t.probability(9, 1), 0.0, "OOV prev");
        // (1,0) ×2 is the only pair with prev 1 → P(0|1) = 1.0.
        assert!((t.probability(1, 0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn g3_tree_wiring_smoke() {
        let t = build_toy(8, 4);
        // Config::micro: draft_lookahead 8, tree_budget 16, vocab 27.
        let config = katgpt_types::Config::micro();
        let mut buf = BigramMarginalBuffer::new(config.draft_lookahead, 8);
        let tree = bigram_build_tree(&t, 0, &config, &mut buf);
        assert!(
            !tree.is_empty(),
            "toy corpus has strong 0→1 signal; tree must propose"
        );
        // Depth-0 nodes must be successors of token 0.
        let (succs, _) = t.successors(0).unwrap();
        for node in tree.iter().filter(|n| n.depth == 0) {
            assert!(succs.contains(&(node.token_idx as u32)));
        }
        // The greedy chain (argmax successor per depth) must be present.
        let mut prev = 0u32;
        for depth in 0..4 {
            let Some((s, _)) = t.successors(prev) else {
                break;
            };
            let chain_token = s[0] as usize;
            assert!(
                tree.iter().any(|n| n.depth == depth && n.token_idx == chain_token),
                "greedy token {chain_token} missing at depth {depth}"
            );
            prev = s[0];
        }
    }

    #[test]
    fn g3_buffer_reuse_content_correct() {
        // Two predicts into the same buffer: entries from call 1 that are not
        // rewritten by call 2 must be reset to 0 (the touched invariant).
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 0, 2, 1, 3, 2, 3]);
        let t = b.build(4, 4);
        let mut buf = BigramMarginalBuffer::new(1, 4);
        bigram_predict(&t, 0, 1, &mut buf);
        assert!(buf.row(0)[1] > 0.0 && buf.row(0)[2] > 0.0);
        bigram_predict(&t, 1, 1, &mut buf);
        // Call 2 writes token 3 only; tokens 1, 2 must be reset.
        assert_eq!(buf.row(0)[1], 0.0);
        assert_eq!(buf.row(0)[2], 0.0);
        assert!(buf.row(0)[3] > 0.0);
    }

    #[test]
    fn g3_build_tree_unseen_root_empty_tree() {
        // Unseen root → all-zero rows → the seam proposes nothing.
        let t = build_toy(8, 4);
        let config = katgpt_types::Config::micro();
        let mut buf = BigramMarginalBuffer::new(config.draft_lookahead, 8);
        let tree = bigram_build_tree(&t, 7, &config, &mut buf);
        assert!(tree.is_empty(), "zero rows must yield an empty tree");
    }

    /// G4 — steady-state emission performs zero allocations. A thread-local
    /// tracking allocator counts allocations made by THIS thread while the
    /// probe is armed (immune to parallel-test noise from other threads).
    #[test]
    fn g4_predict_zero_alloc_steady_state() {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static TRACK: Cell<bool> = const { Cell::new(false) };
            static COUNT: Cell<usize> = const { Cell::new(0) };
        }

        struct Counting;
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                if TRACK.with(|t| t.get()) {
                    COUNT.with(|c| c.set(c.get() + 1));
                }
                unsafe { System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }
        }
        #[global_allocator]
        static A: Counting = Counting;

        let t = build_toy(8, 4);
        let mut buf = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf); // warm-up (touched may grow)

        TRACK.with(|t| t.set(true));
        COUNT.with(|c| c.set(0));
        let steps = bigram_predict(&t, 1, 4, &mut buf);
        TRACK.with(|t| t.set(false));
        let allocs = COUNT.with(|c| c.get());
        assert_eq!(steps, 4);
        assert_eq!(
            allocs, 0,
            "steady-state bigram_predict must be allocation-free"
        );
        assert!(buf.touched_len() <= 4 * t.top_m());
    }

    #[test]
    fn g5_memory_bytes_formula() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 2, 3]);
        let t = b.build(8, 4);
        // Exact accounting: offsets (vocab+1) + successors + probs + totals.
        let n_stored: usize = (0..8u32)
            .map(|p| t.successors(p).map_or(0, |s| s.0.len()))
            .sum();
        let exact = (9 + n_stored + n_stored + 8) * 4;
        assert_eq!(n_stored, 3, "(0,1),(1,2),(2,3)");
        assert_eq!(t.memory_bytes(), exact);

        // Bonsai-scale worst-case projection (documented, not built):
        // vocab 131_072, top_m 16, every row full.
        let bonsai = (131_073 + 131_072 * 16 * 2 + 131_072) * 4;
        assert_eq!(bonsai, 17_825_796, "≈17.0 MiB — the G5 bound claim");
    }

    /// Release-mode cost probe: the mode-2 avoidance made concrete. Sparse
    /// emission at Bonsai scale (vocab 131_072, top_m 16, lookahead 8) must
    /// run in ~µs — versus a 6-layer drafter forward per step (~129 µs/step
    /// per token at the Bench 661 E overhead scale).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn release_bonsai_scale_emission_cost_probe() {
        // Deterministic synthetic corpus: concentrated successor locality.
        let vocab: usize = 131_072;
        let seqs = 64;
        let len = 4_096;
        let mut b = BigramMarkovBuilder::with_capacity(seqs * len);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15; // golden-ratio seed
        let mut xorshift = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..seqs {
            let mut tokens = Vec::with_capacity(len);
            let mut cur = (xorshift() % vocab as u64) as u32;
            tokens.push(cur);
            for _ in 1..len {
                let r = xorshift();
                let next = if r % 4 == 0 {
                    (xorshift() % vocab as u64) as u32
                } else {
                    cur.wrapping_add(1 + (r % 17) as u32) % vocab as u32
                };
                tokens.push(next);
                cur = next;
            }
            b.add_sequence(&tokens);
        }
        let t = b.build(vocab, 16);
        let mut buf = BigramMarginalBuffer::new(8, vocab);
        bigram_predict(&t, 0, 8, &mut buf); // warm-up

        let iters = 1_000;
        let start = std::time::Instant::now();
        for i in 0..iters {
            bigram_predict(&t, (i % 128) as u32, 8, &mut buf);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        let per_step = per_call / 8.0;
        println!(
            "bigram emission @ vocab={vocab} top_m=16 lookahead=8: \
             {per_call:.0} ns/call ({per_step:.0} ns/step, {} touched)",
            buf.touched_len()
        );
        // Generous regression bound: sparse emission must stay µs-scale (a
        // dense fill alone would be ~500 µs/call at this scale).
        assert!(
            per_call < 100_000.0,
            "sparse emission must stay µs-scale, got {per_call:.0} ns/call"
        );
    }

    // Small accessors used by g1_row_offsets_sparse_prevs.
    impl BigramMarkovTable {
        fn row_offsets_len(&self) -> usize {
            self.row_offsets.len()
        }
        fn row_bounds(&self, prev: usize) -> (usize, usize) {
            (
                self.row_offsets[prev] as usize,
                self.row_offsets[prev + 1] as usize,
            )
        }
    }
}
