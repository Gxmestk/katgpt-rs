use super::types::{BpeTokenizer, MergeRule};
use std::collections::HashMap;

/// BPE encoder/decoder implementation.
pub struct BpeTokenizerImpl;

impl BpeTokenizerImpl {
    /// Encode a string into token IDs using BPE merge rules.
    ///
    /// Hot-path design: operates on `Vec<usize>` (token IDs) end-to-end. The
    /// merge-rank lookup uses `merge_ranks_id: HashMap<(usize, usize), usize>`
    /// — no `String` allocation per pair. The replacement ID is resolved via
    /// `merge_target_id[rank]` — no `vocab_to_id` lookup per merge pass.
    ///
    /// Per AGENTS.md hot-loop rules: no allocation inside the merge loop.
    /// The only allocation is the initial char→ID map and the two ping-pong
    /// token buffers, both pre-sized.
    pub fn encode(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Map each char to its token ID up front. Unknown chars map to `unk`.
        // Uses a fixed-size stack buffer (`encode_utf8` writes ≤4 bytes) —
        // zero heap allocation for the entire char→ID map step.
        let unk = tokenizer.unk_id();
        let char_count = text.chars().count();
        let mut tokens: Vec<usize> = Vec::with_capacity(char_count);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            let id = tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
            tokens.push(id);
        }

        // Fast path: no merges configured (or tables not rebuilt).
        if tokenizer.merge_ranks_id.is_empty() {
            return tokens;
        }

        let mut new_tokens: Vec<usize> = Vec::with_capacity(tokens.len());

        // Iteratively merge the highest-priority (lowest-rank) pair.
        loop {
            // Find the lowest-rank applicable merge across all adjacent pairs.
            // `windows(2)` lets LLVM drop the per-iteration bounds check on
            // `tokens[i + 1]` that the manual index loop forces.
            let mut best: Option<(usize, usize)> = None; // (rank, left_idx)
            for (i, w) in tokens.windows(2).enumerate() {
                if let Some(&rank) = tokenizer.merge_ranks_id.get(&(w[0], w[1])) {
                    match best {
                        Some((best_rank, _)) if best_rank <= rank => {}
                        _ => best = Some((rank, i)),
                    }
                }
            }

            let Some((best_rank, left_idx)) = best else {
                break;
            };

            // Resolve the merged token ID via the rank-indexed table — no
            // hashmap lookup, just a slice index.
            let merged_id = tokenizer.merge_target_id[best_rank];
            let left_id = tokens[left_idx];
            let right_id = tokens[left_idx + 1];

            // Apply the merge to all adjacent occurrences of (left, right).
            // Indices are `usize` (Copy) — zero allocation in this loop.
            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left_id && tokens[i + 1] == right_id {
                    new_tokens.push(merged_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        tokens
    }

    /// Decode token IDs back to string.
    pub fn decode(tokenizer: &BpeTokenizer, ids: &[usize]) -> String {
        let mut result = String::with_capacity(ids.len() * 4); // estimate ~4 bytes per token
        for &id in ids {
            match tokenizer.id_to_vocab.get(id) {
                Some(s) => result.push_str(s),
                None => result.push('\u{fffd}'), // replacement character
            }
        }
        result
    }

    /// Fast BPE encode path backed by gigatoken's pure-Rust merge cores
    /// (Issue 191, Research 456).
    ///
    /// **Prefer [`FastBpeEncoder`] over this function** — it caches the
    /// `PairRankTable` once per tokenizer and reuses it across calls. This
    /// function rebuilds the table per call, which dominates the cost on
    /// short inputs (see `.benchmarks/191_fast_bpe_goat.md` G2 short-input
    /// failure). Use this only when you can't amortize a single encoder
    /// across calls.
    ///
    /// Equivalent in semantics to [`encode`]: same input text → same output
    /// token ID sequence. Differs in throughput: builds a [`PairRankTable`]
    /// (dense grid + flat packed open-addressed table) from the tokenizer's
    /// merge rules once on first call, then runs the heap + doubly-linked-list
    /// merge loop with stack-resident scratch. The single-pass table build is
    /// NOT amortized across calls in this function — see [`FastBpeEncoder`]
    /// for the amortized variant.
    ///
    /// # When to use this vs [`encode`] vs [`FastBpeEncoder`]
    ///
    /// - **Use [`FastBpeEncoder`]** when you call `encode_fast` more than
    ///   once with the same tokenizer. Pays the table build once, reuses it
    ///   across calls. This is the production path.
    /// - **Use `encode_fast`** (this function) for one-shot calls where
    ///   the input is long enough (≥~1KB) that the algorithmic win covers
    ///   the table rebuild. The GOAT gate (`tests/fast_bpe_goat.rs`)
    ///   measures 162× speedup on 64KB inputs even WITH the per-call table
    ///   rebuild — so this is a real win on long inputs.
    /// - **Use [`encode`]** for one-shot short inputs (≤~16 chars): the
    ///   iterative-merge loop has lower constants on tiny inputs and there's
    ///   no table build cost.
    ///
    /// # Fallback path
    ///
    /// If the tokenizer's vocab exceeds the 21-bit packed-key lane
    /// (`vocab > 2²¹`), the table build refuses and `encode_fast` falls back
    /// to the slow HashMap-keyed merge loop (still using the heap +
    /// linked-list core, which is faster than [`encode`]'s iterative merge on
    /// long sequences). The fallback is exercised by
    /// `tests/fast_bpe_goat.rs::g1_bit_identical_to_encode_with_table_fallback`.
    #[cfg(feature = "fast_bpe")]
    pub fn encode_fast(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        FastBpeEncoder::from_tokenizer(tokenizer).encode(text)
    }
}

/// Cached fast-BPE encoder (Issue 191 Phase 2.5 — amortized `PairRankTable`).
///
/// Wraps a reference to a [`BpeTokenizer`] with a pre-built
/// [`PairRankTable`] + reusable merge scratch. Construct once per tokenizer
/// (the table build is the dominant cost on short inputs — see
/// `.benchmarks/191_fast_bpe_goat.md` G2 short-input failure for the
/// per-call `encode_fast` path), then call [`Self::encode`] as many times
/// as needed.
///
/// Semantics are bit-identical to [`BpeTokenizerImpl::encode`]. The win is
/// throughput on long inputs and across many calls.
///
/// # Fallback
///
/// If the tokenizer's vocab exceeds the 21-bit packed-key lane
/// (`vocab > 2²¹`), [`Self::from_tokenizer`] keeps `pair_ranks = None` and
/// `encode` falls back to the HashMap-keyed merge loop (still using the
/// heap + linked-list core, which is faster than the iterative-merge loop
/// on long sequences).
#[cfg(feature = "fast_bpe")]
pub struct FastBpeEncoder<'tok> {
    tokenizer: &'tok BpeTokenizer,
    /// Built once; reused across calls. `None` if the vocab is too large
    /// for the packed-key lane — `encode` falls back to the HashMap probe.
    pair_ranks: Option<crate::fast_bpe::PairRankTable>,
    /// `(TokenId, TokenId) → TokenId` map, kept for the fallback path and
    /// for table build. Owned here so the fallback closure can borrow it.
    merges: HashMap<(crate::fast_bpe::TokenId, crate::fast_bpe::TokenId), crate::fast_bpe::TokenId>,
    /// Reused across `encode` calls — no per-call allocation for the
    /// linked-list or merge heap after the first call.
    scratch: crate::fast_bpe::MergeScratch,
    /// Reused across `encode`/`encode_into` calls — the per-call `symbols`
    /// buffer (char→ID map input to the merge loop). Cleared + refilled
    /// each call; only reallocates when an input exceeds the prior peak.
    /// This is the Phase 2.5 G4 unblocker (Issue 191) — v1 allocated a
    /// fresh `Vec` per call.
    symbols: Vec<crate::fast_bpe::TokenId>,
}

#[cfg(feature = "fast_bpe")]
impl<'tok> FastBpeEncoder<'tok> {
    /// Build the encoder + `PairRankTable` (or fallback map) once. Pay this
    /// cost once per tokenizer; reuse across many `encode` calls.
    pub fn from_tokenizer(tokenizer: &'tok BpeTokenizer) -> Self {
        let mut merges: HashMap<
            (crate::fast_bpe::TokenId, crate::fast_bpe::TokenId),
            crate::fast_bpe::TokenId,
        > = HashMap::with_capacity(tokenizer.merge_ranks_id.len());
        for (&(l, r), &rank) in tokenizer.merge_ranks_id.iter() {
            // `merge_target_id[rank]` is the merged ID resolved at
            // `rebuild_ranks` time. `merge_ranks_id`'s rank values are
            // dense `0..merges.len()` by construction (see `rebuild_ranks`),
            // so the slice index is safe.
            let merged = tokenizer.merge_target_id[rank];
            merges.insert(
                (crate::fast_bpe::TokenId(l as u32), crate::fast_bpe::TokenId(r as u32)),
                crate::fast_bpe::TokenId(merged as u32),
            );
        }
        let pair_ranks = crate::fast_bpe::PairRankTable::build(&merges, tokenizer.id_to_vocab.len()).ok();
        FastBpeEncoder {
            tokenizer,
            pair_ranks,
            merges,
            scratch: crate::fast_bpe::MergeScratch::default(),
            symbols: Vec::new(),
        }
    }

    /// Encode `text` to token IDs, reusing the cached `PairRankTable` + scratch.
    /// Bit-identical to [`BpeTokenizerImpl::encode`].
    ///
    /// This allocates the returned `Vec<usize>` per call. For zero-alloc
    /// steady-state (e.g. hot loops, batched encoding), use
    /// [`Self::encode_into`] with a caller-owned reusable output buffer.
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Zero-alloc encode path — writes token IDs into the caller-owned `out`
    /// buffer (cleared first). After warmup (the first call seeds `symbols`
    /// and `scratch` to the prior peak), steady-state `encode_into` performs
    /// **zero heap allocations** for any input ≤ the prior peak size — the
    /// G4 gate (`tests/fast_bpe_goat.rs::g4_zero_alloc_steady_state`) audits
    /// this with a `CountingAllocator`.
    ///
    /// Bit-identical to [`BpeTokenizerImpl::encode`] + [`Self::encode`].
    pub fn encode_into(&mut self, text: &str, out: &mut Vec<usize>) {
        out.clear();
        if text.is_empty() {
            return;
        }

        // Map each char to its token ID up front — same as `encode`. The
        // `symbols` buffer is reused across calls; only reallocates if this
        // input exceeds the prior peak length.
        let unk = self.tokenizer.unk_id();
        let char_count = text.chars().count();
        self.symbols.clear();
        self.symbols.reserve(char_count);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            let id = self.tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
            self.symbols.push(crate::fast_bpe::TokenId(id as u32));
        }

        // Fast path: no merges configured.
        if self.tokenizer.merge_ranks_id.is_empty() {
            out.extend(self.symbols.iter().map(|t| t.0 as usize));
            return;
        }

        match &self.pair_ranks {
            Some(table) => {
                crate::fast_bpe::bpe_merge_symbols_by_rank(table, &mut self.symbols, &mut self.scratch);
            }
            None => {
                crate::fast_bpe::bpe_merge_symbols_by_rank_with_lookup(
                    &|a, b| self.merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                    &mut self.symbols,
                    &mut self.scratch,
                );
            }
        }

        out.extend(self.symbols.iter().map(|t| t.0 as usize));
    }

    /// Diagnostic: current capacity of the internal `symbols` scratch buffer.
    /// After warmup this reflects the peak input size the encoder has seen.
    /// Exposed so callers in hot loops can confirm steady-state capacity has
    /// been reached before relying on the G4 zero-alloc contract.
    #[doc(hidden)]
    pub fn symbols_capacity(&self) -> usize {
        self.symbols.capacity()
    }
}

/// BPE trainer: learns merge rules from a corpus.
pub struct BpeTrainer;

impl BpeTrainer {
    /// Train a BPE tokenizer from a text corpus.
    /// `vocab_size`: target vocabulary size (including special tokens).
    /// `corpus`: training text.
    pub fn train(corpus: &str, vocab_size: usize) -> BpeTokenizer {
        // Pre-allocate: 4 special tokens + up to 256 unique byte-chars + merges.
        let cap = 4usize.saturating_add(vocab_size).min(corpus.len() + 4);
        let mut vocab_to_id: HashMap<String, usize> = HashMap::with_capacity(cap);
        let mut id_to_vocab: Vec<String> = Vec::with_capacity(cap);

        // Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3
        const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, tok) in SPECIAL_TOKENS.iter().enumerate() {
            vocab_to_id.insert((*tok).to_string(), i);
            id_to_vocab.push((*tok).to_string());
        }

        // Add all unique characters from corpus.
        // Use the `entry` API to avoid the double-lookup (contains_key + insert).
        for ch in corpus.chars() {
            vocab_to_id.entry(ch.to_string()).or_insert_with(|| {
                let id = id_to_vocab.len();
                id_to_vocab.push(ch.to_string());
                id
            });
        }

        let mut merges: Vec<MergeRule> = Vec::new();
        let num_merges = vocab_size.saturating_sub(id_to_vocab.len());

        // Split corpus into words (simple whitespace splitting)
        let words: Vec<Vec<String>> = corpus
            .split_whitespace()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        // Learn merge rules
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for _ in 0..num_merges {
            // Count all adjacent pairs
            pair_counts.clear();
            for word in &words {
                let tokens = Self::apply_merges(word, &merges);
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            // Find most frequent pair
            let best_pair = pair_counts.drain().max_by_key(|(_, count)| *count);

            let Some((pair, count)) = best_pair else {
                break;
            };

            if count < 2 {
                break; // Stop if no pair appears more than once
            }

            let merged = format!("{}{}", pair.0, pair.1);

            // Add merged token to vocabulary via entry API (single hash lookup
            // instead of contains_key + insert).
            let id = *vocab_to_id.entry(merged.clone()).or_insert_with(|| {
                let id = id_to_vocab.len();
                id_to_vocab.push(merged.clone());
                id
            });
            let _ = id; // id is already tracked via the merges table below

            merges.push(MergeRule {
                left: pair.0,
                right: pair.1,
                merged,
            });
        }

        let mut tokenizer = BpeTokenizer {
            vocab_to_id,
            id_to_vocab,
            merges,
            merge_ranks: HashMap::new(),
            merge_ranks_id: HashMap::new(),
            merge_target_id: Vec::new(),
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        };
        tokenizer.rebuild_ranks();
        tokenizer
    }

    /// Apply existing merge rules to a sequence of tokens.
    fn apply_merges(tokens: &[String], merges: &[MergeRule]) -> Vec<String> {
        let mut buf_a = tokens.to_vec();
        let mut buf_b = Vec::with_capacity(tokens.len());
        for rule in merges {
            buf_b.clear();
            let mut i = 0;
            while i < buf_a.len() {
                if i + 1 < buf_a.len() && buf_a[i] == rule.left && buf_a[i + 1] == rule.right {
                    buf_b.push(rule.merged.clone());
                    i += 2;
                } else {
                    buf_b.push(buf_a[i].clone());
                    i += 1;
                }
            }
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
        buf_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = "hello world hello rust";
        let tokenizer = BpeTrainer::train(corpus, 64);
        let text = "hello";
        let ids = BpeTokenizerImpl::encode(&tokenizer, text);
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tokenizer = BpeTrainer::train("abc", 32);
        assert_eq!(tokenizer.pad_id, 0);
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
        // unk_id is the last vocab entry
        assert!(tokenizer.unk_id() >= 3);
        // Verify special tokens in vocab
        assert_eq!(tokenizer.vocab_to_id["<pad>"], 0);
        assert_eq!(tokenizer.vocab_to_id["<bos>"], 1);
        assert_eq!(tokenizer.vocab_to_id["<eos>"], 2);
        assert_eq!(tokenizer.vocab_to_id["<unk>"], 3);
    }

    #[test]
    fn test_bpe_train_produces_merges() {
        // Use a corpus with repeated patterns to guarantee merges
        let corpus = "ab ab ab ab ab ab ab ab ab ab";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // "a" + "b" → "ab" should be learned as a merge
        assert!(
            !tokenizer.merges.is_empty(),
            "Expected at least one merge rule from repeated 'ab' patterns"
        );
        // Verify the merge exists
        let has_ab_merge = tokenizer
            .merges
            .iter()
            .any(|m| m.left == "a" && m.right == "b" && m.merged == "ab");
        assert!(has_ab_merge, "Expected merge rule 'a'+'b'→'ab'");
    }

    #[test]
    fn test_bpe_vocab_coverage() {
        let corpus = "hello world";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // All characters from the corpus must be in the vocabulary
        for ch in corpus.chars() {
            let s = ch.to_string();
            assert!(
                tokenizer.vocab_to_id.contains_key(&s),
                "Character '{s}' missing from vocabulary"
            );
        }
    }

    #[test]
    fn test_bpe_encode_empty() {
        let tokenizer = BpeTrainer::train("hello", 32);
        let ids = BpeTokenizerImpl::encode(&tokenizer, "");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bpe_decode_unknown_id() {
        let tokenizer = BpeTrainer::train("hello", 32);
        // Use an out-of-range ID
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &[9999]);
        assert_eq!(decoded, "�");
    }
}
