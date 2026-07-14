use super::partial_parser::PartialParser;
use super::types::PruneResult;
use katgpt_core::ConstraintPruner;
use katgpt_tokenizer::BpeTokenizer;
use katgpt_tokenizer::BpeTokenizerImpl;
use std::cell::RefCell;
use std::sync::Arc;

// Per-thread scratch for the hot path.
//
// `parser` calls `reset()` at the start of every `is_valid()`, so it has no
// persistent state — sharing one instance per thread via `thread_local!` is
// safe and removes all lock contention between rayon workers building the
// DDTree in parallel. `scratch_tokens` is purely a transient token buffer;
// same reasoning.
thread_local! {
    static PARSER: RefCell<PartialParser> = const { RefCell::new(PartialParser::new()) };
    static SCRATCH_TOKENS: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(64));
}

/// Two-tier syntax pruner for Validator.
///
/// Tier 0: Bracket balancer DFA (PartialParser) — O(n), rejects clearly broken code.
/// Tier 1: `syn` parse attempt — accurate, but expensive. Only called if Tier 0 passes.
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
}

impl SynPruner {
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self { tokenizer }
    }

    /// Validate a complete code string through both tiers.
    pub fn validate(&self, code: &str) -> PruneResult {
        // Tier 0: Bracket balance
        let tier0_ok = PARSER.with(|p| p.borrow_mut().is_valid(code));
        if !tier0_ok {
            return PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::UnbalancedBrackets,
            };
        }

        // Tier 1: syn parse
        match syn::parse_str::<syn::Stmt>(code) {
            Ok(_) => PruneResult {
                is_valid: true,
                error_kind: super::types::ErrorKind::None,
            },
            Err(e) => PruneResult {
                is_valid: false,
                error_kind: super::types::ErrorKind::SynError(e.to_string()),
            },
        }
    }

    /// Quick Tier 0 check only (for DDTree hot path).
    pub fn is_valid_quick(&self, code: &str) -> bool {
        PARSER.with(|p| p.borrow_mut().is_valid(code))
    }
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Build the token sequence in thread-local scratch, decode to an owned
        // `code` String, then release the scratch borrow BEFORE touching the
        // parser. This drops one Mutex (scratch) entirely and removes the
        // previous nested-lock pattern (scratch held while acquiring parser).
        let code = SCRATCH_TOKENS.with(|s| {
            let mut s = s.borrow_mut();
            s.clear();
            s.extend_from_slice(parent_tokens);
            s.push(token_idx);
            BpeTokenizerImpl::decode(&self.tokenizer, &s)
        });

        // Only do Tier 0 (bracket balance) in the hot path.
        // Tier 1 (syn) is too expensive for every DDTree node.
        // Thread-local parser: no lock contention between rayon workers.
        // `PartialParser::is_valid` calls `reset()` at entry, so sharing one
        // instance per thread is safe (no persistent state across calls).
        PARSER.with(|p| p.borrow_mut().is_valid(&code))
    }

    #[cfg(feature = "hoare_pruner")]
    fn propagate(&mut self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let code = SCRATCH_TOKENS.with(|s| {
            let mut s = s.borrow_mut();
            s.clear();
            s.extend_from_slice(parent_tokens);
            s.push(token_idx);
            BpeTokenizerImpl::decode(&self.tokenizer, &s)
        });

        PARSER.with(|p| {
            let mut parser = p.borrow_mut();
            parser.reset();
            let valid = parser.is_valid(&code);

            const MAX_BRACKET_DEPTH: i32 = 32;
            valid && parser.total_depth() <= MAX_BRACKET_DEPTH
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syn_pruner_accepts_valid_rust() {
        let tokenizer = Arc::new(katgpt_tokenizer::BpeTrainer::train("fn let mut x", 64));
        let pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let x = 42;");
        assert!(result.is_valid, "expected valid for 'let x = 42;'");
        assert_eq!(result.error_kind, super::super::types::ErrorKind::None);

        let result = pruner.validate("fn main() { }");
        assert!(result.is_valid, "expected valid for 'fn main() {{ }}'");

        let result = pruner.validate("let s = \"hello\";");
        assert!(result.is_valid, "expected valid for string literal");
    }

    #[test]
    fn test_syn_pruner_rejects_invalid_rust() {
        let tokenizer = Arc::new(katgpt_tokenizer::BpeTrainer::train("fn let mut x", 64));
        let pruner = SynPruner::new(tokenizer);

        let result = pruner.validate("let = ;");
        assert!(!result.is_valid, "expected invalid for 'let = ;'");

        match result.error_kind {
            super::super::types::ErrorKind::SynError(msg) => {
                assert!(!msg.is_empty(), "syn error should have a message");
            }
            other => panic!("expected SynError, got {other:?}"),
        }
    }

    #[test]
    fn test_syn_pruner_bracket_tier_rejects() {
        let tokenizer = Arc::new(katgpt_tokenizer::BpeTrainer::train("fn let { }", 64));
        let pruner = SynPruner::new(tokenizer);

        // Unmatched closing brace — Tier 0 should reject before syn sees it
        let result = pruner.validate("fn main() { } }");
        assert!(!result.is_valid, "expected invalid for unbalanced braces");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );

        // Unmatched closing paren
        let result = pruner.validate("foo())");
        assert!(!result.is_valid, "expected invalid for unbalanced parens");
        assert_eq!(
            result.error_kind,
            super::super::types::ErrorKind::UnbalancedBrackets
        );
    }
}
