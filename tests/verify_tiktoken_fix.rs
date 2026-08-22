//! Verify the fixed TiktokenTokenizer against the real Kimi-K3 model.
//! Run: cargo test --features kimi_k3_loader --test verify_tiktoken_fix -- --nocapture --ignored

// `kimi_k3::tiktoken` is gated behind `kimi_k3_loader`; without this guard
// `--all-targets` fails to resolve the import on a default-feature build.
#![cfg(feature = "kimi_k3_loader")]

use katgpt_rs::kimi_k3::tiktoken::{TiktokenTokenizer, load_tiktoken_bpe};

#[test]
fn verify_real_tokenizer_token_count() {
    let model_path = "data/kimi-k3-0.40b/tiktoken.model";
    let data = match std::fs::read(model_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: {model_path} not found ({e})");
            return;
        }
    };
    let ranks = load_tiktoken_bpe(&data).expect("parse tiktoken.model");
    let tok = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);

    eprintln!("Loaded {} base tokens", tok.vocab_size());

    // Test strings (matching the previous session's 5-example comparison)
    let tests = [
        "fn main() { println!(\"hello world\"); }",
        "The quick brown fox jumps over the lazy dog.",
        "use std::collections::HashMap;",
        "pub struct Foo<T> { x: T, y: Vec<u8> }",
        "In machine learning, gradient descent is an iterative optimization algorithm.",
    ];

    let mut total = 0usize;
    for text in &tests {
        let ids = tok.encode(text);
        eprintln!("  {:>4} tokens: {}", ids.len(), &text[..text.len().min(60)]);
        total += ids.len();

        // Verify roundtrip
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text, "roundtrip failed for: {text}");
    }

    eprintln!(
        "\nTotal: {} tokens across {} examples (avg {:.0}/ex)",
        total,
        tests.len(),
        total as f64 / tests.len() as f64
    );
    eprintln!("Previous session's Python tiktoken: 1487 total / 297 avg");
    eprintln!("Previous session's old Rust tok:     2932 total / 586 avg");

    // The fix should bring us close to Python tiktoken (~297 avg)
    let avg = total as f64 / tests.len() as f64;
    assert!(
        avg < 400.0,
        "avg token count {avg} should be < 400 (close to Python's ~297); got {avg}"
    );
}

#[test]
fn verify_exact_token_ids_match_python() {
    let model_path = "data/kimi-k3-0.40b/tiktoken.model";
    let data = match std::fs::read(model_path) {
        Ok(d) => d,
        Err(_) => return,
    };
    let ranks = load_tiktoken_bpe(&data).expect("parse");
    let tok = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);

    // Python tiktoken reference IDs (verified 2026-08-06 via tiktoken.Encoding
    // with the exact Kimi-K3 pat_str from tokenization_kimi.py):
    let cases: &[(&str, &[usize])] = &[
        (
            "fn main() { println!(\"hello world\"); }",
            &[10964, 2777, 539, 384, 47647, 28547, 22931, 2695, 11896, 457],
        ),
        (
            "pub struct Foo<T> { x: T, y: Vec<u8> }",
            &[
                13241, 2901, 52361, 6443, 29, 384, 1288, 25, 377, 11, 364, 25, 20431, 52794, 23,
                29, 457,
            ],
        ),
    ];

    for (text, expected) in cases {
        let got = tok.encode(text);
        assert_eq!(
            got.as_slice(),
            *expected,
            "token ID mismatch for: {text}\n  expected: {expected:?}\n  got:      {got:?}"
        );
    }
}
