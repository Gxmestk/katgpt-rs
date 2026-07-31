// SPDX-License-Identifier: MIT
//
// Vendored from gigatoken https://github.com/marcelroed/gigatoken (Marcel Rød).
// Upstream file: `src/token.rs`. Upstream commit at vendor time: master 2026-07-25.
// Adaptation: none — file is bit-identical to upstream.

use std::fmt::Formatter;

/// Token ID newtype wrapping `u32`. gigatoken's BPE cores operate on `u32`
/// IDs (vocabularies are < 2³²); the katgpt `BpeTokenizer` uses `usize`, so
/// `encode_fast` casts at the adapter boundary.
///
/// `#[repr(transparent)]` so `&[TokenId]` can be reinterpreted as `&[u32]`
/// for bulk `extend_from_slice` emits — see upstream `tiktoken::encode_into`.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
#[repr(transparent)]
pub struct TokenId(pub u32);

impl From<u32> for TokenId {
    fn from(value: u32) -> Self {
        TokenId(value)
    }
}

impl From<TokenId> for u32 {
    fn from(val: TokenId) -> Self {
        val.0
    }
}

impl From<usize> for TokenId {
    fn from(value: usize) -> Self {
        TokenId(value as u32)
    }
}

impl From<TokenId> for usize {
    fn from(val: TokenId) -> Self {
        val.0 as usize
    }
}

impl std::fmt::Debug for TokenId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("<{}>", self.0))
    }
}
