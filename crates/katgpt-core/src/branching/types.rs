//! Type definitions for non-interference memory branches (Plan 329 T1.2).
//!
//! All types are `Clone + Debug`. Pod-compatible types (no `Vec`, no generic
//! payload) are `#[repr(C)]` for sync-friendly layout. The owning containers
//! (`Vec`-backed stores) allocate only at construction / write time, never on
//! the read-side hot path.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `BranchId` | **Raw** | YES (deterministic dense index) |
//! | `BranchLifecycle` | **Raw** | YES (deterministic enum) |
//! | `BranchStats` | **Raw** | YES (deterministic counters) |
//! | `ProceduralRule` (counters) | **Raw** | YES |
//! | `ProceduralRule.direction` | **Latent** | NO (projection vector) |
//! | `EpisodicEntry.embedding` | **Latent** | NO |
//! | `EpisodicEntry.payload` | **Caller-defined** | Caller decides |
//! | `EpisodicEntry.reward` | **Raw** | YES |
//! | `spawn_anchor` | **Latent** | NO (direction vector) |
//! | `token_signature` | **Raw** | YES (deterministic hashes) |

use core::fmt;

// ─── Branch identifier ────────────────────────────────────────────────────

/// Dense index into a [`crate::branching::bank::BranchBank`].
///
/// `#[repr(transparent)]` over `u32` so an array of `BranchId` is byte-compatible
/// with `&[u32]`. Stable for the lifetime of the slot: a pruned branch keeps its
/// `BranchId`; a reused slot inherits the slot's `BranchId`. Callers that need
/// tamper-evident continuity across prune-reuse MUST consult the ARG
/// `RedirectTable` (when `arg_protocol` is enabled) — `BranchId` alone does not
/// distinguish "old branch at this slot" from "reused branch at this slot".
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BranchId(pub u32);

impl BranchId {
    /// Construct from a raw `u32`.
    #[inline]
    #[must_use]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Sentinel for "no branch". Used as a non-match marker by the router;
    /// never a valid index into a `BranchBank`.
    pub const SENTINEL: Self = Self(u32::MAX);

    /// True if this is the sentinel value.
    #[inline]
    pub const fn is_sentinel(self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<u32> for BranchId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<BranchId> for u32 {
    #[inline]
    fn from(id: BranchId) -> u32 {
        id.0
    }
}

impl fmt::Display for BranchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_sentinel() {
            write!(f, "branch:SENTINEL")
        } else {
            write!(f, "branch:{}", self.0)
        }
    }
}

// ─── Branch lifecycle ─────────────────────────────────────────────────────

/// Lifecycle state for a cognitive branch.
///
/// When the `arg_protocol` feature is on, this is a type alias for
/// [`crate::arg::LifecycleState`] — the same type used by the ARG protocol's
/// ontology lifecycle (Step E). When `arg_protocol` is off, a local enum with
/// identical discriminants and semantics is provided so this module compiles
/// standalone.
///
/// Progression is monotonic: `Active → Deprecated → Removed`. `Shadow` is the
/// pre-promotion staging state (reduces blast radius during early adoption).
#[cfg(feature = "arg_protocol")]
pub type BranchLifecycle = crate::arg::LifecycleState;

#[cfg(not(feature = "arg_protocol"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BranchLifecycle {
    /// Visible and routable. The default for newly spawned branches.
    #[default]
    Active = 0,
    /// Pre-promotion staging: suggested but with limited routing weight.
    Shadow = 1,
    /// Superseded by a merged replacement; still resolvable via redirect.
    Deprecated = 2,
    /// Permanently gone. The slot may be reused; the branch id does not
    /// survive reuse. Lookups against a `Removed` branch MUST consult the
    /// `RedirectTable` (when `arg_protocol` is enabled).
    Removed = 3,
}

#[cfg(not(feature = "arg_protocol"))]
impl BranchLifecycle {
    /// Returns `true` when the branch is routable online (router may snap to it).
    #[inline]
    pub fn is_routable(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns `true` when lookups MUST consult a redirect table before
    /// returning (continuity requirement mirroring ARG §3.5).
    #[inline]
    pub fn requires_redirect(self) -> bool {
        matches!(self, Self::Deprecated | Self::Removed)
    }
}

// ─── Episodic codec (caller-supplied E serializer) ────────────────────────
//
// `BranchBank<E>` / `CognitiveBranch<E>` / `EpisodicEntry<E>` /
// `FailureEntry<E>` are generic over the caller-defined episodic payload type
// `E`. Migration serialization needs to know how to encode/decode `E` — the
// `EpisodicCodec` trait is the decoupled seam. The caller (riir-ai's
// `BranchEpisodicPayload`) implements it; katgpt-core stays agnostic to the
// concrete payload shape.
//
// # Determinism contract
//
// `encode_to` MUST be deterministic (same `E` → same bytes). The migration
// envelope's Merkle root is BLAKE3 over the serialized bytes; non-determinism
// breaks the G4 determinism gate.
//
// # Wire format
//
// The codec is responsible for length-prefixing itself. The reader side
// (`decode_from`) consumes exactly the bytes the writer side wrote and returns
// the unconsumed tail. This lets `BranchBank` compose the codec into a larger
// length-prefixed framing without knowing `E`'s shape.
//
// # Blanket impls
//
// No blanket impl is provided. The production `BranchEpisodicPayload` carries
// a data-bearing enum (`ActionClass::Other(u8)`) that breaks `Pod`, so a
// blanket over `bytemuck::Pod` wouldn't cover it anyway. Each caller writes a
// small manual impl. Unit-type `E = ()` has a manual impl here as the
// reference codec (and the unit-test codec).

/// Caller-supplied encode/decode for the `E` payload in `EpisodicEntry<E>` /
/// `FailureEntry<E>`.
///
/// Implement this on your concrete `E` type to enable migration serialization
/// of `NpcCognitiveBranches<E>` / `BranchBank<E>`. See the module docs above
/// for the determinism + framing contract.
pub trait EpisodicCodec: Sized {
    /// Append the canonical encoding of `self` to `out`. MUST be deterministic.
    fn encode_to(&self, out: &mut Vec<u8>);

    /// Decode `Self` from `bytes`, returning the decoded value and the
    /// unconsumed tail. Returns `None` on any structural failure (truncated,
    /// bad magic, etc.).
    fn decode_from(bytes: &[u8]) -> Option<(Self, &[u8])>;
}

/// Reference codec for `E = ()` — the minimal episodic payload (pure
/// direction-based memory, no payload content). Encodes as zero bytes.
impl EpisodicCodec for () {
    #[inline]
    fn encode_to(&self, _out: &mut Vec<u8>) {}

    #[inline]
    fn decode_from(bytes: &[u8]) -> Option<(Self, &[u8])> {
        Some(((), bytes))
    }
}

// ─── Episodic entry ───────────────────────────────────────────────────────

/// A verifier-approved episodic example stored in a branch.
///
/// `embedding` is the latent vector at write time (used for centroid / quarantine
/// checks). `payload` is the caller-defined content (e.g., an Engram shard ref,
/// a closure motif, a game-state snapshot). `reward` is the verifier score that
/// admitted the write. `scope` is an optional caller-defined scope tag
/// (e.g., task family id). `tick` is the deterministic write tick.
#[derive(Clone, Debug)]
pub struct EpisodicEntry<E> {
    /// Latent embedding at write time (caller-normalized).
    pub embedding: Vec<f32>,
    /// Caller-defined payload (Engram ref, motif, snapshot, ...).
    pub payload: E,
    /// Verifier reward `r ∈ [0,1]` that admitted this write.
    pub reward: f32,
    /// Optional caller-defined scope tag (e.g., task family id).
    pub scope: Option<u64>,
    /// Deterministic write tick (raw, syncable).
    pub tick: u64,
}

// ─── Procedural rule ──────────────────────────────────────────────────────

/// An IF-THEN procedural rule with helpful / harmful counters.
///
/// Distilled from RIZZ §"procedural rules" `(u_j, α_j, β_j, H_j, A_j)`:
/// - `direction` is the latent direction this rule fires on (dot-product gate).
/// - `antecedent` is a BLAKE3 commitment of the rule's precondition (the "IF").
/// - `strategy` is a BLAKE3 commitment of the rule's action (the "THEN").
/// - `helpful` counts how often firing this rule improved the outcome.
/// - `harmful` counts how often firing this rule worsened the outcome.
///
/// The net credit is `helpful - harmful`; the rule is pruned when net credit
/// falls below zero for a sustained window. This is the procedural analogue of
/// the CLR reward gate.
#[derive(Clone, Debug)]
pub struct ProceduralRule {
    /// Latent direction this rule fires on (caller-normalized).
    pub direction: Vec<f32>,
    /// BLAKE3 commitment of the precondition (the "IF" side).
    pub antecedent: [u8; 32],
    /// BLAKE3 commitment of the action (the "THEN" side).
    pub strategy: [u8; 32],
    /// Count of outcome-improving firings.
    pub helpful: u32,
    /// Count of outcome-worsening firings.
    pub harmful: u32,
}

impl ProceduralRule {
    /// Net credit (`helpful - harmful`). Positive = keep; negative = prune candidate.
    #[inline]
    #[must_use]
    pub fn net_credit(&self) -> i64 {
        self.helpful as i64 - self.harmful as i64
    }

    /// Increment the helpful counter (rule fired and outcome improved).
    #[inline]
    pub fn record_helpful(&mut self) {
        self.helpful = self.helpful.saturating_add(1);
    }

    /// Increment the harmful counter (rule fired and outcome worsened).
    #[inline]
    pub fn record_harmful(&mut self) {
        self.harmful = self.harmful.saturating_add(1);
    }
}

// ─── Failure entry ────────────────────────────────────────────────────────

/// A substantive failure (anti-pattern) stored in a branch.
///
/// RIZZ §"branch-local memory": failures are concrete anti-patterns — things
/// that demonstrably did not work. Unlike episodic entries (positive examples)
/// and procedural rules (IF-THEN with credit), failures are stored as
/// "do not do this near this branch" anchors. They stay branch-local: a failure
/// in the combat branch never contaminates the crafting branch.
#[derive(Clone, Debug)]
pub struct FailureEntry<E> {
    /// Latent embedding of the failed input.
    pub embedding: Vec<f32>,
    /// Caller-defined payload describing the failure.
    pub payload: E,
    /// Deterministic write tick (raw, syncable).
    pub tick: u64,
}

// ─── Branch statistics ────────────────────────────────────────────────────

/// Per-branch statistics tracked for lifecycle decisions (merge / prune).
///
/// `#[repr(C)]` + all fields are `Copy` → Pod-compatible, sync-friendly,
/// zero-copy mmap-able when the consumer persists a branch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BranchStats {
    /// Number of verifier-approved writes to this branch.
    pub n_writes: u32,
    /// Number of route-side reads (snap-to-this-branch events).
    pub n_reads: u32,
    /// Running average reward of admitted writes (incremental update).
    pub avg_reward: f32,
    /// Last tick this branch was touched (read or write).
    pub last_touch_tick: u64,
}

impl BranchStats {
    /// Record a write and update the incremental average reward.
    #[inline]
    pub fn record_write(&mut self, reward: f32, tick: u64) {
        self.n_writes = self.n_writes.saturating_add(1);
        let n = self.n_writes as f32;
        self.avg_reward += (reward - self.avg_reward) / n;
        self.last_touch_tick = tick;
    }

    /// Record a read (snap-to-this-branch event).
    #[inline]
    pub fn record_read(&mut self, tick: u64) {
        self.n_reads = self.n_reads.saturating_add(1);
        self.last_touch_tick = tick;
    }

    /// True if this branch is stale (no touch within `stale_window` ticks of `now`).
    #[inline]
    #[must_use]
    pub fn is_stale(&self, now: u64, stale_window: u64) -> bool {
        now.saturating_sub(self.last_touch_tick) > stale_window
    }
}

// ─── Cognitive branch ─────────────────────────────────────────────────────

/// A persistent cognitive branch — a "zero-interference zone" (RIZZ §"memory
/// branch").
///
/// Each branch accumulates verifier-approved episodic examples, procedural
/// rules (with helpful/harmful counters), and failure anti-patterns. The
/// `spawn_anchor` is the latent direction this branch represents; the router
/// snaps query embeddings to branches by dot-product against `spawn_anchor`.
/// `token_signature` is an optional sorted set of hash tokens enabling the
/// Jaccard fallback path in the router.
///
/// Non-interference is structural: two branches `b_i`, `b_j` are non-interfering
/// iff their anchor directions are orthogonal (`dot(g_{b_i}, g_{b_j}) ≈ 0`).
/// Writes projected onto one branch's direction have zero component along any
/// orthogonal sibling's direction (Phase 2 `NonInterferenceProjection`).
#[derive(Clone, Debug)]
pub struct CognitiveBranch<E> {
    /// Dense slot index (matches the slot this branch lives in).
    pub id: BranchId,
    /// Latent direction this branch represents (caller-normalized).
    pub spawn_anchor: Vec<f32>,
    /// Sorted, deduplicated hash tokens for Jaccard fallback (empty = disabled).
    pub token_signature: Vec<u64>,
    /// Verifier-approved episodic examples (positive memory).
    pub episodic: Vec<EpisodicEntry<E>>,
    /// Procedural rules with helpful/harmful credit counters.
    pub procedural: Vec<ProceduralRule>,
    /// Failure anti-patterns (branch-local negative memory).
    pub failures: Vec<FailureEntry<E>>,
    /// Optional caller-defined scope context tag.
    pub scope_ctx: Option<u64>,
    /// Per-branch statistics for lifecycle decisions.
    pub stats: BranchStats,
    /// Lifecycle state (Active / Shadow / Deprecated / Removed).
    pub lifecycle: BranchLifecycle,
}

impl<E> CognitiveBranch<E> {
    /// Construct a fresh active branch with the given spawn anchor.
    ///
    /// All memory stores start empty; `token_signature` starts empty (Jaccard
    /// fallback disabled until the caller populates it).
    #[inline]
    #[must_use]
    pub fn new(id: BranchId, spawn_anchor: Vec<f32>) -> Self {
        Self {
            id,
            spawn_anchor,
            token_signature: Vec::new(),
            episodic: Vec::new(),
            procedural: Vec::new(),
            failures: Vec::new(),
            scope_ctx: None,
            stats: BranchStats::default(),
            lifecycle: BranchLifecycle::default(),
        }
    }

    /// Number of memory entries (episodic + procedural + failures).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.episodic.len() + self.procedural.len() + self.failures.len()
    }

    /// True if all memory stores are empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a sorted token onto the signature (maintains sort + dedup).
    /// Call this after writing an episodic entry to enable Jaccard fallback.
    pub fn push_token(&mut self, token: u64) {
        let pos = self
            .token_signature
            .binary_search(&token)
            .unwrap_or_else(|p| p);
        if pos == self.token_signature.len() || self.token_signature[pos] != token {
            self.token_signature.insert(pos, token);
        }
    }
}

// ─── Migration serialization (Issue 456 Phase 2: npc_branch_runtimes) ──────
//
// These serializers cover the per-NPC cognitive branch memory for NPC
// migration (Issue 456). Wire format (all little-endian):
//
// `EpisodicEntry<E>`:
//   `emb_len(u32) || emb(f32 × emb_len) || payload_bytes(E) || reward(f32)
//     || scope_present(u8: 0|1) || [scope(u64)] || tick(u64)`
//
// `FailureEntry<E>`:
//   `emb_len(u32) || emb(f32 × emb_len) || payload_bytes(E) || tick(u64)`
//
// `ProceduralRule` (no E):
//   `dir_len(u32) || dir(f32 × dir_len) || antecedent([u8;32]) || strategy([u8;32])
//     || helpful(u32) || harmful(u32)`
//
// `BranchStats` (Pod-compatible, 20 bytes):
//   `n_writes(u32) || n_reads(u32) || avg_reward(f32) || last_touch_tick(u64)`
//
// `CognitiveBranch<E>`:
//   `id(u32) || anchor_len(u32) || anchor(f32 × anchor_len)
//     || tokens_len(u32) || tokens(u64 × tokens_len)
//     || episodic_len(u32) || (episodic_bytes)*
//     || procedural_len(u32) || (procedural_bytes)*
//     || failures_len(u32) || (failures_bytes)*
//     || scope_ctx_present(u8) || [scope_ctx(u64)]
//     || stats(20 bytes) || lifecycle(u8)`
//
// Each `Vec<T>` slot is length-prefixed. `Option<u64>` slots encode as a
// presence byte (0/1) + optional 8-byte payload. The lifecycle byte is the
// `#[repr(u8)]` discriminant (0..=3, matching both standalone + arg_protocol).
//
// # Determinism
//
// All collections are flat `Vec`s with stable insertion order — no HashMap,
// no papaya iteration. The encoder iterates in natural order; the decoder
// reconstructs in the same order. G4 determinism holds by construction.

/// Wire-format version for the branching-types migration serializers.
///
/// Bump on any incompatible change to the per-type wire formats documented
/// above. The version is NOT embedded in each `EpisodicEntry` / etc. — it's
/// carried by the outer `BranchBank` + `NpcCognitiveBranches` envelope. The
/// per-type decoders assume the envelope version has already been checked.
pub const BRANCH_TYPES_WIRE_VERSION: u64 = 1;

// ── Small read helpers ──────────────────────────────────────────────────
//
// Keeping these local to types.rs avoids a new submodule; the same helpers are
// duplicated (intentionally — each is ~3 lines) in bank.rs and projection.rs.
// They are NOT public API.

#[inline]
fn read_u32_le(bytes: &[u8]) -> Option<(u32, &[u8])> {
    let (head, tail) = bytes.split_first_chunk::<4>()?;
    Some((u32::from_le_bytes(*head), tail))
}

#[inline]
fn read_u64_le(bytes: &[u8]) -> Option<(u64, &[u8])> {
    let (head, tail) = bytes.split_first_chunk::<8>()?;
    Some((u64::from_le_bytes(*head), tail))
}

#[inline]
fn read_f32_le(bytes: &[u8]) -> Option<(f32, &[u8])> {
    let (head, tail) = bytes.split_first_chunk::<4>()?;
    Some((f32::from_le_bytes(*head), tail))
}

#[inline]
fn read_u8(bytes: &[u8]) -> Option<(u8, &[u8])> {
    Some((bytes.first().copied()?, &bytes[1..]))
}

#[inline]
fn read_vec_f32(bytes: &[u8]) -> Option<(Vec<f32>, &[u8])> {
    let (len, rest) = read_u32_le(bytes)?;
    let len = len as usize;
    if rest.len() < len.checked_mul(4)? {
        return None;
    }
    let mut v = Vec::with_capacity(len);
    let mut tail = rest;
    for _ in 0..len {
        let (x, t) = read_f32_le(tail)?;
        v.push(x);
        tail = t;
    }
    Some((v, tail))
}

#[inline]
fn read_vec_u64(bytes: &[u8]) -> Option<(Vec<u64>, &[u8])> {
    let (len, rest) = read_u32_le(bytes)?;
    let len = len as usize;
    if rest.len() < len.checked_mul(8)? {
        return None;
    }
    let mut v = Vec::with_capacity(len);
    let mut tail = rest;
    for _ in 0..len {
        let (x, t) = read_u64_le(tail)?;
        v.push(x);
        tail = t;
    }
    Some((v, tail))
}

#[inline]
fn write_vec_f32(out: &mut Vec<u8>, v: &[f32]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

#[inline]
fn write_vec_u64(out: &mut Vec<u8>, v: &[u64]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

// ── EpisodicEntry<E> serializer ─────────────────────────────────────────

impl<E: EpisodicCodec> EpisodicEntry<E> {
    /// Append the canonical encoding of this entry to `out`.
    ///
    /// The payload codec is responsible for length-prefixing itself; this
    /// method does NOT add an extra length prefix around `payload_bytes`.
    /// The contract: `from_bytes_tail(payload_tail)` must consume exactly the
    /// bytes `E::encode_to` wrote and return the unconsumed tail.
    #[inline]
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        write_vec_f32(out, &self.embedding);
        self.payload.encode_to(out);
        out.extend_from_slice(&self.reward.to_le_bytes());
        // scope: presence byte + optional u64.
        match self.scope {
            Some(s) => {
                out.push(1u8);
                out.extend_from_slice(&s.to_le_bytes());
            }
            None => out.push(0u8),
        }
        out.extend_from_slice(&self.tick.to_le_bytes());
    }

    /// Decode an `EpisodicEntry<E>` from `bytes`, returning the entry and the
    /// unconsumed tail. Returns `None` on any structural failure.
    #[inline]
    pub fn from_bytes_tail(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let (embedding, rest) = read_vec_f32(bytes)?;
        let (payload, rest) = E::decode_from(rest)?;
        let (reward, rest) = read_f32_le(rest)?;
        let (scope_present, rest) = read_u8(rest)?;
        let (scope, rest) = match scope_present {
            0 => (None, rest),
            1 => {
                let (s, t) = read_u64_le(rest)?;
                (Some(s), t)
            }
            _ => return None,
        };
        let (tick, rest) = read_u64_le(rest)?;
        Some((
            Self {
                embedding,
                payload,
                reward,
                scope,
                tick,
            },
            rest,
        ))
    }
}

// ── FailureEntry<E> serializer ──────────────────────────────────────────

impl<E: EpisodicCodec> FailureEntry<E> {
    /// Append the canonical encoding of this entry to `out`.
    #[inline]
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        write_vec_f32(out, &self.embedding);
        self.payload.encode_to(out);
        out.extend_from_slice(&self.tick.to_le_bytes());
    }

    /// Decode a `FailureEntry<E>` from `bytes`, returning the entry and tail.
    #[inline]
    pub fn from_bytes_tail(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let (embedding, rest) = read_vec_f32(bytes)?;
        let (payload, rest) = E::decode_from(rest)?;
        let (tick, rest) = read_u64_le(rest)?;
        Some((
            Self {
                embedding,
                payload,
                tick,
            },
            rest,
        ))
    }
}

// ── ProceduralRule serializer (no E) ────────────────────────────────────

impl ProceduralRule {
    /// Append the canonical encoding of this rule to `out`.
    #[inline]
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        write_vec_f32(out, &self.direction);
        out.extend_from_slice(&self.antecedent);
        out.extend_from_slice(&self.strategy);
        out.extend_from_slice(&self.helpful.to_le_bytes());
        out.extend_from_slice(&self.harmful.to_le_bytes());
    }

    /// Decode a `ProceduralRule` from `bytes`, returning the rule and tail.
    #[inline]
    pub fn from_bytes_tail(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let (direction, rest) = read_vec_f32(bytes)?;
        let (antecedent, rest) = rest.split_first_chunk::<32>()?;
        let (strategy, rest) = rest.split_first_chunk::<32>()?;
        let (helpful, rest) = read_u32_le(rest)?;
        let (harmful, rest) = read_u32_le(rest)?;
        Some((
            Self {
                direction,
                antecedent: *antecedent,
                strategy: *strategy,
                helpful,
                harmful,
            },
            rest,
        ))
    }
}

// ── BranchStats serializer (Pod-compatible, 20 bytes) ───────────────────

impl BranchStats {
    /// Append the canonical encoding of these stats to `out`. 20 bytes.
    #[inline]
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.n_writes.to_le_bytes());
        out.extend_from_slice(&self.n_reads.to_le_bytes());
        out.extend_from_slice(&self.avg_reward.to_le_bytes());
        out.extend_from_slice(&self.last_touch_tick.to_le_bytes());
    }

    /// Decode `BranchStats` from `bytes`, returning the stats and tail.
    /// Exactly 20 bytes consumed.
    #[inline]
    pub fn from_bytes_tail(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let (n_writes, rest) = read_u32_le(bytes)?;
        let (n_reads, rest) = read_u32_le(rest)?;
        let (avg_reward, rest) = read_f32_le(rest)?;
        let (last_touch_tick, rest) = read_u64_le(rest)?;
        Some((
            Self {
                n_writes,
                n_reads,
                avg_reward,
                last_touch_tick,
            },
            rest,
        ))
    }
}

// ── BranchLifecycle serializer (1 byte discriminant) ────────────────────
//
// `BranchLifecycle` is either the local enum (no `arg_protocol`) or a type
// alias for `arg::LifecycleState` (with `arg_protocol`). Both are `#[repr(u8)]`
// with identical discriminants 0..=3 (Active/Shadow/Deprecated/Removed). The
// serializer treats it as a raw `u8` on the wire; the decoder rejects unknown
// discriminants (>3) for forwards-compat.

/// Encode a `BranchLifecycle` as a single `u8` discriminant.
#[inline]
pub(crate) fn encode_lifecycle(out: &mut Vec<u8>, lifecycle: BranchLifecycle) {
    out.push(lifecycle as u8);
}

/// Decode a `BranchLifecycle` from the leading byte, rejecting unknown
/// discriminants (>3).
#[inline]
pub(crate) fn decode_lifecycle(bytes: &[u8]) -> Option<(BranchLifecycle, &[u8])> {
    let (b, rest) = read_u8(bytes)?;
    let lc = match b {
        0 => BranchLifecycle::Active,
        1 => BranchLifecycle::Shadow,
        2 => BranchLifecycle::Deprecated,
        3 => BranchLifecycle::Removed,
        _ => return None,
    };
    Some((lc, rest))
}

// ── CognitiveBranch<E> serializer ───────────────────────────────────────

impl<E: EpisodicCodec> CognitiveBranch<E> {
    /// Append the canonical encoding of this branch to `out`.
    ///
    /// The 9 fields are written in declaration order: id, spawn_anchor,
    /// token_signature, episodic, procedural, failures, scope_ctx, stats,
    /// lifecycle. Each `Vec<T>` is length-prefixed; `Option<u64>` is a presence
    /// byte + optional payload; `BranchStats` is 20 flat bytes; `lifecycle`
    /// is 1 byte.
    #[inline]
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.id.0.to_le_bytes());
        write_vec_f32(out, &self.spawn_anchor);
        write_vec_u64(out, &self.token_signature);
        // episodic: length-prefix + each entry's self-framed bytes.
        out.extend_from_slice(&(self.episodic.len() as u32).to_le_bytes());
        for entry in &self.episodic {
            entry.encode_to(out);
        }
        // procedural: length-prefix + each rule's self-framed bytes.
        out.extend_from_slice(&(self.procedural.len() as u32).to_le_bytes());
        for rule in &self.procedural {
            rule.encode_to(out);
        }
        // failures: length-prefix + each entry's self-framed bytes.
        out.extend_from_slice(&(self.failures.len() as u32).to_le_bytes());
        for entry in &self.failures {
            entry.encode_to(out);
        }
        // scope_ctx: presence byte + optional u64.
        match self.scope_ctx {
            Some(s) => {
                out.push(1u8);
                out.extend_from_slice(&s.to_le_bytes());
            }
            None => out.push(0u8),
        }
        self.stats.encode_to(out);
        encode_lifecycle(out, self.lifecycle);
    }

    /// Decode a `CognitiveBranch<E>` from `bytes`, returning the branch and tail.
    #[inline]
    pub fn from_bytes_tail(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let (id_raw, rest) = read_u32_le(bytes)?;
        let (spawn_anchor, rest) = read_vec_f32(rest)?;
        let (token_signature, rest) = read_vec_u64(rest)?;
        // episodic
        let (n, mut rest) = read_u32_le(rest)?;
        let mut episodic = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let (entry, t) = EpisodicEntry::<E>::from_bytes_tail(rest)?;
            episodic.push(entry);
            rest = t;
        }
        // procedural
        let (n, mut rest) = read_u32_le(rest)?;
        let mut procedural = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let (rule, t) = ProceduralRule::from_bytes_tail(rest)?;
            procedural.push(rule);
            rest = t;
        }
        // failures
        let (n, mut rest) = read_u32_le(rest)?;
        let mut failures = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let (entry, t) = FailureEntry::<E>::from_bytes_tail(rest)?;
            failures.push(entry);
            rest = t;
        }
        // scope_ctx
        let (scope_present, rest) = read_u8(rest)?;
        let (scope_ctx, rest) = match scope_present {
            0 => (None, rest),
            1 => {
                let (s, t) = read_u64_le(rest)?;
                (Some(s), t)
            }
            _ => return None,
        };
        let (stats, rest) = BranchStats::from_bytes_tail(rest)?;
        let (lifecycle, rest) = decode_lifecycle(rest)?;
        Some((
            Self {
                id: BranchId(id_raw),
                spawn_anchor,
                token_signature,
                episodic,
                procedural,
                failures,
                scope_ctx,
                stats,
                lifecycle,
            },
            rest,
        ))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_id_round_trip() {
        let id = BranchId::new(42);
        assert_eq!(u32::from(id), 42);
        assert_eq!(BranchId::from(42u32), id);
        assert!(!id.is_sentinel());
        assert!(BranchId::SENTINEL.is_sentinel());
        assert_eq!(format!("{id}"), "branch:42");
        assert_eq!(format!("{}", BranchId::SENTINEL), "branch:SENTINEL");
    }

    #[test]
    fn branch_id_default_is_zero() {
        assert_eq!(BranchId::default(), BranchId::new(0));
    }

    #[test]
    fn lifecycle_active_is_routable() {
        assert!(BranchLifecycle::default().is_routable());
        assert!(BranchLifecycle::Active.is_routable());
        assert!(!BranchLifecycle::Shadow.is_routable());
        assert!(!BranchLifecycle::Deprecated.is_routable());
        assert!(!BranchLifecycle::Removed.is_routable());
    }

    #[test]
    fn lifecycle_redirect_requirements() {
        assert!(!BranchLifecycle::Active.requires_redirect());
        assert!(!BranchLifecycle::Shadow.requires_redirect());
        assert!(BranchLifecycle::Deprecated.requires_redirect());
        assert!(BranchLifecycle::Removed.requires_redirect());
    }

    #[test]
    fn procedural_rule_credit_arithmetic() {
        let mut rule = ProceduralRule {
            direction: vec![1.0, 0.0],
            antecedent: [0u8; 32],
            strategy: [1u8; 32],
            helpful: 5,
            harmful: 2,
        };
        assert_eq!(rule.net_credit(), 3);
        rule.record_helpful();
        assert_eq!(rule.net_credit(), 4);
        rule.record_harmful();
        rule.record_harmful();
        assert_eq!(rule.net_credit(), 2);
    }

    #[test]
    fn procedural_rule_saturating_counters() {
        let mut rule = ProceduralRule {
            direction: vec![],
            antecedent: [0u8; 32],
            strategy: [0u8; 32],
            helpful: u32::MAX,
            harmful: 0,
        };
        rule.record_helpful();
        assert_eq!(rule.helpful, u32::MAX); // saturated
        assert_eq!(rule.net_credit(), u32::MAX as i64);
    }

    #[test]
    fn branch_stats_incremental_average() {
        let mut stats = BranchStats::default();
        stats.record_write(0.5, 1);
        assert!((stats.avg_reward - 0.5).abs() < 1e-6);
        stats.record_write(1.0, 2);
        assert!((stats.avg_reward - 0.75).abs() < 1e-6);
        stats.record_write(0.25, 3);
        // (0.5 + 1.0 + 0.25) / 3 = 0.583...
        assert!((stats.avg_reward - 0.58333).abs() < 1e-3);
        assert_eq!(stats.n_writes, 3);
        assert_eq!(stats.last_touch_tick, 3);
    }

    #[test]
    fn branch_stats_staleness() {
        let stats = BranchStats {
            last_touch_tick: 100,
            ..Default::default()
        };
        assert!(!stats.is_stale(150, 100)); // 50 ticks since touch, window 100
        assert!(stats.is_stale(250, 100)); // 150 ticks since touch, window 100
        assert!(!stats.is_stale(50, 100)); // now < touch (saturating_sub → 0)
    }

    #[test]
    fn cognitive_branch_new_is_empty_active() {
        let branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![1.0, 0.0, 0.0]);
        assert!(branch.is_empty());
        assert_eq!(branch.len(), 0);
        assert!(branch.lifecycle.is_routable());
        assert!(branch.token_signature.is_empty());
        assert!(branch.episodic.is_empty());
        assert!(branch.procedural.is_empty());
        assert!(branch.failures.is_empty());
        assert_eq!(branch.spawn_anchor, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn cognitive_branch_push_token_maintains_sorted_dedup() {
        let mut branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![]);
        branch.push_token(30);
        branch.push_token(10);
        branch.push_token(20);
        branch.push_token(10); // duplicate
        branch.push_token(30); // duplicate
        assert_eq!(branch.token_signature, vec![10, 20, 30]);
    }

    #[test]
    fn cognitive_branch_clone_preserves_all_fields() {
        let mut branch = CognitiveBranch::<&'static str>::new(BranchId::new(3), vec![0.0, 1.0]);
        branch.episodic.push(EpisodicEntry {
            embedding: vec![1.0],
            payload: "hello",
            reward: 0.8,
            scope: Some(7),
            tick: 42,
        });
        branch.stats.n_writes = 5;
        branch.push_token(99);

        let cloned = branch.clone();
        assert_eq!(cloned.id, branch.id);
        assert_eq!(cloned.spawn_anchor, branch.spawn_anchor);
        assert_eq!(cloned.token_signature, branch.token_signature);
        assert_eq!(cloned.episodic.len(), 1);
        assert_eq!(cloned.episodic[0].payload, "hello");
        assert_eq!(cloned.stats.n_writes, 5);
    }

    // ── Migration serialization tests (Issue 456 Phase 2) ──────────────
    //
    // The mock codec `U32Codec` wraps a `u32` as a 4-byte LE blob. It's the
    // simplest non-trivial codec (length-prefixed by fixed size). The unit
    // codec (`E = ()`) is also exercised for zero-payload round-trips.

    /// Mock codec wrapping a `u32` — 4 bytes LE, no length prefix (fixed size).
    #[derive(Clone, Debug, PartialEq)]
    struct U32Codec(u32);

    impl EpisodicCodec for U32Codec {
        fn encode_to(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.0.to_le_bytes());
        }
        fn decode_from(bytes: &[u8]) -> Option<(Self, &[u8])> {
            let (head, tail) = bytes.split_first_chunk::<4>()?;
            Some((Self(u32::from_le_bytes(*head)), tail))
        }
    }

    #[test]
    fn episodic_entry_round_trip_with_u32_codec() {
        let entry = EpisodicEntry {
            embedding: vec![0.1, 0.2, 0.3],
            payload: U32Codec(42),
            reward: 0.75,
            scope: Some(99),
            tick: 7,
        };
        let mut buf = Vec::new();
        entry.encode_to(&mut buf);
        let (decoded, tail) = EpisodicEntry::<U32Codec>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty(), "no trailing bytes");
        assert_eq!(decoded.embedding, entry.embedding);
        assert_eq!(decoded.payload, entry.payload);
        assert_eq!(decoded.reward.to_bits(), entry.reward.to_bits());
        assert_eq!(decoded.scope, entry.scope);
        assert_eq!(decoded.tick, entry.tick);
    }

    #[test]
    fn episodic_entry_scope_none_round_trip() {
        let entry = EpisodicEntry::<()> {
            embedding: vec![],
            payload: (),
            reward: 0.0,
            scope: None,
            tick: 0,
        };
        let mut buf = Vec::new();
        entry.encode_to(&mut buf);
        let (decoded, tail) = EpisodicEntry::<()>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded.scope, None);
    }

    #[test]
    fn episodic_entry_truncated_returns_none() {
        let entry = EpisodicEntry::<U32Codec> {
            embedding: vec![1.0],
            payload: U32Codec(1),
            reward: 0.5,
            scope: None,
            tick: 1,
        };
        let mut buf = Vec::new();
        entry.encode_to(&mut buf);
        // Truncate at every position 0..len-1; each must return None.
        for cut in 0..buf.len() - 1 {
            assert!(
                EpisodicEntry::<U32Codec>::from_bytes_tail(&buf[..cut]).is_none(),
                "expected None at cut={cut}"
            );
        }
    }

    #[test]
    fn failure_entry_round_trip() {
        let entry = FailureEntry {
            embedding: vec![-1.0, 2.5],
            payload: U32Codec(7),
            tick: 99,
        };
        let mut buf = Vec::new();
        entry.encode_to(&mut buf);
        let (decoded, tail) = FailureEntry::<U32Codec>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded.embedding, entry.embedding);
        assert_eq!(decoded.payload, entry.payload);
        assert_eq!(decoded.tick, entry.tick);
    }

    #[test]
    fn procedural_rule_round_trip() {
        let rule = ProceduralRule {
            direction: vec![0.5, 0.5],
            antecedent: [1u8; 32],
            strategy: [2u8; 32],
            helpful: 10,
            harmful: 3,
        };
        let mut buf = Vec::new();
        rule.encode_to(&mut buf);
        let (decoded, tail) = ProceduralRule::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded.direction, rule.direction);
        assert_eq!(decoded.antecedent, rule.antecedent);
        assert_eq!(decoded.strategy, rule.strategy);
        assert_eq!(decoded.helpful, rule.helpful);
        assert_eq!(decoded.harmful, rule.harmful);
    }

    #[test]
    fn branch_stats_round_trip() {
        let stats = BranchStats {
            n_writes: 100,
            n_reads: 250,
            avg_reward: 0.625,
            last_touch_tick: 1234,
        };
        let mut buf = Vec::new();
        stats.encode_to(&mut buf);
        assert_eq!(buf.len(), 20);
        let (decoded, tail) = BranchStats::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded, stats);
    }

    #[test]
    fn cognitive_branch_round_trip_full() {
        let mut branch = CognitiveBranch::<U32Codec>::new(BranchId::new(2), vec![0.0, 1.0, 0.0]);
        branch.push_token(15);
        branch.push_token(7);
        branch.episodic.push(EpisodicEntry {
            embedding: vec![0.1, 0.2],
            payload: U32Codec(100),
            reward: 0.9,
            scope: Some(42),
            tick: 10,
        });
        branch.procedural.push(ProceduralRule {
            direction: vec![1.0],
            antecedent: [0xaa; 32],
            strategy: [0xbb; 32],
            helpful: 1,
            harmful: 0,
        });
        branch.failures.push(FailureEntry {
            embedding: vec![-0.5],
            payload: U32Codec(200),
            tick: 11,
        });
        branch.scope_ctx = Some(99);
        branch.stats = BranchStats {
            n_writes: 1,
            n_reads: 0,
            avg_reward: 0.9,
            last_touch_tick: 10,
        };

        let mut buf = Vec::new();
        branch.encode_to(&mut buf);
        let (decoded, tail) = CognitiveBranch::<U32Codec>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());

        // Verify every field.
        assert_eq!(decoded.id, branch.id);
        assert_eq!(decoded.spawn_anchor, branch.spawn_anchor);
        assert_eq!(decoded.token_signature, branch.token_signature);
        assert_eq!(decoded.episodic.len(), 1);
        assert_eq!(decoded.episodic[0].payload, U32Codec(100));
        assert_eq!(decoded.episodic[0].reward.to_bits(), 0.9f32.to_bits());
        assert_eq!(decoded.episodic[0].scope, Some(42));
        assert_eq!(decoded.episodic[0].tick, 10);
        assert_eq!(decoded.procedural.len(), 1);
        assert_eq!(decoded.procedural[0].helpful, 1);
        assert_eq!(decoded.failures.len(), 1);
        assert_eq!(decoded.failures[0].payload, U32Codec(200));
        assert_eq!(decoded.scope_ctx, Some(99));
        assert_eq!(decoded.stats, branch.stats);
        assert_eq!(decoded.lifecycle, branch.lifecycle);
    }

    #[test]
    fn cognitive_branch_round_trip_empty() {
        // Minimal branch: just an id + anchor, no memory. Exercises the
        // length-zero Vec paths.
        let branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![1.0]);
        let mut buf = Vec::new();
        branch.encode_to(&mut buf);
        let (decoded, tail) = CognitiveBranch::<()>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded.id, branch.id);
        assert!(decoded.episodic.is_empty());
        assert!(decoded.procedural.is_empty());
        assert!(decoded.failures.is_empty());
    }

    #[test]
    fn cognitive_branch_round_trip_shadow_lifecycle() {
        // Non-default lifecycle — exercises the lifecycle encoder/decoder.
        let mut branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![1.0]);
        branch.lifecycle = BranchLifecycle::Shadow;
        let mut buf = Vec::new();
        branch.encode_to(&mut buf);
        let (decoded, tail) = CognitiveBranch::<()>::from_bytes_tail(&buf).expect("decode");
        assert!(tail.is_empty());
        assert_eq!(decoded.lifecycle, BranchLifecycle::Shadow);
    }

    #[test]
    fn cognitive_branch_rejects_truncated_header() {
        let branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![1.0]);
        let mut buf = Vec::new();
        branch.encode_to(&mut buf);
        // Truncate to 2 bytes (only partial id).
        assert!(CognitiveBranch::<()>::from_bytes_tail(&buf[..2]).is_none());
    }
}
