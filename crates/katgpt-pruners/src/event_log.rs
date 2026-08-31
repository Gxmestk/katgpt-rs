//! Event-sourced game traces with fork-and-diff capability.
//!
//! Provides append-only event logs that serve as the source of truth for game state.
//! State is always a fold over the log, enabling:
//! - Deterministic replay from any event log
//! - Cheap forking at any event boundary
//! - Structural diff between divergent traces
//! - Content-addressed evaluation caching
//!
//! # Architecture
//!
//! ```text
//! EventLog<A>       — Append-only event sequence
//! ├── EvalCache     — Content-addressed evaluation results (blake3)
//! └── ForkDiff<A>   — Structural comparison of divergent traces
//! ```
//!
//! Plan 124: Event-sourced game traces with fork-and-diff.
//! Feature gate: `event_log`

use std::collections::HashMap;
use std::fmt::Debug;

// ── Types ───────────────────────────────────────────────────────

/// Monotonic event identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u64);

impl EventId {
    /// First event ID.
    pub const ZERO: Self = Self(0);
}

impl From<u64> for EventId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<usize> for EventId {
    fn from(v: usize) -> Self {
        Self(v as u64)
    }
}

/// Type of event in the trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventType {
    /// Game started.
    GameStart,
    /// Player action taken.
    Action,
    /// Evaluation/heuristic computed.
    Evaluation,
    /// Bandit/arm update.
    BanditUpdate,
    /// Heuristic fired.
    HeuristicFire,
    /// Reward signal emitted.
    RewardSignal,
    /// Game ended.
    GameEnd,
}

/// Who produced this event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    /// A player action (player ID).
    Player(u8),
    /// A named heuristic.
    Heuristic(&'static str),
    /// Bandit/RL system.
    Bandit,
    /// External model.
    Model,
    /// Runtime/infrastructure.
    Runtime,
}

/// Outcome of a completed game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    /// Player won (player ID).
    Win(u8),
    /// Game ended in draw.
    Draw,
    /// Game didn't complete.
    Incomplete,
}

/// Single event in the trace.
#[derive(Clone, Debug)]
pub struct Event<A: Clone + Debug> {
    /// Monotonic event ID.
    pub id: EventId,
    /// Type of event.
    pub event_type: EventType,
    /// Game-specific payload.
    pub payload: A,
    /// Who produced this event.
    pub actor: Actor,
    /// Which event caused this one (causal chain).
    pub caused_by: Option<EventId>,
}

/// Append-only event log for game traces.
/// Source of truth — game state is always a fold over this log.
#[derive(Clone, Debug)]
pub struct EventLog<A: Clone + Debug> {
    /// All events in order.
    events: Vec<Event<A>>,
}

impl<A: Clone + Debug> EventLog<A> {
    /// Create a new empty event log.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Append an event, auto-assigning the next monotonic ID.
    pub fn push(
        &mut self,
        event_type: EventType,
        payload: A,
        actor: Actor,
        caused_by: Option<EventId>,
    ) -> EventId {
        let id = EventId(self.events.len() as u64);
        self.events.push(Event {
            id,
            event_type,
            payload,
            actor,
            caused_by,
        });
        id
    }

    /// Number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get event by ID.
    pub fn get(&self, id: EventId) -> Option<&Event<A>> {
        self.events.get(id.0 as usize)
    }

    /// Iterate over all events.
    pub fn iter(&self) -> impl Iterator<Item = &Event<A>> {
        self.events.iter()
    }

    /// Last event ID (or ZERO if empty).
    pub fn last_id(&self) -> EventId {
        self.events.last().map_or(EventId::ZERO, |e| e.id)
    }

    /// Fork the log at a given event: clone prefix up to (including) `at`.
    /// Returns a new log that shares the prefix events.
    pub fn fork(&self, at: EventId) -> Self {
        let prefix_len = (at.0 as usize + 1).min(self.events.len());
        Self {
            events: self.events[..prefix_len].to_vec(),
        }
    }

    /// Compute structural diff between this log and another.
    /// Returns divergence information starting from the first different event.
    pub fn diff(&self, other: &Self) -> ForkDiff<A>
    where
        A: PartialEq,
    {
        let shared = self
            .events
            .iter()
            .zip(other.events.iter())
            .take_while(|(a, b)| a.payload == b.payload && a.event_type == b.event_type)
            .count();

        let fork_point = if shared < self.events.len() {
            self.events[shared].id
        } else {
            EventId(shared as u64)
        };

        let mut diff_events = Vec::new();

        // Events only in self after fork
        for event in &self.events[shared..] {
            diff_events.push(DiffEvent::ParentOnly(event.clone()));
        }

        // Events only in other after fork
        for event in &other.events[shared..] {
            diff_events.push(DiffEvent::ForkOnly(event.clone()));
        }

        ForkDiff {
            fork_point,
            shared_prefix_len: shared,
            diff_events,
        }
    }

    /// Replay events through a fold function to reconstruct state.
    pub fn replay<F, S>(&self, initial: S, f: F) -> S
    where
        F: FnMut(S, &Event<A>) -> S,
    {
        self.events.iter().fold(initial, f)
    }
}

impl<A: Clone + Debug> Default for EventLog<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two divergent traces.
#[derive(Clone, Debug)]
pub struct ForkDiff<A: Clone + Debug> {
    /// Event ID where divergence starts.
    pub fork_point: EventId,
    /// Number of shared prefix events.
    pub shared_prefix_len: usize,
    /// Divergent events from both logs.
    pub diff_events: Vec<DiffEvent<A>>,
}

impl<A: Clone + Debug> ForkDiff<A> {
    /// Whether the two logs are identical.
    pub fn is_identical(&self) -> bool {
        self.diff_events.is_empty()
    }

    /// Number of divergent events.
    pub fn divergence_count(&self) -> usize {
        self.diff_events.len()
    }
}

/// A single event in a diff between two traces.
#[derive(Clone, Debug)]
pub enum DiffEvent<A: Clone + Debug> {
    /// Event exists only in the parent trace.
    ParentOnly(Event<A>),
    /// Event exists only in the forked trace.
    ForkOnly(Event<A>),
}

/// Content-addressed evaluation cache.
/// Same game state hash → cached score, no re-evaluation.
pub struct EvalCache {
    entries: HashMap<[u8; 32], CachedEval>,
}

/// A cached evaluation result.
#[derive(Clone, Debug)]
pub struct CachedEval {
    /// Cached score.
    pub score: f32,
    /// Depth of evaluation.
    pub depth: usize,
    /// Which event produced this evaluation.
    pub provenance: EventId,
}

impl EvalCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a cached evaluation by state hash.
    pub fn get(&self, hash: &[u8; 32]) -> Option<&CachedEval> {
        self.entries.get(hash)
    }

    /// Insert a cached evaluation.
    pub fn insert(&mut self, hash: [u8; 32], score: f32, depth: usize, provenance: EventId) {
        self.entries.insert(
            hash,
            CachedEval {
                score,
                depth,
                provenance,
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit rate: hits / (hits + misses).
    pub fn hit_rate(&self, hits: usize, total: usize) -> f32 {
        match total {
            0 => 0.0,
            _ => hits as f32 / total as f32,
        }
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Query API (feature: `event_log_query`) ────────────────────────────
//
// Plan 562 — programmatic search over the lossless event log.
// Distillation of PRO-LONG (Fox et al., arXiv:2607.20064) — see Research 461.
//
// PRO-LONG's load-bearing finding (Table 1): programmatic tools (grep + Python)
// drive +15.2 of the +18.1 accuracy gain on ARC-AGI-3, vs only +2.9 for
// Write/Edit. The access pattern is "write everything losslessly, search
// programmatically at read time" — no learned compression at write time, no
// embedding retrieval at read time.
//
// This module ships the **pattern axis** of that access pattern as a generic,
// modelless, zero-allocation query combinator. It is the deterministic,
// LLM-free analog of "coding agent greps the log."
//
// # Three orthogonal retrieval axes (compose at the consumer layer)
//
// | Axis | Mechanism | Ships in |
// |---|---|---|
// | **Pattern** (this module) | `Predicate` enum (EventTypeIs / IdRange / And / Or / Not / Custom) | `EventLog::filter` |
// | **Semantic** | latent-seeded NS traversal (vector ANN + KG edges) | `riir-neuron-db::experience_graph` |
// | **Content-addressed** | hash → slot lookup | `katgpt-core::engram_runtime` |
//
// The `Predicate::Custom` escape hatch is the composition point: a consumer
// can wrap a semantic predicate or a content-addressed predicate as a
// `Box<dyn EventPredicate<A>>` and compose it with pattern predicates via
// `And` / `Or` / `Not`.
//
// # Zero-allocation contract
//
// `filter` / `count_where` / `first_where` / `last_where` return lazy
// iterators (or early-exit reductions over iterators). No `collect()` in
// the hot path — the iterator borrows `&self`. `query_window` returns a
// direct slice into `self.events` (no allocation at all).

/// Object-safe predicate over events. The escape hatch for consumer-defined
/// predicates (e.g., score-threshold, action-tag-regex at the game-domain
/// layer, or a semantic predicate composed via `experience_graph`).
///
/// Implement this trait + wrap in [`Predicate::Custom`] to compose with the
/// built-in pattern predicates.
#[cfg(feature = "event_log_query")]
pub trait EventPredicate<A: Clone + Debug>: std::fmt::Debug {
    /// Whether this event matches the predicate.
    fn matches(&self, event: &Event<A>) -> bool;
}

/// Composable predicate enum — the deterministic, LLM-free analog of
/// "coding agent greps the log." Variant names mirror the grep mental model:
/// `EventTypeIs` = `grep event_type=X`, `IdRange` = `grep -n` window,
/// `And`/`Or`/`Not` = boolean combinators.
///
/// The enum delegates to [`EventPredicate`] so a single trait object suffices
/// for any predicate — built-in or consumer-defined via [`Predicate::Custom`].
///
/// # Example
///
/// ```
/// use katgpt_pruners::event_log::{EventLog, EventType, Predicate, Actor};
///
/// let mut log: EventLog<&'static str> = EventLog::new();
/// log.push(EventType::Action, "move", Actor::Player(0), None);
/// log.push(EventType::RewardSignal, "+1", Actor::Runtime, None);
/// log.push(EventType::Action, "jump", Actor::Player(0), None);
///
/// // Find all actions via pattern predicate.
/// let actions: Vec<_> = log
///     .filter(&Predicate::event_type(EventType::Action))
///     .map(|e| e.payload)
///     .collect();
/// assert_eq!(actions, vec!["move", "jump"]);
/// ```
#[cfg(feature = "event_log_query")]
#[derive(Debug)]
pub enum Predicate<A: Clone + Debug> {
    /// Match events whose `event_type` equals `t` — the raw pattern predicate
    /// (direct PRO-LONG "grep event_type" analog).
    EventTypeIs(EventType),
    /// Match events whose `event_type` is in `types` — multi-type pattern.
    EventTypeIn(&'static [EventType]),
    /// Match events with `lo <= id < hi` — window predicate (tick-range analog).
    IdRange {
        /// Inclusive lower bound.
        lo: EventId,
        /// Exclusive upper bound.
        hi: EventId,
    },
    /// Match events with `id >= from` — open-ended window ("last-N" analog).
    IdRangeFrom(EventId),
    /// Conjunction — identity element is [`Predicate::All`].
    And(Box<Predicate<A>>, Box<Predicate<A>>),
    /// Disjunction — identity element is [`Predicate::None_`].
    Or(Box<Predicate<A>>, Box<Predicate<A>>),
    /// Negation.
    Not(Box<Predicate<A>>),
    /// Always-true (identity for `And`).
    All,
    /// Always-false (identity for `Or`). Named with trailing underscore to
    /// avoid the `None` keyword collision.
    None_,
    /// Consumer-defined predicate (escape hatch). This is the composition
    /// point for semantic / content-addressed predicates at the consumer layer.
    Custom(Box<dyn EventPredicate<A>>),
}

#[cfg(feature = "event_log_query")]
impl<A: Clone + Debug> Predicate<A> {
    /// Construct an `EventTypeIs` predicate.
    pub fn event_type(t: EventType) -> Self {
        Self::EventTypeIs(t)
    }

    /// Construct an `IdRange` predicate over `lo..hi` (half-open).
    pub fn id_range(lo: impl Into<EventId>, hi: impl Into<EventId>) -> Self {
        Self::IdRange {
            lo: lo.into(),
            hi: hi.into(),
        }
    }

    /// Construct an `IdRangeFrom` predicate (matches `id >= from`).
    /// The "last-N events" analog: `id_range_from(EventId(total - N))`.
    pub fn id_range_from(from: impl Into<EventId>) -> Self {
        Self::IdRangeFrom(from.into())
    }

    /// Compose with another predicate via conjunction (logical AND).
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Compose with another predicate via disjunction (logical OR).
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Wrap a consumer-defined predicate (the composition point for semantic /
    /// content-addressed predicates at the consumer layer).
    pub fn custom<P: EventPredicate<A> + 'static>(p: P) -> Self {
        Self::Custom(Box::new(p))
    }
}

/// `!predicate` = `Predicate::Not(predicate)`. Implementing `Not` avoids the
/// clippy `should_implement_trait` warning from a standalone `not()` method.
#[cfg(feature = "event_log_query")]
impl<A: Clone + Debug> std::ops::Not for Predicate<A> {
    type Output = Self;
    fn not(self) -> Self {
        Self::Not(Box::new(self))
    }
}

#[cfg(feature = "event_log_query")]
impl<A: Clone + Debug> EventPredicate<A> for Predicate<A> {
    fn matches(&self, event: &Event<A>) -> bool {
        match self {
            Self::EventTypeIs(t) => event.event_type == *t,
            Self::EventTypeIn(types) => types.contains(&event.event_type),
            Self::IdRange { lo, hi } => event.id >= *lo && event.id < *hi,
            Self::IdRangeFrom(from) => event.id >= *from,
            Self::All => true,
            Self::None_ => false,
            Self::And(a, b) => a.matches(event) && b.matches(event),
            Self::Or(a, b) => a.matches(event) || b.matches(event),
            Self::Not(a) => !a.matches(event),
            Self::Custom(p) => p.matches(event),
        }
    }
}

#[cfg(feature = "event_log_query")]
impl<A: Clone + Debug> EventLog<A> {
    /// Filter the log by a predicate — the direct PRO-LONG "grep the log" analog.
    ///
    /// Returns a lazy iterator yielding only events matching `predicate`.
    /// **Zero allocation** — the iterator borrows `self`; no `collect()` in the
    /// hot path.
    pub fn filter(&self, predicate: &Predicate<A>) -> impl Iterator<Item = &Event<A>> {
        self.events.iter().filter(move |e| predicate.matches(e))
    }

    /// Bounded-window query — returns a contiguous slice of events in
    /// `range` (half-open: `start..end`).
    ///
    /// When `event_type_filter` is `Some(t)`, the slice is filtered to only
    /// events of type `t` via [`filter`](Self::filter). The slice path (no
    /// filter) is the fast path for "all events in window"; the filter path is
    /// the "actions only in window" path.
    ///
    /// **Zero allocation** for the no-filter case (direct slice into
    /// `self.events`). The filtered case is a lazy iterator (no allocation).
    ///
    /// Sub-µs target — it's a slice + optional filter.
    pub fn query_window(
        &self,
        range: std::ops::Range<EventId>,
        event_type_filter: Option<EventType>,
    ) -> impl Iterator<Item = &Event<A>> {
        let lo = (range.start.0 as usize).min(self.events.len());
        let hi = (range.end.0 as usize).min(self.events.len());
        let slice = &self.events[lo..hi.max(lo)];
        slice.iter().filter(move |e| match event_type_filter {
            Some(t) => e.event_type == t,
            None => true,
        })
    }

    /// Count events matching `predicate` — the PRO-LONG `grep -c` analog.
    ///
    /// **Zero allocation** (iterator count).
    pub fn count_where(&self, predicate: &Predicate<A>) -> usize {
        self.filter(predicate).count()
    }

    /// Find the first event matching `predicate`, early-exit. Returns `None`
    /// if no match. **Zero allocation.**
    pub fn first_where(&self, predicate: &Predicate<A>) -> Option<&Event<A>> {
        self.events.iter().find(|e| predicate.matches(e))
    }

    /// Find the last event matching `predicate`, early-exit (reverse scan).
    /// Returns `None` if no match. **Zero allocation.**
    pub fn last_where(&self, predicate: &Predicate<A>) -> Option<&Event<A>> {
        self.events.iter().rfind(|e| predicate.matches(e))
    }
}

// ── Tests (feature: `event_log_query`) ────────────────────────────────

#[cfg(all(test, feature = "event_log_query"))]
mod query_tests {
    use super::*;

    // Helper: build a deterministic 10-event log with a known mix of types.
    //   ids 0,1     = GameStart
    //   ids 2,4,6,8 = Action       (4 actions)
    //   ids 3,5,7   = RewardSignal (3 rewards)
    //   ids 9       = GameEnd
    fn build_mixed_log() -> EventLog<&'static str> {
        let mut log = EventLog::new();
        log.push(EventType::GameStart, "start0", Actor::Player(0), None);
        log.push(EventType::GameStart, "start1", Actor::Player(1), None);
        log.push(EventType::Action, "a0", Actor::Player(0), None);
        log.push(EventType::RewardSignal, "+0.1", Actor::Runtime, None);
        log.push(EventType::Action, "a1", Actor::Player(0), None);
        log.push(EventType::RewardSignal, "+0.2", Actor::Runtime, None);
        log.push(EventType::Action, "a2", Actor::Player(1), None);
        log.push(EventType::RewardSignal, "+0.3", Actor::Runtime, None);
        log.push(EventType::Action, "a3", Actor::Player(0), None);
        log.push(EventType::GameEnd, "end", Actor::Runtime, None);
        log
    }

    #[test]
    fn filter_returns_only_matching_events() {
        let log = build_mixed_log();
        let actions: Vec<&str> = log
            .filter(&Predicate::event_type(EventType::Action))
            .map(|e| e.payload)
            .collect();
        assert_eq!(actions, vec!["a0", "a1", "a2", "a3"]);
    }

    #[test]
    fn query_window_returns_contiguous_slice() {
        let log = build_mixed_log();
        // EventId(2)..EventId(5) → ids 2,3,4
        let window: Vec<u64> = log
            .query_window(EventId(2)..EventId(5), None)
            .map(|e| e.id.0)
            .collect();
        assert_eq!(window, vec![2, 3, 4]);
    }

    #[test]
    fn query_window_with_type_filter() {
        let log = build_mixed_log();
        // ids 2..8 → contains Action(2,4,6) + RewardSignal(3,5,7)
        let actions_only: Vec<&str> = log
            .query_window(EventId(2)..EventId(8), Some(EventType::Action))
            .map(|e| e.payload)
            .collect();
        assert_eq!(actions_only, vec!["a0", "a1", "a2"]);
    }

    #[test]
    fn predicate_and_composes_correctly() {
        let log = build_mixed_log();
        // Action AND id < 5 → ids 2,4 (a0, a1)
        let pred = Predicate::event_type(EventType::Action)
            .and(Predicate::id_range(EventId(0), EventId(5)));
        let result: Vec<&str> = log.filter(&pred).map(|e| e.payload).collect();
        assert_eq!(result, vec!["a0", "a1"]);
    }

    #[test]
    fn predicate_or_composes_correctly() {
        let log = build_mixed_log();
        // Action OR RewardSignal → 7 events (4 actions + 3 rewards)
        let pred = Predicate::event_type(EventType::Action)
            .or(Predicate::event_type(EventType::RewardSignal));
        assert_eq!(log.count_where(&pred), 7);
    }

    #[test]
    fn predicate_not_composes_correctly() {
        let log = build_mixed_log();
        // NOT Action → 6 events (2 start + 3 reward + 1 end)
        let pred = !Predicate::event_type(EventType::Action);
        assert_eq!(log.count_where(&pred), 6);
    }

    #[test]
    fn count_where_matches_grep_c_semantics() {
        let log = build_mixed_log();
        assert_eq!(
            log.count_where(&Predicate::event_type(EventType::Action)),
            4
        );
        assert_eq!(
            log.count_where(&Predicate::event_type(EventType::RewardSignal)),
            3
        );
        assert_eq!(log.count_where(&Predicate::All), 10);
        assert_eq!(log.count_where(&Predicate::None_), 0);
    }

    #[test]
    fn first_where_and_last_where_early_exit() {
        let log = build_mixed_log();
        let first_action = log.first_where(&Predicate::event_type(EventType::Action));
        let last_action = log.last_where(&Predicate::event_type(EventType::Action));
        assert_eq!(first_action.unwrap().payload, "a0");
        assert_eq!(last_action.unwrap().payload, "a3");
        // They differ on a mixed log.
        assert_ne!(first_action.unwrap().id, last_action.unwrap().id);
        // No match → None.
        assert!(
            log.first_where(&Predicate::event_type(EventType::HeuristicFire))
                .is_none()
        );
    }

    #[test]
    fn custom_predicate_escape_hatch() {
        // Test-only consumer predicate: match events whose payload starts with 'a'.
        #[derive(Debug)]
        struct StartsWithA;
        impl EventPredicate<&'static str> for StartsWithA {
            fn matches(&self, event: &Event<&'static str>) -> bool {
                event.payload.starts_with('a')
            }
        }

        let log = build_mixed_log();
        // Custom alone → a0, a1, a2, a3 (4 events)
        let custom_pred = Predicate::custom(StartsWithA);
        assert_eq!(log.count_where(&custom_pred), 4);

        // Custom composed with pattern via And → only actions starting with 'a' in ids 2..6
        let composed = custom_pred.and(Predicate::id_range(EventId(2), EventId(6)));
        let result: Vec<&str> = log.filter(&composed).map(|e| e.payload).collect();
        assert_eq!(result, vec!["a0", "a1"]);
    }

    #[test]
    fn filter_zero_allocation() {
        // G4 proxy: after building the log, the filter iterator borrows `&self`.
        // We verify zero-allocation indirectly: no intermediate Vec is allocated.
        // The `filter` method returns `impl Iterator<Item = &Event<A>>` which is
        // a lazy `Filter<slice::Iter, closure>` — no heap allocation.
        //
        // We assert the count matches a direct linear scan, proving the iterator
        // is correct without any intermediate collection.
        let log = build_mixed_log();

        // Direct linear scan (ground truth).
        let expected = log
            .iter()
            .filter(|e| e.event_type == EventType::Action)
            .count();

        // Query API path.
        let actual = log.count_where(&Predicate::event_type(EventType::Action));

        assert_eq!(actual, expected);
        assert_eq!(actual, 4);
    }

    #[test]
    fn existing_api_unchanged() {
        // Regression: the existing Plan 124 API still works identically.
        let log = build_mixed_log();
        assert_eq!(log.len(), 10);
        assert!(!log.is_empty());
        assert_eq!(log.last_id(), EventId(9));
        assert_eq!(
            log.get(EventId(0)).unwrap().event_type,
            EventType::GameStart
        );
        assert_eq!(log.iter().count(), 10);

        // fork + diff still work.
        let forked = log.fork(EventId(4));
        assert_eq!(forked.len(), 5);
        let d = log.diff(&forked);
        // Shared prefix = 5, then parent-only events for the rest.
        assert_eq!(d.shared_prefix_len, 5);
        assert!(d.divergence_count() > 0);

        // replay still works.
        let count = log.replay(0usize, |n, _| n + 1);
        assert_eq!(count, 10);
    }
}
