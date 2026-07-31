//! Plan 562 — `event_log_query` basic demo.
//!
//! Distillation of PRO-LONG (Fox et al., arXiv:2607.20064, Research 461):
//! programmatic search over a lossless event log. This example builds a
//! deterministic 100-event log and exercises the five query API methods:
//!
//! 1. `filter`           — grep the log by event type
//! 2. `query_window`     — bounded slice + optional type filter
//! 3. `count_where`      — grep -c (count matching events)
//! 4. `first_where`      — first match (early exit)
//! 5. `Custom` predicate — consumer-defined escape hatch
//!
//! Run:
//! ```bash
//! cargo run -p katgpt-pruners --example event_log_query_basic --features event_log_query
//! ```

use katgpt_pruners::event_log::{
    Actor, EventId, EventLog, EventPredicate, EventType, Predicate,
};
use std::fmt::Debug;

/// A tiny game-action payload for the demo.
#[derive(Clone, Debug)]
struct GameAction {
    name: &'static str,
    score: f32,
}

fn main() {
    let mut log: EventLog<GameAction> = EventLog::new();

    // Build a 100-event log with a deterministic mix.
    // Pattern: GameStart(0) → then 32 cycles of [Action, RewardSignal, Evaluation]
    //        → GameEnd(97) → 2 trailing HeuristicFire(98,99).
    log.push(
        EventType::GameStart,
        GameAction { name: "boot", score: 0.0 },
        Actor::Runtime,
        None,
    );
    for i in 0..32 {
        let action_score = 0.5 + (i as f32) * 0.01;
        log.push(
            EventType::Action,
            GameAction { name: "move", score: action_score },
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::RewardSignal,
            GameAction { name: "reward", score: action_score - 0.3 },
            Actor::Runtime,
            None,
        );
        log.push(
            EventType::Evaluation,
            GameAction { name: "eval", score: action_score + 0.1 },
            Actor::Heuristic("positional"),
            None,
        );
    }
    log.push(
        EventType::GameEnd,
        GameAction { name: "end", score: 1.0 },
        Actor::Runtime,
        None,
    );
    log.push(
        EventType::HeuristicFire,
        GameAction { name: "blunder_check", score: -0.5 },
        Actor::Heuristic("tactical"),
        None,
    );
    log.push(
        EventType::HeuristicFire,
        GameAction { name: "time_pressure", score: -0.2 },
        Actor::Heuristic("clock"),
        None,
    );

    println!("=== Plan 562: event_log_query demo ({} events) ===\n", log.len());

    // 1. filter — "grep event_type=Action" → print the action sequence
    println!("--- 1. filter: all Action events (first 5) ---");
    for (i, e) in log.filter(&Predicate::event_type(EventType::Action)).take(5).enumerate() {
        println!("  [{}] id={} payload={:?} score={:.3}",
            i, e.id.0, e.payload.name, e.payload.score);
    }
    let total_actions = log.count_where(&Predicate::event_type(EventType::Action));
    println!("  ... ({} total actions)\n", total_actions);

    // 2. query_window — bounded slice + optional type filter
    println!("--- 2. query_window: EventId(10)..EventId(20), type=Evaluation ---");
    for e in log.query_window(EventId(10)..EventId(20), Some(EventType::Evaluation)) {
        println!("  id={} score={:.3}", e.id.0, e.payload.score);
    }
    println!();

    // 3. count_where — composed predicate (grep -c with And)
    println!("--- 3. count_where: RewardSignal AND id >= 50 ---");
    let back_half_rewards = log.count_where(
        &Predicate::event_type(EventType::RewardSignal)
            .and(Predicate::id_range_from(EventId(50))),
    );
    println!("  {} reward signals in the back half (id >= 50)\n", back_half_rewards);

    // 4. first_where / last_where — early exit
    println!("--- 4. first_where / last_where ---");
    if let Some(first_eval) = log.first_where(&Predicate::event_type(EventType::Evaluation)) {
        println!("  first evaluation: id={} score={:.3}", first_eval.id.0, first_eval.payload.score);
    }
    if let Some(last_action) = log.last_where(&Predicate::event_type(EventType::Action)) {
        println!("  last action:      id={} score={:.3}", last_action.id.0, last_action.payload.score);
    }
    println!();

    // 5. Custom predicate — consumer-defined escape hatch (score > threshold)
    println!("--- 5. Custom predicate: Evaluation events with score > 0.70 ---");
    #[derive(Debug)]
    struct HighScoreEval { threshold: f32 }
    impl EventPredicate<GameAction> for HighScoreEval {
        fn matches(&self, event: &katgpt_pruners::event_log::Event<GameAction>) -> bool {
            event.event_type == EventType::Evaluation && event.payload.score > self.threshold
        }
    }
    let high_score = Predicate::custom(HighScoreEval { threshold: 0.70 });
    let high_count = log.count_where(&high_score);
    println!("  {} evaluations scored above 0.70:", high_count);
    for e in log.filter(&high_score) {
        println!("    id={} score={:.3}", e.id.0, e.payload.score);
    }

    // Composed: high-score evaluations in the first half via And
    let composed = high_score.and(Predicate::id_range(EventId(0), EventId(50)));
    let composed_count = log.count_where(&composed);
    println!("\n  Composed (high-score AND id < 50): {} events", composed_count);

    println!("\n=== Demo complete ===");
    println!("All three retrieval axes compose at the consumer layer:");
    println!("  - Pattern (this module):    EventTypeIs / IdRange / And / Or / Not / Custom");
    println!("  - Semantic (riir-neuron-db): experience_graph latent-seeded traversal");
    println!("  - Content-addressed (katgpt-core): engram_runtime hash → slot lookup");
    println!("The Custom escape hatch is the composition point.");
}
