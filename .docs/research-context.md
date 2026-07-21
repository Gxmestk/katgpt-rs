# katgpt-rs Research Context

Project context distilled for anyone learning, explaining, or reasoning about this
codebase — humans and coding agents alike. Originally captured as a Kiro steering
file; relocated here so it lives with the rest of the project documentation.

## What this project is

- **Modelless** neuro-symbolic micro-Transformer in Rust — no training, no
  backprop, no gradient descent at runtime.
- 27 in-tree crates, 378 feature flags (152 default-on, all GOAT-proved), MIT
  licensed.
- Explanations should focus on mathematical / algorithmic concepts, not just
  code mechanics.

## How to explain a feature

For any feature, reference three things:

1. The **paper** it distills.
2. The **GOAT gate** results that validate it.
3. Which **crate** it lives in.

Use the crate dependency DAG when explaining architecture:

```
katgpt-types → katgpt-core → domain crates → root
```

## First-class design primitives

These core traits are the load-bearing abstractions — treat them as primitives,
not implementation details:

- `ConstraintPruner`
- `ScreeningPruner`
- `SpeculativeGenerator`
- `BeliefKernel`
- `GameState`

## Hard design rules

- **Sigmoid, not softmax** — for all gating / routing decisions. This is a hard
  project rule.
- The **inference pipeline** is fixed:
  `LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies`.

## Allowed runtime weight mutations

Only three — nothing else mutates weights at runtime:

1. **freeze / thaw**
2. **raw / lora hot-swap** (deterministic)
3. **latent-space updates**

## Session mode note

This codebase is frequently used as a learning / research subject. In that mode,
prefer concise, technically precise explanations aimed at an AI-researcher
audience, and do not suggest modifications to the code unless explicitly asked.
