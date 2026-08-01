# Proposal 011 — Rust-SWE-bench as a Latent-Space Benchmark via WASM Constraint Pruner

## TL;DR

**Should we use Rust-SWE-bench (500 real-world Rust SWE tasks from 34 popular repos) as a latent-space validation target, by compiling each task's test suite to WASM and loading it as a `ConstraintPruner` inside the model's inference loop?**

This is the **moka pattern applied to SWE**: Benchmark 205 brought the Go game INTO Moka's native policy/value heads via PUCT search (98% win) instead of treating Moka as an external black box. This proposal brings the Rust test suite INTO the inference loop as a WASM-compiled symbolic pruner, instead of running `cargo test` externally.

**The enabling substrate already ships:**
- `WasmPruner` / `BomberWasmPruner` — loads WASM modules as `ConstraintPruner` impls with a `is_valid(depth, action_idx, state_ptr, state_len) -> i32` ABI, papaya instance pool, fuel-limited sandboxed execution, zero-copy state buffer, batch API.
- `HotSwapPruner` — BLAKE3-hash-detected runtime reload of WASM pruners from disk.
- `SpecAsPruner` — compiles NL specs into symbolic bitmap rules (4400× smaller than LoRA, O(1) per token, zero training, exact verification).
- `WasmTestGate` — validates pruner skills against WASM-sandboxed state checks.
- [rubrc](https://github.com/oligamiq/rubrc) — a port of `rustc` to WebAssembly (WIP, external). This is the compiler that would turn a Rust-SWE-bench task's test logic into a WASM pruner module.

**The fusion this enables:**
- **Proposal 010** (Non-Hidden-State Canonical Construction) — extracts source features (AST histogram, Clippy fingerprint, ownership graph) from the buggy + fixed code → defines canonical "fix directions" in source-feature space. Rust-SWE-bench provides the probe corpus (500 (buggy→fixed) pairs from 34 real repos).
- **Proposal 009** (Canonical Intent Space) — projects the fix direction into any model's latent space via `ModelAdapter`.
- **Proposal 032** (riir-ai, Kimi-K3 Native Support) — the model being validated. Phase 6 GOAT gate currently only tests "logits match PyTorch ref on a fixed prompt" — a weak numerical-correctness test. Rust-SWE-bench as a WASM pruner would provide a **functional correctness test** that exercises real Rust semantics inside the inference loop.

**Honest verdict: HIGHLY SPECULATIVE — same risk class as Proposal 010.** The proposal exists to establish the architectural path so it can be evaluated and potentially rejected with reasoning, NOT because it is likely to succeed. The #1 risk is whether rubrc (WIP) can actually compile real Rust-SWE-bench test suites to WASM. The #2 risk is whether a 0.40B model has enough capability to propose patches that the WASM pruner can meaningfully validate.

## The problem this solves

### Problem 1: Proposal 032's Phase 6 GOAT gate is too weak

P032 Phase 6 currently tests: "load real 0.40B weights, run a forward pass, compare logits against the reference PyTorch implementation on a fixed prompt." This is a **numerical correctness** test — it verifies the forward pass math is right, but it does NOT verify the model produces **semantically meaningful** outputs on real Rust code.

The moka lesson (Benchmark 204, negative result): "blind heuristics cannot improve on a trained policy within the policy's training distribution." For Kimi-K3, the question is: does the 0.40B distillation carry enough Rust knowledge to produce coherent representations on real Rust repos? A single fixed-prompt logits comparison cannot answer this.

### Problem 2: Proposal 010 needs a probe corpus

P010 Phase 3 T3.1 needs paired `{(source_features(code_i), model_activations(code_i))}` samples for the ridge-regression `SourceFeatureAdapter` fit. The current plan says "Curate probe corpus (Rust code samples with known style properties)" — a hand-curated corpus that may be biased or too small.

Rust-SWE-bench provides 500 tasks from 34 popular Rust repos (ripgrep, bevy, tokio, clap, serde, nushell, axum, bytes, tracing, burn...). Each task is a controlled semantic transformation: (buggy code, fixed code, issue description, test patch). This is a **ready-made, diverse, ground-truth-labeled probe corpus** — far better than anything hand-curated.

### Problem 3: SWE benchmarks evaluate via external cargo test — slow + non-latent

The entire SWE-bench ecosystem (SWE-bench, Multi-SWE-bench, Rust-SWE-bench, RustForger) evaluates resolution by: LLM agent reads issue → writes patch → runs `cargo test` → checks pass/fail. This is:
- **Slow** — cargo test takes seconds to minutes per task.
- **External** — the validation happens outside the model's inference loop.
- **Non-latent** — it measures the OUTPUT (patch text), not the model's INTERNAL representations.

The moka pattern (Benchmark 205) showed that bringing the benchmark INTO the model's native inference path extracts dramatically more signal. For Go, this meant PUCT search using the policy/value heads natively. For SWE, the analogous move is: **compile the test suite to WASM, load it as a `ConstraintPruner`, and let the model's inference loop validate proposed patches in-sandbox.**

## The proposed design

### Architecture: the WASM-compiled test suite as a ConstraintPruner

```text
┌─────────────────────────────────────────────────────────────────┐
│  Rust-SWE-bench task                                            │
│  (issue description + repo snapshot + test patch + fix patch)   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────┐
│  rubrc (WASM rustc, external WIP dependency)      │
│  Compiles the task's test logic to a WASM module  │
│  with the WasmPruner ABI:                         │
│    is_valid(patch_bytes_ptr, patch_len) -> i32    │
└───────────────┬───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────┐
│  WasmPruner / HotSwapPruner (existing substrate)  │
│  Loads the WASM module as a ConstraintPruner.     │
│  Fuel-limited, sandboxed, papaya instance pool.   │
│  Runs INSIDE the model's inference loop.          │
└───────────────┬───────────────────────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
┌─────────────────┐  ┌──────────────────────────────────┐
│ Kimi-K3 (P032)  │  │ Source features (P010)            │
│ proposes patch  │  │ AST histogram of buggy + fixed   │
│ in latent space │  │ → canonical "fix direction"       │
│ via P009 adapter│  │ → projected into latent space     │
└────────┬────────┘  └──────────────┬───────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  WASM ConstraintPruner validates the patch IN-LOOP              │
│  is_valid() runs the compiled test logic against the patch.     │
│  Result feeds back into the inference loop (prune invalid       │
│  branches, steer toward valid ones) — the PUCT analogy.         │
└─────────────────────────────────────────────────────────────────┘
```

### The three layers

**Layer 1 — Corpus (Rust-SWE-bench as P010 probe corpus):**
- Each task provides `(buggy_code, fixed_code)` — a controlled semantic transformation.
- P010's `ast_histogram(buggy)` and `ast_histogram(fixed)` produce source-feature vectors.
- The "fix direction" = `normalize(features(fixed) - features(buggy))` — a canonical direction in source-feature space.
- P010's `SourceFeatureAdapter` ridge-regression fit uses these pairs as the training corpus.
- This layer is **concrete and actionable today** — it just needs P010 Phase 1 (the `syn`-based AST histogram extractor) to ship.

**Layer 2 — Functional test (Rust-SWE-bench as P032 Phase 6 gate):**
- Run Kimi-K3's forward pass on Rust-SWE-bench task inputs (issue + relevant code context).
- Check that the model's latent representations are coherent:
  - Does attention highlight issue-relevant code regions?
  - Does MoE routing assign issue tokens to consistent experts?
  - Does KDA recurrent state differentiate buggy vs fixed code?
- This is **mechanistic interpretability for code models** — measuring whether the model's internals "light up" correctly on real Rust code.
- This layer is **partially actionable** — needs P032 Phase 5 (safetensors loader) to complete, then the latent-extraction harness.

**Layer 3 — In-loop validation (the wild idea — WASM pruner):**
- Compile the task's test logic to WASM via rubrc.
- Load as a `ConstraintPruner` via `WasmPruner`.
- The model's inference loop proposes patches; the WASM pruner validates them in-sandbox.
- The PUCT analogy: just as PUCT search used Moka's policy/value heads to explore + evaluate moves, the inference loop uses the WASM pruner to explore + validate patches.
- This layer is **HIGHLY SPECULATIVE** — depends on rubrc maturity, WASM compilability of real Rust test suites, and the 0.40B model's capability.

### The WasmPruner ABI extension

The existing `BomberWasmPruner` ABI is:
```text
is_valid(depth, action_idx, state_ptr, state_len) -> i32
```

For SWE validation, the ABI would extend to:
```text
is_patch_valid(patch_bytes_ptr, patch_len, test_state_ptr, test_state_len) -> i32
```

Where:
- `patch_bytes` = the model's proposed patch (as a unified diff or token sequence).
- `test_state` = the serialized test runner state (compiled test functions + fixtures).
- Returns 1 if the patch passes the tests, 0 otherwise.

This is a straightforward extension of the existing zero-copy state buffer pattern (`ZeroCopyStateBuffer` in `bomber/wasm_state.rs`). The WASM module applies the patch to the in-memory codebase representation, runs the test functions, and returns pass/fail — all sandboxed with fuel limits.

### Why this might work where external cargo test doesn't

The moka lesson (Benchmark 205): **bringing the evaluator into the model's native inference path extracts more signal.** For Go, PUCT search with the policy/value heads natively achieved 98% win vs 74% for external alpha-beta. The mechanism: the evaluator's feedback steers the search in real-time, pruning bad branches early.

For SWE, the analogous mechanism:
- The model proposes a patch (in latent space / as token drafts).
- The WASM pruner immediately validates it (no cargo build delay).
- Invalid patches are pruned from the DDTree before expansion.
- Valid patches get higher relevance scores, steering the search.
- The feedback loop is **microseconds** (WASM sandbox) not **seconds** (cargo test).

This is the SpeculativeGenerator + ConstraintPruner pattern (katgpt-rs's core architecture) applied to SWE: the model generates draft patches, the WASM pruner filters them, the DDTree explores valid branches.

## Honest caveats — READ BEFORE IMPLEMENTING

1. **rubrc is WIP + has hard dependency/proc-macro limitations.** Per the [rubrc README](https://github.com/oligamiq/rubrc) (verified 2026-08-01): rubrc runs `rustc` + `cargo` + `clang` + `llvm` + `rust-analyzer` as in-browser WASM modules via `wasi_virt_layer` (no OS subprocesses — a good fit for direct dispatch). Supported targets: only `wasm32-wasip1` + `x86_64-unknown-linux-musl`. **CRITICAL:** "external dependencies and procedural macros are currently unsupported" for Cargo. This is a near-total blocker for Rust-SWE-bench — the 34 repos (bevy, tokio, clap, serde, axum, tracing, burn...) ALL have external deps + many use proc-macros (serde derives, clap derives, tokio macros, bevy reflection). The realistic rubrc-compilable subset of the 500 tasks is **near-zero today**. **This is the #1 risk, upgraded from "WIP" to "hard blocker".** Mitigation paths: (a) wait for rubrc to add dependency/proc-macro support (timeline unknown); (b) bypass rubrc entirely — hand-extract the test assertion logic + hand-compile minimal WASM modules with the `is_patch_valid` ABI (Phase 3 T3.1 already proposes this); (c) target only the simplest tasks (single-file crates with no deps — likely <10 of the 500). Layer 1 (probe corpus) and Layer 2 (functional test) do NOT depend on rubrc — they work today once P010/P032 ship.

2. **Rust-SWE-bench repos are large.** Average: 993 files, 128K lines. The largest (bevy) is 15K files, 753K lines. Compiling these to WASM may be intractable. Mitigation: the WASM pruner only needs the **test logic + the code under test**, not the entire repo. A subset extraction step (identify which files the test patch touches, compile only those) would reduce scope.

3. **The 0.40B model may not have enough capability.** RustForger (with Claude-Sonnet-3.7, a much larger model) achieves only 28.6% resolution. A 0.40B distillation will be dramatically weaker. The WASM pruner may reject everything the model proposes. **This is the #2 risk.** Mitigation: the goal is NOT to match RustForger's resolution rate. The goal is to measure whether the model's latent representations are coherent on real Rust code (Layer 2) and whether the in-loop validation pattern works at all (Layer 3 — a POC on a tiny subset).

4. **WASM compilation of test suites with external dependencies may fail.** Many Rust-SWE-bench repos use crates that don't compile to `wasm32-unknown-unknown` (e.g., crates using `std::process`, file I/O, networking). Mitigation: filter tasks to those whose test suites are WASM-compatible (no `std::process`, no networking, no filesystem). This likely reduces the 500-task corpus significantly, but even 50 WASM-compatible tasks would be a useful POC.

5. **The "resolution in latent space" claim (Thread C) is unproven.** SWE resolution has no closed-form win condition like Go's territory score. The honest version of Layer 3 is "in-loop validation via WASM pruner" (the pruner gives pass/fail feedback), NOT "latent-space resolution" (the latent state magically contains the fix). The R463 lesson applies: "storage format ≠ capability" → "WASM in-loop validation ≠ resolution capability." If the model can't propose a valid patch, the pruner will reject it regardless of whether validation is in-loop or external.

6. **This is architecturally adjacent to Proposal 010's speculation level.** P010's verdict: "HIGHLY SPECULATIVE... exists because it's the ONLY remaining path, not because it's likely to succeed." This proposal inherits that risk. If P010's G5 (cross-arch agreement) fails, the source-feature directions are meaningless, and Layer 1's "fix directions" are noise. Layer 3 can still work as a pure WASM-pruner POC (independent of P010), but the fusion value collapses.

7. **`syn` dependency weight.** Same as P010 — adding a full Rust parser to the canon crate's feature surface. Mitigation: same as P010 (behind `canon_source_features` feature flag).

## Fusion lineage

This proposal combines **five** existing substrate pieces + **one** external dependency + **one** benchmark dataset:

1. **`WasmPruner` / `BomberWasmPruner`** (`katgpt-pruners/src/hot_swap.rs`, `src/pruners/bomber/wasm_pruner.rs`) — the WASM-module-as-ConstraintPruner substrate. Fuel-limited, sandboxed, papaya instance pool, zero-copy state buffer, batch API. This is the load-bearing substrate for Layer 3.
2. **`SpecAsPruner`** (`katgpt-pruners/src/spec_compile/`) — compiles NL specs into symbolic bitmap rules. Layer 1's "fix direction" is a symbolic rule compiled from source features, not a neural adapter. This is the ideological ancestor.
3. **`HotSwapPruner`** (`katgpt-pruners/src/hot_swap.rs`) — BLAKE3-hash-detected runtime reload. Enables swapping the WASM pruner when the Rust-SWE-bench task changes.
4. **Proposal 010** (`katgpt-rs/.proposals/010_non_hidden_state_canonical_construction.md`) — `SourceFeatureDirection` (AST histogram, Clippy fingerprint, ownership graph) → canonical directions from source code. Rust-SWE-bench is its probe corpus.
5. **Proposal 009** (`katgpt-rs/.proposals/009_canonical_intent_space.md`) — `CanonicalIntent` + `ModelAdapter` (Procrustes, Subspace, Mask). Projects source-feature directions into model latent space.
6. **Proposal 032** (`riir-ai/.proposals/032_kimi_k3_native_support.md`) — Kimi-K3 native support (MLA + MoE + KDA + SiTU). The model being validated.
7. **rubrc** ([github.com/oligamiq/rubrc](https://github.com/oligamiq/rubrc), MIT OR Apache-2.0) — WASM-hosted Rust toolchain running in a browser worker via `wasi_virt_layer`. Embeds `rustc_opt.wasm` + `cargo_opt.wasm` + `llvm_opt.wasm` + `lsp_opt.wasm`. Supported targets: `wasm32-wasip1` + `x86_64-unknown-linux-musl` only. **Hard limitation (verified 2026-08-01):** external dependencies + procedural macros are unsupported in Cargo. Serialized execution via `CARGO_RUN_LOCK` / `RUSTC_RUN_LOCK`. The in-process module-dispatch architecture (no OS subprocesses) is a good fit for the WASM-pruner pattern, but the dependency/proc-macro blocker makes the rubrc path near-viable for real Rust-SWE-bench tasks today. The compiler that *would* turn Rust test suites into WASM pruner modules — once the dependency limitation lifts.
8. **Rust-SWE-bench** ([arXiv:2602.22764](https://arxiv.org/abs/2602.22764), Xiang et al. ICSE '26) — 500 real-world Rust SWE tasks from 34 repos. The benchmark dataset.

The combination produces what none alone can: a path to evaluate SWE capability **inside the model's inference loop** using the existing WASM-pruner substrate, validated against a real-world Rust benchmark, with source-feature-based canonical directions connecting the code structure to the model's latent space.

## GOAT gate

This proposal does NOT request default-on promotion. It requests **research validation** behind an opt-in feature flag. The gates differ per layer:

| Layer | Feature flag | G1 correctness | G2 perf | G3 no-reg | G4 alloc | G5 (decisive) |
|---|---|---|---|---|---|---|
| 1 (corpus) | `swe_bench_corpus` | source-feature extraction deterministic (BLAKE3) | AST histogram on 10K-line crate < 1s (same as P010) | opt-in, no default impact | projection apply zero-alloc | **fix directions correlate with ground-truth patches** (threshold TBD) |
| 2 (functional test) | `kimi_k3_swe_probe` | latent extraction matches PyTptorch ref attention pattern | forward pass on 500 tasks < 60s total | existing P032 tests pass | per-task latent extraction alloc-free | **attention highlights issue-relevant code regions** (measurable via attention-rollout correlation with ground-truth fix locations) |
| 3 (WASM pruner) | `swe_bench_wasm_pruner` | WASM module produces same pass/fail as cargo test | WASM validation < 100ms per patch (vs seconds for cargo) | existing WasmPruner tests pass | sandbox state buffer reused | **in-loop validation prunes more invalid patches than no-validation baseline, improving resolution rate on a WASM-compatible subset** |

**No "Report the Floor" rule applies** — this is not a UQ-bearing primitive (no probability distribution / coverage claim).

**Promotion criterion:** Layer 3 G5 is the load-bearing gate. If it passes on even a 50-task WASM-compatible subset, the pattern is proven. If it fails, Layer 1 + Layer 2 still ship as research validation (probe corpus + functional test) — they don't depend on Layer 3.

## What ships now (katgpt-rs) vs deferred

### Ships now — katgpt-rs (if validated)
- `RustSweBenchTask` struct (task ID, repo, issue text, buggy commit, fixed commit, test patch path).
- `RustSweBenchCorpus` loader (reads the 500-task dataset, filters by WASM-compatibility heuristic).
- `source_features` extraction adapter (P010 integration — uses P010's AST histogram on buggy + fixed code to compute fix directions).
- `SweBenchWasmPruner` (Layer 3 — extends `WasmPruner` with the `is_patch_valid` ABI). Behind `swe_bench_wasm_pruner` feature.
- G1/G2/G4 gates on extraction + WASM validation.
- **G5 runs in riir-ai** (needs Kimi-K3 loaded).

### Deferred — riir-ai
- Kimi-K3 latent extraction on Rust-SWE-bench tasks (Layer 2 functional test).
- In-loop validation wiring (Layer 3 — the WASM pruner integrated into Kimi-K3's DDTree inference loop).
- Real-model GOAT gate (P032 Phase 6 extension).

### Deferred — external dependency
- **rubrc maturity** — the entire Layer 3 depends on rubrc being able to compile real Rust test suites. If rubrc cannot, Layer 3 is blocked and only Layer 1 + Layer 2 ship.

### Explicitly NOT shipped by this proposal
- **A competing SWE agent** — this is NOT "build our own RustForger." The goal is latent-space validation, not agent resolution rate competition.
- **Training on Rust-SWE-bench** — modelless-first mandate. If the 0.40B model can't resolve tasks, the answer is "needs training" (→ riir-train), NOT "train on the benchmark."
- **Default-on promotion** — this is research validation. Promotion (if G5 passes) requires a separate proposal.
- **Full 500-task WASM compilation** — even if rubrc works, compiling all 500 tasks' test suites is a massive effort. The POC targets a WASM-compatible subset (estimated 50-100 tasks after filtering).

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — Corpus loader + source-feature extraction (Layer 1)
- [ ] T1.1 Download Rust-SWE-bench dataset (500 tasks from [GitHub](https://github.com/GhabiX/Rust-SWE-Bench))
- [ ] T1.2 `RustSweBenchCorpus` loader struct (task ID, repo path, commit hashes, patch paths)
- [ ] T1.3 P010 integration: `ast_histogram` on buggy + fixed code → fix direction computation
- [ ] T1.4 WASM-compatibility filter (heuristic: reject tasks whose test deps use `std::process` / networking / filesystem)
- [ ] T1.5 G1 correctness: deterministic extraction (BLAKE3 commitment)
- [ ] T1.6 G2 perf: AST histogram on the largest repo (bevy subset) < 5s

### Phase 2 — Functional test harness (Layer 2, gated on P032 Phase 5)
- [ ] T2.1 Latent extraction: run Kimi-K3 forward pass on task inputs, extract MLA attention + MoE routing + KDA state
- [ ] T2.2 Attention-rollout correlation: does attention correlate with ground-truth fix locations?
- [ ] T2.3 MoE routing consistency: do issue-relevant tokens route to consistent experts?
- [ ] T2.4 G5: measurable signal on a 50-task subset

### Phase 3 — WASM pruner POC (Layer 3, gated on rubrc maturity)
- [ ] T3.1 Manual WASM compilation of 1 task's test suite (hand-compiled, not via rubrc — proves the ABI works)
- [ ] T3.2 `SweBenchWasmPruner` impl (extends `WasmPruner` with `is_patch_valid`)
- [ ] T3.3 Integration with DDTree: invalid patches pruned before expansion
- [ ] T3.4 rubrc evaluation: can it compile a real Rust-SWE-bench test suite?
- [ ] T3.5 G5: in-loop validation improves resolution rate vs no-validation baseline on the WASM-compatible subset

### Phase 4 — Scale + fusion (only if Phase 3 G5 passes)
- [ ] T4.1 Scale to 50+ WASM-compatible tasks
- [ ] T4.2 P009 adapter: project P010 fix directions into Kimi-K3 latent space, steer toward valid patches
- [ ] T4.3 PUCT-style search: combine WASM pruner feedback + policy prior + value estimate
- [ ] T4.4 Full GOAT gate + honest negative result if it fails

## Risks

1. **rubrc dependency/proc-macro blocker (highest, upgraded).** Verified 2026-08-01 via the rubrc README: external dependencies + procedural macros are unsupported. ALL 34 Rust-SWE-bench repos have external deps + most use proc-macros (serde/clap/tokio/bevy derives). The rubrc-compilable subset is near-zero today. Layer 3 via rubrc is blocked until rubrc adds dependency support (timeline unknown). Mitigation: Phase 3 T3.1 (hand-compiled WASM module) proves the ABI without rubrc. If the hand-compiled POC shows the pattern works, the question becomes "when does rubrc unblock deps?" not "does the architecture work?".

2. **WASM-compatibility filtering reduces corpus severely.** Many Rust-SWE-bench tasks have test suites with external deps (tokio, filesystem, networking) that don't compile to wasm32. The WASM-compatible subset may be < 50 tasks. Mitigation: even a small subset proves the pattern; scaling is a follow-up.

3. **0.40B model capability ceiling.** The model may not propose any valid patches, making the WASM pruner reject everything. Mitigation: Layer 2 (functional test) measures internal coherence regardless of output quality — it's the mechanistic-interpretability angle, not the resolution-rate angle.

4. **Source features too coarse (inherited from P010).** If P010's G5 (cross-arch agreement) fails, the fix directions are noise. Mitigation: Layer 3 (WASM pruner) works independently of P010 — it validates patches against real test logic, regardless of whether the source-feature direction is meaningful.

5. **Scope creep into "build a SWE agent."** This proposal is about latent-space validation, not agent competition. Mitigation: the GOAT gate measures internal coherence (Layer 2) + in-loop validation pattern (Layer 3), NOT resolution rate vs RustForger. Comparing to RustForger's 28.6% is a category error — they're different goals.

6. **Dataset licensing.** Rust-SWE-bench is CC-BY 4.0 (academic). Using it as a probe corpus in a commercial product needs license review. Mitigation: the corpus is used for adapter fitting (setup-time, not shipped), and the WASM pruner modules are derived artifacts. Legal review before any production use.

## Out of scope

- **Building a competing SWE agent.** This is latent-space validation, not "our RustForger."
- **Training on Rust-SWE-bench.** Modelless-first mandate. Training → riir-train.
- **Cross-language SWE benchmarks.** Rust-only (the source features + rubrc are Rust-specific).
- **Full 500-task WASM compilation.** POC on a WASM-compatible subset first.
- **Default-on promotion.** Research validation only.
- **Production steering runtime.** The adapter fit + WASM validation is the substrate; runtime integration (riir-ai NPC cognition, seal consumer) is a separate plan.

## References

1. **Rust-SWE-bench / RustForger** — Xiang, He, Wang, Tian, Zhang (SUSTech / Ant Group, ICSE '26). [arXiv:2602.22764](https://arxiv.org/abs/2602.22764). 500 real-world Rust SWE tasks from 34 repos + RustForger agent (proc-macro AST Trace command, 28.6% resolution). Used here as the probe corpus + functional test target. The RustForger agent itself is PASS (LLM-dependent semantic code generation); the benchmark dataset is Gain.
2. **Proposal 010** — [katgpt-rs/.proposals/010_non_hidden_state_canonical_construction.md](../.proposals/010_non_hidden_state_canonical_construction.md). Source-feature directions (AST histogram, Clippy fingerprint, ownership graph). The enabling substrate for Layer 1.
3. **Proposal 009** — [katgpt-rs/.proposals/009_canonical_intent_space.md](../.proposals/009_canonical_intent_space.md). Canonical Intent Space + ModelAdapter. The enabling substrate for projecting fix directions into model latent space.
4. **Proposal 032** — [riir-ai/.proposals/032_kimi_k3_native_support.md](../../riir-ai/.proposals/032_kimi_k3_native_support.md). Kimi-K3 native support (MLA + MoE + KDA + SiTU). The model being validated (Phase 6 GOAT gate).
5. **Benchmark 205** — [katgpt-rs/.benchmarks/205_puct_search_vs_moka_win.md](../.benchmarks/205_puct_search_vs_moka_win.md). PUCT search vs Moka (98% win). The moka precedent: bringing the benchmark INTO the model's native inference path.
6. **Research 463** — [katgpt-rs/.research/463_moka_freeze_thaw_lever_audit.md](../.research/463_moka_freeze_thaw_lever_audit.md). The "storage format ≠ capability" caveat. Applies to Layer 3: WASM in-loop validation ≠ resolution capability.
7. **rubrc** — [github.com/oligamiq/rubrc](https://github.com/oligamiq/rubrc). WASM-compiled rustc (WIP). External dependency for Layer 3.
8. **`WasmPruner` / `BomberWasmPruner`** — `katgpt-rs/crates/katgpt-pruners/src/hot_swap.rs`, `katgpt-rs/src/pruners/bomber/wasm_pruner.rs`. The existing WASM-module-as-ConstraintPruner substrate.
9. **`SpecAsPruner`** — `katgpt-rs/crates/katgpt-pruners/src/spec_compile/mod.rs`. Compiles NL specs into symbolic bitmap rules. The ideological ancestor (symbolic rules, not neural adapters).
10. **code2vec** — Alon et al. (ICLR 2019, [arXiv:1803.09473](https://arxiv.org/abs/1803.09473)). AST path embeddings. P010 borrows the AST→fixed-vector idea (deterministic, not learned).

## TL;DR

**Verdict: HIGHLY SPECULATIVE proposal for research validation, NOT production.** This sketches the path to bring Rust-SWE-bench INTO the model's inference loop via the existing WASM-pruner substrate, using P010 source features + P009 adapters to connect code structure to latent space, validated against P032 (Kimi-K3). Layer 1 (probe corpus) is concrete + actionable today. Layer 2 (functional test) is gated on P032 Phase 5. Layer 3 (WASM pruner) is HIGHLY SPECULATIVE — gated on rubrc maturity + WASM-compilability of real Rust test suites. The proposal exists to be evaluated and potentially rejected with reasoning. Next action: if the user approves, the cheapest validation step is Phase 1 T1.1-T1.3 (download Rust-SWE-bench + run P010's AST histogram on the (buggy, fixed) pairs to compute fix directions). If those directions show no signal, Layer 1 fails early and the proposal is demoted.
