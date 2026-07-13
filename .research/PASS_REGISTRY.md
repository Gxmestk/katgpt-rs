# PASS Registry — Papers Evaluated and Declined

> **One row per PASS verdict. Prevents re-evaluation.** Created 2026-07-13 per
> research-skill §3.6 "PASS verdict output contract" refinement.
>
> PASS = **zero new `.md` files** — no numbered note, no plan, no guide. The
> row IS the verdict record and the re-evaluation guard. The full mandatory
> pre-flight still runs before any PASS — zero files ≠ zero diligence.
>
> See `.agents/skills/research/SKILL.md` §3.6 for the full contract.

## Column format

| Column | Content |
|---|---|
| `#` | Monotonic row number (never reused, never resequenced). |
| Date | Verdict date (YYYY-MM-DD). |
| arxiv | arxiv ID (or other stable source identifier). |
| Title | Short paper title. |
| Repo | Which repo(s) the verdict applies to (usually "all" — PASS means no repo benefits). |
| One-line reason | The honest reason. Must distinguish: (a) LLM-orchestration class, (b) training-only (→ riir-train), (c) already-ships architectural coverage, (d) out-of-scope. |
| Closest shipped cousin | The shipped primitive that covers the mechanism (grep target for re-evaluation). |
| Quality claim | `architectural-only` (default), `PoC-confirmed parity` (rare), or `N/A` (training-only / out-of-scope). |
| Trace | Pointer to full reasoning: a legacy numbered note (R133, R169, R289) or `(registry-only)` for new PASSes. |

## Registry

| # | Date | arxiv | Title | Repo | One-line reason | Closest shipped cousin | Quality claim | Trace |
|---|------|-------|-------|------|-----------------|------------------------|---------------|-------|
| 1 | 2026-05-29 | 2605.28773 | FluxMem (Connectivity-Evolving Memory) | all | LLM-orchestration memory class; every architecture class ships modellessly; LLM-call-heavy contradicts 20Hz budget | Engram + DeltaMemory + AnyRAG + Four-Tier Memory | architectural-only | [R133](133_FluxMem_Connectivity_Evolving_Memory.md) |
| 2 | 2026-07-02 | 2606.24775 | AgentMemBench (Agent-Native Memory) | all | Benchmark paper, no mechanism; LLM-orchestration class; all 12 evaluated systems make ≥1 LLM call/step | Five-Tier Memory + Raven/δ-Mem + MAPE-K | architectural-only | [riir-ai R169](../../riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md) |
| 3 | 2026-06-22 | 2604.25917 | RecursiveMAS (Recursive Multi-Agent Systems) | all | All primitives ship at higher fidelity; training recipe (inner-outer loop co-optimization) → riir-train | Plan 311 NPC mind-reading + latent_functor + FuncAttn rank-k | architectural-only | [R289](289_RecursiveMAS_Pass_Already_Shipped.md) |
| 4 | 2026-07-13 | 2607.01942 | Atomic Task Graph (ATG) | all | LLM-orchestration agent-control framework; all 3 mechanisms ship (recursive graph compilation, thought experiment, minimal subgraph repair) | TrajectoryDoctor + AND-OR DDTree + ReestimationScheduler + CUCG + GdnTreeVerifier | architectural-only | (registry-only) |

## Notes

- **Rows 1–3** are backfills from legacy full PASS notes (R133, R169, R289) that
  predate the zero-file rule. Those notes remain as historical record; new
  PASSes are registry-only.
- **Row 4** (ATG) is the first registry-only PASS. The original 163-line note
  (R416) was removed as noise per the zero-file refinement; the reasoning is
  captured in this row + git history (commit `2c5a43f1`, now reverted by the
  deletion commit).
- **Numbering:** research-note number 416 is permanently burned (allocated then
  deleted per the AGENTS.md numbering-discipline rule). `.research/.highwater`
  is at 417 (bumped by a concurrent agent). The registry's `#` column is
  independent of the note numbering.
