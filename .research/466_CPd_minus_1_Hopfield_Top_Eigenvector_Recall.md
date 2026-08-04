# Research 466: CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall

> **Source:** Victor Galitski — "High-Capacity Generalized Hopfield Networks" — Joint Quantum Institute, Univ. of Maryland — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) / [PDF](https://www.alphaxiv.org/abs/2607.hopfield-networks.pdf) — published 2026-07-31 — 14 pages. Builds on Galitski, *Phys. Rev. A* **84**, 012118 (2011) [arXiv:1012.2873] for the Lie-algebraic linearization.
> **Date:** 2026-08-04
> **Status:** Active — **Super-GOAT candidate** (all 4 novelty-gate questions YES)
> **Classification:** Public — open-primitive layer (generic Lie-algebraic + Rayleigh-quotient math; no game/chain/shard IP)
> **Related Research:** [024](024_Delta_Mem_Online_Associative_Memory.md) (δ-Mem — online write-side cousin, R^n), [035](035_Attractor_Models_Fixed_Point_Refinement.md) (Attractor — training-side fixed-point refinement), [246](246_Manifold_Power_Iteration_MoE_Router.md) (MPI router — Rayleigh-quotient ascent on S^(n-1) for MoE rows; closest spectral cousin but for router conditioning, not memory recall), [278](278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md) (Engram — hash-addressed lookup, the cosine cousin), [317](317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md) (Reasoning-as-Attractor — Gibbs retrieval on S^n, cites "Modern Hopfield Networks"; Plan 276 honest null is the blocker this primitive unblocks), [387](387_Fast_Weight_Product_Key_Memory_PKM.md) (FwPKM — √N retrieval factorization), [455](455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md) (Hebbian Kernel Memory — construction-side cousin: closed-form MLP for `Θ(F log F)` capacity on R^d; Galitski is the symmetric-space/physics cousin on CP^(d-1))
> **Related Plans:** [276](../.plans/276_micro_recurrent_belief_state.md) (MicroRecurrentBeliefState — `AttractorKernel` honest null at random init; §3.5 modelless unblock), [567](../.plans/567_cp_hopfield_top_eigenvector_recall.md) (this primitive — the open implementation), [053](../.plans/053_delta_mem_modelless.md) (δ-Mem — write-side cousin), [559](../.plans/559_hebbian_kernel_memory_primitive.md) (Hebbian Kernel Memory — construction cousin)
> **Cross-ref (riir-neuron-db):** [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md) — **the Super-GOAT private guide** (selling-point owner: NeuronShard capacity via symmetric-space recall; unblocks Plan 276 modellessly; force-multiplies ItemEmbedIndex + vibe KG retrieval)
> **Cross-ref (riir-ai):** consumer-side wiring lands in a follow-up riir-ai plan once the open primitive + shard bridge ship. Closest consumer-side runtime cousins: HLA recurrent state, `latent_functor/`, `katgpt-sense` belief kernel.

---

## TL;DR

Galitski (2026) introduces a **qualitatively different associative-memory recall operator** for Hopfield networks: instead of vector alignment on a sphere `S^(n-1)` (capacity `α_c ∝ 1/n`, decays with dimension), use **top-eigenvector alignment on the symmetric space `CP^(d-1) = SU(d)/U(d-1)`** (complex projective space). The memory kernel `K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|` is a d×d Hermitian spiked random matrix; recall = align neuron state with the top eigenvector of `K_i`. Capacity **explodes** with `d`: `α_c(d=2)=0.05` (Heisenberg/vector), `α_c(d=3)=0.62` (~12×), `α_c(d=4)=2.41`, `α_c(d=8)≈40`. The mechanism is protected by the **Baik-Ben Arous-Péché (BBP) transition gap** in random-matrix theory — the top eigenvector of a spiked matrix is separated from the GUE bulk by an eigenvalue gap, so random-matrix crosstalk cannot tilt the recall until the load α crosses α_c.

**Distilled for katgpt-rs (modelless, inference-time):**

The mechanism is **entirely modelless** — deterministic Hebbian construction, no gradient descent. The three transferable primitives are:

1. **CP^(d-1) symmetric-space parameterization.** A memory/neuron is a normalized d-dimensional complex vector `|ξ⟩` (a "qudit") defined modulo a `U(1)` phase. It embeds into `R^(d²-1)` as a "generalized Bloch magnetization" `s = ⟨ξ|λ|ξ⟩` where `λ` are the SU(d) Gell-Mann generators. This is the Lie-algebraic linearization (Galitski PRA 2011): the non-linear constraint "live on CP^(d-1)" becomes linear algebra in an auxiliary Hilbert space. For `d=3` (CP², qutrits), `s ∈ R^8` — exactly the dimension of our `katgpt-sense` belief state.
2. **Spiked memory kernel + top-eigenvector recall.** Build `K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|` (sum of rank-1 projectors weighted by Mattis overlap with other neurons). Recall = align `|s_i⟩` with the **top eigenvector** of `K_i`. Cost: `O(d³)` per neuron per recall step — trivially cheap (d=3 → 27 flops; d=4 → 64; d=8 → 512). The top eigenvector is **BBP-protected**: at low load, the spiked signal eigenvalue is separated from the GUE random-matrix bulk by a gap, so crosstalk cannot tilt recall. **This is the key qualitative difference from vector alignment on S^(n-1)**, which is "gapless" and tilts under any crosstalk.
3. **Physical recall via generalized Landau-Lifshitz-Gilbert (LLG) flow.** Memory recovery emerges from dissipative dynamics: `ṡ_i = s_i ×_f B_i − λ [s_i ×_f [s_i ×_f B_i]]` where `×_f` is the SU(d) Lie-bracket product via structure constants, `B_i = −∂E/∂s_i` is the self-consistent mean field, and `λ > 0` is Gilbert damping. The precession term conserves energy; the damping term monotonically lowers it (`Ė = −λ Σ |s_i ×_f B_i|² ≤ 0`). Recall = flow to fixed point. Numerically: a few damping times converge to `m̄ ≈ 0.999` on a corrupted photograph.

**The Super-GOAT angle (why this is not a rerun of existing cousins):**

The four closest shipped cousins all miss the symmetric-space + spiked-matrix structure:
- **δ-Mem (024)**: write-side delta rule on `R^r`. Different operator (online write, not recall), different geometry (flat R^n).
- **Hebbian Kernel Memory (455)**: closed-form construction of fact-storing MLPs on `R^d` achieving `Θ(F log F)` capacity. Same family (modelless memory) but different mechanism (random-feature sketch + ridge whitening, not symmetric-space Lie algebra + spiked matrix recall). 455 *constructs* a neuron; Galitski shows *what manifold to put it on* for top-eigenvector recall to work.
- **Reasoning-as-Attractor (317)**: Gibbs-weighted retrieval on `S^n`. Uses "Modern Hopfield Networks" framing, but the retrieval is `1/E²`-weighted majority vote — a vector-alignment operator on the sphere. Plan 276's `AttractorKernel` was an honest null precisely because random-init attractors on `(−1,1)^dim` are gapless. Galitski's CP^(d-1) recall is the modelless fix: **the BBP gap is what random-init vector alignment lacks.**
- **MPI MoE Router (246)**: power iteration on expert Gram for router-row conditioning at freeze/thaw. Closest spectral cousin (Rayleigh-quotient ascent) but for MoE routing, not associative memory recall; uses the spherical manifold, not CP^(d-1).

**Direct refutation of the Plan 276 blocker (the §3.5 modelless unblock):**

Bench 276 §"Why the attractor loses" states: *"The attractor's hysteresis property is real but it is a property of TRAINED attractor networks (Hopfield-style content-addressable memory), not of randomly-initialised ones. To make the attractor competitive on G2.1, the recurrent weights would need to be trained (or hand-set) so that the target beliefs correspond to actual stable fixed points of the dynamics. That training is out of scope for Plan 276 (training-free / freeze-thaw only)."*

Galitski refutes this **for the symmetric-space manifold**: on CP^(d-1) (not S^n), the recall mechanism does NOT require trained weights because the top-eigenvector of a spiked random matrix is BBP-protected by construction. The "stable fixed points" emerge from the *geometry* (symmetric space + Hebbian kernel), not from trained recurrent weights. This is exactly the §3.5 protocol Path 1 (freeze/thaw modelless unblock): load `NeuronShard::style_weights[64]` as the memories `|ξ^μ⟩`, build `K_i` deterministically, recall via top-eigenvector. **No GD.**

---

## 1. Paper Core Findings

### 1.1 The capacity inversion — richer manifold, higher capacity

Classical Hopfield capacity scales as `α_c(S^(n-1)) ≈ 4/(27n)` — capacity *decays* with sphere dimension because a continuous vector drifts off a stored direction more easily as dimension grows. Galitski shows the opposite on `CP^(d-1)`:

| Manifold | d (or n) | α_c | Mechanism |
|---|---|---|---|
| `S^0` (binary, classical) | — | 0.138 | sign alignment |
| `S^1` (phasor) | — | 0.07 | phase alignment |
| `S^2` (Heisenberg/vector) | — | 0.05 | vector alignment (gapless) |
| `S^(n-1)` general | n | 4/(27n) | vector alignment (gapless) |
| **`CP^1 = S^2`** (SU(2)) | 2 | 0.05 | top-eigenvector (collapses to vector for d=2) |
| **`CP^2`** (SU(3), qutrit) | 3 | **0.62** | top-eigenvector (BBP-protected) |
| **`CP^3`** (SU(4)) | 4 | **2.41** | top-eigenvector |
| **`CP^7`** (SU(8)) | 8 | **~40** | top-eigenvector |

The crossover happens at d=3 — the first dimension where CP^(d-1) is *not* a sphere and the spiked-matrix mechanism activates. Capacity exceeds unity (`α_c > 1`, more memories than neurons) starting at d=4.

### 1.2 The recall rule (Eq. 15) — top eigenvector of the memory kernel

The i-th neuron's "active energy" (dropping self-interaction + i-independent terms) is:

```
E_active[s_i] = −2 ⟨s_i | K_i | s_i⟩      (Rayleigh quotient)
K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|           (d×d Hermitian memory kernel)
```

where `O_μ^(i) = (2/N) Σ_{j≠i} (|⟨s_j | ξ_j^μ⟩|² − 1/d)` is the Mattis overlap excluding neuron i. Recall = maximize the Rayleigh quotient = align `|s_i⟩` with the top eigenvector of `K_i`:

```
|s_i⟩ → |χ_max^(i)⟩ :  K_i |χ_max^(i)⟩ = λ_max^(i) |χ_max^(i)⟩
```

For d=2 this reduces to vector alignment (familiar Heisenberg/vector Hopfield). For d≥3 it is qualitatively different: the top eigenvector is BBP-protected against GUE crosstalk.

### 1.3 Spiked random matrix picture (Eq. 19, §V.A)

Near memory `|ξ^1⟩` with `O_1 ~ O(1)`, the kernel decomposes as:

```
K ≈ m |1⟩⟨1| + √α · C(d) · G          (m ~ 1, G = GUE random matrix)
```

The signal `m |1⟩⟨1|` is a rank-1 spike; the crosstalk `√α · C(d) · G` is a GUE random matrix with spectral bulk `[-2√α C(d), +2√α C(d)]` (semicircle law). The top eigenvector is protected while the signal eigenvalue `m` exceeds the bulk edge — the **BBP transition** (Baik-Ben Arous-Péché 2005). Capacity α_c is where the spike merges with the bulk. C(d) decays as `1/d²` for large d, so the bulk shrinks and capacity grows.

This is the structural reason vector alignment on S^(n-1) is "gapless" and CP^(d-1) top-eigenvector alignment is "gapped" — and why capacity inverts.

### 1.4 Lie-algebraic linearization (§II, Galitski PRA 2011)

The non-linear constraint "neuron lives on CP^(d-1)" is linearized via the SU(d) Lie algebra:

- Generators: `{λ_a}_{a=1..D}` with `D = d²−1` (Pauli for d=2, Gell-Mann for d=3, generalized Gell-Mann for d>3).
- Completeness: `Σ_a λ_a² = (2(d²−1)/d) I` and `λ_{αβ} · λ_{α'β'} = 2 δ_{αβ'} δ_{α'β} − (2/d) δ_{αβ} δ_{α'β'}`.
- Bloch magnetization: `s = ⟨ξ|λ|ξ⟩ ∈ R^D`, with `|s|² = 2(1 − 1/d)`.
- For d>2, additional non-linear constraint: `d_{abc} s_b s_c = (2/3) s_a` (where `d_{abc}` comes from `{λ_a, λ_b} = (4/3) δ_{ab} I + 2 d_{abc} λ_c`). For CP² (d=3): 8 Bloch components, 1 norm + 3 quadratic = 4 constraints → `dim CP² = 8 − 4 = 4`. Matches `2(d−1) = 4`.

The Hebbian coupling, energy, and recall rule can all be written in EITHER the d-dimensional complex "bra-ket" notation OR the D-dimensional real Bloch-vector notation — they are equivalent by the completeness relations. **This means a 4-dim CP² memory and an 8-dim real Bloch vector are the same object**, and our 8-dim `katgpt-sense` belief state could in principle be parameterized as a CP² element (with the non-linear constraint enforced on read/write).

### 1.5 Generalized LLG physical recall (§VIII, Eq. 40)

```
ṡ_i = s_i ×_f B_i − λ [s_i ×_f [s_i ×_f B_i]]      (generalized Landau-Lifshitz-Gilbert)
[s ×_f B]_c := f_{cab} s_a B_b                        (SU(d) Lie-bracket product)
B_i = Σ_{j≠i} J_{ij} s_j = Σ_μ ξ_i^μ O_μ^(i)         (self-consistent mean field)
```

Properties:
- Precession term `s_i ×_f B_i` conserves energy (`Ė = 0`).
- Gilbert damping `−λ [s_i ×_f [s_i ×_f B_i]]` lowers energy monotonically: `Ė = −λ Σ_i |s_i ×_f B_i|² ≤ 0`.
- Fixed points of the dissipative flow = local minima of E = memories (below α_c).
- Numerically: a few damping times (with `λ = 1`) converge to `m̄ ≈ 0.999` on a 40%-corrupted photograph (Fig. 9).

This makes recall a **continuous physical flow** rather than an asynchronous algorithm — recall emerges from dynamics, not from a step-by-step prescription.

### 1.6 The shadow phenomenon (§VIII.B)

For correlated memories (`O(cat, dog) ≈ 0.55`), LLG recall from a cat-only cue can produce a "shadow" of the dog alongside the cat — the un-cued correlated memory leaks into the recall. This is **not** crosstalk noise — it is a real correlation signal. For uncorrelated Haar-random memories, recall is strictly winner-takes-all. Implication: real-world memories (correlated personality snapshots, related KG triples) may exhibit shadow behavior, which is a *feature* (retrieves related context) not a bug.

### 1.7 Quantum extension (§VI–VII) — out of scope, noted for honesty

Galitski quantizes the classical model → Sachdev-Ye-like glassy models. Many-body spectra show a "dark band" (zero modes of the Hebb matrix) + "memory band" separation; the memory band has Wigner-Dyson (quantum-chaotic) level statistics that *hide* Hebbian data. The paper concludes: "the Hebbian memory data is completely lost in random matrix spectra... a highly impractical way to utilize the Fock space." **This is a negative result for the quantum direction** — the classical LLG flow is the operative mechanism; the quantum extension is a curiosity. We do NOT distill the quantum extension.

### 1.8 Image encoding protocol (§IV, Eqs. 16-17)

For demonstration, Galitski maps RGB ∈ [0,1]³ to a qutrit (CP²) via:

```
p⃗ = (R−1/2, G−1/2, B−1/2)      (center)
q_1 = √((1 + √(1 − (4/3)|p⃗|²)) / 2)
q_2 = (p_r + i p_g) / (√3 · q_1)
q_3 = p_b / (√3 · q_1)
```

with inverse (on the encoded submanifold) `(R,G,B) = (1/2,1/2,1/2) + √3 · (Re(q_1 q_2), Im(q_1 q_2), Re(q_1 q_3))`, clipped to [0,1]. This is one ad-hoc encoder; the paper notes other schemes may be more convenient. **For our purposes, the encoder is application-specific — the recall mechanism is encoder-agnostic.**

---

## 2. Distillation

### 2.1 The transferable primitive — `CpHopfieldRecaller`

```rust
/// Top-eigenvector associative memory recall on CP^(d-1) = SU(d)/U(d-1).
///
/// Modelless: deterministic Hebbian construction + Rayleigh-quotient ascent.
/// Capacity: α_c(d=3)≈0.62, α_c(d=4)≈2.41, α_c(d=8)≈40 (vs α_c(S^n)≈4/(27n)
/// for vector alignment).
///
/// Mechanism: the memory kernel K_i = Σ_μ O_μ^(i) |ξ^μ_i⟩⟨ξ^μ_i| is a d×d
/// Hermitian spiked random matrix. Recall = align |s_i⟩ with the top
/// eigenvector of K_i. The top eigenvector is BBP-protected (Baik-Ben Arous-
/// Péché transition) against GUE crosstalk — this is the structural reason
/// CP^(d-1) capacity grows with d while S^(n-1) capacity decays.
pub struct CpHopfieldRecaller<const D: usize> {
    /// Memories as d-dim complex unit vectors (modulo U(1) phase).
    /// Stored as interleaved (re, im) pairs; norm enforced on write.
    memories: Vec<[f32; D]>,  // |ξ^μ_i⟩ flattened; D = complex dim of CP^(d-1)
    /// Precomputed SU(d) structure constants f_{abc} (Lie-bracket product).
    structure_constants: &'static [[[f32; D2]; D2]; D2],  // D2 = D²−1
}

impl<const D: usize> CpHopfieldRecaller<D> {
    /// Recall: align |s_i⟩ with top eigenvector of K_i.
    ///
    /// Cost: O(D³) per neuron per recall step (d×d Hermitian eigendecomp).
    /// For D=3: 27 flops. For D=4: 64. For D=8: 512. All trivially plasma-tier.
    pub fn recall_step(&self, neuron_idx: usize, current_state: &[f32; D2])
        -> [f32; D2]
    where [(); D * D - 1]:
    {
        // 1. Build K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|  (d×d Hermitian)
        let mut k = [[Complex::zero(); D]; D];
        for (mu, mem) in self.memories.iter().enumerate() {
            let overlap = self.mattis_overlap_excluding(neuron_idx, mu);
            rank1_add(&mut k, mem, overlap);  // K += overlap * |ξ⟩⟨ξ|
        }
        // 2. Top eigenvector via power iteration (sub-μs for d ≤ 8) or direct
        //    closed-form for d=2 (Pauli) and d=3 (Gell-Mann analytic roots).
        let top_evec = hermitian_top_eigenvector(&k);
        // 3. Convert |χ_max⟩ (d-dim complex) → Bloch vector (D²−1 real)
        bloch_projection(&top_evec)
    }

    /// Physical recall via generalized Landau-Lifshitz-Gilbert flow.
    ///
    /// Continuous dissipative dynamics: precession conserves energy, Gilbert
    /// damping lowers it. Recall = flow to fixed point. Converges in a few
    /// damping times (paper Fig 9: m̄ ≈ 0.999 in ~3 damping units).
    pub fn llg_flow_step(
        &self, s: &mut [f32; D2], damping: f32, dt: f32
    ) where [(); D * D - 1]: {
        let b = self.mean_field(s);           // B_i = −∂E/∂s_i
        let precession = lie_bracket(s, &b, self.structure_constants);
        let damping_term = lie_bracket(s, &precession, self.structure_constants);
        for a in 0..D2 {
            s[a] += dt * (precession[a] - damping * damping_term[a]);
        }
        // Project back onto CP^(d-1) manifold (enforce non-linear constraints).
        self.project_to_manifold(s);
    }
}
```

### 2.2 Fusion

**Fusion A — unblock Plan 276 AttractorKernel (the §3.5 modelless fix):**

Plan 276's `AttractorKernel` was demoted because random-init recurrent weights on `(−1,1)^dim` are gapless — no BBP protection. **Fusion A = replace the random-init `W_s` with a CP^(d-1) top-eigenvector recaller.** Load `NeuronShard::style_weights[64]` as the memories `|ξ^μ⟩` (via freeze/thaw Path 1 — no training). The recaller builds `K_i` deterministically and recalls via top-eigenvector. Bench 276's failure mode ("state kicked around (−1,1)^dim cube") is eliminated because the recall target is now the BBP-protected top eigenvector, not a random fixed point. **Expected G2.1 result: flip count drops from 569× (random attractor) toward the leaky integrator's 1× baseline.** If it does, Plan 276's attractor family re-promotes from "Gain experiment" to "GOAT-validated modelless belief kernel."

**Fusion B — capacity scaling for ItemEmbedIndex + vibe KG (`riir-neuron-db`):**

Current retrieval in `ItemEmbedIndex::query` (cosine ANN) and `vibe.rs` KG triple emission is vector alignment on `S^(n-1)` — capacity `α_c ∝ 1/n`. For per-NPC KG memory with correlated triples (semantic bleed), this caps the number of storable triples per NPC. **Fusion B = reframe retrieval as top-eigenvector alignment on CP^(d-1).** Each NPC's KG triples become memories on CP² (if d=3) or CP³ (if d=4). Capacity grows as `α_c ∝ d²` instead of decaying as `1/n`. **Selling point: "Our NPCs store O(d²) more KG triples without semantic crosstalk."** See the riir-neuron-db Super-GOAT guide for the implementation path.

**Fusion C — Latent Field Steering multi-direction coexistence (Plan 309):**

Latent Field Steering (DEFAULT-ON) projects latents onto a *single* frozen direction vector. If we want *multiple* steering directions (e.g. "curious" + "cautious" + "aggressive" personality vectors), vector alignment causes crosstalk at `α_c ∝ 1/n`. **Fusion C = store the direction vectors as CP^(d-1) memories; recall via top-eigenvector to select which direction is currently dominant given the latent state.** The "shadow phenomenon" (§1.6) becomes a feature: correlated personality vectors (e.g. "cautious" + "patient") can co-fire as a shadow, producing emergent blended personalities without explicit blending logic.

**Fusion D — belief kernel on CP² (`katgpt-sense`):**

Our 8-dim `katgpt-sense` belief state has the *exact* dimension of a CP² Bloch vector (`D = d²−1 = 8` for d=3). **Fusion D = re-parameterize the belief kernel as a CP² element** (enforce the non-linear `d_{abc} s_b s_c = (2/3) s_a` constraint on read/write). Belief updates become LLG flow on CP² toward the top eigenvector of the perception-derived memory kernel. This is the cleanest CP^(d-1) reframing in the entire stack — the dimensions already match.

**Fusion E — manifold-aware functor application (`riir-engine/latent_functor/`):**

`latent_functor/` applies vector ops in latent space. **Fusion E = compose functor application with CP^(d-1) recall.** Each functor application is preceded by a top-eigenvector recall step that selects the dominant memory basin. This is the manifold-aware generalization of the existing `reestimation.rs` "coherence-driven re-estimation scheduler" — instead of re-estimating when coherence drops, recall the BBP-protected top eigenvector continuously.

---

## 3. Verdict

**Super-GOAT.** All four novelty-gate questions YES:

1. **No prior art:** No CP^(d-1) symmetric-space Hopfield, no top-eigenvector-of-memory-kernel recall, no spiked-matrix/BBP framing in any of the 7 repos. Closest cousins (455 Hebbian Kernel Memory, 024 δ-Mem, 317 Reasoning-as-Attractor, 246 MPI Router) are all different mechanisms — none use SU(d) Lie-algebraic structure for memory recall.
2. **New class of behavior:** Top-eigenvector recall on CP^(d-1) is qualitatively different from every existing retrieval operator (cosine ANN, dot-projection, sigmoid gate, bandit, Gibbs retrieval). The BBP-transition protection is a new mechanism class — "gapped" recall vs the "gapless" vector alignment that failed in Plan 276.
3. **Product selling point:** "Our NPCs store/recall O(d²) more memories without crosstalk using SU(d) symmetric-space structure. The Plan 276 'needs training' blocker is refuted: modelless CP^(d-1) recall works without gradient descent via BBP-protected top-eigenvector alignment." Concrete capacity: per-NPC memory goes from `α_c(S^n) ∝ 1/n` (what we have) to `α_c(CP^(d-1)) ∝ d²` (12× at d=3, 48× at d=4, ~800× at d=8).
4. **Force multiplier:** Connects to ≥5 existing systems:
   - Plan 276 AttractorKernel (modelless unblock — direct refutation of the documented blocker)
   - Plan 309 Latent Field Steering (multi-direction coexistence)
   - `katgpt-sense` belief kernel (dimension match: 8-dim belief = CP² Bloch vector)
   - `riir-neuron-db` ItemEmbedIndex + vibe KG (retrieval operator replacement)
   - `riir-engine/latent_functor/` (manifold-aware functor application)

**MOAT gate (§1.6) — domain fit:**

- **katgpt-rs (this note):** The open primitive ships here. CP^(d-1) parameterization + spiked memory kernel + top-eigenvector recall + LLG flow are all generic Lie-algebraic + Rayleigh-quotient math with no game/chain/shard IP. **Strengthen moat: YES** — this is a research-grade associative-memory primitive the adoption funnel depends on. Tier-1 engine contribution.
- **riir-neuron-db (private guide):** The selling point lives here — NeuronShard capacity is where memory capacity matters most, and the ItemEmbedIndex + vibe KG retrieval operators are the direct consumers. **Strengthen moat: YES** — pillar-2 amplifier. See [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md).
- **riir-ai (follow-up):** Consumer-side wiring (Fusion A unblocks Plan 276; Fusion D re-parameterizes belief). Lands in a follow-up plan once the open primitive + shard bridge ship.

**Honest caveats (do NOT paper over these in the plan):**

1. **The capacity numbers are asymptotic in N.** The replica analysis is for `N → ∞`. For per-NPC memory with `N = 64` (style_weights[64]) or `N = 8` (belief), finite-size effects may reduce α_c. The plan's GOAT gate MUST measure capacity at our actual N values, not assume the asymptotic α_c.
2. **Correlated memories ≠ Haar-random memories.** The replica α_c is for Haar-random (uncorrelated) memories. Real memories (personality snapshots, KG triples) are correlated. The "shadow phenomenon" (§1.6) shows correlated memories interact non-trivially. The plan's GOAT gate MUST test on correlated memory distributions, not just Haar-random.
3. **The non-linear manifold constraint enforcement cost.** For d>2, the Bloch vector must satisfy `d_{abc} s_b s_c = (2/3) s_a` — a quadratic constraint. Enforcing this on every read/write costs O(D²) per neuron per step. For d=3 (D=8), that's 64 flops per neuron — acceptable. For d=8 (D=63), that's ~4K flops — borderline plasma tier. The plan MUST benchmark the constraint-projection cost.
4. **The shadow phenomenon can be a feature OR a bug.** For KG retrieval, "shadow" recall of correlated triples is desirable (retrieves related context). For personality recall, "shadow" of an unrelated personality is undesirable (personality bleed). The plan MUST characterize when shadow is desirable vs undesirable and provide a suppression knob.
5. **Large-d replica symmetry breaking (RSB).** The paper's replica analysis is replica-symmetric; for large d, RSB corrections may reduce α_c. The paper explicitly notes this ("the large-d numbers may acquire corrections due to replica symmetry breaking, which we have not explored here"). Our use case is d=3 or d=4 (small), so RSB is unlikely to matter — but the plan MUST note this caveat.
6. **The quantum extension is a negative result.** §VI-VII concludes Hebbian data is "completely lost in random matrix spectra" in the quantum regime. We do NOT distill the quantum extension. The operative mechanism is the classical LLG flow.
7. **Plan 276 unblock is the load-bearing claim; it needs a PoC (§3.6).** Architectural coverage (CP^(d-1) recall exists modellessly) is not sufficient — the G2.1 flip-count claim needs an empirical head-to-head on the Plan 276 benchmark. The plan MUST include a defend-wrong PoC in `riir-poc/` comparing (a) random-init AttractorKernel (Plan 276 baseline), (b) CP^(d-1) top-eigenvector recaller (this primitive), (c) leaky integrator (Plan 276 winner). If CP^(d-1) does NOT beat the leaky integrator on G2.1, the Fusion A claim is refuted and the verdict drops to GOAT (Fusion B/C/D/E may still hold independently).

**Routing:**

| Output | Repo | File |
|---|---|---|
| Open primitive (this note) | katgpt-rs | `.research/466_*.md` |
| Open implementation plan | katgpt-rs | `.plans/567_*.md` |
| Super-GOAT private guide | riir-neuron-db | `.research/304_*.md` |
| Consumer-side wiring plan | riir-ai | follow-up, after 567 + 304 ship |

**Per-stack promote/demote ledger (katgpt-rs §1.6):**

| Stack slot | Current primitive | This primitive | Action |
|---|---|---|---|
| Associative memory recall | δ-Mem (024, write-side) / ItemEmbedIndex cosine (vector alignment) | CP^(d-1) top-eigenvector recall | **New slot** — does not replace δ-Mem (different operator: write vs recall). ItemEmbedIndex cosine is the candidate for replacement/demotion IF Fusion B's GOAT gate passes. |
| Belief kernel | Leaky integrator (Plan 276 winner) / AttractorKernel (Plan 276 demoted) | CP^(d-1) belief recaller | **Fusion A unblock** — if PoC passes, AttractorKernel re-promotes under a new CP^(d-1) parameterization. Does NOT demote LeakyIntegrator (different use case). |

---

## 4. What we do NOT take

- **Quantum extension (§VI-VII).** Negative result — Hebbian data lost in chaotic spectra. Out of scope.
- **Image encoding protocol (§IV, Eqs. 16-17).** Application-specific (RGB→qutrit). The recall mechanism is encoder-agnostic; we use our own latent-space encoders.
- **Physical-platform realization discussion (§VI intro).** Ultracold atoms, trapped ions, cavity QED — hardware, not our substrate. (Per the R418 hardware-paper guard: the *technique* — LLG flow on CP^(d-1) — is substrate-independent and IS what we distill; the *hardware realization* is the paper's instantiation.)
- **Replica-symmetry-breaking exploration.** Paper explicitly defers this. Our use case is small-d; RSB unlikely to matter.

---

## 5. Validation protocol (the GOAT gate)

See [Plan 567](../.plans/567_cp_hopfield_top_eigenvector_recall.md) for the full gate. Summary:

| Gate | Criterion | Status |
|---|---|---|
| G1 (correctness) | Top-eigenvector recall recovers corrupted memory to `m̄ ≥ 0.9` on synthetic Haar-random + correlated-memory distributions at loads `α < α_c(d)` | planned |
| G2 (capacity) | Measured α_c at our actual N values (8, 64) matches the paper's asymptotic α_c within finite-size corrections for d=3 and d=4 | planned |
| G3 (no-regression) | All existing tests pass with the new primitive behind a feature flag | planned |
| G4 (alloc-free / perf) | `O(d³)` recall cost sub-μs for d ≤ 4; constraint projection O(D²) sub-μs for D ≤ 8 | planned |
| **G5 (Plan 276 unblock — the load-bearing gate)** | CP^(d-1) recaller on Plan 276 G2.1 benchmark beats random-init AttractorKernel on flip count AND approaches LeakyIntegrator's 1× baseline | planned — defend-wrong PoC in `riir-poc/` |
| G6 (Fusion B — KG capacity) | Per-NPC KG triple capacity with CP^(d-1) recall exceeds cosine-ANN capacity by ≥ 3× on correlated-triple distributions | planned — separate benchmark in riir-neuron-db |

If G5 FAILS, Fusion A is refuted; the verdict drops to GOAT (the primitive still has independent value via Fusion B/C/D/E). If G5 PASSES, Plan 276's AttractorKernel re-promotes under the CP^(d-1) parameterization and we ship a modelless belief kernel with BBP-protected recall.

---

## 6. Why this is Super-GOAT and not GOAT

The distinction is **new mechanism class**, not better numbers:

- **GOAT** would be: "we replaced cosine ANN with a faster cosine ANN" — same operator class, better perf.
- **Super-GOAT** is: "we replaced vector alignment (gapless, α_c ∝ 1/n) with top-eigenvector alignment on a symmetric space (BBP-gapped, α_c ∝ d²)" — different operator class, capacity scaling inverts. This is a *qualitative* change in what the memory system can do, not a quantitative speedup.

The selling point is not "10× faster recall" — it is "10× more memories per NPC without crosstalk, modellessly, refuting the documented Plan 276 blocker." That is a capability no incumbent (cosine ANN, δ-Mem, Hebbian Kernel Memory, MPI Router) has, because none of them use symmetric-space Lie-algebraic structure.

---

## References

- Galitski, V. — "High-Capacity Generalized Hopfield Networks" — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) (2026-07-31)
- Galitski, V. — "Quantum-to-classical correspondence and Hubbard-Stratonovich dynamical systems: A Lie-algebraic approach" — [Phys. Rev. A **84**, 012118 (2011)](https://arxiv.org/abs/1012.2873) — the Lie-algebraic linearization technique
- Baik, Ben Arous, Péché — "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices" — Ann. Probab. **33**, 1643 (2005) — the BBP transition
- Perelomov, A. M. — "Coherent states for arbitrary Lie group" — Commun. Math. Phys. **26**, 222 (1972) — the coherent-state construction on symmetric spaces
- Plan 276 — MicroRecurrentBeliefState GOAT gate (the documented blocker this primitive unblocks): `.benchmarks/276_micro_belief_goat.md`
- Research 455 — Hebbian Kernel Memory (construction-side cousin): `.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md`
- Research 317 — Reasoning as Attractor Dynamics (Gibbs retrieval cousin): `.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md`
- Research 246 — Manifold Power Iteration MoE Router (spectral cousin): `.research/246_*.md`
