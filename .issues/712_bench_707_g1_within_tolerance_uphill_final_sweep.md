# Issue 712: bench_707 G1 red at HEAD — the within-tolerance uphill final sweep ships a worse-than-best artifact

**Status:** G1 half **RESOLVED** (best-iterate guard landed, this session — G1 `returned == min → true` at HEAD on the 4090 box; 23/23 tpr lib tests, default 1992/0 unchanged). **G2 half OPEN** — the surgery perf bar is environment-sensitive on this box (5–8× vs the landing commit's recorded numbers, same binary; the calibrated-bar proposal below is the owner's call).

## Reproduction (HEAD = 08310b4c, release, this box)

```
cargo test --release -p katgpt-core --features tpr --bench bench_707_tpr_binding_goat
```

```
  monotone: 0 rejected proposal(s); returned objective 1.0246975682245041e-4 == trajectory min 9.727195864998823e-5 → false
  → G1 FAIL (is_best)
```

Trajectory (dumped with a temporary print, since removed):

```
[12294.4, 838.4, 1.854, 7.67e-3, 1.3438e-4, 9.7272e-5, 1.0247e-4]
```

Sweep 6 reaches `9.7272e-5`; sweep 7 goes UPHILL to `1.0247e-4` (+5.2%) and is
**accepted with 0 violations** — then the artifact ships sweep 7 while the bench
correctly asserts `a.fit_objective <= min(ssr_per_sweep)·(1+1e-9)`.

## Root cause

`als.rs` (~line 679):

```rust
let monotone_tol = (prev.abs() * 1e-9).max(1e-12);   // prev = the INITIAL ssr = 12294.4
...
if ssr > prev + monotone_tol {                        // tol = 1.23e-5 — ABSOLUTE, scaled by ssr_0
```

The initial-scale tolerance is deliberate (the comment documents why: near
convergence the SSR's last digits are f32 noise, and a `prev·1e-9` bar at the
CURRENT prev reports noise as violations — a measured 2e-9 blip). But the
side effect: a genuine +5.2e-6 uphill step near convergence is UNDER the
`ssr_0·1e-9` bar, accepted, and shipped — violating the ALS's OWN documented
invariant ("the artifact is then, by construction, the minimum of the recorded
trajectory", the descent-guard comment) and the bench's `is_best` gate.

The two policies are inconsistent: the acceptance bar is absolute-scaled
(correct for its purpose), but nothing enforces the shipped artifact == best
iterate.

## Proposed fix (small, surgical) — LANDED for the G1 half

The best-iterate guard as proposed below is implemented in `als.rs` (track
`best_ssr`/`best_snap` on accepted improving sweeps; restore at loop exit
when the loop ended on a within-tolerance uphill step; `prev` reset so the
L2,1 prune phase and the residual certificate evaluate the restored best).
Measured at HEAD: `returned objective 9.727195864998823e-5 == trajectory min
9.727195864998823e-5 → true`, G1 PASS with the same corpus that reproduced
the failure deterministically pre-fix.

Track the best iterate and restore it at loop exit — keeps the loose
acceptance bar (its noise rationale stands) while making the shipped artifact
the trajectory min BY CONSTRUCTION:

```rust
// in the sweep loop, after the accept/reject:
if ssr < best_ssr {
    best_ssr = ssr;
    best_snap = (fillers.clone(), scheme.clone(), w.clone(), bias.clone(), cores.clone());
}
// after the loop (before the L2,1 prune phase + residual certificate):
if prev > best_ssr {
    (fillers, scheme, w, bias, cores) = best_snap;
    // (the prune phase and final_ssr then evaluate the restored best)
}
```

One clone per improving sweep — the snapshot cost class already paid by the
existing `snap`. The G1 `is_best` assertion then holds by construction for
every corpus, on every box, independent of fp-path luck.

## Also observed in the same run (G2, environment class — NOT a code bug)

```
surgery D=64   p50  400 ns  p99  500 ns   (bar 1000 ns) PASS
surgery D=256  p50 1200 ns  p99 1400 ns   (bar 1000 ns) FAIL
surgery D=768  p50 3400 ns  p99 3500 ns   (bar 1000 ns) FAIL
```

The landing commit recorded `84/167/417 ns` for the same harness. This box
(i7-13700K, idle-for-compute, Zed/GUI only) measures 5–8× slower — consistent
with the process not getting P-core boost (power plan / E-core scheduling /
thermal state at measure time), not with a code change: the binary is the
same. The G2 perf bar is environment-sensitive on this machine; consider a
calibrated bar (e.g. vs a same-binary reference op like `simd_dot_f32` at the
same shape) or recording the box's boost state alongside, the Bench-831
cooled-window protocol. Filing here rather than tuning the bar — the owner
decides the gate's environment policy.

## Validation bar for the fix

- `bench_707_tpr_binding_goat` G1 green at HEAD on this box (the corpus
  above reproduces the failure deterministically pre-fix).
- G8/G3/G4 unchanged (the fix touches only which iterate ships).
- Double-fit bit-identical still holds (the restore is deterministic).
