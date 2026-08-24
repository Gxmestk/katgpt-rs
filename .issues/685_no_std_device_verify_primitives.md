# Issue 685 — Host the two `no_std` device-verify primitives (`Satellite` receiving side)

> **Opened:** 2026-08-24 · **Status:** ✅ **CLOSED 2026-08-24** — landed, released, and consumed
> **Requesting side:** `riir-chain` Issue 108 (fair-roll verify) + Issue 109 (Merkle verify)
> **Design:** `riir-chain/.proposals/006_esp32_device_tier_ws_fallback.md` §10 Q6/Q7

## Ask

Two verification primitives need a home that a 512 KB microcontroller can link.
`riir-chain` cannot host them: it is std-only by construction and fails to
compile for a bare-metal target at its **first** dependency
(`error[E0463]: can't find crate for std` in `rustc-hash`, with **zero** features
enabled — measured 2026-08-24 on `riscv32imc-unknown-none-elf`).

| Primitive | Shape | Source today |
|---|---|---|
| **Fair-roll verify** | `from_combined_seed(seed)` + `roll_die(sides)` — one `blake3::hash` of 32 B + integer rejection sampling. No std, no alloc | `riir-chain/src/split_key.rs:110` |
| **Merkle verify** | BLAKE3 root compare + `subtree_inclusion`. Host cost 4.55 ns/zone, 847 ps/root compare | `riir-chain` `catchup::merkle` |

## Why here

- **Domain test passes verbatim:** both are modelless verification primitives
  with **no riir dep**, and this repo is upstream of everything.
- **`no_std` discipline already exists** here (`riir-wallet-signer` is the
  precedent for the shape, and it is PROVEN to compile bare-metal:
  `cargo check -p riir-wallet-signer --no-default-features --features ed25519
  --target riscv32imc-unknown-none-elf` → 0 errors).
- **Plan 040 lesson:** cargo reads path-dep manifests recursively, so homing an
  MCU-facing crate inside `riir-chain` drags a 100,707-LOC std workspace into
  every device build's resolution — measured in that plan's `/tmp/ctxtest` lab.

## The invariant that matters more than the location

**One implementation, consumed — never copied.** Two implementations of a
rejection-sampling threshold will drift, and when they do **every honest claim
looks like fraud**: device and node compute different items from the same seed,
indistinguishable from cheating. So `riir-chain` must *consume* whatever lands
here, with its own function becoming a thin re-export.

## Tasks

- [x] **BOUNDARY.md first.** Read it, decide the crate (new leaf vs. an existing
      `no_std` home), and record the allowlist row **before** writing code. File
      the boundary drift row if one is needed.
- [x] Land fair-roll verify (`riir-chain` Issue 108) with the pinned
      `(seed, sides) → die` vector fixture. **Ship the vectors first** — they are
      the deliverable; the code is the easy half. Cover `sides` values that do
      and do not divide 256 (the rejection branch is where drift hides).
- [x] Land Merkle verify (`riir-chain` Issue 109) with pinned root-compare vectors.
- [x] Gate `alloc`-requiring surface separately: `roll_dice` (plural) returns
      `Vec`; `roll_die` (single) must stay alloc-free.
- [x] Prove all three targets in CI: host, `wasm32-unknown-unknown`,
      `riscv32imc-unknown-none-elf`. A green host test proves nothing about the
      MCU path.
- [x] Do **not** add a `getrandom` dependency. Verification needs no entropy;
      the signer path deliberately has none.

## Done when

The same `(seed, sides)` yields a bit-identical die, and the same proof yields a
bit-identical root verdict, on host / wasm32 / bare-metal RISC-V — from one
implementation that `riir-chain` consumes rather than duplicates.


---

## What landed (2026-08-24)

**Crate:** `crates/katgpt-device-verify` — `#![no_std]`, `#![forbid(unsafe_code)]`,
one dependency (`blake3`, `default-features = false`), no `getrandom`.

| Module | Surface |
|---|---|
| `fair_roll` | `combine_seed`, `FairRollVerifier::{roll_die, checked_roll_die, verify_die, roll_unit}`, `roll_dice` behind `alloc` |
| `merkle_verify` | `hash_pair`, `compute_root_from_proof`, `verify_proof`, `verify_proof_bounded`, `roots_match`, `EMPTY_HASH` |
| `vectors` | 26 fair-roll + 19 Merkle pinned vectors, plus the committed generator |

**BOUNDARY.md decision, recorded not waved through.** Admitted on the domain
test read as written — modelless, no riir dep, upstream of everything. It is a
*verification* rather than an *inference* primitive, which widens the Owns
line; that widening is written down in `BOUNDARY.md` along with why the
alternatives fail (both siblings are downstream, std-only, and drag a 100 K-LOC
manifest into every MCU build's resolution). Workspace boundary gate re-run
after the change: **clean, 15 repos, 180 edges**.

### Targets proven — compiled *and*, where possible, executed

| Target | Result |
|---|---|
| host (aarch64-apple-darwin) | 17/17 tests pass |
| `wasm32-wasip2` (via wasmtime) | **17/17 tests pass** — 32-bit `usize` executed, not just compiled |
| `wasm32-unknown-unknown` | builds |
| `riscv32imc-unknown-none-elf` | builds (bare metal) |
| `xtensa-esp32s3-none-elf` | builds (the real board's ISA, `+esp -Z build-std=core`) |

The wasip2 row is the one that matters beyond a compile check: `compute_root_from_proof`
shifts a `usize` index, and "compiles on 32-bit" is not "agrees on 32-bit".

### The fixture earned its keep on the first run

It caught a live drift bug before any device existed — `riir-neuron-db`'s
`EMPTY_HASH` is documented as `BLAKE3("")` and **is not**. The first draft of
the device Merkle port was written from that comment and would have disagreed
with the node on every odd-sized tree. Filed + fixed as `riir-neuron-db`
Issue 606. This is the entire argument for shipping vectors before code,
demonstrated rather than asserted.

### Second finding, deliberately NOT fixed

`roll_die`'s rejection sampling is incomplete: the first byte is threshold-tested,
but the **fallback byte is used unconditionally**, so for a `sides` that does
not divide 256 the low faces are over-represented. Reproduced bit-for-bit here
on purpose — the node computes the same skew, and bit-identity is what keeps an
honest claim from looking like fraud. Correcting it is a consensus-visible
change to every historical roll and belongs in a versioned `_v2` seam agreed on
both sides. Documented on `FairRollVerifier::roll_die`; tracked in
`riir-chain` Issue 108.

## Consumed — released the same day

`develop` → `main` promoted as a fast-forward (`51be354a` → `4d6749fa`),
verified first: `main` was 0 commits ahead, and `katgpt-core` was
check-built under the sibling-consumed feature set
(`engram, subspace_phase_gate, dec_operators, mag_mining, tropical_algebra,
chunked_content_store, rtdc_subtree_inclusion`) before the push.

Both siblings now **consume** rather than duplicate:

| Repo | Was | Now |
|---|---|---|
| `riir-chain` | own `FairRng::roll_die` etc. | delegates to `FairRollVerifier` (`optional`, pulled in by `chain`) |
| `riir-neuron-db` | own `hash_pair` / `compute_root_from_proof` / `verify_proof` / `EMPTY_HASH` | `pub use` from `merkle_verify` (non-optional) |

**There is now exactly one implementation of each, and it is the one the
device links.** The invariant this issue opened with — *one implementation,
consumed, never copied* — holds structurally, not by convention.

### The proof the move was behaviour-preserving

The vector fixtures were written against the **old, in-repo** implementations
and still pass against the delegated ones:

| Gate | Result |
|---|---|
| `katgpt-device-verify` (host + wasip2) | 17/17 each |
| `riir-chain --test fair_roll_device_vectors` | 3/3 |
| `riir-neuron-db --test merkle_device_vectors` | 5/5 |
| `riir-neuron-db --test merkle_soundness_spec_match` | 7/7 — the **Lean** corpus survived the move |
| `riir-chain --lib` | 343, unchanged |
| `riir-neuron-db --lib` | 412 passed, 4 ignored |
| workspace boundary contract | clean, 15 repos, 181 edges |

That is what makes this a refactor rather than a rewrite: the measurement
predates the change.

### One thing the release forced, and how it was handled

Adding a second package from the same git source moves the **whole**
`katgpt-rs` rev, so both siblings jumped 30 upstream commits. Every
`katgpt-core`-consuming `riir-chain` feature was re-checked
(`chain_engram_commit`, `chain_rtdc`, `chain_rtdc_subtree`,
`chain_vessel_delivery`, `chain_wasm`, `chain_guard`) — all clean. Worth
knowing for next time: a git-dep addition is never *only* an addition.

## Still open elsewhere (not this issue)

`roll_die`'s residual bias — reproduced here bit-for-bit on purpose, and now
the single definition, so a `_v2` fix would be a one-place change agreed on
both sides. Tracked in `riir-chain` Issue 108.
