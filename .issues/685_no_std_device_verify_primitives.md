# Issue 685 — Host the two `no_std` device-verify primitives (`Satellite` receiving side)

> **Opened:** 2026-08-24 · **Status:** OPEN — **BOUNDARY.md decision first**
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

- [ ] **BOUNDARY.md first.** Read it, decide the crate (new leaf vs. an existing
      `no_std` home), and record the allowlist row **before** writing code. File
      the boundary drift row if one is needed.
- [ ] Land fair-roll verify (`riir-chain` Issue 108) with the pinned
      `(seed, sides) → die` vector fixture. **Ship the vectors first** — they are
      the deliverable; the code is the easy half. Cover `sides` values that do
      and do not divide 256 (the rejection branch is where drift hides).
- [ ] Land Merkle verify (`riir-chain` Issue 109) with pinned root-compare vectors.
- [ ] Gate `alloc`-requiring surface separately: `roll_dice` (plural) returns
      `Vec`; `roll_die` (single) must stay alloc-free.
- [ ] Prove all three targets in CI: host, `wasm32-unknown-unknown`,
      `riscv32imc-unknown-none-elf`. A green host test proves nothing about the
      MCU path.
- [ ] Do **not** add a `getrandom` dependency. Verification needs no entropy;
      the signer path deliberately has none.

## Done when

The same `(seed, sides)` yields a bit-identical die, and the same proof yields a
bit-identical root verdict, on host / wasm32 / bare-metal RISC-V — from one
implementation that `riir-chain` consumes rather than duplicates.
