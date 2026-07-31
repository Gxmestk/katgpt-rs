# Moka v1 — Vendored Model Weights

Source: https://github.com/millionco/moka (commit at time of vendoring: `develop`, 2026-07-29)
Files: `go-model.bin` (113,648 bytes, sha256 `d808d09f4b9dab959fc2764a16485448ae407bbd90cbdf8de6e8e8605b2c2de9`, matches the manifest's declared `sha256`), `go-model.json` (tensor manifest).

## License (MIT)

```
MIT License

Copyright (c) 2026 Million Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Why these files are here

Used by `crates/katgpt-pruners/src/go/moka_net.rs` (Plan 563) to run Moka's real
105,353-parameter Go policy/value network natively in Rust, for a head-to-head
benchmark against this repo's modelless Go players. See
`.docs/06_game_arenas/go_arena.md` and `.plans/563_go_moka_baseline_poc.md`.
