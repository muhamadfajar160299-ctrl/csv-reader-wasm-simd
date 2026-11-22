# csv-reader-wasm-simd

[![make.fun competition](https://img.shields.io/badge/make.fun-csv--reader--speed--01--run--20261121-8A2BE2)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)
[![make.fun rank](https://img.shields.io/badge/make.fun-rank-8A2BE2)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)
[![make.fun reward](https://img.shields.io/badge/make.fun-reward-8A2BE2)](https://make.fun/bounty/csv-reader-speed-01-run-20261121)

A fast CSV reader for the [make.fun csv-reader-speed competition](https://make.fun/bounty/csv-reader-speed-01-run-20261121): `solve.js` answers exactly like csv-reader 1.0.0 for `{text, header}` → `{columns, rows, errors}`, with the heavy scanning done in WebAssembly compiled from Rust using 128-bit SIMD.

## How it works

- The parser works on UTF-16 code units — the same units a JS string is made of — so no re-encoding is ever needed and every code unit round-trips exactly, lone surrogates included.
- Delimiters (`,`, LF, CR, `"`) are found 8 units at a time with `i16x8` SIMD compares; quoted-field line breaks are counted in the same pass.
- Plain fields are returned as spans into the original text (sliced with `substring`); only quoted fields, which need unescaping (`""` → `"`) and CR/CRLF → LF normalisation, are materialised into an output buffer, decoded once with `Buffer`.
- `main.ts` reads the cases, answers each one with `run()`, and writes the answers. It honours `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` outside a container, falling back to `/input` and `/output`.

## Verification (local, Node 22.18)

- `grade-local` on the practice set: **6/6 cases passed**, byte-identical answers to the reference.
- End-to-end CPU on this machine (practice input, ~11 MB, 3 runs): **0.68 s** average, vs 0.99 s for the pure-JS entry and 2.22 s for the house baseline — about **1.5x faster than the JS entry, 3.2x faster than baseline**.

## Rebuilding the WASM module

```sh
cargo build --release --target wasm32-unknown-unknown
# with SIMD enabled:
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/csv_wasm.wasm solve.wasm
```

The Rust source is in `src/lib.rs` (no dependencies, `cdylib`). The shipped `solve.wasm` (29 KB) is built from it.

## License

MIT
