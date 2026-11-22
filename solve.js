// solve.js — fast CSV reader backed by a WASM SIMD parser.
//
// run(input) answers exactly like csv-reader 1.0.0 for {text, header}:
//   {columns, rows, errors}
//
// The heavy scanning runs in WebAssembly (solve.wasm, compiled from Rust
// with 128-bit SIMD): it walks the text as UTF-16 code units — the same
// units a JS string is made of, so no re-encoding is ever needed and every
// code unit round-trips exactly, lone surrogates included. Plain fields come
// back as spans into the original text (sliced here with substring); only
// quoted fields, which need unescaping and CR/LF normalisation, are
// materialised by the parser into an output buffer, which is decoded here
// once with Buffer (whose utf16le round-trip preserves lone surrogates).

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmModule = new WebAssembly.Module(readFileSync(join(__dirname, 'solve.wasm')));
const wasm = new WebAssembly.Instance(wasmModule, {}).exports;
const memory = wasm.memory;

const CODE_TO_NAME = {
  1: 'stray_cr',
  2: 'quote_junk',
  3: 'unterminated_quote',
  4: 'short_row',
  5: 'long_row',
};
const NULL_ROW = 0xffffffff;

const F_TRANSFORMED = 1;

export function run(input) {
  const text = input.text;
  const header = !!input.header;
  const n = text.length;

  // Copy the text as raw UTF-16LE units into the parser's input buffer,
  // in a single native pass: a Buffer view over the WASM memory, written
  // to directly (no intermediate copy).
  const inPtr = wasm.input_buf(n);
  const written = Buffer.from(memory.buffer, inPtr, n * 2).write(text, 'utf16le');

  // Parse. Returns a pointer to the serialised result blob.
  const outPtr = wasm.parse(inPtr, n, header ? 1 : 0);

  // Header: 8 little-endian u32s.
  const h = new Uint32Array(memory.buffer, outPtr, 8);
  const fieldCount = h[0];
  const recordCount = h[1];
  const errorCount = h[2];
  const outLen = h[3];
  const fieldsOff = h[4];
  const recordsOff = h[5];
  const errorsOff = h[6];
  const contentOff = h[7];

  // Decode the transformed (quoted) content once, then slice fields out of
  // it; plain fields are sliced out of the original text.
  const outStr =
    outLen > 0 ? Buffer.from(memory.buffer, outPtr + contentOff, outLen * 2).toString('utf16le') : '';
  const f = new Uint32Array(memory.buffer, outPtr + fieldsOff, fieldCount * 3);
  const r = new Uint32Array(memory.buffer, outPtr + recordsOff, recordCount * 2);
  const e = new Uint32Array(memory.buffer, outPtr + errorsOff, errorCount * 4);

  const fieldString = (fi) => {
    const start = f[fi * 3];
    const len = f[fi * 3 + 1];
    if (f[fi * 3 + 2] & F_TRANSFORMED) return outStr.substring(start, start + len);
    return text.substring(start, start + len);
  };

  // Raw records (fields as parsed, before padding/truncation).
  const rawRecords = new Array(recordCount);
  for (let ri = 0; ri < recordCount; ri++) {
    const fs = r[ri * 2];
    const fc = r[ri * 2 + 1];
    const arr = new Array(fc);
    for (let k = 0; k < fc; k++) arr[k] = fieldString(fs + k);
    rawRecords[ri] = arr;
  }

  let columns = null;
  let rows;
  let width = 0;
  if (recordCount === 0) {
    rows = [];
  } else {
    width = rawRecords[0].length;
    const pad = (rec) => {
      if (rec.length === width) return rec;
      if (rec.length < width) {
        const out = rec.slice();
        while (out.length < width) out.push('');
        return out;
      }
      return rec.slice(0, width);
    };
    if (header) {
      columns = rawRecords[0];
      rows = new Array(recordCount - 1);
      for (let i = 1; i < recordCount; i++) rows[i - 1] = pad(rawRecords[i]);
    } else {
      rows = new Array(recordCount);
      for (let i = 0; i < recordCount; i++) rows[i] = pad(rawRecords[i]);
    }
  }

  const errors = new Array(errorCount);
  for (let i = 0; i < errorCount; i++) {
    const code = e[i * 4];
    const line = e[i * 4 + 1];
    const row = e[i * 4 + 2];
    const recIdx = e[i * 4 + 3];
    const err = { code: CODE_TO_NAME[code], line, row: row === NULL_ROW ? null : row };
    if (code === 4) {
      // short_row: the record's raw field count.
      err.fields = rawRecords[recIdx].length;
    } else if (code === 5) {
      // long_row: the fields that were cut off.
      err.extra = rawRecords[recIdx].slice(width);
    }
    errors[i] = err;
  }

  return { columns, rows, errors };
}
