import { setFlagsFromString } from 'node:v8';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { run } from './solve.js';

// V8's incremental marking off, set before solve.js is imported so all of
// its work (including the WASM parse) runs under it, like the house
// baseline. The driver keeps the input/output locations the harness
// provides: outside a container MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output.
setFlagsFromString('--no-incremental-marking');

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const parsed = JSON.parse(readFileSync(join(INPUT_DIR, 'cases.json'), 'utf8'));
const cases = (Array.isArray(parsed) ? parsed : parsed.cases) as {
  id: string;
  input: unknown;
}[];
const answers: unknown[] = [];
for (const c of cases) answers.push(run(c.input));
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, 'answers.json'), JSON.stringify(answers));
