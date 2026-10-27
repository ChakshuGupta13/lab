import assert from 'node:assert/strict';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const PQCLEAN = process.env.PQCLEAN_ROOT;
assert(PQCLEAN, 'set PQCLEAN_ROOT to the pinned PQClean checkout');
const CLEAN = join(PQCLEAN, 'crypto_sign/ml-dsa-44/clean');
const COMMON = join(PQCLEAN, 'common');
const OUTPUT = join(ROOT, 'results', `mldsa-pqclean-${new Date().toISOString().replaceAll(':', '-')}`);
const ENV = {PATH: process.env.PATH, HOME: process.env.HOME, LANG: 'C'};

function execute(command, args) {
  const result = spawnSync(command, args, {cwd: OUTPUT, env: ENV, encoding: 'utf8', timeout: 120000});
  assert.ifError(result.error);
  assert.equal(result.status, 0, `${command}: ${result.stderr}\n${result.stdout}`);
  return result.stdout;
}

const ntt = readFileSync(join(CLEAN, 'ntt.c'), 'utf8')
  .replace('#include <stdint.h>\n', '#include <stdint.h>\nextern int pqc_fault_calls_remaining;\n')
  .replace('    k = 0;\n', '    k = 0;\n    const int fault_this_call = pqc_fault_calls_remaining > 0;\n    if (fault_this_call) pqc_fault_calls_remaining--;\n')
  .replace('            zeta = zetas[++k];\n', '            zeta = zetas[++k];\n            if (fault_this_call && k <= 128 && (k & (k - 1)) == 0) zeta = 0;\n');
assert(ntt.includes('fault_this_call'), 'ntt_patch_missed');
mkdirSync(OUTPUT, {recursive: true});
writeFileSync(join(OUTPUT, 'ntt.c'), ntt);
const sources = ['packing.c', 'poly.c', 'polyvec.c', 'reduce.c', 'rounding.c', 'sign.c', 'symmetric-shake.c']
  .map(name => join(CLEAN, name));
const flags = ['-std=c99', '-O2', '-Wall', '-Wextra', '-Werror', `-I${CLEAN}`, `-I${COMMON}`];
execute('clang', [...flags, ...sources, join(COMMON, 'fips202.c'), join(OUTPUT, 'ntt.c'),
  join(ROOT, 'mldsa_pqclean_pair.c'), '-o', join(OUTPUT, 'pair')]);
const report = JSON.parse(execute(join(OUTPUT, 'pair'), []));
writeFileSync(join(OUTPUT, 'summary.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({output: OUTPUT, ...report}, null, 2));
