import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {copyFileSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {spawn, spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {observe} from './observer.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const MANIFEST = JSON.parse(readFileSync(join(ROOT, 'MANIFEST.json'), 'utf8'));
const CONFIG = MANIFEST.experiment;
const FAULTS = [1, 2, 4, 8, 16, 32, 64];
const LOADED_WORDS = [381889553, 372858381, 21932846];
const ARM_BIN = process.env.ARM_GNU_BIN ?? '/Applications/ArmGNUToolchain/15.3.rel2/arm-none-eabi/bin';
const ARM_GCC = join(ARM_BIN, 'arm-none-eabi-gcc');
const ARM_GDB = join(ARM_BIN, 'arm-none-eabi-gdb');
const ARM_OBJDUMP = join(ARM_BIN, 'arm-none-eabi-objdump');
const OUTPUT = join(ROOT, 'results', `m4-keygen-theorem2-${new Date().toISOString().replaceAll(':', '-')}`);
const FLAGS = ['-mcpu=cortex-m4', '-mthumb', '-mfpu=fpv4-sp-d16', '-mfloat-abi=hard',
  '-O3', '-g', '-ffreestanding', '-fno-builtin', '-nostdlib', '-Wall', '-Wextra', '-Wshadow', '-Werror'];
const LOCKED = ['src/m4_keygen_probe.c', 'src/m4_probe.ld', 'src/observer.mjs', 'vendor/mupq.tar.gz', 'vendor/pqm4.tar.gz'];
const ENV = {PATH: process.env.PATH, HOME: process.env.HOME, LANG: 'C'};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const save = (name, value) => writeFileSync(join(OUTPUT, name), `${JSON.stringify(value, null, 2)}\n`, {flag: 'wx'});
const mod = value => ((value % CONFIG.modulus) + CONFIG.modulus) % CONFIG.modulus;

function execute(command, args, options = {}) {
  const result = spawnSync(command, args, {cwd: ROOT, env: ENV, encoding: 'utf8',
    timeout: CONFIG.commandTimeoutMs, maxBuffer: 8 * 1024 * 1024, ...options});
  assert.ifError(result.error);
  assert.equal(result.status, 0, `${command}: ${result.stderr}\n${result.stdout}`);
  return result.stdout;
}

function power(base, exponent) {
  let result = 1;
  while (exponent > 0) {
    if (exponent % 2) result = mod(result * base);
    base = mod(base * base);
    exponent = Math.floor(exponent / 2);
  }
  return result;
}

function reverseBits(value) {
  let result = 0;
  for (let bit = 0; bit < 7; bit++) {
    result = 2 * result + value % 2;
    value = Math.floor(value / 2);
  }
  return result;
}

function transform(input, faults) {
  const values = [...input];
  let index = 1;
  for (let length = CONFIG.degree / 2; length >= 2; length /= 2) {
    for (let start = 0; start < CONFIG.degree; start += 2 * length) {
      const twiddle = faults.includes(index) ? CONFIG.replacement : power(CONFIG.root, reverseBits(index));
      index++;
      for (let offset = start; offset < start + length; offset++) {
        const product = mod(twiddle * values[offset + length]);
        values[offset + length] = mod(values[offset] - product);
        values[offset] = mod(values[offset] + product);
      }
    }
  }
  return values;
}

function readVector(path) {
  const bytes = readFileSync(path);
  assert.equal(bytes.length, 2 * CONFIG.degree);
  return Array.from({length: CONFIG.degree}, (_, index) => mod(bytes.readInt16LE(2 * index)));
}

function extract(archive, destination) {
  mkdirSync(destination, {recursive: true});
  execute('tar', ['-xzf', archive, '-C', destination]);
}

function sitesFrom(disassembly) {
  const forwardBlock = disassembly.match(/^([0-9a-f]+) <ntt_fast>:\n([\s\S]*?)(?=^\s*$)/m);
  assert(forwardBlock, 'forward_ntt_symbol_not_found');
  const instructions = [...forwardBlock[0].matchAll(/^\s*([0-9a-f]+):\s+(?:[0-9a-f]{4}\s+){1,2}(.+)$/gm)]
    .map(match => ({address: Number.parseInt(match[1], 16), assembly: match[2]}));
  const preload = instructions.findIndex(item => /vldmia\s+r1!.*\{s8-s22\}/.test(item.assembly));
  const loads = instructions
    .map((item, index) => ({...item, index}))
    .filter(item => /ldr\.w\s+(?:sl|r10),\s*\[r1\],\s*#4|ldrd\s+(?:sl|r10),\s*(?:fp|r11),\s*\[r1\],\s*#8/.test(item.assembly));
  assert(preload !== -1 && loads.length >= 3, 'theorem2_intervention_sites_not_found');
  return {
    afterPreload: instructions[preload + 1].address,
    afterLayer3: instructions[loads[0].index + 1].address,
    afterLayer2: instructions[loads[1].index + 1].address,
    afterLayer1: instructions[loads[2].index + 1].address,
  };
}

async function runMode(mode, elf, sites) {
  const socketRoot = mkdtempSync(join(tmpdir(), 'pqc-keygen-'));
  const socket = join(socketRoot, 'gdb.sock');
  const commands = ['set pagination off', 'set confirm off', 'set remotetimeout 5',
    `target remote ${socket}`, 'tbreak keygen_before', 'continue'];
  for (let call = 0; call < CONFIG.expectedForwardCalls; call++) {
    const pointer = `$p${call}`;
    commands.push('tbreak ntt_fast', 'continue', `set ${pointer} = $r0`, `set $return${call} = $lr`,
      `dump binary memory ${join(OUTPUT, `${mode}-input-${call}.bin`)} ${pointer} ${pointer}+512`);
    if (mode === 'faulted' && call === CONFIG.targetForwardCall) {
      commands.push(
        `tbreak *0x${sites.afterPreload.toString(16)}`, 'continue',
        'set $s8 = 0.0', 'set $s9 = 0.0', 'set $s11 = 0.0', 'set $s15 = 0.0',
        `tbreak *0x${sites.afterLayer3.toString(16)}`, 'continue', 'printf "LOADED16\\n"', 'print $r10', 'set $r10 = 0',
        `tbreak *0x${sites.afterLayer2.toString(16)}`, 'continue', 'printf "LOADED32\\n"', 'print $r10', 'set $r10 = 0',
        `tbreak *0x${sites.afterLayer1.toString(16)}`, 'continue', 'printf "LOADED64\\n"', 'print $r10', 'set $r10 = 0');
    }
    commands.push(`tbreak *($return${call} & ~1)`, 'continue',
      `dump binary memory ${join(OUTPUT, `${mode}-output-${call}.bin`)} ${pointer} ${pointer}+512`);
  }
  commands.push('break ntt_fast', 'commands', 'silent', 'printf "UNEXPECTED_EXTRA_NTT\\n"', 'quit 1', 'end',
    'tbreak keygen_after', 'continue', 'print keygen_status', 'print keygen_random_calls',
    `dump binary memory ${join(OUTPUT, `${mode}-public-key.bin`)} keygen_public_key keygen_public_key+800`,
    'disconnect', 'quit');
  const script = join(OUTPUT, `${mode}.gdb`);
  writeFileSync(script, `${commands.join('\n')}\n`, {flag: 'wx'});
  let stderr = '';
  const emulator = spawn('qemu-system-arm', ['-M', 'mps2-an386', '-display', 'none',
    '-serial', 'none', '-monitor', 'none', '-semihosting', '-kernel', elf,
    '-chardev', `socket,path=${socket},server=on,wait=off,id=debug`,
    '-gdb', 'chardev:debug', '-S'], {cwd: ROOT, env: ENV, stdio: ['ignore', 'ignore', 'pipe']});
  emulator.stderr.on('data', chunk => { stderr += chunk; });
  const exited = new Promise((resolve, reject) => {
    emulator.once('error', reject);
    emulator.once('exit', (code, signal) => resolve({code, signal}));
  });
  const deadline = setTimeout(() => emulator.kill('SIGKILL'), CONFIG.commandTimeoutMs);
  try {
    const log = execute(ARM_GDB, ['-nx', '-batch', '-ex', 'set tcp auto-retry on',
      '-ex', 'set tcp connect-timeout 5', elf, '-x', script]);
    writeFileSync(join(OUTPUT, `${mode}-gdb.log`), log, {flag: 'wx'});
    emulator.kill('SIGTERM');
    const result = await exited;
    assert(result.code === 0 || result.signal === 'SIGTERM', `${mode}: ${stderr}`);
    assert(!log.includes('UNEXPECTED_EXTRA_NTT'));
    assert(/\$\d+ = 2/.test(log), 'keygen_did_not_finish');
    assert(/\$\d+ = 1/.test(log), 'keygen_randomness_count_unexpected');
    return {mode, log, inputs: Array.from({length: CONFIG.expectedForwardCalls}, (_, call) =>
      readVector(join(OUTPUT, `${mode}-input-${call}.bin`))), outputs: Array.from({length: CONFIG.expectedForwardCalls}, (_, call) =>
      readVector(join(OUTPUT, `${mode}-output-${call}.bin`))), publicKey: readFileSync(join(OUTPUT, `${mode}-public-key.bin`))};
  } finally {
    clearTimeout(deadline);
    if (emulator.exitCode === null && emulator.signalCode === null) {
      emulator.kill('SIGKILL');
      await exited;
    }
    writeFileSync(join(OUTPUT, `${mode}-qemu.log`), stderr, {flag: 'wx'});
  }
}

async function main() {
  for (const path of LOCKED) assert.equal(sha(readFileSync(join(ROOT, path))), MANIFEST.sha256[path], path);
  mkdirSync(OUTPUT, {recursive: true});
  mkdirSync(join(OUTPUT, 'source'));
  for (const name of ['m4_keygen_probe.c', 'm4_probe.ld', 'observer.mjs', 'run_theorem2_keygen.mjs']) {
    copyFileSync(join(ROOT, 'src', name), join(OUTPUT, 'source', name));
  }
  const frozen = join(OUTPUT, 'pqm4');
  extract(join(ROOT, 'vendor', 'pqm4.tar.gz'), frozen);
  extract(join(ROOT, 'vendor', 'mupq.tar.gz'), join(frozen, 'mupq'));
  const implementation = join(frozen, CONFIG.implementation);
  const sources = ['cbd.c', 'indcpa.c', 'kem.c', 'matacc.c', 'ntt.c', 'poly.c', 'polyvec.c', 'symmetric-fips202.c', 'verify.c',
    'cmov_int16.S', 'fastaddsub.S', 'fastbasemul.S', 'fastinvntt.S', 'fastntt.S', 'matacc_asm.S', 'poly_asm.S', 'reduce.S']
    .map(name => join(implementation, name));
  const elf = join(OUTPUT, 'm4_keygen.elf');
  execute(ARM_GCC, [...FLAGS, `-I${implementation}`, `-I${join(frozen, 'mupq/common')}`,
    join(ROOT, 'src', 'm4_keygen_probe.c'), join(frozen, 'mupq/common/fips202.c'), join(frozen, 'common/keccakf1600.S'),
    ...sources, '-Wl,-T,src/m4_probe.ld', `-Wl,-Map,${join(OUTPUT, 'm4_keygen.map')}`, '-Wl,--gc-sections', '-lgcc', '-o', elf]);
  const disassembly = execute(ARM_OBJDUMP, ['-d', elf]);
  writeFileSync(join(OUTPUT, 'disassembly.txt'), disassembly, {flag: 'wx'});
  const sites = sitesFrom(disassembly);
  const baseline = await runMode('baseline', elf, sites);
  const faulted = await runMode('faulted', elf, sites);
  const loaded = ['LOADED16', 'LOADED32', 'LOADED64'].map(marker => {
    const match = faulted.log.match(new RegExp(`${marker}\\n\\$\\d+ = (-?\\d+)`));
    assert(match, `${marker}_missing`);
    return Number(match[1]);
  });
  assert.deepEqual(loaded, LOADED_WORDS, 'pointer_loaded_words_differ_from_mapped_probe');
  assert.deepEqual(baseline.inputs[0], faulted.inputs[0], 'target_secret_polynomial_input_changed');
  assert.deepEqual(baseline.inputs[1], faulted.inputs[1], 'second_secret_polynomial_input_changed');
  assert.deepEqual(baseline.outputs[1], faulted.outputs[1], 'second_secret_polynomial_ntt_changed');
  assert.notDeepEqual(baseline.outputs[0], faulted.outputs[0], 'target_secret_polynomial_ntt_unchanged');
  assert.deepEqual(baseline.outputs[0], transform(baseline.inputs[0], []), 'baseline_target_ntt_differs_from_reference');
  assert.deepEqual(faulted.outputs[0], transform(baseline.inputs[0], FAULTS), 'faulted_target_ntt_differs_from_declared_map');
  assert.notDeepEqual(baseline.publicKey, faulted.publicKey, 'public_key_unchanged');
  const observed = observe({level: 512, target: 0, faults: FAULTS, replacement: 0,
    correct_pk: baseline.publicKey.toString('hex'), faulted_pk: faulted.publicKey.toString('hex')});
  const internalDifference = baseline.outputs[0].map((value, index) => mod(value - faulted.outputs[0][index]));
  assert.deepEqual(observed.difference, internalDifference, 'public_observer_difference_mismatch');
  assert.equal(observed.rank, 254, 'one_per_layer_rank_mismatch');
  assert.deepEqual(observed.freeColumns, [0, 1], 'one_per_layer_unresolved_coordinates_mismatch');
  const summary = {status: 'FULL_KEYGEN_ONE_PER_LAYER_MODEL_MATCH', parameterSet: 'ML-KEM-512',
    forwardNttCalls: CONFIG.expectedForwardCalls, faultedForwardCall: CONFIG.targetForwardCall,
    declaredFaultIndices: FAULTS, cachedRegisters: {s8: 1, s9: 2, s11: 4, s15: 8},
    pointerLoadedRegisters: {r10AfterFirstLength8Load: 16, r10AfterFirstLength4Load: 32, r10AfterFirstLength2Load: 64},
    loadedWords: loaded, interventionCount: 7, unchangedSecondSecretPolynomialNtt: true,
    targetNttMatchesDeclaredMap: true, publicKeysDiffer: true,
    publicObserverMatchesInternalDifference: true, rank: observed.rank,
    unresolvedCoordinates: observed.freeColumns, instructionSkipDemonstrated: false,
    physicalFaultDemonstrated: false, allInputsProved: false, runtimeFaultRepeatabilityMeasured: false,
    singletonResultPreserved: true};
  save('summary.json', summary);
  console.log(JSON.stringify({output: OUTPUT, sites, ...summary}, null, 2));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
