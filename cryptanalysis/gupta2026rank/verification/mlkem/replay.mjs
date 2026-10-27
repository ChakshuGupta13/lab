import assert from 'node:assert/strict';
import {createHash, randomBytes} from 'node:crypto';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {CONFIG, LEVELS, observe} from './observer.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const PIN = 'd1b2fe782888bdb761a50336012923180be7f502';
const LEVEL_LIST = [512, 768, 1024];
const SEEDS = 10;
const TIMEOUT_MS = 120000;
const FLAGS = ['-std=c99', '-O2', '-Wall', '-Wextra', '-Werror'];
const OUTPUT = join(ROOT, 'results', `repeat-filter-${new Date().toISOString().replaceAll(':', '-')}`);
const ENV = {PATH: process.env.PATH, HOME: process.env.HOME, LANG: 'C'};
const mod = value => ((value % CONFIG.modulus) + CONFIG.modulus) % CONFIG.modulus;

function execute(command, args, options = {}) {
  const result = spawnSync(command, args, {cwd: ROOT, env: ENV, encoding: 'utf8',
    timeout: TIMEOUT_MS, maxBuffer: 8 * 1024 * 1024, ...options});
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
  const values = input.map(mod);
  let index = 1;
  for (let length = 128; length >= 2; length /= 2) {
    for (let start = 0; start < 256; start += 2 * length) {
      const twiddle = faults.includes(index) ? 0 : power(CONFIG.root, reverseBits(index));
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

function alphabet(level) {
  const eta = LEVELS[level].eta;
  return Array.from({length: 2 * eta + 1}, (_, index) => index - eta);
}

function prepare(pinnedSource) {
  const binaries = {};
  for (const level of LEVEL_LIST) {
    const binary = join(OUTPUT, `producer-${level}`);
    execute('clang', [...FLAGS, '-DPQC_FAULT_VALIDATION', `-DMLK_CONFIG_PARAMETER_SET=${level}`,
      `-I${join(pinnedSource, 'mlkem')}`, 'pilot.c', '-o', binary]);
    binaries[level] = binary;
  }
  return binaries;
}

function produce(binary, coins, args = []) {
  return JSON.parse(execute(binary, [coins, ...args]));
}

function sameDifference(left, right) {
  return left.difference.every((value, index) => value === right.difference[index]);
}

function filterSeed(level, binary, coins) {
  const values = alphabet(level);
  const base = produce(binary, coins);
  const baseObserved = observe(base.public);
  assert.equal(baseObserved.rank, 254);
  assert.deepEqual(baseObserved.freeColumns, [0, 1]);
  assert.deepEqual(base.private.faulted_ntt.map(mod), transform(base.private.input, CONFIG.faultIndices));
  const truePair = [base.private.input[0], base.private.input[1]];
  let sameDifferenceCount = 0;
  let samePublicKeyCount = 0;
  let truePairSeen = false;
  let truePairMatchesAbsoluteKey = false;
  for (const c0 of values) {
    for (const c1 of values) {
      const run = produce(binary, coins, ['replace', String(c0), String(c1)]);
      assert.deepEqual([run.private.input[0], run.private.input[1]], [c0, c1]);
      const observed = observe(run.public);
      const sameKey = run.public.correct_pk === base.public.correct_pk
        && run.public.faulted_pk === base.public.faulted_pk;
      if (sameDifference(observed, baseObserved)) sameDifferenceCount++;
      if (sameKey) samePublicKeyCount++;
      if (c0 === truePair[0] && c1 === truePair[1]) {
        truePairSeen = true;
        truePairMatchesAbsoluteKey = sameKey;
      }
    }
  }
  return {rank: baseObserved.rank, freeColumns: baseObserved.freeColumns,
    unchangedOtherOutputs: base.private.unchanged_other_outputs,
    alphabetCandidates: values.length ** 2, sameDifferenceCount, samePublicKeyCount,
    truePairSeen, truePairMatchesAbsoluteKey,
    selectedPairIsTrue: samePublicKeyCount === 1 && truePairMatchesAbsoluteKey};
}

function coinDigest(coins) {
  return createHash('sha256').update(coins).digest('hex');
}

function main() {
  mkdirSync(OUTPUT, {recursive: true});
  const replayPath = process.argv[2];
  const expectPath = process.argv[3];
  const coinsByLevel = replayPath
    ? JSON.parse(readFileSync(replayPath, 'utf8'))
    : Object.fromEntries(LEVEL_LIST.map(level => [String(level),
      Array.from({length: SEEDS}, () => randomBytes(64).toString('hex'))]));
  for (const level of LEVEL_LIST) assert.equal(coinsByLevel[level].length, SEEDS);
  if (!replayPath) {
    const privateDir = join(ROOT, 'private', OUTPUT.split('/').at(-1));
    mkdirSync(privateDir, {recursive: true, mode: 0o700});
    writeFileSync(join(privateDir, 'seeds.json'), JSON.stringify(coinsByLevel), {mode: 0o600, flag: 'wx'});
  }
  const upstream = join(ROOT, 'upstream', 'mlkem-native');
  assert.equal(execute('git', ['rev-parse', 'HEAD'], {cwd: upstream}).trim(), PIN);
  const patch = JSON.parse(readFileSync(join(ROOT, 'mlkem-native-inputs.json'))).upstreamPatch;
  const pinnedSource = join(OUTPUT, 'upstream');
  mkdirSync(pinnedSource);
  execute('git', ['archive', '--format=tar', `--output=${join(OUTPUT, 'upstream.tar')}`, PIN], {cwd: upstream});
  execute('tar', ['-xf', join(OUTPUT, 'upstream.tar'), '-C', pinnedSource]);
  execute('patch', ['--batch', '--fuzz=0', '-p1', '-d', pinnedSource], {input: patch});
  const binaries = prepare(pinnedSource);
  const records = [];
  const levels = [];
  for (const level of LEVEL_LIST) {
    const seeds = coinsByLevel[level].map(coins => ({coinSha256: coinDigest(coins), ...filterSeed(level, binaries[level], coins)}));
    records.push(...seeds.map(item => ({level, ...item})));
    levels.push({level, seeds: SEEDS,
      rank254: seeds.filter(item => item.rank === 254).length,
      freeColumns01: seeds.filter(item => item.freeColumns.length === 2 && item.freeColumns[0] === 0 && item.freeColumns[1] === 1).length,
      otherOutputsUnchanged: seeds.filter(item => item.unchangedOtherOutputs).length,
      alphabetCandidates: seeds[0].alphabetCandidates,
      pairedDifferenceInvariant: seeds.filter(item => item.sameDifferenceCount === item.alphabetCandidates).length,
      absoluteKeySelectsOne: seeds.filter(item => item.samePublicKeyCount === 1).length,
      truePairIncluded: seeds.filter(item => item.truePairSeen).length,
      selectedPairIsTrue: seeds.filter(item => item.selectedPairIsTrue).length});
  }
  if (expectPath) assert.deepEqual(records, JSON.parse(readFileSync(expectPath, 'utf8')));
  writeFileSync(join(OUTPUT, 'seeds-public.json'), `${JSON.stringify(records, null, 2)}\n`);
  const summary = {status: 'REPEAT_AND_FREE_COORDINATE_FILTER', seedsPerLevel: SEEDS, levels,
    coinsRetainedPrivately: !replayPath, perSeedRecords: 'seeds-public.json',
    replayOf: replayPath || null,
    publicOnlySolver: false, absoluteKeySelectionHoldsOtherInputsFixed: true,
    knownAnswerTestsRun: false};
  writeFileSync(join(OUTPUT, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify({output: OUTPUT, ...summary}, null, 2));
}

main();
