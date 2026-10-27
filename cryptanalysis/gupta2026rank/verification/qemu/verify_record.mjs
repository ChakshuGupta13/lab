import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {observe} from './source/observer.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const RECORD = join(ROOT, 'record');
const DEGREE = 256;
const MODULUS = 3329;
const ROOT_OF_UNITY = 17;
const FAULTS = [1, 2, 4, 8, 16, 32, 64];
const mod = value => ((value % MODULUS) + MODULUS) % MODULUS;

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
  for (let length = DEGREE / 2; length >= 2; length /= 2) {
    for (let start = 0; start < DEGREE; start += 2 * length) {
      const twiddle = faults.includes(index) ? 0 : power(ROOT_OF_UNITY, reverseBits(index));
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

function readVector(name) {
  const bytes = readFileSync(join(RECORD, name));
  assert.equal(bytes.length, 2 * DEGREE);
  return Array.from({length: DEGREE}, (_, index) => mod(bytes.readInt16LE(2 * index)));
}

const baselineInput = readVector('baseline-input-0.bin');
const faultedInput = readVector('faulted-input-0.bin');
const baselineOutput = readVector('baseline-output-0.bin');
const faultedOutput = readVector('faulted-output-0.bin');
assert.deepEqual(faultedInput, baselineInput);
assert.deepEqual(baselineOutput, transform(baselineInput, []));
assert.deepEqual(faultedOutput, transform(baselineInput, FAULTS));
assert.deepEqual(readVector('baseline-input-1.bin'), readVector('faulted-input-1.bin'));
assert.deepEqual(readVector('baseline-output-1.bin'), readVector('faulted-output-1.bin'));

const correctKey = readFileSync(join(RECORD, 'baseline-public-key.bin'));
const faultedKey = readFileSync(join(RECORD, 'faulted-public-key.bin'));
assert.notDeepEqual(correctKey, faultedKey);
const observed = observe({level: 512, target: 0, faults: FAULTS, replacement: 0,
  correct_pk: correctKey.toString('hex'), faulted_pk: faultedKey.toString('hex')});
const internalDifference = baselineOutput.map((value, index) => mod(value - faultedOutput[index]));
assert.deepEqual(observed.difference, internalDifference);
assert.equal(observed.rank, 254);
assert.deepEqual(observed.freeColumns, [0, 1]);

const summary = JSON.parse(readFileSync(join(ROOT, 'summary.json'), 'utf8'));
assert.equal(summary.publicObserverMatchesInternalDifference, true);
assert.equal(summary.instructionSkipDemonstrated, false);
assert.equal(summary.physicalFaultDemonstrated, false);
console.log(JSON.stringify({status: 'QEMU_RECORD_VERIFIED', rank: observed.rank,
  unresolvedCoordinates: observed.freeColumns}));