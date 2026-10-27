import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

export const LEVELS = Object.freeze({
  512: {dimension: 2, eta: 3, publicKeyBytes: 800},
  768: {dimension: 3, eta: 2, publicKeyBytes: 1184},
  1024: {dimension: 4, eta: 2, publicKeyBytes: 1568},
});

export const CONFIG = Object.freeze({
  level: 512,
  degree: 256,
  modulus: 3329,
  root: 17,
  dimension: LEVELS[512].dimension,
  eta: LEVELS[512].eta,
  publicKeyBytes: LEVELS[512].publicKeyBytes,
  shakeBytes: 4096,
  faultIndices: [1, 2, 4, 8, 16, 32, 64],
  expectedRank: 254,
});

const {degree, modulus, root, dimension} = CONFIG;
const mod = value => ((value % modulus) + modulus) % modulus;

function power(base, exponent) {
  let result = 1;
  while (exponent > 0) {
    if (exponent % 2 === 1) result = mod(result * base);
    base = mod(base * base);
    exponent = Math.floor(exponent / 2);
  }
  return result;
}

function reverseBits(value, width) {
  let result = 0;
  for (let bit = 0; bit < width; bit++) {
    result = 2 * result + value % 2;
    value = Math.floor(value / 2);
  }
  return result;
}

function transform(input, faults) {
  const values = [...input];
  let index = 1;
  for (let length = degree / 2; length >= 2; length /= 2) {
    for (let start = 0; start < degree; start += 2 * length) {
      const twiddle = faults.includes(index) ? 0 : power(root, reverseBits(index, 7));
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

function unpackPublicKey(encoded, spec) {
  assert.match(encoded, /^[0-9a-f]+$/);
  const bytes = Buffer.from(encoded, 'hex');
  assert.equal(bytes.length, spec.publicKeyBytes);
  const polynomials = [];
  for (let row = 0; row < spec.dimension; row++) {
    const coefficients = [];
    for (let offset = 384 * row; offset < 384 * (row + 1); offset += 3) {
      coefficients.push(bytes[offset] + 256 * (bytes[offset + 1] & 15));
      coefficients.push((bytes[offset + 1] >> 4) + 16 * bytes[offset + 2]);
    }
    assert(coefficients.every(value => value < modulus));
    polynomials.push(coefficients);
  }
  return {polynomials, rho: bytes.subarray(-32)};
}

function matrixEntry(rho, row, column) {
  const bytes = createHash('shake128', {outputLength: CONFIG.shakeBytes})
    .update(rho).update(Buffer.from([column, row])).digest();
  const coefficients = [];
  for (let offset = 0; offset + 2 < bytes.length && coefficients.length < degree; offset += 3) {
    const candidates = [bytes[offset] + 256 * (bytes[offset + 1] & 15),
      (bytes[offset + 1] >> 4) + 16 * bytes[offset + 2]];
    for (const value of candidates) {
      if (value < modulus && coefficients.length < degree) coefficients.push(value);
    }
  }
  assert.equal(coefficients.length, degree, 'matrix_sampling_buffer_exhausted');
  return coefficients;
}

function reconstructDifference(correct, faulted, target, spec) {
  assert(correct.rho.equals(faulted.rho), 'different_public_matrix_seed');
  const column = Array.from({length: spec.dimension}, (_, row) => matrixEntry(correct.rho, row, target));
  const differences = correct.polynomials.map((values, row) =>
    values.map((value, offset) => mod(value - faulted.polynomials[row][offset])));
  const result = Array(degree).fill(0);
  for (let block = 0; block < degree / 2; block++) {
    const gamma = power(root, 2 * reverseBits(block, 7) + 1);
    let solved = false;
    for (let row = 0; row < dimension; row++) {
      const even = column[row][2 * block];
      const odd = column[row][2 * block + 1];
      const determinant = mod(even * even - gamma * odd * odd);
      if (determinant === 0) continue;
      const inverse = power(determinant, modulus - 2);
      const first = differences[row][2 * block];
      const second = differences[row][2 * block + 1];
      result[2 * block] = mod(mod(even * first - gamma * odd * second) * inverse);
      result[2 * block + 1] = mod(mod(even * second - odd * first) * inverse);
      solved = true;
      break;
    }
    assert(solved, 'no_invertible_public_entry_in_block');
    for (let row = 0; row < spec.dimension; row++) {
      const even = column[row][2 * block];
      const odd = column[row][2 * block + 1];
      assert.equal(mod(even * result[2 * block] + gamma * odd * result[2 * block + 1]),
        differences[row][2 * block], 'public_key_rows_inconsistent');
      assert.equal(mod(odd * result[2 * block] + even * result[2 * block + 1]),
        differences[row][2 * block + 1], 'public_key_rows_inconsistent');
    }
  }
  return result;
}

function solveDifference(difference, faults) {
  const matrix = Array.from({length: degree}, (_, row) => {
    const values = new Float64Array(degree + 1);
    values[degree] = difference[row];
    return values;
  });
  for (let column = 0; column < degree; column++) {
    const basis = Array(degree).fill(0);
    basis[column] = 1;
    const correct = transform(basis, []);
    const faulted = transform(basis, faults);
    for (let row = 0; row < degree; row++) matrix[row][column] = mod(correct[row] - faulted[row]);
  }
  const pivots = [];
  for (let column = 0; column < degree; column++) {
    const candidate = matrix.findIndex((values, row) => row >= pivots.length && values[column] !== 0);
    if (candidate === -1) continue;
    const pivot = pivots.length;
    [matrix[pivot], matrix[candidate]] = [matrix[candidate], matrix[pivot]];
    const inverse = power(matrix[pivot][column], modulus - 2);
    for (let offset = column; offset <= degree; offset++) matrix[pivot][offset] = mod(matrix[pivot][offset] * inverse);
    for (let row = 0; row < degree; row++) {
      if (row === pivot) continue;
      const factor = matrix[row][column];
      if (factor === 0) continue;
      for (let offset = column; offset <= degree; offset++) {
        matrix[row][offset] = mod(matrix[row][offset] - factor * matrix[pivot][offset]);
      }
    }
    pivots.push(column);
  }
  for (let row = pivots.length; row < degree; row++) {
    assert.equal(matrix[row][degree], 0, 'difference_outside_declared_map_image');
  }
  const particular = Array(degree).fill(0);
  pivots.forEach((column, row) => { particular[column] = matrix[row][degree]; });
  const freeColumns = Array.from({length: degree}, (_, column) => column).filter(column => !pivots.includes(column));
  if (faults.length !== 0) {
    assert.equal(pivots.length, CONFIG.expectedRank);
    assert.deepEqual(freeColumns, [0, 1]);
    for (const column of pivots) {
      for (const free of freeColumns) assert.equal(matrix[pivots.indexOf(column)][free], 0);
    }
  }
  return {rank: pivots.length, freeColumns, particular};
}

export function observe(packet) {
  assert.deepEqual(Object.keys(packet).sort(),
    ['correct_pk', 'faulted_pk', 'faults', 'level', 'replacement', 'target']);
  const spec = LEVELS[packet.level];
  assert(spec, 'unsupported_level');
  assert.equal(packet.target, 0);
  assert.equal(packet.replacement, 0);
  assert(packet.faults.length === 0 || JSON.stringify(packet.faults) === JSON.stringify(CONFIG.faultIndices));
  const difference = reconstructDifference(unpackPublicKey(packet.correct_pk, spec),
    unpackPublicKey(packet.faulted_pk, spec), packet.target, spec);
  const solution = solveDifference(difference, packet.faults);
  return {difference, ...solution};
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  console.log(JSON.stringify(observe(JSON.parse(readFileSync(process.argv[2], 'utf8')))));
}