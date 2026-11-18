import assert from 'node:assert/strict';
import {execFileSync, spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const metadata = JSON.parse(readFileSync(join(root, 'sources.json'), 'utf8'));
const sourcesOnly = process.argv.includes('--sources-only');
const requireClean = process.argv.includes('--require-clean');
assert(process.argv.slice(2).every(argument => ['--sources-only', '--require-clean'].includes(argument)));
const hash = value => createHash('sha256').update(value).digest('hex');
const modules = new Map(metadata.sources.map(entry => [entry.module, entry]));
const imports = new Map();
const forbidden = /\bsorry\b|\bnative_decide\b|^\s*axiom\s/m;

for (const entry of metadata.sources) {
  assert(/^[A-Za-z0-9_./]+\.lean$/.test(entry.file) && !entry.file.includes('..'));
  const contents = readFileSync(join(root, entry.file), 'utf8');
  assert.equal(hash(contents), entry.sha256, entry.file);
  assert(!forbidden.test(contents), entry.file);
  const dependencies = [];
  for (const line of contents.split('\n')) {
    const match = line.match(/^import\s+([A-Za-z0-9_. ]+)$/);
    if (!match) continue;
    for (const module of match[1].trim().split(/\s+/)) {
      assert(modules.has(module) || /^(Mathlib|Lean|Std|Init|Batteries)(\.|$)/.test(module), module);
      if (modules.has(module)) dependencies.push(module);
    }
  }
  imports.set(entry.module, dependencies);
}
const toolchain = readFileSync(join(root, 'lean-toolchain'), 'utf8').trim();
const lock = JSON.parse(readFileSync(join(root, 'lake-manifest.json'), 'utf8'));
assert.equal(toolchain, metadata.leanToolchain);
assert.equal(lock.packages.find(entry => entry.name === 'mathlib').rev, metadata.mathlibRevision);
const order = [];
const active = new Set();
const complete = new Set();
function visit(module) {
  if (complete.has(module)) return;
  assert(!active.has(module), 'Import cycle: ' + module);
  active.add(module);
  for (const dependency of imports.get(module)) visit(dependency);
  active.delete(module);
  complete.add(module);
  order.push(module);
}
for (const module of modules.keys()) visit(module);
const ownBuild = join(root, '.lake/build/lib/lean');
if (requireClean) {
  for (const module of [...order, 'MiuraCounting']) {
    assert(!existsSync(join(ownBuild, module.replaceAll('.', '/') + '.olean')), 'Existing local proof object: ' + module);
  }
}
console.log(JSON.stringify({sources: order.length, sourceHashes: 'pass', imports: 'pass', hygiene: 'pass', requireClean}));
if (sourcesOnly) process.exit(0);

const started = Date.now();
const results = [];
function run(argumentsList, label, timeout = 180000) {
  assert(Date.now() - started < 30 * 60 * 1000, 'Thirty-minute build budget exceeded');
  console.log(label);
  const result = spawnSync('lake', argumentsList, {
    cwd: root, encoding: 'utf8', timeout, maxBuffer: 8 * 1024 * 1024,
    env: {...process.env, LEAN_NUM_THREADS: '2'}
  });
  const output = (result.stdout ?? '') + (result.stderr ?? '');
  assert.equal(result.status, 0, output + (result.error?.message ?? ''));
  assert(!/sorryAx/.test(output), output);
  const warnings = [...output.matchAll(/^warning: ([^\n]+)/gm)].map(match => match[1]);
  assert(warnings.every(warning => warning.startsWith('OrigamiCone/')), output);
  assert(!/\bwarning(?:\(|:)/.test(output.replace(/^warning: OrigamiCone\/[^\n]+/gm, '')), output);
  if (warnings.length > 0) console.log(JSON.stringify({retainedLegacyWarnings: warnings.length, module: label}));
  results.push({label, exitCode: result.status, outputSha256: hash(output), retainedLegacyWarnings: warnings});
  return output;
}
for (const dependency of lock.packages) {
  const path = join(root, lock.packagesDir, dependency.name);
  assert(existsSync(path), 'Missing dependency: ' + dependency.name + '; run lake exe cache get first');
  const revision = execFileSync('git', ['-C', path, 'rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  assert.equal(revision, dependency.rev, 'Dependency revision mismatch: ' + dependency.name);
}
const leanVersion = run(['env', 'lean', '--version'], 'Checking pinned Lean compiler');
assert(leanVersion.includes('version 4.27.0'), leanVersion);
run(['--no-cache', '--no-build', 'build', 'Mathlib'], 'Checking third-party dependency cache; no dependency build allowed');
for (const [index, module] of order.entries()) {
  run(['--no-cache', 'build', module + ':olean'], `${index + 1}/${order.length} ${module}`);
}
run(['--no-cache', 'build', 'MiuraCounting'], 'Building the public library entry point');
const axiomOutput = run(['env', 'lean', '-j2', '-M8192', 'Check.lean'], 'Checking public theorem axiom closures');
assert(!/\bwarning(?:\(|:)/.test(axiomOutput), axiomOutput);
const traces = [...axiomOutput.matchAll(/'([^']+)' (?:depends on axioms:\s*\[([^\]]*)\]|does not depend on any axioms)/g)]
  .map(match => ({name: match[1], axioms: (match[2] ?? '').split(',').map(value => value.trim()).filter(Boolean)}));
assert.equal(traces.length, metadata.checkedDeclarations.length, axiomOutput);
assert.equal(new Set(traces.map(trace => trace.name)).size, traces.length);
for (const name of metadata.checkedDeclarations) assert(traces.some(trace => trace.name === name), name);
for (const trace of traces) assert(trace.axioms.every(axiom => ['propext', 'Classical.choice', 'Quot.sound'].includes(axiom)), trace.name);
const receipt = {schema: 1, leanToolchain: toolchain, mathlibRevision: metadata.mathlibRevision,
  sourceCount: order.length, sourcesSha256: hash(readFileSync(join(root, 'sources.json'))),
  requireClean, localProofObjectsReusedAtStart: !requireClean, elapsedMs: Date.now() - started,
  dependencyDownloadsPerformed: false, results, axiomTraces: traces};
writeFileSync(join(root, 'validation.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({build: 'pass', declarations: traces.length, standardAxiomsOnly: true, elapsedMs: receipt.elapsedMs}));