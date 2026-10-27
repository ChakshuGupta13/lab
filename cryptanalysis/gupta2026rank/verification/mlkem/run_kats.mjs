import assert from 'node:assert/strict';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const PIN = 'd1b2fe782888bdb761a50336012923180be7f502';
const OUTPUT = join(ROOT, 'results', `kat-levels-${new Date().toISOString().replaceAll(':', '-')}`);
const ENV = {PATH: process.env.PATH, HOME: process.env.HOME, LANG: 'C'};

function execute(command, args, options = {}) {
  const result = spawnSync(command, args, {cwd: ROOT, env: ENV, encoding: 'utf8',
    timeout: 240000, maxBuffer: 8 * 1024 * 1024, ...options});
  assert.ifError(result.error);
  assert.equal(result.status, 0, `${command}: ${result.stderr}\n${result.stdout}`);
  return result.stdout;
}

mkdirSync(OUTPUT, {recursive: true});
const upstream = join(ROOT, 'upstream', 'mlkem-native');
assert.equal(execute('git', ['rev-parse', 'HEAD'], {cwd: upstream}).trim(), PIN);
const pinned = join(OUTPUT, 'upstream');
mkdirSync(pinned);
execute('git', ['archive', '--format=tar', `--output=${join(OUTPUT, 'upstream.tar')}`, PIN], {cwd: upstream});
execute('tar', ['-xf', join(OUTPUT, 'upstream.tar'), '-C', pinned]);
const patch = JSON.parse(readFileSync(join(ROOT, 'mlkem-native-inputs.json'))).upstreamPatch;
execute('patch', ['--batch', '--fuzz=0', '-p1', '-d', pinned], {input: patch});
const log = execute('make', ['OPT=0', 'run_kat_512', 'run_kat_768', 'run_kat_1024'], {cwd: pinned});
const passes = (log.match(/kat-sha256: OK/g) || []).length;
assert.equal(passes, 3, log);
writeFileSync(join(OUTPUT, 'kat.log'), log);
const summary = {status: 'KNOWN_ANSWER_TESTS_PASSED', pin: PIN, levels: [512, 768, 1024],
  katSha256Ok: passes, faultHookCompiled: false,
  reasonHookExcluded: 'PQC_FAULT_VALIDATION is not defined by the upstream KAT build'};
writeFileSync(join(OUTPUT, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify({output: OUTPUT, ...summary}, null, 2));
