const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require('../Yueqin/Yueqin/ABCScore/ABCScoreModel.js');
const score = globalThis.YueqinScoreModel;

function fixture(vocal, ins = 'Z|', key = 'C') {
  return score.template({ key }).replace('V: Vocal\nZ|\nV: Ins\nZ|', `V: Vocal\n${vocal}\nV: Ins\n${ins}`);
}
function parsed(source) {
  const result = score.parse(source);
  assert.equal(result.compatible, true, JSON.stringify(result.issues));
  return result;
}
const notes = model => model.events.filter(e => e.kind === 'note');
const signature = model => notes(model).map(e => [e.voice, e.time, e.midi, e.duration, e.tie]);
const first = source => parsed(source).events[0].start;

// Native accidentals are deliberately not ordinary per-octave ABC semantics.
test('native accidental state crosses octaves but tied carry does not leak into a new bar', () => {
  assert.deepEqual(notes(parsed(fixture('^F8f8F16|'))).map(e => e.midi), [66, 78, 66]);
  assert.deepEqual(notes(parsed(fixture('^F32-|F8F24|', 'Z2|'))).map(e => e.midi), [66, 66, 65]);
});

test('editing a tied continuation changes the whole sounding note, not later untied notes', () => {
  const source = fixture('^F32-|F8F24|', 'Z2|');
  const original = parsed(source);
  const output = parsed(score.edit(source, [notes(original)[1].start], 'transpose', '1').source);
  assert.deepEqual(notes(output).map(e => e.midi), [67, 67, 65]);
  assert.deepEqual(notes(output).map(e => [e.time, e.duration, e.tie]), [[0, 1, true], [1, .25, false], [1.25, .75, false]]);
});

test('phrase transposition ignores rests and keeps the phrase selected for a second edit', () => {
  const source = fixture('C8z8E8G8|');
  const selected = parsed(source).events.filter(e => e.voice === 'Vocal').map(e => e.start);
  const once = score.edit(source, selected, 'transpose', '1');
  const twice = parsed(score.edit(once.source, once.selection, 'transpose', '1').source);
  assert.deepEqual(notes(twice).map(e => e.midi), [62, 66, 69]);
  assert.deepEqual(twice.events.filter(e => e.voice === 'Vocal').map(e => e.time), [0, .25, .5, .75]);
});

test('duration changes preserve subsequent onsets and cannot consume melody', () => {
  const source = fixture('C8z8D16|');
  const longer = parsed(score.edit(source, [first(source)], 'duration', '1/2').source);
  assert.deepEqual(notes(longer).map(e => [e.midi, e.time, e.duration]), [[60, 0, .5], [62, .5, .5]]);
  assert.throws(() => score.edit(source, [first(source)], 'duration', '3/4'), /melody/);
  const shorter = parsed(score.edit(source, [first(source)], 'duration', '1/8').source);
  assert.deepEqual(notes(shorter).map(e => [e.time, e.duration]), [[0, .125], [.5, .5]]);
});

test('overwrite splits a held note at barlines and preserves a harmony change', () => {
  const source = fixture('z24"C"z8|"Am"z32|', 'Z2|');
  const start = parsed(source).events.find(e => e.voice === 'Vocal' && e.time === .75).start;
  const output = parsed(score.edit(source, [start], 'write', JSON.stringify({ pitch:'E', duration:'1/2', voice:'Vocal' })).source);
  assert.deepEqual(notes(output).map(e => [e.midi, e.time, e.duration, e.tie]), [[64, .75, .25, true], [64, 1, .25, false]]);
  assert.deepEqual(output.events.filter(e => e.chord).map(e => [e.time, e.chord]), [[.75, 'C'], [1, 'Am']]);
});

test('note entry advances past the last bar instead of overwriting the last note', () => {
  let source = score.template(), selection = [first(source)];
  for (const pitch of ['C','D','E','F','G']) {
    const change = score.edit(source, selection, 'write', JSON.stringify({pitch, duration:'1/4', voice:'Vocal'}));
    source = change.source; selection = change.selection;
  }
  const output = parsed(source);
  assert.deepEqual(notes(output).map(e => [e.time, e.midi]), [[0,60],[.25,62],[.5,64],[.75,65],[1,67]]);
  assert.equal(output.measures.filter(b => b.voice === 'Vocal').length, 2);
  assert.equal(output.measures.filter(b => b.voice === 'Ins').length, 2);
});

test('harmony on a compressed rest affects only its first logical bar', () => {
  const source = fixture('Z4|', 'Z4|');
  const output = parsed(score.edit(source, [first(source)], 'chord', 'F#m7/C#').source);
  assert.deepEqual(output.events.filter(e => e.chord).map(e => [e.time,e.chord]), [[0,'F#m7/C#']]);
  assert.equal(output.measures.filter(b => b.voice === 'Vocal').length, 4);
  assert.throws(() => score.edit(source, [first(source)], 'chord', 'C13'), /supported chord/);
});

test('structural operations keep both parts aligned and retain sections', () => {
  const source = fixture('C32|D32|', 'E32|F32|').replace('V: Vocal\nC', '% verse\nV: Vocal\nC');
  const second = parsed(source).events.find(e => e.voice === 'Vocal' && e.measure === 2).start;
  const inserted = parsed(score.edit(source, [second], 'insertBars', '2').source);
  assert.deepEqual(notes(inserted).map(e => [e.voice,e.time,e.midi]), [['Vocal',0,60],['Vocal',3,62],['Ins',0,64],['Ins',3,65]]);
  assert.match(inserted.source, /% verse/);
  const duplicate = parsed(score.edit(source, [second], 'duplicateBars').source);
  assert.deepEqual(notes(duplicate).filter(e => e.voice === 'Vocal').map(e => e.midi), [60,62,62]);
  const removed = parsed(score.edit(source, [second], 'deleteBars').source);
  assert.deepEqual(notes(removed).map(e => [e.voice,e.midi]), [['Vocal',60],['Ins',64]]);
});

test('key changes and rhythmic unit refinement retain the sounding music', () => {
  const source = fixture('F8^F8f8=F8|');
  const original = parsed(source);
  const changed = parsed(score.edit(source, [], 'key', 'D').source);
  assert.deepEqual(signature(changed), signature(original));
  const refined = parsed(score.edit(source, [], 'unit', '1/64').source);
  assert.deepEqual(signature(refined), signature(original));
});

test('repair fills incomplete time and missing parts without discarding unknown content', () => {
  const partial = fixture('C8-', '').replace('V: Ins\n\n', '');
  const repaired = parsed(score.edit(partial, [], 'repair').source);
  assert.deepEqual(notes(repaired).map(e => [e.midi,e.duration,e.tie]), [[60,.25,false]]);
  assert.equal(repaired.measures.length, 2);
  const unsupported = fixture('(3C8D8E8z8|');
  assert.equal(score.parse(unsupported).compatible, false);
  assert.throws(() => score.edit(unsupported, [], 'repair'), /Unsupported|unsupported/);
});

test('explicit simple import adaptation preserves music and refuses ambiguous accidental semantics', () => {
  const generic = 'X:1\nT:Imported theme\nM:4/4\nL:1/8\nQ:1/4=120\nK:G\nF2G2A2B2|\n';
  assert.equal(score.parse(generic).compatible, false);
  const output = parsed(score.edit(generic, [], 'adapt').source);
  assert.deepEqual(notes(output).map(e => [e.midi,e.time,e.duration]), [[66,0,.25],[67,.25,.25],[69,.5,.25],[71,.75,.25]]);
  assert.equal(output.headers.bpm, 120);
  assert.throws(() => score.edit(generic.replace('K:G','K:C').replace('F2G2A2B2|','^F2f2F4|'), [], 'adapt'), /ambiguous/);
});

test('musical clipboard preserves dotted rhythm and rejects mismatched payloads', () => {
  const source = fixture('C12D4z16|');
  const chosen = notes(parsed(source)).map(e => e.start);
  const clipboard = score.copy(source, chosen);
  const target = score.template();
  const output = parsed(score.edit(target, [first(target)], 'paste', clipboard).source);
  assert.deepEqual(notes(output).map(e => [e.midi,e.time,e.duration]), [[60,0,.375],[62,.375,.125]]);
  assert.throws(() => score.edit(target, [first(target)], 'paste', clipboard.replace('=C12','=E12')), /disagree/);
});

test('editing published native examples retains a compatible score', () => {
  const source = fs.readFileSync(path.join(__dirname, '../vendor/yue/examples/score.abc'), 'utf8');
  const original = parsed(source);
  const output = parsed(score.edit(source, [notes(original)[0].start], 'transpose', '-1').source);
  assert.equal(notes(output)[0].midi, notes(original)[0].midi - 1);
  assert.deepEqual(signature(output).slice(1), signature(original).slice(1));
});
