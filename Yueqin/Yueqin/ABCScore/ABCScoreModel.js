/* Native two-part notation semantics. No renderer or host dependencies. */
(() => {
    'use strict';
    const VOICES = ['Vocal', 'Ins'];
    const LEGAL = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];
    const NATURAL = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
    const KEYS = {};
    ['Cb Gb Db Ab Eb Bb F C G D A E B F# C#', 'Abm Ebm Bbm Fm Cm Gm Dm Am Em Bm F#m C#m G#m D#m A#m'].forEach(row => row.split(' ').forEach((key, i) => { KEYS[key] = i - 7; }));
    const CHORD = /^[A-G](?:bb|##|b|#)?(?:m\(maj7\)|7sus4|m7b5|maj7|dim7|sus4|sus2|dim|aug|m7|m6|m|7|6)?(?:\/[A-G](?:bb|##|b|#)?)?$/;
    const PITCH = /^(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)$/;
    const fail = message => { throw new Error(message); };
    const pow2 = n => Number.isSafeInteger(n) && n > 0 && Number.isInteger(Math.log2(n));
    function fraction(text) {
        const m = /^(\d+)(?:\/(\d+))?$/.exec(String(text).trim());
        if (!m || !Number.isSafeInteger(+m[1]) || !pow2(+(m[2] || 1)) || +m[1] <= 0) fail('Use a positive duration with a power-of-two denominator.');
        return +m[1] / +(m[2] || 1);
    }
    function meter(text) {
        if (!/^\d+\/\d+$/.test(text)) fail('Use an explicit meter such as 4/4 or 6/8.');
        return fraction(text);
    }
    function signature(key) {
        if (!Object.prototype.hasOwnProperty.call(KEYS, key)) fail(`Unsupported key ${key}; use a standard major or minor key.`);
        const result = { C: 0, D: 0, E: 0, F: 0, G: 0, A: 0, B: 0 }, count = KEYS[key];
        for (const letter of (count > 0 ? 'FCGDAEB' : 'BEADGCF').slice(0, Math.abs(count))) result[letter] = Math.sign(count);
        return result;
    }
    function writtenPitch(pitch) {
        const m = PITCH.exec(pitch);
        if (!m || (m[3].includes(',') && m[3].includes("'"))) fail('Use a single ABC pitch, for example C, ^F or c\'.');
        const letter = m[2].toUpperCase();
        const written = 60 + NATURAL[letter] + (m[2] === letter ? 0 : 12) + 12 * ([...m[3]].filter(x => x === "'").length - [...m[3]].filter(x => x === ',').length);
        return { letter, written, accidental: m[1] || '', alteration: { '^': 1, '^^': 2, '_': -1, '__': -2, '=': 0 }[m[1]] };
    }
    function spelling(midi, preferred) {
        if (!Number.isInteger(midi) || midi < 0 || midi > 127) fail('The edited pitch is outside MIDI 0–127.');
        let written, letter;
        if (preferred) {
            const p = writtenPitch(preferred);
            if (Math.abs(midi - p.written) <= 2) { written = p.written; letter = p.letter; }
        }
        if (written === undefined) {
            const names = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B'];
            letter = names[midi % 12]; written = Math.floor(midi / 12) * 12 + NATURAL[letter];
        }
        const octave = Math.floor(written / 12) - 5;
        const base = octave >= 1 ? letter.toLowerCase() + "'".repeat(octave - 1) : letter + ','.repeat(-octave);
        return ({ '-2': '__', '-1': '_', 0: '=', 1: '^', 2: '^^' })[midi - written] + base;
    }
    function parseInternal(source, generic = false) {
        source = String(source);
        const result = { source, headers: { key: 'C', meter: '4/4', unit: 1 / 32, bpm: 90 }, voices: [], events: [], measures: [], issues: [], editable: false, compatible: false, _bars: Object.create(null), _removed: [] };
        const issue = (message, start, end, repairable = false) => result.issues.push({ severity: 'error', message, start, end, repairable });
        const states = Object.create(null);
        let active = null, body = false, offset = 0, comments = [], definitions = [], headerSeen = new Set();
        let gapStart = null;
        const state = name => {
            if (!states[name]) {
                states[name] = { name, key: result.headers.key, meter: result.headers.meter, time: 0, local: {}, pending: null, bar: null };
                result._bars[name] = []; result.voices.push(name);
            }
            return states[name];
        };
        const begin = (s, start) => {
            if (!s.bar) {
                s.bar = { voice: s.name, number: result._bars[s.name].length + 1, start, end: start, duration: 0, expectedDuration: meter(s.meter), time: s.time, key: s.key, meter: s.meter, events: [], keys: [], comments: comments.splice(0), closed: false };
            }
            return s.bar;
        };
        const close = (s, end, closed) => {
            const b = begin(s, end); b.end = end; b.closed = closed;
            if (b.duration !== b.expectedDuration) issue(`${s.name} bar ${b.number}: ${b.duration} whole notes; expected ${b.expectedDuration}. Use Repair to fill missing time.`, b.start, end, b.duration < b.expectedDuration);
            if (!closed) issue(`${s.name} bar ${b.number} is missing its closing barline.`, b.start, end, true);
            result._bars[s.name].push(b); result.measures.push(b); s.time += b.expectedDuration; s.bar = null; s.local = {};
        };
        for (const raw of source.split(/\n/)) {
            const line = raw.replace(/\r$/, ''), leading = line.search(/\S|$/), text = line.trim(), start = offset + leading;
            offset += raw.length + 1;
            if (!text) { if (headerSeen.size && gapStart === null) gapStart = start; continue; }
            if (gapStart !== null) {
                issue('A blank line terminates an ABC tune. Remove the interior blank line before adapting or editing.', gapStart, start);
                gapStart = null;
            }
            if (text.startsWith('%')) {
                if (text.startsWith('%%')) issue('Custom ABC directives are outside the editable native dialect.', start, start + text.length);
                else comments.push(text.replace(/^%\s?/, ''));
                continue;
            }
            const field = /^([A-Za-z]):\s*(.*)$/.exec(text);
            if (field) {
                const [, name, value] = field;
                try {
                    if (name === 'V') {
                        const vm = /^(\S+)(.*)$/.exec(value);
                        if (!vm) fail('Missing voice name.');
                        if (body && vm[2].trim()) fail('Voice switches with additional options are ambiguous; keep display options in header definitions only.');
                        if (!body && (vm[2].trim() || !headerSeen.has('K'))) {
                            definitions.push(vm[1]);
                            const expected = vm[1] === 'Vocal' ? 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"' : 'V: Ins clef=treble name="Ins Melody" snm="Inst."';
                            if (!generic && text !== expected) issue('Use the native Vocal and Ins voice definitions, or explicitly Adapt this score.', start, start + text.length);
                            if (generic && vm[2].replace(/\s+(?:clef=(?:treble|bass|alto|tenor)|(?:name|snm)="[^"]*")/g, '').trim()) issue('Adapt cannot infer voice transposition, octave directives or custom voice options.', start, start + text.length);
                            if (generic && text !== expected) result._removed.push(text);
                        } else { body = true; active = state(vm[1]); }
                        continue;
                    }
                    if (!body && ['X', 'T', 'M', 'L', 'Q', 'K'].includes(name)) {
                        if (headerSeen.has(name)) fail(`Repeated ${name}: header or multiple tunes cannot be adapted safely.`);
                        headerSeen.add(name);
                        if (name === 'X' && value !== '1') { if (generic) result._removed.push(`X:${value}`); else issue('Native authoring uses X:1; use Adapt.', start); }
                        if (name === 'T' && value) { if (generic) result._removed.push(`T:${value}`); else issue('Native authoring uses a blank T:; use Adapt to remove the display title.', start); }
                        if (name === 'M') { result.headers.meter = generic && value === 'C' ? '4/4' : generic && value === 'C|' ? '2/2' : value; meter(result.headers.meter); }
                        if (name === 'K') { signature(value); result.headers.key = value; }
                        if (name === 'L') { const u = fraction(value); if (!pow2(1 / u)) fail('L must be 1/<power of two>.'); result.headers.unit = u; }
                        if (name === 'Q') { const q = /^(?:1\/4=)([1-9]\d*)$/.exec(value) || (generic && /^([1-9]\d*)$/.exec(value)); if (!q || !Number.isSafeInteger(+q[1])) fail('Use an integer quarter-note tempo: Q:1/4=90.'); result.headers.bpm = +q[1]; }
                        continue;
                    }
                    if (body && active && (name === 'K' || name === 'M')) {
                        if (active.bar) fail('A line key/meter field must be at a bar boundary.');
                        if (name === 'K') { signature(value); active.key = value; active.local = {}; } else { meter(value); active.meter = value; }
                        continue;
                    }
                    if (generic && !body && ['C', 'O', 'A', 'B', 'D', 'F', 'H', 'N', 'R', 'S', 'Z'].includes(name)) { result._removed.push(text); continue; }
                    fail(`Unsupported field ${name}:; it will not be discarded by an edit.`);
                } catch (error) { issue(error.message, start, start + text.length); }
                continue;
            }
            body = true;
            if (!active) {
                if (generic && definitions.length <= 1) active = state(definitions[0] || 'Vocal');
                else { issue('Music must follow V: Vocal or V: Ins.', start, start + text.length); continue; }
            }
            const s = active;
            let i = leading;
            while (i < line.length) {
                const at = offset - raw.length - 1 + i, tail = line.slice(i);
                if (/^\s/.test(tail)) { i++; continue; }
                if (tail[0] === '%') { if (s.bar) s.bar.comments.push(tail.slice(1).trim()); else comments.push(tail.slice(1).trim()); break; }
                if (tail[0] === '|') {
                    if (/^\|[:\]|]/.test(tail) || (i > 0 && line[i - 1] === ':')) { issue('Repeat/double barlines need deliberate adaptation; playback order is not guessed.', at, at + 2); i += 2; continue; }
                    close(s, at + 1, true); i++; continue;
                }
                const b = begin(s, at);
                const quoted = /^"([^"\n]*)"/.exec(tail);
                if (quoted) {
                    if (!CHORD.test(quoted[1])) issue(`Unsupported harmony or annotation "${quoted[1]}".`, at, at + quoted[0].length);
                    if (!generic && s.name !== 'Vocal') issue('Quoted harmony belongs in Vocal, not Ins.', at, at + quoted[0].length);
                    if (b.pendingChord) issue('Multiple harmonies at one onset are ambiguous.', at, at + quoted[0].length);
                    b.pendingChord = { text: quoted[1], start: at, end: at + quoted[0].length }; i += quoted[0].length; continue;
                }
                const inline = /^\[K:([^\]]+)\]/.exec(tail);
                if (inline) {
                    try { signature(inline[1]); s.key = inline[1]; s.local = {}; b.keys.push({ offset: b.duration, key: s.key }); }
                    catch (error) { issue(error.message, at, at + inline[0].length); }
                    i += inline[0].length; continue;
                }
                const multi = /^Z(\d*)/.exec(tail);
                if (multi) {
                    const count = +(multi[1] || 1);
                    if (!Number.isSafeInteger(count) || count < 1 || count > 4) { issue('Supported compressed rests span one to four bars. Split longer Z spans explicitly.', at, at + multi[0].length); i += multi[0].length; continue; }
                    if (b.duration || b.pendingChord || b.keys.length || s.pending) issue('Compressed rests cannot cover notes, harmony, key changes or ties.', at, at + multi[0].length);
                    for (let n = 0; n < count; n++) {
                        const bar = begin(s, at);
                        const e = { start: at, end: at + multi[0].length, voice: s.name, measure: bar.number, kind: 'multirest', pitch: 'Z', midi: null, duration: bar.expectedDuration, time: s.time, chord: '', tie: false, key: s.key, meter: s.meter, _barOffset: 0 };
                        bar.events.push(e); result.events.push(e); bar.duration = bar.expectedDuration;
                        if (n < count - 1) close(s, e.end, true);
                    }
                    i += multi[0].length; continue;
                }
                const note = /^(\^\^|__|\^|_|=)?([A-Ga-gz])([,']*)(\d*(?:\/\d*)?)(-?)/.exec(tail);
                if (!note) { issue(`Unsupported notation near ${tail.slice(0, 16)}; use advanced source repair.`, at, at + 1); i++; continue; }
                const [, accidental, letter, oct, length, tied] = note;
                i += note[0].length;
                let units = 1;
                if (length.includes('/')) {
                    const parts = length.split('/'); units = +(parts[0] || 1) / +(parts[1] || 2);
                    if (!generic) issue('Fractional duration tokens require explicit Adapt or source repair.', at, at + note[0].length);
                } else units = +(length || 1);
                if (!(units > 0) || !Number.isFinite(units) || (length.includes('/') && !pow2(+(length.split('/')[1] || 2)))) issue('Duration is not on a binary rhythmic grid.', at, at + note[0].length);
                if (!generic && !LEGAL.includes(units)) issue(`Duration multiplier ${length} must be split into native supported lengths.`, at, at + note[0].length);
                const pitch = (accidental || '') + letter + oct;
                const e = { start: at, end: at + note[0].length, voice: s.name, measure: b.number, kind: letter === 'z' ? 'rest' : 'note', pitch, midi: null, duration: units * result.headers.unit, time: s.time + b.duration, chord: b.pendingChord?.text || '', tie: !!tied, key: s.key, meter: s.meter, _barOffset: b.duration, _chordStart: b.pendingChord?.start, _chordEnd: b.pendingChord?.end };
                b.pendingChord = null;
                try {
                    if (letter === 'z') {
                        if (accidental || oct || tied) fail('Rests cannot carry accidentals, octave marks or ties.');
                        if (s.pending) issue('A tie enters a rest; Repair can remove it.', s.pending.start, e.end, true);
                        s.pending = null;
                    } else {
                        const p = writtenPitch(pitch), localID = generic ? p.written : p.letter;
                        e._written = p.written;
                        e.midi = p.written + (p.alteration ?? s.local[localID] ?? signature(s.key)[p.letter]);
                        if (generic && !p.accidental && Object.entries(s.local).some(([written, alteration]) => +written !== p.written && +written % 12 === NATURAL[p.letter] && alteration !== (s.local[localID] ?? signature(s.key)[p.letter]))) issue('Cross-octave accidental propagation is ambiguous in generic ABC. Write explicit accidentals before adapting.', at, e.end);
                        if (p.accidental) s.local[localID] = p.alteration;
                        if (s.pending) {
                            if (!p.accidental && p.written === s.pending._written) e.midi = s.pending.midi;
                            if (e.midi !== s.pending.midi) issue('A tie changes sounding pitch; Repair can remove it.', s.pending.start, e.end, true);
                        }
                        if (e.midi < 0 || e.midi > 127) fail('Pitch is outside MIDI 0–127.');
                        s.pending = e.tie ? e : null;
                    }
                } catch (error) { issue(error.message, at, e.end); }
                b.events.push(e); result.events.push(e); b.duration += e.duration;
            }
        }
        for (const s of Object.values(states)) {
            if (s.bar) close(s, source.length, false);
            if (s.pending) issue(`${s.name} ends with an unresolved tie; use Repair.`, s.pending.start, s.pending.end, true);
            for (const b of result._bars[s.name]) if (b.pendingChord) issue('A harmony has no following rhythmic event.', b.pendingChord.start, b.pendingChord.end);
        }
        if (comments.length) issue('A section comment has no following music; attach it to a bar in source.', source.length, source.length);
        for (const field of ['X', 'T', 'M', 'L', 'Q', 'K']) if (!headerSeen.has(field)) issue(`Missing ${field}: header; supply it in advanced source repair.`, 0, 0);
        if (!generic && (definitions.length !== 2 || definitions[0] !== 'Vocal' || definitions[1] !== 'Ins')) issue('Expected native Vocal and Ins definitions; use Adapt for a generic score.', 0, 0);
        if (!generic && (result.voices.length !== 2 || !VOICES.every(v => result.voices.includes(v)))) issue('Both Vocal and Ins parts are required. Repair can add a missing trailing part.', 0, 0, result.voices.every(v => VOICES.includes(v)));
        if (generic && (result.voices.length < 1 || result.voices.length > 2)) issue('Adapt supports one or two monophonic voices only.', 0, 0);
        if (result.voices.length === 2) {
            const a = result._bars[result.voices[0]], b = result._bars[result.voices[1]];
            if (a.length !== b.length) issue('The two parts have different bar counts; Repair can fill trailing bars.', 0, 0, true);
            for (let n = 0; n < Math.min(a.length, b.length); n++) {
                if (a[n].meter !== b[n].meter || a[n].time !== b[n].time) issue(`Parts have different meter grids at bar ${n + 1}.`, a[n].start, b[n].end);
                if (a[n].key !== b[n].key || JSON.stringify(a[n].keys) !== JSON.stringify(b[n].keys)) issue(`Parts have different key timelines at bar ${n + 1}.`, a[n].start, b[n].end);
            }
        }
        result.editable = result.issues.every(x => x.repairable);
        result.compatible = result.issues.length === 0;
        return result;
    }
    function parse(source) {
        try { return parseInternal(source); }
        catch (error) { return { source: String(source), headers: { key: 'C', meter: '4/4', unit: 1 / 32, bpm: 90 }, voices: [], events: [], measures: [], issues: [{ severity: 'error', message: error.message }], editable: false, compatible: false }; }
    }
    function template(options = {}) {
        const headers = { meter: '4/4', key: 'C', bpm: 90, unit: 1 / 32, ...options };
        signature(headers.key); meter(headers.meter);
        if (!pow2(1 / headers.unit) || !Number.isSafeInteger(+headers.bpm) || +headers.bpm <= 0) fail('Use a binary unit and positive integer tempo.');
        return header(headers) + 'V: Vocal\nZ|\nV: Ins\nZ|\n';
    }
    function header(h) {
        return `X:1\nT:\nM:${h.meter}\nL:1/${1 / h.unit}\nQ:1/4=${h.bpm}\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:${h.key}\n`;
    }
    const rest = duration => ({ kind: 'rest', pitch: 'z', midi: null, duration, chord: '', tie: false });
    function engravedPitch(event, state) {
        const token = spelling(event.midi, event.pitch), p = writtenPitch(token);
        const alteration = event.midi - p.written;
        const native = state.native[p.letter] ?? state.signature[p.letter];
        const engraved = state.written[p.written] ?? state.signature[p.letter];
        if (native === alteration && engraved === alteration) return token.slice(p.accidental.length);
        state.native[p.letter] = alteration;
        state.written[p.written] = alteration;
        return token;
    }
    function extendTo(model, end) {
        let total = model._bars.Vocal.reduce((sum, bar) => sum + bar.expectedDuration, 0);
        while (total < end) {
            for (const voice of VOICES) {
                const bars = model._bars[voice], previous = bars.at(-1);
                bars.push({
                    ...previous, time: total, key: previous.keys.at(-1)?.key || previous.key,
                    events: [rest(previous.expectedDuration)], keys: [], comments: []
                });
            }
            total += model._bars.Vocal.at(-1).expectedDuration;
        }
    }
    function cloneBar(b) { return { ...b, events: b.events.map(e => ({ ...e })), keys: b.keys.map(k => ({ ...k })), comments: [...b.comments] }; }
    function refineUnit(model) {
        let unit = model.headers.unit;
        for (const bars of Object.values(model._bars)) for (const b of bars) {
            for (const duration of [b.expectedDuration, ...b.events.map(e => e.duration), ...b.keys.map(k => k.offset)]) {
                while (!Number.isInteger(duration / unit)) { unit /= 2; if (!Number.isFinite(duration / unit) || !unit) fail('Timing cannot be represented exactly on a binary grid.'); }
            }
        }
        model.headers.unit = unit;
    }
    function serialize(model) {
        refineUnit(model);
        let source = header(model.headers);
        const count = model._bars.Vocal.length;
        const current = { Vocal: { key: model.headers.key, meter: model.headers.meter }, Ins: { key: model.headers.key, meter: model.headers.meter } };
        for (let first = 0; first < count;) {
            let last = first + 1;
            while (last < count && last - first < 4 && VOICES.every(v => {
                const b = model._bars[v][last], previous = model._bars[v][last - 1];
                return b && !b.comments.length && b.key === (previous.keys.at(-1)?.key || previous.key) && b.meter === previous.meter;
            })) last++;
            const comments = [...new Set(VOICES.flatMap(v => model._bars[v][first].comments))];
            for (const comment of comments) source += `% ${comment.replace(/[\r\n]/g, ' ')}\n`;
            for (const voice of VOICES) {
                source += `V: ${voice}\n`;
                const initial = model._bars[voice][first];
                if (initial.meter !== current[voice].meter) source += `M:${initial.meter}\n`;
                if (initial.key !== current[voice].key) source += `K:${initial.key}\n`;
                for (let n = first; n < last; n++) {
                    const b = model._bars[voice][n];
                    let time = 0, keyIndex = 0;
                    let pitchState = { signature: signature(b.key), native: {}, written: {} };
                    for (const event of b.events) {
                        while (keyIndex < b.keys.length && b.keys[keyIndex].offset === time) {
                            const key = b.keys[keyIndex++].key;
                            source += `[K:${key}]`;
                            pitchState = { signature: signature(key), native: {}, written: {} };
                        }
                        if (event.chord) source += `"${event.chord}"`;
                        let units = event.duration / model.headers.unit;
                        if (!Number.isSafeInteger(units) || units < 1) fail('An event has unrepresentable duration.');
                        while (units) {
                            const amount = LEGAL.find(n => n <= units);
                            units -= amount;
                            source += event.kind === 'note' ? engravedPitch(event, pitchState) : 'z';
                            if (amount !== 1) source += amount;
                            if (event.kind === 'note' && (units || event.tie)) source += '-';
                        }
                        time += event.duration;
                        event._targetTime = time - event.duration;
                    }
                    if (keyIndex !== b.keys.length) fail('A key change is not on an event boundary; split the affected event before editing.');
                    source += '|';
                    current[voice] = { key: b.keys.at(-1)?.key || b.key, meter: b.meter };
                }
                source += '\n';
            }
            first = last;
        }
        return source;
    }
    function selection(model, starts) {
        const set = new Set(starts);
        return model.events.filter(e => set.has(e.start) && set.delete(e.start));
    }
    function requireSelection(events) { if (!events.length) fail('Select a note or rest first.'); }
    function chain(model, events) {
        const chosen = new Set(events);
        for (const voice of model.voices) {
            const all = model._bars[voice].flatMap(b => b.events);
            for (let i = 0; i < all.length; i++) if (chosen.has(all[i])) {
                let left = i;
                while (left && all[left - 1].tie) chosen.add(all[--left]);
                let right = i;
                while (all[right]?.tie && all[right + 1]) chosen.add(all[++right]);
            }
        }
        return [...chosen];
    }
    function fixTies(model) {
        for (const bars of Object.values(model._bars)) {
            const events = bars.flatMap(b => b.events);
            events.forEach((e, i) => { if (e.tie && (e.kind !== 'note' || events[i + 1]?.kind !== 'note' || events[i + 1].midi !== e.midi)) e.tie = false; });
        }
    }
    function splitAt(b, at) {
        let time = 0;
        for (let i = 0; i < b.events.length; i++) {
            const e = b.events[i], end = time + e.duration;
            if (time < at && at < end) {
                b.events.splice(i, 1, { ...e, duration: at - time, tie: e.kind === 'note' }, { ...e, duration: end - at, chord: '', tie: e.tie });
                return;
            }
            time = end;
        }
    }
    function overwrite(model, voice, startTime, duration, incoming) {
        const bars = model._bars[voice];
        const total = bars.reduce((n, b) => n + b.expectedDuration, 0);
        if (startTime < 0 || startTime + duration > total) fail('Entry exceeds the part. Append bars first.');
        const preceding = model.events.find(e => e.voice === voice && e.time + e.duration === startTime);
        if (preceding) preceding.tie = false;
        let barTime = 0;
        for (const b of bars) {
            const from = Math.max(0, startTime - barTime), to = Math.min(b.expectedDuration, startTime + duration - barTime);
            if (from < to) {
                splitAt(b, from); splitAt(b, to);
                const boundaries = new Set([from, to]);
                let t = 0;
                const chords = new Map();
                for (const e of b.events) { if (t >= from && t < to && e.chord) { chords.set(t, e.chord); boundaries.add(t); } t += e.duration; }
                for (const k of b.keys) if (k.offset > from && k.offset < to) boundaries.add(k.offset);
                let incomingTime = startTime;
                for (const e of incoming) {
                    if (incomingTime > barTime + from && incomingTime < barTime + to) boundaries.add(incomingTime - barTime);
                    incomingTime += e.duration;
                }
                const points = [...boundaries].sort((a, b) => a - b), replacements = [];
                for (let i = 0; i < points.length - 1; i++) {
                    const global = barTime + points[i];
                    let onset = startTime, item = null;
                    for (const e of incoming) { if (global >= onset && global < onset + e.duration) { item = e; break; } onset += e.duration; }
                    if (!item) fail('Clipboard timing does not cover the requested span.');
                    replacements.push({ ...item, duration: points[i + 1] - points[i], chord: (global === onset && item.chord) || chords.get(points[i]) || '', tie: item.kind === 'note' && (global + points[i + 1] - points[i] < onset + item.duration || item.tie) });
                }
                t = 0;
                const kept = [];
                let inserted = false;
                for (const e of b.events) {
                    if (t === from) { kept.push(...replacements); inserted = true; }
                    if (t < from || t >= to) kept.push(e);
                    t += e.duration;
                }
                if (!inserted) fail('The selected bar has unknown or incomplete timing; Repair first.');
                b.events = kept;
            }
            barTime += b.expectedDuration;
        }
        fixTies(model);
    }
    function copy(source, starts) {
        const model = parse(source);
        if (!model.compatible) fail('Repair or explicitly adapt this score before copying musical notation.');
        const chosen = selection(model, starts); requireSelection(chosen);
        if (new Set(chosen.map(e => e.voice)).size !== 1) fail('Copy one part at a time.');
        chosen.sort((a, b) => a.time - b.time);
        for (let i = 1; i < chosen.length; i++) if (chosen[i].time !== chosen[i - 1].time + chosen[i - 1].duration) fail('Copy a contiguous rhythmic span.');
        const payload = { format: 'Yueqin musical clipboard 1', voice: chosen[0].voice, unit: model.headers.unit, key: chosen[0].key, meter: chosen[0].meter, bpm: model.headers.bpm, events: chosen.map(e => ({ kind: e.kind === 'note' ? 'note' : 'rest', midi: e.midi, pitch: e.pitch, duration: e.duration, chord: e.chord, tie: e.tie, key: e.key })) };
        payload.events.at(-1).tie = false;
        let text = `X:1\nT:\nM:${payload.meter}\nL:1/${1 / payload.unit}\nQ:1/4=${payload.bpm}\nK:${payload.key}\n`;
        for (const e of payload.events) {
            if (e.key !== payload.key) { text += `[K:${e.key}]`; payload.key = e.key; }
            if (e.chord) text += `"${e.chord}"`;
            let units = e.duration / payload.unit;
            while (units) { const n = LEGAL.find(n => n <= units); units -= n; text += (e.kind === 'note' ? spelling(e.midi, e.pitch) : 'z') + (n === 1 ? '' : n) + (e.kind === 'note' && (units || e.tie) ? '-' : ''); }
        }
        return text + '\n% Yueqin-clipboard: ' + JSON.stringify(payload) + '\n';
    }
    function edit(source, starts, action, value = '') {
        if (action === 'adapt') return adapt(source);
        const model = parse(source);
        if (!model.editable || (!model.compatible && action !== 'repair')) fail(model.issues[0]?.message || 'This source cannot be edited safely; use advanced source repair.');
        const chosen = selection(model, starts || []);
        let target = chosen[0] ? { voice: chosen[0].voice, time: chosen[0].time } : null;
        let notice;
        if (['pitch', 'transpose', 'diatonic', 'accidental'].includes(action)) {
            requireSelection(chosen);
            const bases = new Map();
            const pitched = action === 'pitch' ? chosen : chosen.filter(e => e.kind === 'note');
            if (!pitched.length) fail('Select at least one pitched note for this operation.');
            for (const selected of pitched) for (const member of chain(model, [selected])) if (!bases.has(member)) bases.set(member, { ...selected });
            for (const [e, basis] of bases) {
                let midi, preferred = basis.pitch;
                if (action === 'pitch') { const p = writtenPitch(value); midi = p.written + (p.alteration ?? signature(basis.key)[p.letter]); preferred = value; e.kind = 'note'; }
                if (action === 'transpose') { if (!/^[+-]?\d+$/.test(value)) fail('Transpose requires integer semitones.'); midi = basis.midi + +value; }
                if (action === 'diatonic') {
                    if (!/^[+-]?\d+$/.test(value)) fail('Diatonic movement requires integer staff steps.');
                    const p = writtenPitch(basis.pitch), index = Math.floor(p.written / 12) * 7 + 'CDEFGAB'.indexOf(p.letter) + +value;
                    const letter = 'CDEFGAB'[((index % 7) + 7) % 7], written = Math.floor(index / 7) * 12 + NATURAL[letter];
                    midi = written + signature(basis.key)[letter]; preferred = spelling(written);
                }
                if (action === 'accidental') {
                    const p = writtenPitch(basis.pitch), amounts = { natural: 0, sharp: 1, flat: -1, doubleSharp: 2, doubleFlat: -2, auto: signature(basis.key)[p.letter] };
                    if (!Object.prototype.hasOwnProperty.call(amounts, value)) fail('Unknown accidental.'); midi = p.written + amounts[value];
                }
                e.pitch = spelling(midi, preferred); e.midi = midi;
            }
        } else if (action === 'rest' || action === 'delete') {
            requireSelection(chosen);
            for (const e of chosen) { e.kind = 'rest'; e.pitch = 'z'; e.midi = null; e.tie = false; }
            fixTies(model);
        } else if (action === 'tie') {
            requireSelection(chosen);
            for (const e of chosen) {
                if (e.kind !== 'note') fail('Rests cannot be tied.');
                if (e.tie) { e.tie = false; continue; }
                const all = model._bars[e.voice].flatMap(b => b.events), next = all[all.indexOf(e) + 1];
                if (!next || next.kind !== 'note' || next.midi !== e.midi) fail('A tie needs a following note with the same sounding pitch.');
                e.tie = true;
            }
        } else if (action === 'chord') {
            requireSelection(chosen);
            if (value && !CHORD.test(value)) fail('Use a supported chord root/quality and optional note-name slash bass.');
            for (const e of chosen) { if (e.voice !== 'Vocal') fail('Harmony belongs in Vocal.'); e.chord = value; if (e.kind === 'multirest') { e.kind = 'rest'; e.pitch = 'z'; } }
            if (chosen.every(e => source[e.start] !== 'Z')) {
                let output = source;
                for (const e of [...chosen].sort((a, b) => b.start - a.start)) {
                    const from = e._chordStart ?? e.start, to = e._chordEnd ?? e.start;
                    output = output.slice(0, from) + (value ? `"${value}"` : '') + output.slice(to);
                }
                const checked = parse(output);
                if (!checked.compatible) fail(`Harmony edit rejected: ${checked.issues[0]?.message}`);
                const onsets = new Set(chosen.map(e => e.time));
                return { source: output, selection: checked.events.filter(e => e.voice === 'Vocal' && onsets.has(e.time)).map(e => e.start), label: 'Change harmony' };
            }
        } else if (action === 'duration') {
            requireSelection(chosen);
            const duration = fraction(value);
            for (const e of [...chosen].sort((a, b) => b.time - a.time)) {
                const b = model._bars[e.voice][e.measure - 1], index = b.events.indexOf(e), original = e.duration;
                if (e._barOffset + duration > b.expectedDuration) fail('Duration crosses the barline. Use note entry to write across bars.');
                if (duration < original) { e.duration = duration; e.tie = false; b.events.splice(index + 1, 0, rest(original - duration)); }
                if (duration > original) {
                    let remaining = duration - original, cursor = index + 1;
                    while (remaining > 0) {
                        const next = b.events[cursor];
                        if (!next || next.kind === 'note') fail('Lengthening would erase melody. Only adjacent rests can be consumed.');
                        if (next.chord || b.keys.some(k => k.offset >= e._barOffset + original && k.offset < e._barOffset + duration)) fail('Lengthening crosses harmony or a key change. Use note entry, which preserves those onsets.');
                        const take = Math.min(remaining, next.duration); remaining -= take;
                        if (take === next.duration) b.events.splice(cursor, 1); else { next.duration -= take; cursor++; }
                    }
                    e.duration = duration;
                }
            }
            fixTies(model);
        } else if (action === 'write' || action === 'paste') {
            let incoming, voice = chosen[0]?.voice || 'Vocal', duration;
            let startTime = chosen[0]?.time;
            if (action === 'write') {
                let input; try { input = JSON.parse(value); } catch { fail('Note entry requires pitch, duration and voice.'); }
                voice = input.voice || voice;
                if (!VOICES.includes(voice)) fail('Choose Vocal or Ins.');
                startTime ??= model._bars[voice].reduce((sum, bar) => sum + bar.expectedDuration, 0);
                duration = fraction(input.duration);
                extendTo(model, startTime + duration);
                if (input.pitch === 'z') incoming = [rest(duration)];
                else { const p = writtenPitch(input.pitch), bar = model._bars[voice].find(b => startTime >= b.time && startTime < b.time + b.expectedDuration); if (!bar) fail('The selected onset is outside this part.'); const key = [...bar.keys].reverse().find(k => k.offset <= startTime - bar.time)?.key || bar.key; const midi = p.written + (p.alteration ?? signature(key)[p.letter]); incoming = [{ kind: 'note', pitch: input.pitch, midi, duration, chord: '', tie: false }]; }
            } else {
                requireSelection(chosen);
                const match = /^% Yueqin-clipboard: (.+)$/m.exec(value);
                if (!match) fail('Paste musical notation copied from this editor; generic ABC must be opened and explicitly adapted.');
                let payload; try { payload = JSON.parse(match[1]); } catch { fail('Malformed musical clipboard.'); }
                if (payload.format !== 'Yueqin musical clipboard 1' || !Array.isArray(payload.events) || !payload.events.length) fail('Incompatible musical clipboard.');
                incoming = payload.events;
                for (const e of incoming) {
                    if (!['note', 'rest'].includes(e.kind) || !Number.isFinite(e.duration) || e.duration <= 0 || !Number.isSafeInteger(e.duration / payload.unit) || !pow2(1 / payload.unit) || typeof e.tie !== 'boolean' || typeof e.chord !== 'string' || (e.chord && (!CHORD.test(e.chord) || voice !== 'Vocal'))) fail('Clipboard contains unsupported timing or harmony.');
                    signature(e.key); if (e.kind === 'note') spelling(e.midi, e.pitch);
                }
                const clipboardBody = value.slice(0, match.index).trimEnd();
                let canonical = `X:1\nT:\nM:${incoming[0].meter || payload.meter}\nL:1/${1 / payload.unit}\nQ:1/4=${payload.bpm}\nK:${incoming[0].key}\n`;
                let clipboardKey = incoming[0].key;
                for (const e of incoming) {
                    if (e.key !== clipboardKey) { canonical += `[K:${e.key}]`; clipboardKey = e.key; }
                    if (e.chord) canonical += `"${e.chord}"`;
                    let units = e.duration / payload.unit;
                    while (units) { const n = LEGAL.find(n => n <= units); units -= n; canonical += (e.kind === 'note' ? spelling(e.midi, e.pitch) : 'z') + (n === 1 ? '' : n) + (e.kind === 'note' && (units || e.tie) ? '-' : ''); }
                }
                if (canonical.trimEnd() !== clipboardBody || value.slice(match.index + match[0].length).trim()) fail('The clipboard notation and musical payload disagree. Copy the music again.');
                for (let i = 0; i < incoming.length; i++) if (incoming[i].tie && (incoming[i].kind !== 'note' || incoming[i + 1]?.kind !== 'note' || incoming[i + 1].midi !== incoming[i].midi)) fail('Clipboard contains an invalid or dangling tie.');
                duration = incoming.reduce((n, e) => n + e.duration, 0);
                let onset = chosen[0].time;
                for (const e of incoming) {
                    const b = model._bars[voice].find(b => onset >= b.time && onset < b.time + b.expectedDuration);
                    const key = b && ([...b.keys].reverse().find(k => k.offset <= onset - b.time)?.key || b.key);
                    if (key !== e.key) fail('Clipboard key timeline differs from the destination. Align key changes before pasting; pitches will not be guessed.');
                    onset += e.duration;
                }
            }
            overwrite(model, voice, startTime, duration, incoming);
            target = { voice, time: startTime + duration };
        } else if (['appendBars', 'insertBars', 'deleteBars', 'duplicateBars'].includes(action)) {
            if (action !== 'appendBars') requireSelection(chosen);
            const numbers = [...new Set(chosen.map(e => e.measure - 1))].sort((a, b) => a - b);
            if (numbers.some((n, i) => i && n !== numbers[i - 1] + 1)) fail('Select a contiguous range of bars.');
            let count = 1;
            if (action === 'appendBars' || action === 'insertBars') { if (!/^[1-9]\d*$/.test(value) || !Number.isSafeInteger(+value)) fail('Enter a positive whole number of bars.'); count = +value; }
            const index = action === 'appendBars' ? model._bars.Vocal.length : numbers[0];
            for (const voice of VOICES) {
                const bars = model._bars[voice];
                if (action === 'deleteBars') {
                    if (numbers.length === bars.length) fail('Keep at least one bar; replace its notes with rests instead.');
                    if (index > 0) bars[index - 1].events.at(-1).tie = false;
                    bars.splice(index, numbers.length);
                } else if (action === 'duplicateBars') {
                    const copies = bars.slice(index, index + numbers.length).map(cloneBar);
                    copies.at(-1).events.at(-1).tie = false;
                    bars[index + numbers.length - 1].events.at(-1).tie = false;
                    bars.splice(index + numbers.length, 0, ...copies);
                } else {
                    const reference = bars[Math.min(index, bars.length - 1)];
                    const key = action === 'appendBars' ? reference.keys.at(-1)?.key || reference.key : reference.key;
                    const additions = Array.from({ length: count }, () => ({ ...reference, key, events: [rest(reference.expectedDuration)], keys: [], comments: [] }));
                    if (index > 0) bars[index - 1].events.at(-1).tie = false;
                    bars.splice(index, 0, ...additions);
                }
            }
            fixTies(model);
        } else if (action === 'meter') {
            const expected = meter(value);
            for (const bars of Object.values(model._bars)) for (const b of bars) {
                if (expected < b.expectedDuration) {
                    let time = 0;
                    for (const e of b.events) { if (time + e.duration > expected && (e.kind === 'note' || e.chord)) fail('Shorter meter would discard notes or harmony. Shorten or move that music explicitly first.'); time += e.duration; }
                    if (b.keys.some(k => k.offset >= expected)) fail('Shorter meter would discard a key change.');
                    splitAt(b, expected); time = 0; b.events = b.events.filter(e => { const keep = time < expected; time += e.duration; return keep; });
                } else if (expected > b.expectedDuration) { b.events.at(-1).tie = false; b.events.push(rest(expected - b.expectedDuration)); }
                b.expectedDuration = expected; b.meter = value;
            }
            model.headers.meter = value; fixTies(model);
        } else if (action === 'key') {
            signature(value);
            const original = model.headers.key;
            for (const bars of Object.values(model._bars)) {
                let initial = true;
                for (const b of bars) { if (b.key !== original) initial = false; if (initial) b.key = value; if (b.keys.length) initial = false; }
            }
            model.headers.key = value;
        } else if (action === 'bpm') {
            if (!/^[1-9]\d*$/.test(value) || !Number.isSafeInteger(+value)) fail('Tempo must be a positive integer in quarter notes per minute.');
            const replacement = source.replace(/^Q:\s*1\/4=\d+\s*$/m, `Q:1/4=${value}`);
            return { source: replacement, selection: starts || [], label: 'Change tempo' };
        } else if (action === 'unit') {
            const unit = fraction(value);
            if (!pow2(1 / unit)) fail('Unit must be 1/<power of two>.');
            if (model.events.some(e => !Number.isInteger(e.duration / unit))) fail('That unit is too coarse for the existing rhythms. Choose a finer unit.');
            model.headers.unit = unit;
        } else if (action === 'section') {
            requireSelection(chosen);
            if (!/^[A-Za-z][A-Za-z0-9 _-]*$/.test(value)) fail('Use a plain section name such as verse, chorus, bridge or interlude.');
            const index = Math.min(...chosen.map(e => e.measure)) - 1;
            for (const voice of VOICES) model._bars[voice][index].comments.push(value.trim());
        } else if (action === 'repair') {
            if (!model.editable) fail('Unknown notation cannot be repaired automatically. Correct the flagged source first.');
            for (const voice of VOICES) model._bars[voice] ||= [];
            const count = Math.max(...VOICES.map(v => model._bars[v].length));
            if (!count) fail('There is no timed music to repair. Create a new score or supply a bar in source.');
            for (const voice of VOICES) {
                const bars = model._bars[voice], other = model._bars[voice === 'Vocal' ? 'Ins' : 'Vocal'];
                while (bars.length < count) { const reference = other[bars.length]; bars.push({ ...cloneBar(reference), voice, events: [rest(reference.expectedDuration)], keys: reference.keys.map(k => ({ ...k })) }); const b = bars.at(-1); for (const k of b.keys) splitAt(b, k.offset); }
                for (const b of bars) { const duration = b.events.reduce((n, e) => n + e.duration, 0); if (duration > b.expectedDuration) fail('Repair never discards overfull music. Move the extra notes explicitly.'); if (duration < b.expectedDuration) b.events.push(rest(b.expectedDuration - duration)); }
            }
            fixTies(model); notice = 'Filled missing time and trailing bars; removed only invalid or dangling ties.';
        } else fail(`Unknown score action: ${action}`);
        const output = serialize(model), checked = parse(output);
        if (!checked.compatible) fail(`Edit rejected without changing the source: ${checked.issues[0]?.message}`);
        const retainSelection = ['pitch', 'transpose', 'diatonic', 'accidental', 'rest', 'delete', 'tie', 'chord', 'duration', 'key', 'unit'].includes(action);
        const retained = retainSelection
            ? checked.events.filter(e => chosen.some(old => old.voice === e.voice && old.time === e.time)).map(e => e.start)
            : [];
        const next = target ? checked.events.find(e => e.voice === target.voice && e.time >= target.time) : null;
        return { source: output, selection: retainSelection ? retained : next ? [next.start] : [], label: ({ pitch: 'Change pitch', transpose: 'Transpose', diatonic: 'Move pitch', accidental: 'Change accidental', rest: 'Replace with rests', delete: 'Delete notes', tie: 'Toggle tie', chord: 'Change harmony', duration: 'Change duration', write: 'Enter notes (overwrite)', paste: 'Paste music', appendBars: 'Append bars', insertBars: 'Insert bars', deleteBars: 'Delete bars', duplicateBars: 'Duplicate bars', meter: 'Change meter', key: 'Change key (preserve pitches)', unit: 'Change rhythmic unit', section: 'Add section', repair: 'Repair score' })[action], notice };
    }
    function adapt(source) {
        const native = parse(source);
        if (native.compatible) return { source, selection: native.events.length ? [native.events[0].start] : [], label: 'Adapt to native score', notice: 'This score already uses the editable native authoring dialect; no adaptation was needed.' };
        const model = parseInternal(source, true);
        if (model.issues.length) fail(`Adapt refused: ${model.issues[0].message}`);
        const names = model.voices;
        const vocal = names.includes('Vocal') ? 'Vocal' : names[0], ins = names.find(n => n !== vocal);
        if (ins && model._bars[ins].some(b => b.events.some(e => e.chord))) fail('The second part contains harmony. Move it deliberately to Vocal before adapting.');
        const bars = { Vocal: model._bars[vocal], Ins: ins ? model._bars[ins] : model._bars[vocal].map(b => { const copy = cloneBar(b); copy.events = [rest(b.expectedDuration)]; for (const k of copy.keys) splitAt(copy, k.offset); return copy; }) };
        model._bars = bars;
        const output = serialize(model), checked = parse(output);
        if (!checked.compatible) fail(`Adapt refused: ${checked.issues[0].message}`);
        return { source: output, selection: checked.events.length ? [checked.events[0].start] : [], label: 'Adapt to native score', notice: `Adapted explicit simple monophonic music to Vocal and Ins, preserving sounding pitches and timing.${model._removed.length ? ' Removed display-only headers: ' + model._removed.join(', ') + '.' : ''} This checks editor authoring compatibility, not a hard engine whitelist or generated-audio adherence.` };
    }
    globalThis.YueqinScoreModel = Object.freeze({ parse, edit, template, copy });
})();
