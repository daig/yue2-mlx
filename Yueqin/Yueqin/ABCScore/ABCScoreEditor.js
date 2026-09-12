(() => {
    "use strict";
    const $ = id => document.getElementById(id);
    const post = message => window.webkit?.messageHandlers.abcScore?.postMessage(message);
    let host, documentID = null, version = 0, abc = "", parsed, selection = [], anchor = null;
    let entering = false, duration = "1/4", dotted = false, playing = null;
    const model = () => window.YueqinScoreModel;
    const selected = () => parsed?.events.filter(event => selection.includes(event.start)) || [];
    const notice = message => { $("editor-notice").textContent = message || ""; };
    function native(action) { post({ kind: "native", id: documentID, action }); }
    function entryDuration() {
        const [n, d = 1] = duration.split("/").map(Number);
        return dotted ? `${n * 3}/${d * 2}` : duration;
    }
    function stop() {
        if (playing) { cancelAnimationFrame(playing.frame); void playing.context.close(); playing = null; }
        $("preview-button").textContent = "Play";
        document.querySelectorAll(".score-playing").forEach(node => node.classList.remove("score-playing"));
    }
    function refresh() {
        try { parsed = model().parse(abc); }
        catch (error) { parsed = { events: [], voices: [], headers: {}, issues: [{severity:"error", message:error.message}], editable:false, compatible:false }; }
        selection = selection.filter(start => parsed.events.some(event => event.start === start));
        const events = selected(), first = events[0];
        $("selection-info").textContent = first ? `${first.voice} · bar ${first.measure} · ${events.length > 1 ? `${events.length} events` : `${first.pitch || first.kind} · ${first.duration} whole notes`}` : entering ? "End of part · continue typing to extend the score" : "Select a note or rest";
        $("entry-state").textContent = entering ? `Overwrite entry · ${entryDuration()}` : "Select mode";
        $("note-input").setAttribute("aria-pressed", String(entering));
        $("dot-button").setAttribute("aria-pressed", String(dotted));
        document.querySelectorAll("[data-duration]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.duration === duration)));
        $("compatibility-summary").textContent = parsed.compatible ? "Within the supported two-part authoring dialect. Model adherence is not guaranteed." : "Outside the supported authoring dialect. Source and save remain available; review the issues below.";
        $("compatibility-list").replaceChildren(...parsed.issues.map(issue => {
            const li = document.createElement("li"), button = document.createElement("button");
            button.textContent = `${issue.severity}: ${issue.message}`;
            button.addEventListener("click", () => { $("source-panel").open = true; $("source-text").focus(); $("source-text").setSelectionRange(issue.start || 0, issue.end ?? issue.start ?? 0); });
            li.append(button); return li;
        }));
        if ($("source-text").value !== abc) {
            const start = $("source-text").selectionStart, end = $("source-text").selectionEnd;
            $("source-text").value = abc;
            $("source-text").setSelectionRange(Math.min(start, abc.length), Math.min(end, abc.length));
        }
        for (const [field, value] of Object.entries({key:parsed.headers.key, meter:parsed.headers.meter, bpm:parsed.headers.bpm, unit:parsed.headers.unit ? `1/${1 / parsed.headers.unit}` : undefined})) {
            if (value !== undefined && document.activeElement !== $(`setup-${field}`)) $(`setup-${field}`).value = String(value);
        }
        if (first) {
            $("entry-voice").value = first.voice;
            const chord = (first.chord || "").match(/^([A-G](?:bb|##|b|#)?)([^/]*)(?:\/([A-G](?:bb|##|b|#)?))?$/);
            if (chord) { $("chord-root").value = chord[1]; $("chord-quality").value = chord[2]; $("chord-bass").value = chord[3] || ""; }
        }
        $("apply-chord").disabled = !first || first.voice !== "Vocal";
        host?.highlight(selection);
    }
    function status() { return documentID === null ? undefined : { id:documentID, version, compatible:!!parsed?.compatible, editable:!!parsed?.editable, selection, issues:parsed?.issues || [] }; }
    function report() { host?.report(); }
    function commit(next, nextSelection, label) {
        if (next === abc) { selection = nextSelection; refresh(); report(); return; }
        stop();
        let start = 0, oldEnd = abc.length, newEnd = next.length;
        while (start < oldEnd && start < newEnd && abc.charCodeAt(start) === next.charCodeAt(start)) start++;
        // A replacement boundary must never bisect a UTF-16 surrogate pair.
        if (start && /[\uDC00-\uDFFF]/.test(abc[start] || next[start] || "")) start--;
        while (oldEnd > start && newEnd > start && abc.charCodeAt(oldEnd - 1) === next.charCodeAt(newEnd - 1)) { oldEnd--; newEnd--; }
        if (oldEnd < abc.length && /[\uDC00-\uDFFF]/.test(abc[oldEnd])) { oldEnd++; newEnd++; }
        const message = { kind:"edit", id:documentID, baseVersion:version, start, end:oldEnd, text:next.slice(start,newEnd), label, selectionStart:nextSelection[0] ?? null };
        abc = next; version++; selection = nextSelection; anchor = selection[0] ?? null;
        refresh(); host.render(abc); post(message); report();
    }
    function action(name, value = "") {
        try {
            const result = model().edit(abc, selection, name, value);
            commit(result.source, result.selection || selection, result.label || "Edit score"); notice(result.notice);
        } catch (error) { notice(error.message || String(error)); host.render(abc); report(); }
    }
    function move(delta, extend) {
        const first = selected().at(-1), voice = first?.voice || $("entry-voice").value;
        const events = parsed.events.filter(event => event.voice === voice);
        const index = first ? events.findIndex(event => event.start === first.start) : delta > 0 ? -1 : events.length;
        const next = events[Math.max(0, Math.min(events.length - 1, index + delta))];
        if (next) select(next, extend);
    }
    function select(event, extend) {
        const anchored = parsed.events.find(item => item.start === anchor);
        if (extend && anchored?.voice === event.voice) {
            const low = Math.min(anchor,event.start), high = Math.max(anchor,event.start);
            selection = [...new Set(parsed.events.filter(item => item.voice === event.voice && item.start >= low && item.start <= high).map(item => item.start))];
        } else { selection = [event.start]; anchor = event.start; }
        refresh(); report();
    }
    function enterPitch(letter) {
        let octave = Number($("entry-octave").value);
        if (!Number.isInteger(octave) || octave < 0 || octave > 9) { notice("Choose an octave from 0 to 9."); return; }
        let pitch = letter === "z" ? "z" : octave >= 5 ? letter.toLowerCase() + "'".repeat(octave - 5) : letter + ",".repeat(4 - octave);
        if (letter !== "z") pitch = ({auto:"", natural:"=", sharp:"^", flat:"_", doubleSharp:"^^", doubleFlat:"__"}[$("accidental").value] || "") + pitch;
        if (entering) action("write", JSON.stringify({ pitch, duration:entryDuration(), voice:$("entry-voice").value }));
        else action(letter === "z" ? "rest" : "pitch", pitch);
    }
    async function preview() {
        if (playing) { stop(); return; }
        if (!parsed?.editable || !parsed.compatible) { notice("Repair or explicitly adapt this source before preview; unsupported music cannot be auditioned faithfully."); return; }
        const Audio = window.AudioContext || window.webkitAudioContext;
        if (!Audio) { notice("Local tone preview is unavailable in this web view."); return; }
        try {
            const context = new Audio(), state = {context,frame:0}; playing = state;
            await context.resume(); if (playing !== state) return;
            const from = selected()[0]?.time || 0, scale = 240 / parsed.headers.bpm, origin = context.currentTime + .04;
            let endTime = from;
            for (const voice of parsed.voices) {
                const events = parsed.events.filter(event => event.voice === voice);
                for (let i = 0; i < events.length; i++) {
                    const event = events[i]; let end = event.time + event.duration;
                    if (event.kind !== "note" || !Number.isFinite(event.midi)) { endTime = Math.max(endTime,end); continue; }
                    while (events[i].tie && events[i+1]?.kind === "note" && events[i+1].midi === event.midi && Math.abs(events[i+1].time - end) < 1e-8) { i++; end += events[i].duration; }
                    endTime = Math.max(endTime,end); if (end <= from) continue;
                    const start = origin + (Math.max(event.time,from) - from) * scale, finish = origin + (end - from) * scale;
                    const oscillator = context.createOscillator(), gain = context.createGain();
                    oscillator.type = voice === "Vocal" ? "sine" : "triangle";
                    oscillator.frequency.value = 440 * 2 ** ((event.midi - 69) / 12);
                    gain.gain.setValueAtTime(0,start); gain.gain.linearRampToValueAtTime(.10, Math.min(start+.008,finish));
                    gain.gain.setValueAtTime(.10,Math.max(start+.008,finish-.018)); gain.gain.linearRampToValueAtTime(0,finish);
                    oscillator.connect(gain); gain.connect(context.destination); oscillator.start(start); oscillator.stop(finish+.01);
                }
            }
            $("preview-button").textContent = "Stop"; notice("Preview tones — not generated audio. Both monophonic parts; no synthesized harmony.");
            const frame = () => {
                if (playing !== state) return;
                const time = from + (context.currentTime-origin) / scale;
                host.playhead(parsed.events.filter(event => event.time <= time && event.time + event.duration > time).map(event => event.start));
                if (time >= endTime) stop(); else state.frame = requestAnimationFrame(frame);
            }; state.frame = requestAnimationFrame(frame);
        } catch (error) { stop(); notice(`Tone preview failed: ${error.message}`); }
    }
    function command(name, value = "") {
        if (documentID === null) return snapshot();
        const sourceField = $("source-text");
        if (document.activeElement === sourceField && ["copy","cut","paste","selectAll"].includes(name)) {
            const start = sourceField.selectionStart, end = sourceField.selectionEnd;
            if (name === "selectAll") sourceField.select();
            else if (name === "paste" && !value) native("paste");
            else {
                if (name === "copy" || name === "cut") post({kind:"clipboard",id:documentID,text:abc.slice(start,end)});
                if (name === "cut" || name === "paste") {
                    const insertion = name === "paste" ? String(value) : "";
                    commit(abc.slice(0,start) + insertion + abc.slice(end), [], "Edit ABC source");
                    sourceField.setSelectionRange(start + insertion.length,start + insertion.length);
                }
            }
            return snapshot();
        }
        if (["undo","redo","save"].includes(name) || (name === "paste" && !value)) native(name);
        else if (name === "copy" || name === "cut") {
            try { const text = model().copy(abc,selection); post({kind:"clipboard",id:documentID,text}); if (name === "cut") action("delete"); }
            catch (error) { notice(error.message); }
        } else if (name === "showSource" || name === "source" || name === "showCompatibility" || name === "help") { const panel = name === "help" ? $("editor-help").parentElement : $(name === "showCompatibility" ? "compatibility-panel" : "source-panel"); panel.open = !panel.open; if (panel.open) panel.scrollIntoView({block:"nearest"}); }
        else if (name === "noteInput" || name === "toggleNoteInput" || name === "input") { entering = !entering; if (!selection.length) move(1,false); refresh(); }
        else if (name === "selectAll") { selection = [...new Set(parsed.events.filter(event => event.voice === ($("entry-voice").value || "Vocal")).map(event => event.start))]; refresh(); report(); }
        else if (name === "play" || name === "preview") void preview();
        else if (name === "stop") stop();
        else if (name === "rest" && entering) enterPitch("z");
        else action(name,String(value));
        return snapshot();
    }
    function snapshot() { return { id:documentID, version, abc, compatible:!!parsed?.compatible, editable:!!parsed?.editable }; }
    function adopt(payload) {
        const next = payload.editor, changedEpoch = (next?.id ?? null) !== documentID;
        if (!changedEpoch && next && !next.force && next.version < version) return {abc, reset:false};
        const changed = changedEpoch || abc !== payload.abc;
        if (changed) stop();
        documentID = next?.id ?? null; version = next?.version ?? 0; abc = payload.abc;
        if (changedEpoch) { selection = []; anchor = null; entering = false; notice(""); }
        if (next && Number.isInteger(next.selectionStart) && (changed || !selection.length)) selection = [next.selectionStart];
        for (const id of ["editor-toolbar","note-palette","editor-lower"]) $(id).hidden = !next;
        if (next) refresh(); else parsed = undefined;
        return {abc,reset:changedEpoch};
    }
    document.querySelectorAll("[data-command]").forEach(button => button.addEventListener("click", () => command(button.dataset.command,button.dataset.value || "")));
    document.querySelectorAll("[data-pitch]").forEach(button => button.addEventListener("click", () => enterPitch(button.dataset.pitch)));
    function setDuration(value) { duration = value; if (!entering && selection.length) action("duration",entryDuration()); refresh(); }
    document.querySelectorAll("[data-duration]").forEach(button => button.addEventListener("click", () => setDuration(button.dataset.duration)));
    $("note-input").addEventListener("click", () => command("noteInput"));
    $("dot-button").addEventListener("click", () => { dotted = !dotted; setDuration(duration); });
    $("preview-button").addEventListener("click", () => void preview());
    $("entry-voice").addEventListener("change", () => { const current = selected()[0]; const next = parsed.events.find(event => event.voice === $("entry-voice").value && (!current || event.time >= current.time)) || parsed.events.find(event => event.voice === $("entry-voice").value); if (next) select(next,false); });
    $("accidental").addEventListener("change", () => { if (!entering) action("accidental",$("accidental").value); });
    $("apply-chord").addEventListener("click", () => { const bass = $("chord-bass").value.trim(); action("chord",$("chord-root").value.trim()+$("chord-quality").value+(bass ? "/"+bass : "")); });
    for (const field of ["key","meter","bpm","unit"]) $(`apply-${field}`).addEventListener("click", () => action(field,$(`setup-${field}`).value));
    for (const verb of ["append","insert"]) $(`${verb}-bars`).addEventListener("click", () => action(`${verb}Bars`,$("bar-count").value));
    $("apply-section").addEventListener("click", () => action("section",$("section-name").value));
    $("source-text").addEventListener("input", () => { if (documentID !== null) commit($("source-text").value,[],"Edit ABC source"); });
    document.addEventListener("keydown", event => {
        if (documentID === null || event.isComposing) return;
        const textField = event.target.closest?.("input,textarea,select,[contenteditable]");
        const key = event.key.toLowerCase(), mod = event.metaKey || event.ctrlKey;
        if (mod && key === "s") { event.preventDefault(); native("save"); return; }
        if (mod && key === "z" && event.target === $("source-text")) { event.preventDefault(); native(event.shiftKey ? "redo" : "undo"); return; }
        if (textField) return;
        if (mod && ["z","c","x","v","a"].includes(key)) { event.preventDefault(); command(({z:event.shiftKey ? "redo":"undo",c:"copy",x:"cut",v:"paste",a:"selectAll"})[key]); return; }
        if (mod || event.altKey) return;
        let handled = true;
        if (key === "enter" || (key === "n" && event.shiftKey)) command("noteInput");
        else if (key === "escape") { entering = false; refresh(); }
        else if (key === "arrowleft" || key === "arrowright") move(key === "arrowleft" ? -1:1,event.shiftKey);
        else if (key === "arrowup" || key === "arrowdown") action("diatonic",key === "arrowup" ? "1":"-1");
        else if (key === "+" || key === "=" || key === "-" || key === "_") action("transpose",key === "-" || key === "_" ? "-1":"1");
        else if (/^[a-g]$/.test(key)) enterPitch(key.toUpperCase());
        else if (key === "r") enterPitch("z");
        else if (key === "t") action("tie");
        else if (key === "delete" || key === "backspace") action("delete");
        else if (key === ".") { dotted = !dotted; setDuration(duration); }
        else if (key >= "3" && key <= "8") setDuration({3:"1/32",4:"1/16",5:"1/8",6:"1/4",7:"1/2",8:"1/1"}[key]);
        else if (key === " ") void preview();
        else handled = false;
        if (handled) event.preventDefault();
    });
    window.addEventListener("pagehide",stop);
    window.YueqinScoreEditor = Object.freeze({
        attach(value) { host = value; }, adopt, status, snapshot, command, stop,
        active:() => documentID !== null,
        didRender() { if (documentID !== null) host.highlight(selection); },
        click(element, drag, mouseEvent) {
            if (documentID === null) return;
            const event = parsed.events.find(item => item.start < element.endChar && item.end > element.startChar);
            if (!event) { notice("This notation element is outside the editable event model. Use ABC repair to inspect it."); if (drag?.step) host.render(abc); return; }
            if (!drag?.step || !selection.includes(event.start)) select(event,!!mouseEvent?.shiftKey);
            if (drag?.step) action("diatonic",String(drag.step));
            $("viewport").focus({preventScroll:true});
        }
    });
})();
