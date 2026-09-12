(() => {
    "use strict";

    const viewport = document.getElementById("viewport");
    const score = document.getElementById("score");
    let source;
    let revision = -1;
    let zoom = 1;
    let result = { state: "empty", tuneCount: 0, warnings: [] };
    const editor = window.YueqinScoreEditor;
    let tunes = [];
    let renderedEditable = false;

    // abcjs warning strings contain this formatting wrapper, with escaped source
    // around it. Decode as text, never by assigning source to an HTML parser.
    function plainWarning(value) {
        return String(value)
            .replaceAll('<span style="text-decoration:underline;font-size:1.3em;font-weight:bold;">', "")
            .replaceAll("</span>", "")
            .replace(/&(amp|lt|gt);/g, (_, entity) => ({ amp: "&", lt: "<", gt: ">" })[entity]);
    }

    function errorMessage(error) {
        return error instanceof Error ? error.message : String(error);
    }

    function report() {
        window.webkit?.messageHandlers.abcScore?.postMessage({ kind: "status", revision, ...result, editor: editor?.status() });
    }

    function applyZoom() {
        // Percentage widths resolve inside the zoomed coordinate space. Scale
        // the width too, so zoom changes both the painting and scrollable area.
        score.style.zoom = zoom;
        score.style.width = `${zoom * 100}%`;
    }

    function hasNotation(tune) {
        return tune?.lines?.some(line => line.staff?.some(staff =>
            staff.voices?.some(voice => voice.some(element => element.el_type === "note"))));
    }

    function render() {
        tunes = [];
        score.replaceChildren();
        result = { state: "empty", tuneCount: 0, warnings: [] };
        if (!source.trim()) {
            applyZoom();
            return;
        }

        try {
            if (!window.ABCJS?.renderAbc || !window.ABCJS?.TuneBook) {
                throw new Error("The bundled ABC notation renderer could not be loaded.");
            }
            const book = new window.ABCJS.TuneBook(source);
            let failedTunes = 0;
            for (let index = 0; index < book.tunes.length; index += 1) {
                const entry = book.tunes[index];
                const label = `Tune ${index + 1}: `;
                const warnings = new Set();
                const paper = document.createElement("section");
                paper.className = "tune";
                score.appendChild(paper);
                let parsedTune;
                try {
                    // TuneBook carries file-wide directives into each entry.
                    // Isolating renders lets a malformed tune leave other tunes useful.
                    const rendered = window.ABCJS.renderAbc(paper, source, {
                        startingTune: index,
                        responsive: "resize",
                        paddingleft: 15,
                        paddingright: 15,
                        paddingtop: 15,
                        paddingbottom: 15,
                        // Preserve source systems at the engraver's natural width.
                        // Reflowing dense multi-voice scores into a narrow viewport
                        // can misalign multi-measure rests; scale the SVG instead.
                        expandToWidest: true,
                        foregroundColor: "currentColor",
                        selectionColor: "#778bea",
                        dragColor: "#778bea",
                        add_classes: true,
                        selectTypes: editor?.active() ? ["note"] : false,
                        dragging: !!editor?.active(),
                        clickListener(element, tuneNumber, classes, analysis, drag, mouseEvent) {
                            editor?.click(element, drag, mouseEvent);
                        },
                        afterParsing(tune) {
                            parsedTune = tune;
                            for (const warning of tune.warnings || []) warnings.add(plainWarning(warning));
                        }
                    });
                    parsedTune = rendered?.[0] || parsedTune;
                    if (parsedTune) tunes.push(parsedTune);
                    for (const warning of parsedTune?.warnings || []) warnings.add(plainWarning(warning));
                    if (hasNotation(parsedTune) && paper.querySelector("svg")) {
                        result.tuneCount += 1;
                    } else {
                        paper.remove();
                        warnings.add("No notes or rests could be rendered from this tune.");
                    }
                } catch (error) {
                    paper.remove();
                    failedTunes += 1;
                    warnings.add(`Could not engrave this tune: ${errorMessage(error)}`);
                }

                // ABC ends a tune at a blank line. Make ignored non-comment
                // material visible to the user instead of silently losing it.
                const nextStart = book.tunes[index + 1]?.startPos ?? source.length;
                const trailing = source.slice(entry.startPos + entry.pure.length, nextStart);
                if (trailing.split("\n").some(line => line.trim() && !line.trimStart().startsWith("%"))) {
                    warnings.add("Text after the tune's terminating blank line was not interpreted as notation.");
                }
                result.warnings.push(...Array.from(warnings, warning => label + warning));
            }
            result.state = result.tuneCount > 0 ? "rendered" : failedTunes > 0 ? "failed" : "empty";
            if (result.state === "failed") result.message = "The ABC notation could not be engraved. See parser warnings for details.";
        } catch (error) {
            result.state = "failed";
            result.message = errorMessage(error);
        }
        applyZoom();
        editor?.didRender();
    }

    function highlight(starts, playing = false) {
        const selected = new Set(starts);
        for (const tune of tunes) {
            const engraver = tune.engraver;
            for (const item of engraver?.selectables || []) {
                const element = item.absEl?.abcelem;
                const node = item.svgEl;
                if (!element || !node?.classList) continue;
                const contains = starts.some(start => start >= element.startChar && start < element.endChar);
                node.classList.toggle(playing ? "score-playing" : "score-selected", contains || selected.has(element.startChar));
                if (!playing) {
                    item.absEl.unhighlight(undefined, "currentColor");
                    if (contains) item.absEl.highlight(undefined, getComputedStyle(document.documentElement).getPropertyValue("--accent").trim());
                }
            }
        }
    }
    editor?.attach({
        render(value) { source = value; render(); },
        report, highlight,
        playhead(starts) { highlight(starts, true); }
    });
    window.YueqinABCScore = Object.freeze({
        update(payload) {
            const newEpoch = (payload.editor?.id ?? null) !== (editor?.snapshot().id ?? null);
            if (!newEpoch && !payload.editor && payload.revision < revision) return;
            revision = payload.revision;
            document.documentElement.dataset.theme = payload.theme === "dark" ? "dark" : "light";
            zoom = Number.isFinite(payload.zoom) ? Math.min(3, Math.max(0.5, payload.zoom)) : 1;
            const adopted = editor?.adopt(payload) || { abc: payload.abc, reset: true };
            const editable = !!payload.editor;
            const changed = source !== adopted.abc || renderedEditable !== editable;
            source = adopted.abc;
            renderedEditable = editable;
            if (changed) render(); else applyZoom();
            if (adopted.reset) { viewport.scrollLeft = 0; viewport.scrollTop = 0; }
            report();
        },
        command(action, value = "") { return editor?.command(action, value); },
        snapshot() { return editor?.snapshot(); }
    });

    // This surface is notation only, never a link, form, or drag destination.
    document.addEventListener("click", event => {
        if (event.target.closest?.("a")) event.preventDefault();
    }, true);
    document.addEventListener("submit", event => event.preventDefault(), true);
    document.addEventListener("dragover", event => event.preventDefault());
    document.addEventListener("drop", event => event.preventDefault());
})();
