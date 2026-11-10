// The practice panel: Watershed's controls for the engine in the server (src/player.rs, player.js).
//   Tempo 25-200 % and transpose in semitones, each without changing the other; drums can keep their pitch and the
//   vocal its formants. A click on the track's beat grid, with a count-in. The stems as a mixer: volume, pan, mute,
//   solo, offset. The output device. What you set is saved with the track and comes back when you open it again.
// Loaded after app.js: it uses its helpers (h, icon, $, state, player, fmtBpm...). Only shown when the engine plays.

const PRACTICE = { open: false, el: null, binds: [], saved: "", pendingSave: null, applied: null };
const PRACTICE_DEFAULT = { tempo: 1, semitones: 0 };

// What is saved per track: everything a musician sets while practicing, nothing about where the playhead is.
function practiceSettings(s) {
  return {
    tempo: Math.round(s.tempo * 100) / 100, semitones: s.semitones,
    vocal: { formants: s.vocal_formants, tonality_limit: s.vocal_tonality_limit },
    click: { on: s.click.on, count_in: s.click.count_in, gain: Math.round(s.click.gain * 100) / 100, sound: s.click.sound, channels: s.click.channels },
    // Only real stems: the original mix, loaded as one track called "mix", has no mixer.
    stems: Object.fromEntries(s.stems.filter((x) => STEM_ORDER.includes(x.name)).map((x) => [x.name, { gain: Math.round(x.gain * 100) / 100, pan: Math.round(x.pan * 100) / 100, mute: x.mute, pitch_lock: x.pitch_lock, offset_ms: Math.round(x.offset_ms) }])),
  };
}
// Nothing to remember: the defaults (the saved value is then cleared).
function isDefaultPractice(p) {
  return p.tempo === 1 && p.semitones === 0 && !p.vocal.formants && !p.vocal.tonality_limit && !p.click.on && !p.click.count_in && p.click.gain === 0.5
    && p.click.sound === "click" && p.click.channels === "both"
    && Object.entries(p.stems).every(([n, x]) => x.gain === 1 && x.pan === 0 && !x.mute && x.offset_ms === 0 && x.pitch_lock === (n === "drums"));
}

// Put a track's saved settings (or the defaults) on the engine after it loaded.
async function applyPractice(t, whole = true) {
  if (!player.engine || !t) return;
  const p = typeof t.practice === "string" && t.practice ? (() => { try { return JSON.parse(t.practice); } catch { return null; } })() : t.practice;
  PRACTICE.applied = t.id;
  if (whole) await player.set({ tempo: p?.tempo ?? PRACTICE_DEFAULT.tempo, semitones: p?.semitones ?? PRACTICE_DEFAULT.semitones,
                                vocal: p?.vocal || { formants: false, tonality_limit: false }, click: p?.click || { on: false, count_in: false } });
  if (player.snap?.stems_loaded) for (const [name, v] of Object.entries(p?.stems || {})) { if (STEM_ORDER.includes(name)) await player.stem(name, v).catch(() => {}); }
  PRACTICE.saved = JSON.stringify(practiceSettings(player.snap));
}

// Save what changed, 0.8 s after the last change, as one edit.
function watchPractice() {
  player.addEventListener("snapshot", () => {
    const s = player.snap;
    if (!s || s.track_id == null || s.track_id !== state.current?.id || PRACTICE.applied !== s.track_id) return;
    const p = practiceSettings(s), now = JSON.stringify(p);
    if (now === PRACTICE.saved) { clearTimeout(PRACTICE.pendingSave); PRACTICE.pendingSave = null; PRACTICE.pendingFor = null; return; }
    if (PRACTICE.pendingFor === now) return;
    PRACTICE.pendingFor = now;
    clearTimeout(PRACTICE.pendingSave);
    const id = s.track_id;
    PRACTICE.pendingSave = setTimeout(() => {
      PRACTICE.saved = now; PRACTICE.pendingFor = null;
      const value = isDefaultPractice(p) ? null : now;
      api(`/api/tracks/${id}`, { method: "PATCH", body: { practice: value } }).then(() => { if (state.current?.id === id) state.current.practice = value; }).catch(() => {});
    }, 800);
  });
  player.addEventListener("modechange", () => { if (state.system) renderSystem(); if (state.focus && state.current) renderDetail(); });
  // A track (or its stems) just loaded: put its saved settings on the engine.
  player.addEventListener("loadedmetadata", () => {
    const s = player.snap, t = state.current;
    if (!player.engine || !t || s?.track_id !== t.id) return;
    const fresh = PRACTICE.applied !== t.id, stems = s.stems_loaded && PRACTICE.appliedStems !== t.id;
    if (!fresh && !stems) return;
    if (s.stems_loaded) PRACTICE.appliedStems = t.id;
    applyPractice(t, fresh).then(syncPractice);
  });
  player.addEventListener("snapshot", syncPractice);
}

function practiceButton() {
  if (!player.engine) return null;
  const s = player.snap, changed = s && (Math.abs(s.tempo - 1) > 0.004 || s.semitones !== 0 || s.click.on);
  return h("button", { class: "tb" + (changed ? " lit" : ""), id: "practiceBtn", "aria-pressed": String(PRACTICE.open), "aria-label": "Practice: tempo, transpose, click, stem mixer",
    title: changed ? "Practice (settings are not at their defaults)" : "Practice: tempo, transpose, click, stem mixer", onclick: togglePractice }, icon("sliders"));
}

function togglePractice(force) {
  PRACTICE.open = typeof force === "boolean" ? force : !PRACTICE.open;
  const b = $("#practiceBtn"); if (b) b.setAttribute("aria-pressed", String(PRACTICE.open));
  if (!PRACTICE.open) { PRACTICE.el?.remove(); PRACTICE.el = null; return; }
  PRACTICE.el = buildPractice();
  $("#detail").append(PRACTICE.el);
  syncPractice();
}

function buildPractice() {
  const t = state.current, s = player.snap;
  PRACTICE.binds = [];
  const bind = (el, get, set = (v) => { el.value = v; }) => { PRACTICE.binds.push({ el, get, set }); return el; };
  const range = (min, max, step, label, onInput) => h("input", { type: "range", min, max, step, "aria-label": label, oninput: (e) => onInput(+e.target.value) });
  const row = (label, ...ctl) => h("div", { class: "pr-row" }, h("span", { class: "pr-l" }, label), ...ctl);

  // Tempo and key
  const tempoOut = h("output", { class: "pr-out" }), semiOut = h("output", { class: "pr-out" });
  const tempo = bind(range(25, 200, 1, "Tempo in percent", (v) => player.set({ tempo: v / 100 })), (x) => Math.round(x.tempo * 100));
  const semi = bind(range(-12, 12, 1, "Transpose in semitones", (v) => player.set({ semitones: v })), (x) => x.semitones);
  PRACTICE.binds.push({ el: tempoOut, get: (x) => `${Math.round(x.tempo * 100)} %${t?.bpm ? ` · ${Math.round(t.bpm * x.tempo)} BPM` : ""}`, set: (v) => { tempoOut.textContent = v; } });
  PRACTICE.binds.push({ el: semiOut, get: (x) => (x.semitones > 0 ? "+" : "") + x.semitones + " st", set: (v) => { semiOut.textContent = v; } });
  const reset = (what) => h("button", { class: "tb small", "aria-label": `Reset ${what}`, title: `Reset ${what}`, onclick: () => player.set(what === "tempo" ? { tempo: 1 } : { semitones: 0 }) }, icon("refresh", "ic sm"));
  const step = (d) => h("button", { class: "tb small", "aria-label": `Tempo ${d > 0 ? "+" : "−"}5 %`, title: `${d > 0 ? "+" : "−"}5 %`, onclick: () => player.set({ tempo: Math.max(0.25, Math.min(2, Math.round((player.snap.tempo + d * 0.05) * 100) / 100)) }) }, d > 0 ? "+5" : "−5");
  const sec1 = h("section", {}, h("h4", {}, "Tempo and key"),
    row("Tempo", step(-1), tempo, step(1), tempoOut, reset("tempo")),
    row("Transpose", semi, semiOut, reset("transpose")),
    t?.camelot ? h("div", { class: "pr-note" }, `${t.shifted_camelot || t.camelot} as the deck plays it. Practice transposing is only for listening: it is not written to the card.`) : null);

  // Click
  const sel = (opts, label, key) => bind(h("select", { "aria-label": label, onchange: (e) => player.set({ click: { [key]: e.target.value } }) }, opts.map(([v, l]) => h("option", { value: v }, l))), (x) => x.click[key]);
  const toggle = (label, key) => { const c = bind(h("input", { type: "checkbox", onchange: (e) => player.set({ click: { [key]: e.target.checked } }) }), (x) => x.click[key], (v) => { c.checked = !!v; }); return h("label", { class: "pr-chk" }, c, ` ${label}`); };
  const clickGain = bind(range(0, 200, 5, "Click volume", (v) => player.set({ click: { gain: v / 100 } })), (x) => Math.round(x.click.gain * 100));
  const sec2 = h("section", {}, h("h4", {}, "Click"),
    h("div", { class: "pr-row" }, toggle("Click", "on"), toggle("Count-in", "count_in")),
    row("Sound", sel([["click", "Click"], ["metronome", "Metronome"], ["wood", "Wood block"]], "Click sound", "sound"),
      sel([["both", "Both ears"], ["left", "Left"], ["right", "Right"]], "Click channels", "channels")),
    row("Volume", clickGain),
    s && !s.click.beats ? h("div", { class: "pr-note" }, "This track has no beat grid yet: analyse it for the click.") : h("div", { class: "pr-note" }, "The click follows the track's beat grid, bars and time signature as you set them."));

  // Stems
  let sec3;
  if (s?.stems_loaded && s.stems.length) {
    const lbl = { drums: "Drums", bass: "Bass", other: "Melody", vocals: "Vocals" };
    sec3 = h("section", {}, h("h4", {}, "Stems"), ...s.stems.map((x) => {
      const n = x.name, c = STEM_COLORS[n] || "var(--ink-2)";
      const vol = bind(range(0, 150, 1, `${lbl[n] || n} volume`, (v) => player.stem(n, { gain: v / 100 })), (st) => Math.round(st.stems.find((q) => q.name === n)?.gain * 100));
      const pan = bind(range(-100, 100, 1, `${lbl[n] || n} pan`, (v) => player.stem(n, { pan: v / 100 })), (st) => Math.round(st.stems.find((q) => q.name === n)?.pan * 100));
      pan.addEventListener("dblclick", () => player.stem(n, { pan: 0 })); vol.addEventListener("dblclick", () => player.stem(n, { gain: 1 }));
      const flag = (k, text, title) => { const b = h("button", { class: "tb small flag", "aria-label": `${title} ${lbl[n] || n}`, title, onclick: () => player.stem(n, { [k]: !player.snap.stems.find((q) => q.name === n)?.[k] }) }, text);
        PRACTICE.binds.push({ el: b, get: (st) => !!st.stems.find((q) => q.name === n)?.[k], set: (v) => b.setAttribute("aria-pressed", String(!!v)) }); return b; };
      const off = bind(h("input", { type: "number", step: 5, class: "pr-off", "aria-label": `${lbl[n] || n} offset in ms`, title: "Offset in ms: move this stem in time (for stems that start late)",
        onchange: (e) => player.stem(n, { offset_ms: +e.target.value || 0 }) }), (st) => Math.round(st.stems.find((q) => q.name === n)?.offset_ms));
      return h("div", { class: "pr-stem" }, h("i", { style: `background:${c}` }), h("b", {}, lbl[n] || n), flag("mute", "M", "Mute"), flag("solo", "S", "Solo"),
        h("span", { class: "pr-sl" }, vol), h("span", { class: "pr-sl pan" }, pan), off);
    }),
      row("Pitch", (() => { const dr = bind(h("input", { type: "checkbox", onchange: (e) => player.stem("drums", { pitch_lock: e.target.checked }) }), (st) => st.stems.find((q) => q.name === "drums")?.pitch_lock, (v) => { dr.checked = !!v; });
        return h("label", { class: "pr-chk" }, dr, " Drums keep their pitch"); })(),
        (() => { const f = bind(h("input", { type: "checkbox", onchange: (e) => player.set({ vocal: { formants: e.target.checked, tonality_limit: player.snap.vocal_tonality_limit } }) }), (st) => st.vocal_formants, (v) => { f.checked = !!v; });
          return h("label", { class: "pr-chk" }, f, " Vocal keeps its formants"); })()));
  } else if (t?.stem_files?.length) {
    sec3 = h("section", {}, h("h4", {}, "Stems"), h("button", { class: "ghost small", onclick: () => { player.open(t.id, { stems: true, position: player.currentTime }); }, title: "Load the stems into the player: mute, solo and mix them" }, "Load the stems to mix them"));
  } else {
    sec3 = h("section", {}, h("h4", {}, "Stems"), h("div", { class: "pr-note" }, "Make stems for this track to mix them here (Make stems in the track's menu)."));
  }

  // Output
  const dev = h("select", { "aria-label": "Output device", onchange: (e) => { player.chooseDevice(e.target.value || null).catch(fail); } }, h("option", { value: "" }, "System default"));
  player.devices().then((d) => { for (const x of d.devices) dev.append(h("option", { value: x.id, selected: d.current?.id === x.id ? true : null }, x.name)); }).catch(() => {});
  const sec4 = h("section", {}, h("h4", {}, "Output"), h("div", { class: "pr-row" }, dev),
    (() => { const xr = h("output", { class: "pr-note" }); PRACTICE.binds.push({ el: xr, get: (x) => (x.xruns ? `${x.xruns} audio dropouts so far: close other audio apps or raise the buffer` : ""), set: (v) => { xr.textContent = v; } }); return xr; })());

  return h("div", { id: "practice", class: "practice", role: "dialog", "aria-label": "Practice" },
    h("div", { class: "pr-head" }, h("b", {}, "Practice"), h("span", { class: "spacer" }), h("button", { class: "tb small", "aria-label": "Close", onclick: () => togglePractice(false) }, icon("x", "ic sm"))),
    sec1, sec2, sec3, sec4);
}

function syncPractice() {
  const s = player.snap;
  if (!PRACTICE.el || !s) return;
  // New stems or none: rebuild the panel (its rows depend on them).
  const key = `${s.track_id}:${s.stems_loaded}:${s.stems.length}`;
  if (PRACTICE.key !== key) { PRACTICE.key = key; PRACTICE.el.remove(); PRACTICE.el = buildPractice(); $("#detail").append(PRACTICE.el); }
  for (const b of PRACTICE.binds) {
    if (b.el === document.activeElement) continue;
    const v = b.get(s);
    if (v == null) continue;
    if (b.el.type === "checkbox") b.el.checked = !!v; else b.set(v);
  }
}

watchPractice();
