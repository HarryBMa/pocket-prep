// Pocket Prep UI. Plain JavaScript, no build step: edit and reload.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(k));
  return el;
};

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body instanceof FormData) init.body = opts.body;
  else if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

const fmtTime = (s) => { if (s == null) return ""; s = Math.max(0, s); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const fmtBytes = (b) => b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`;
const COLORS = ["", "#e85d9c", "#e5484d", "#f08c3c", "#e6c84a", "#46a758", "#3fb8c4", "#3e7bd6", "#8e4ec6"];
const COLOR_NAMES = ["None", "Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"];
const PAD = "ABCDEFGH";

const state = { tracks: [], selected: new Set(), anchor: null, playlist: 0, playlists: [], open: new Set(),
                current: null, wave: null, jobs: [], lastUndo: null, settings: {}, system: {} };

// ── Toasts and undo ──────────────────────────────────────────────────────────
function toast(text, undoId = null, alert = false) {
  const t = h("div", { class: "toast" + (alert ? " alert" : "") }, text);
  if (undoId) {
    state.lastUndo = undoId;
    t.append(h("button", { onclick: async () => { await doUndo(undoId); t.remove(); } }, "Undo"));
  }
  $("#toasts").append(t);
  setTimeout(() => t.remove(), undoId ? 9000 : 4500);
}
async function doUndo(id) {
  try { await api(`/api/undo/${id}`, { body: {} }); toast("Undone"); state.lastUndo = null; await refreshAll(); }
  catch (e) { toast(e.message, null, true); }
}
const fail = (e) => toast(e.message || String(e), null, true);

// ── Modal (simple prompt) ────────────────────────────────────────────────────
function ask(title, value = "", okLabel = "OK") {
  return new Promise((resolve) => {
    const box = $("#modal .modal-box");
    const input = h("input", { value });
    const close = (v) => { $("#modal").hidden = true; resolve(v); };
    box.replaceChildren(h("h3", {}, title), input,
      h("div", { class: "buttons" }, h("button", { class: "ghost", onclick: () => close(null) }, "Cancel"),
        h("button", { class: "primary", onclick: () => close(input.value.trim()) }, okLabel)));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") close(input.value.trim()); if (e.key === "Escape") close(null); });
    $("#modal").hidden = false;
    input.focus(); input.select();
  });
}

// ── Tabs and theme ───────────────────────────────────────────────────────────
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
function showView(v) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  if (v === "export") loadExport();
  if (v === "deck") loadDeck();
  if (v === "settings") loadSettings();
}
let cssCache = {};   // colours read from CSS variables; reading them every animation frame is slow
function applyTheme(t) { cssCache = {}; if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { cssCache = {}; if (state.wave) buildWaves(); });
try { applyTheme(localStorage.getItem("theme")); } catch {}
$("#themeBtn").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch {}
  if (state.wave) buildWaves();
});

// ── Library: playlists sidebar ───────────────────────────────────────────────
async function loadPlaylists() {
  state.playlists = await api("/api/playlists");
  const kids = (pid) => state.playlists.filter((p) => p.parent_id === pid);
  const build = (pid) => kids(pid).map((p) => {
    const btn = h("button", { class: "nav-item" + (state.playlist === p.id ? " active" : ""), "data-playlist": p.id },
      p.name, p.is_folder ? null : h("span", { class: "count" }, String(p.count)));
    btn.addEventListener("click", () => {
      if (p.is_folder) { if (state.open.has(p.id)) { state.open.delete(p.id); state.openFolderTarget = null; } else { state.open.add(p.id); state.openFolderTarget = p.id; } loadPlaylists(); }
      else selectPlaylist(p.id);
    });
    btn.addEventListener("contextmenu", (e) => { e.preventDefault(); playlistMenu(p); });
    if (!p.is_folder) dropTarget(btn, (ids) => addToPlaylist(p.id, ids));
    if (!p.is_folder) return btn;
    return h("div", { class: "folder" + (state.open.has(p.id) ? " open" : "") }, btn, h("div", { class: "children" }, build(p.id)));
  });
  $("#tree").replaceChildren(...build(0));
  $$(".side > .nav-item").forEach((b) => b.classList.toggle("active", state.playlist === 0));
}
async function playlistMenu(p) {
  const name = await ask(`Rename "${p.name}" (leave empty to delete)`, p.name, "Save");
  if (name === null) return;
  if (name === "") {
    await api(`/api/playlists/${p.id}`, { method: "DELETE" });
    toast(`Deleted ${p.is_folder ? "folder" : "playlist"} "${p.name}". Tracks stay in the library.`);
    if (state.playlist === p.id) selectPlaylist(0); else loadPlaylists();
  } else { await api(`/api/playlists/${p.id}`, { method: "PATCH", body: { name } }); loadPlaylists(); }
}
async function newPlaylist(isFolder) {
  const inFolder = state.playlists.find((p) => p.id === state.openFolderTarget);
  const name = await ask((isFolder ? "New folder" : "New playlist") + (inFolder ? ` in "${inFolder.name}"` : ""), "", "Create");
  if (!name) return;
  const parent = state.playlists.find((p) => p.id === state.openFolderTarget) ? state.openFolderTarget : 0;
  const p = await api("/api/playlists", { body: { name, parent_id: parent, is_folder: isFolder } });
  if (!isFolder && state.selected.size) await addToPlaylist(p.id, [...state.selected]);
  loadPlaylists();
}
$("#newPlaylist").addEventListener("click", () => newPlaylist(false));
$("#newFolder").addEventListener("click", () => newPlaylist(true));
$(".side > .nav-item").addEventListener("click", () => selectPlaylist(0));

function selectPlaylist(id) { state.playlist = id; state.selected.clear(); loadPlaylists(); loadTracks(); }
async function addToPlaylist(pid, ids) {
  const r = await api(`/api/playlists/${pid}/add`, { body: { track_ids: ids } });
  const p = state.playlists.find((x) => x.id === pid);
  toast(`Added ${ids.length} to "${p ? p.name : "playlist"}"`, r.undo);
  loadPlaylists();
}

// ── Library: track table ─────────────────────────────────────────────────────
async function loadTracks() {
  const q = encodeURIComponent($("#search").value.trim());
  state.tracks = await api(`/api/tracks?q=${q}&playlist=${state.playlist}`);
  renderTracks();
  if (!state.playlist && !q) $("#allCount").textContent = state.tracks.length || "";
}
function stemPill(t) {
  const label = { none: "–", queued: "Queued", running: "Making", done: "Ready", failed: "Failed" }[t.stems] || t.stems;
  return t.stems === "none" ? h("span", { class: "muted" }, "–") : h("span", { class: `pill ${t.stems}` }, label);
}
function renderTracks() {
  const body = $("#tracks tbody");
  const rows = state.tracks.map((t, i) => {
    const tr = h("tr", { "data-id": t.id, draggable: "true", class: (state.selected.has(t.id) ? "sel" : "") + (state.current?.id === t.id ? " playing" : "") },
      h("td", { class: "c-col" }, t.color ? h("span", { class: "swatch", style: `background:${COLORS[t.color]}` }) : null),
      h("td", { title: t.path }, t.title || ""),
      h("td", {}, t.artist || h("span", { class: "muted" }, "Unknown")),
      h("td", { class: "num" }, t.bpm ? t.bpm.toFixed(t.bpm % 1 ? 2 : 0) : h("span", { class: "muted" }, t.analysed_at ? "?" : "…")),
      h("td", {}, t.camelot ? `${t.camelot} · ${t.musical_key}` : ""),
      h("td", { class: "num" }, t.energy ?? ""),
      h("td", { class: "num" }, fmtTime(t.duration)),
      h("td", {}, stemPill(t)));
    tr.addEventListener("click", (e) => clickRow(e, t.id, i));
    tr.addEventListener("dblclick", () => openTrack(t.id, true));
    tr.addEventListener("dragstart", (e) => {
      if (!state.selected.has(t.id)) { state.selected = new Set([t.id]); renderTracks(); }
      e.dataTransfer.setData("application/x-tracks", JSON.stringify([...state.selected]));
      e.dataTransfer.effectAllowed = "copyMove";
    });
    if (state.playlist) {   // reorder inside a playlist by dragging
      tr.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("application/x-tracks")) e.preventDefault(); });
      tr.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); reorder(JSON.parse(e.dataTransfer.getData("application/x-tracks")), t.id); });
    }
    return tr;
  });
  body.replaceChildren(...rows);
  $("#empty").hidden = state.tracks.length > 0 || !!$("#search").value || state.playlist !== 0;
  renderContext();
}
function clickRow(e, id, i) {
  if (e.shiftKey && state.anchor != null) {
    const a = state.tracks.findIndex((t) => t.id === state.anchor);
    const [lo, hi] = [Math.min(a, i), Math.max(a, i)];
    state.selected = new Set(state.tracks.slice(lo, hi + 1).map((t) => t.id));
  } else if (e.ctrlKey || e.metaKey) {
    state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
    state.anchor = id;
  } else {
    state.selected = new Set([id]); state.anchor = id;
    openTrack(id, false);
  }
  renderTracks();
}
async function reorder(ids, beforeId) {
  const order = state.tracks.map((t) => t.id).filter((x) => !ids.includes(x));
  const at = order.indexOf(beforeId);
  order.splice(at < 0 ? order.length : at, 0, ...ids);
  const r = await api(`/api/playlists/${state.playlist}/tracks`, { method: "PUT", body: { track_ids: order } });
  toast("Order changed", r.undo);
  loadTracks();
}

// Actions shown only when tracks are selected (context-aware toolbar)
function renderContext() {
  const box = $("#contextActions");
  const ids = [...state.selected];
  if (!ids.length) { box.replaceChildren(); return; }
  const sel = state.tracks.filter((t) => state.selected.has(t.id));
  const parts = [h("span", { class: "sel-count" }, `${ids.length} selected`)];
  parts.push(h("button", { class: "ghost small", onclick: () => api("/api/analyse", { body: { ids } }).then(() => { toast(`Analysing ${ids.length}`); pollJobs(); }).catch(fail) },
    sel.every((t) => t.analysed_at) ? "Re-analyse" : "Analyse"));
  const noStems = sel.filter((t) => t.stems !== "done" && t.stems !== "running" && t.stems !== "queued").map((t) => t.id);
  if (noStems.length) parts.push(h("button", { class: "ghost small", onclick: () => api("/api/stems", { body: { ids: noStems } }).then(() => { toast(`Making stems for ${noStems.length}`); loadTracks(); pollJobs(); }).catch(fail) }, "Make stems"));
  const lists = state.playlists.filter((p) => !p.is_folder);
  if (lists.length) {
    const s = h("select", { class: "small", "aria-label": "Add to playlist" }, h("option", { value: "" }, "Add to playlist…"), lists.map((p) => h("option", { value: p.id }, p.name)));
    s.addEventListener("change", () => { if (s.value) addToPlaylist(+s.value, ids); s.value = ""; });
    parts.push(s);
  }
  if (state.playlist) parts.push(h("button", { class: "ghost small", onclick: async () => {
    const keep = state.tracks.map((t) => t.id).filter((x) => !state.selected.has(x));
    const r = await api(`/api/playlists/${state.playlist}/tracks`, { method: "PUT", body: { track_ids: keep } });
    toast(`Removed ${ids.length} from playlist`, r.undo); state.selected.clear(); loadTracks(); loadPlaylists();
  } }, "Remove from playlist"));
  parts.push(h("button", { class: "danger small", onclick: removeSelected }, "Remove from library"));
  box.replaceChildren(...parts);
}
async function removeSelected() {
  const ids = [...state.selected];
  if (!ids.length) return;
  const r = await api("/api/tracks/remove", { body: { ids } });
  toast(`Removed ${ids.length} from the library (files are kept)`, r.undo);
  state.selected.clear();
  if (state.current && ids.includes(state.current.id)) closeDetail();
  loadTracks();
}
$("#search").addEventListener("input", () => { clearTimeout(state.searchT); state.searchT = setTimeout(loadTracks, 150); });

// Dropping onto a playlist
function dropTarget(el, onIds) {
  el.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("application/x-tracks")) { e.preventDefault(); el.classList.add("drop"); } });
  el.addEventListener("dragleave", () => el.classList.remove("drop"));
  el.addEventListener("drop", (e) => { e.preventDefault(); el.classList.remove("drop"); onIds(JSON.parse(e.dataTransfer.getData("application/x-tracks"))); });
}

// ── Importing ────────────────────────────────────────────────────────────────
async function importKind(kind) {
  let paths = [];
  const r = await api("/api/pick", { body: { kind } });
  if (r.unavailable) {
    const p = await ask("Folder or file path to add", "", "Add");
    if (!p) return;
    paths = [p];
  } else paths = r.paths;
  if (!paths.length) return;
  const res = await api("/api/import", { body: { paths } });
  toast(res.imported ? `Added ${res.imported} tracks. Analysing…` : "Nothing new to add");
  loadTracks(); pollJobs();
}
$("#importFolder").addEventListener("click", () => importKind("folder").catch(fail));
$("#emptyImport").addEventListener("click", () => importKind("folder").catch(fail));
$("#importFiles").addEventListener("click", () => importKind("files").catch(fail));
// Files dragged in from Explorer/Finder: web views don't reveal their paths, so point to the buttons.
// (The window keeps HTML drag and drop on, which dragging tracks onto playlists needs.)
const dz = $("#dropZone");
dz.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); dz.classList.add("drop"); } });
dz.addEventListener("dragleave", () => dz.classList.remove("drop"));
dz.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault(); dz.classList.remove("drop");
  toast("Use Add files or Add folder to bring in music", null, true);
});

// ── Track detail: player, waveform, cues, tags ───────────────────────────────
const player = $("#player");
async function openTrack(id, play) {
  const t = await api(`/api/tracks/${id}`);
  const same = state.current?.id === id;
  state.current = t;
  if (!same) {
    player.src = `/api/tracks/${id}/audio`;
    state.wave = null;
    buildWaves();
    api(`/api/tracks/${id}/waveform`).then((w) => { if (state.current?.id === id) { state.wave = decodeWave(w); buildWaves(); } }).catch(() => {});
  }
  renderDetail();
  if (play) player.play().catch(() => {});
  renderTracks();
}
function closeDetail() { player.pause(); state.ws?.destroy(); state.ws = null; state.current = null; state.wave = null; $("#detail").hidden = true; }

function decodeWave(w) {
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  return { detail: b64(w.detail), duration: w.duration_ms / 1000 };
}

function renderDetail() {
  const t = state.current;
  const d = $("#detail");
  if (!t) { d.hidden = true; return; }
  d.hidden = false;
  const field = (label, key, type = "text") => {
    const inp = h("input", { type, value: t[key] ?? "" });
    inp.addEventListener("change", () => saveField(key, type === "number" ? (inp.value === "" ? null : +inp.value) : inp.value));
    return h("label", {}, label, inp);
  };
  const color = h("select", {}, COLOR_NAMES.map((n, i) => h("option", { value: i, selected: (t.color || 0) === i }, n)));
  color.addEventListener("change", () => saveField("color", +color.value));
  const rating = h("select", {}, [0, 1, 2, 3, 4, 5].map((r) => h("option", { value: r, selected: (t.rating || 0) === r }, r ? "★".repeat(r) : "No rating")));
  rating.addEventListener("change", () => saveField("rating", +rating.value));

  const pads = h("div", { class: "pads" }, [...PAD].map((letter, i) => {
    const cue = t.cues.find((c) => c.slot === i + 1);
    const pad = h("button", { class: "pad" + (cue ? " set" : ""), title: cue ? "Jump (right-click to clear)" : "Set at the playhead" },
      letter, h("small", {}, cue ? (cue.label || fmtTime(cue.start_ms / 1000)) : "empty"), cue ? h("span", { class: "x" }, "×") : null);
    pad.addEventListener("click", (e) => {
      if (cue && !e.target.classList.contains("x")) { player.currentTime = cue.start_ms / 1000; return; }
      setCue(i + 1, cue && e.target.classList.contains("x") ? null : player.currentTime * 1000);
    });
    pad.addEventListener("contextmenu", (e) => { e.preventDefault(); if (cue) setCue(i + 1, null); });
    return pad;
  }));

  const stemRow = h("div", { class: "stems-row" }, h("b", {}, "Stems"));
  if (t.stem_files.length) {
    t.stem_files.forEach((s) => {
      const b = h("button", { class: "ghost small" + (state.stemSolo === s ? " on" : "") }, s === "other" ? "melody" : s);
      b.addEventListener("click", () => soloStem(s));
      stemRow.append(b);
    });
    if (state.stemSolo) stemRow.append(h("button", { class: "ghost small", onclick: () => soloStem(null) }, "Full mix"));
  } else {
    stemRow.append(h("span", { class: "muted" }, { none: "Not made", queued: "Queued", running: "Making…", failed: "Failed" }[t.stems] || ""),
      t.stems === "none" || t.stems === "failed" ? h("button", { class: "ghost small", onclick: () => api("/api/stems", { body: { ids: [t.id] } }).then(() => { pollJobs(); loadTracks(); openTrack(t.id); }) }, "Make stems") : null);
  }

  const ph = t.phrases;
  d.replaceChildren(
    h("div", { class: "d-head" },
      t.has_artwork ? h("img", { class: "d-art", src: `/api/tracks/${t.id}/artwork`, alt: "" }) : h("div", { class: "d-art" }),
      h("div", {}, h("div", { class: "d-title" }, t.title || ""),
        h("div", { class: "d-meta" }, [t.artist, t.bpm ? `${t.bpm} BPM` : null, t.camelot ? `${t.camelot} (${t.musical_key})` : null, t.energy ? `energy ${t.energy}/10` : null].filter(Boolean).join(" · "))),
      h("div", { class: "spacer" }),
      h("div", { class: "transport" },
        h("button", { class: "ghost", id: "playBtn", onclick: togglePlay }, player.paused ? "Play" : "Pause"),
        h("span", { class: "time", id: "timeLabel" }, ""),
        h("button", { class: "ghost small", onclick: closeDetail }, "Close"))),
    waveBox,
    h("div", { class: "d-row" },
      h("div", {}, pads,
        h("div", { class: "muted", style: "font-size:13px;margin-top:6px" },
          ph ? `Intro ends ${ph.intro_end_ms ? fmtTime(ph.intro_end_ms / 1000) : "–"} · outro ${ph.outro_start_ms ? fmtTime(ph.outro_start_ms / 1000) : "–"} · ${ph.drops.length} drop${ph.drops.length === 1 ? "" : "s"}` : "Not analysed yet"),
        h("div", { style: "margin-top:8px" }, stemRow)),
      h("div", { class: "fields" },
        field("Title", "title"), field("Artist", "artist"), field("Album", "album"),
        field("Genre", "genre"), field("BPM", "bpm", "number"), field("Comment", "comment"),
        h("label", {}, "Colour", color), h("label", {}, "Rating", rating), field("Year", "year", "number"))));
  syncMarkers();
}
async function saveField(key, value) {
  try {
    const r = await api(`/api/tracks/${state.current.id}`, { method: "PATCH", body: { [key]: value } });
    Object.assign(state.current, r.track);
    toast("Saved", r.undo);
    loadTracks();
  } catch (e) { fail(e); }
}
async function setCue(slot, ms) {
  const cues = state.current.cues.filter((c) => c.slot !== slot).map((c) => ({ ...c }));
  if (ms != null) cues.push({ slot, start_ms: Math.round(snapToBeat(ms)), label: null, auto: 0 });
  // A user edit makes all remaining cues "user" cues, so re-analysis keeps them.
  cues.forEach((c) => (c.auto = 0));
  const r = await api(`/api/tracks/${state.current.id}/cues`, { method: "PUT", body: cues });
  state.current.cues = r.cues;
  toast(ms == null ? `Cleared hot cue ${PAD[slot - 1]}` : `Hot cue ${PAD[slot - 1]} set`, r.undo);
  renderDetail();
}
function snapToBeat(ms) {
  const g = state.current?.grid;
  if (!g) return ms;
  const n = Math.round((ms - g.first_beat_ms) / g.period_ms);
  const snapped = g.first_beat_ms + n * g.period_ms;
  return Math.abs(snapped - ms) < g.period_ms / 4 ? snapped : ms;
}
function togglePlay() { player.paused ? player.play().catch(() => {}) : player.pause(); }
function soloStem(name) {
  const t = state.current;
  const pos = player.currentTime, playing = !player.paused;
  state.stemSolo = name;
  player.src = name ? `/api/tracks/${t.id}/stem/${name}` : `/api/tracks/${t.id}/audio`;
  player.addEventListener("loadedmetadata", () => { player.currentTime = pos; if (playing) player.play(); }, { once: true });
  renderDetail();
}
player.addEventListener("play", () => { const b = $("#playBtn"); if (b) b.textContent = "Pause"; });
player.addEventListener("pause", () => { const b = $("#playBtn"); if (b) b.textContent = "Play"; });

// Waveforms: wavesurfer.js (ui/vendor/wavesurfer, BSD-3-Clause) draws the scrolling detail
// view and the whole-track overview (Minimap plugin) from the analysis peaks, and follows
// the shared <audio> player. Beat grid and cues are Regions-plugin markers.
function cssVar(n) { return cssCache[n] ??= getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const waveBox = h("div", { class: "waves", id: "waves" });

// PWV5 (150 columns per second: height plus a 3-bit bass/mid/high mix) as three layers
// drawn over each other, rekordbox style: bass colour at full height, mids, then highs.
function wavePeaks(w) {
  const n = w.detail.length / 2, lo = new Float32Array(n), md = new Float32Array(n), hi = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = (w.detail[i * 2] << 8) | w.detail[i * 2 + 1];
    const g = (v >> 10) & 7, b = (v >> 7) & 7, ht = ((v >> 2) & 31) / 31;
    lo[i] = ht; md[i] = (ht * Math.max(g, b)) / 7; hi[i] = (ht * b) / 7;
  }
  return [lo, md, hi];
}

function buildWaves() {
  state.ws?.destroy(); state.ws = null; state.regions = null;
  const t = state.current, w = state.wave;
  if (!w) { waveBox.replaceChildren(h("div", { class: "wave-empty muted" }, t?.analysed_at ? "Loading waveform…" : "Waveform appears after analysis")); return; }
  waveBox.replaceChildren();
  const layers = [cssVar("--wave-low"), cssVar("--wave-mid"), cssVar("--wave-high")].map((c) => ({ waveColor: c, progressColor: c, overlay: true }));
  state.regions = WaveSurfer.Regions.create();
  state.ws = WaveSurfer.create({
    container: waveBox, media: player, peaks: wavePeaks(w), duration: w.duration,
    height: 110, minPxPerSec: 100, autoScroll: true, autoCenter: true, hideScrollbar: true,
    dragToSeek: true, normalize: false, splitChannels: layers, cursorColor: cssVar("--ink"), cursorWidth: 2,
    plugins: [state.regions, WaveSurfer.Minimap.create({ height: 46, splitChannels: layers, dragToSeek: true,
      cursorColor: cssVar("--ink"), overlayColor: "rgba(128,128,128,.22)" })],
  });
  state.ws.on("timeupdate", (time) => { const l = $("#timeLabel"); if (l) l.textContent = `${fmtTime(time)} / ${fmtTime(t?.duration)}`; });
  state.ws.on("ready", syncMarkers);
}

// Downbeats of the beat grid, then hot cues (A-H) and memory cues (M) on top.
function syncMarkers() {
  const t = state.current, g = t?.grid, reg = state.regions;
  if (!reg || !state.wave) return;
  reg.clearRegions();
  if (g) {
    for (let ms = g.first_beat_ms + g.downbeat_index * g.period_ms; ms < g.duration_ms; ms += 4 * g.period_ms) {
      // Background colour, so bar lines cut through the waveform in light and dark themes.
      reg.addRegion({ start: ms / 1000, color: cssVar("--bg"), drag: false, resize: false });
    }
  }
  for (const c of t.cues) {
    reg.addRegion({ start: c.start_ms / 1000, color: c.slot ? cssVar("--accent") : cssVar("--ink"),
                    content: c.slot ? PAD[c.slot - 1] : "M", drag: false, resize: false });
  }
}

// ── Jobs ─────────────────────────────────────────────────────────────────────
let pollTimer = null, lastActive = 0;
async function pollJobs() {
  clearTimeout(pollTimer);
  try { state.jobs = await api("/api/jobs"); } catch { state.jobs = []; }
  const active = state.jobs.filter((j) => j.status === "running" || j.status === "queued");
  $("#jobsDot").classList.toggle("busy", active.length > 0);
  $("#jobsLabel").textContent = active.length ? `${active.length} job${active.length > 1 ? "s" : ""} running` : "No jobs";
  if (!$("#jobsPanel").hidden) renderJobs();
  renderInlineJobs();
  if (lastActive && active.length < lastActive) {     // something finished
    loadTracks();
    if (state.current) openTrack(state.current.id);
  }
  lastActive = active.length;
  pollTimer = setTimeout(pollJobs, active.length ? 1000 : 4000);
}
function jobEl(j) {
  const label = { queued: "Queued", running: `${Math.round(j.progress * 100)}%`, done: "Done", failed: "Failed", cancelled: "Cancelled" }[j.status];
  return h("div", { class: `job ${j.status}` },
    h("div", { class: "row" }, h("b", {}, j.title), h("span", { class: `pill ${j.status}` }, label),
      j.status === "running" || j.status === "queued" ? h("button", { class: "ghost small", onclick: () => api(`/api/jobs/${j.id}/cancel`, { body: {} }).then(pollJobs) }, "Cancel") : null),
    j.status === "running" ? h("div", { class: "progress" }, h("i", { style: `width:${j.progress * 100}%` })) : null,
    j.message ? h("div", { class: "msg" }, j.message) : null);
}
function renderJobs() {
  $("#jobsList").replaceChildren(...(state.jobs.length ? state.jobs.map(jobEl) : [h("p", { class: "muted", style: "padding:12px" }, "Nothing running.")]));
}
function renderInlineJobs() {
  for (const [slot, kind] of [["#exportJob", "export"], ["#otaJob", "ota"], ["#flashJob", "flash"]]) {
    const j = state.jobs.find((x) => x.kind === kind);
    const el = $(slot);
    if (!el) continue;
    if (!j) { el.replaceChildren(); continue; }
    el.replaceChildren(jobEl(j));
    if (kind === "export" && j.status === "done" && j.result) {
      el.append(h("div", { class: "plan" }, `Exported ${j.result.tracks} tracks (${j.result.copied} files copied), ${j.result.playlists} playlists, ${j.result.stems} with stems.` +
        (j.result.missing_files ? ` ${j.result.missing_files} files were missing and skipped.` : "") + " Eject the drive before unplugging it.",
        ...deckWarnings(j.result.warnings)));
    }
  }
}
$("#jobsBtn").addEventListener("click", () => { const p = $("#jobsPanel"); p.hidden = !p.hidden; if (!p.hidden) renderJobs(); });
$("#closeJobs").addEventListener("click", () => ($("#jobsPanel").hidden = true));
$("#clearJobs").addEventListener("click", () => api("/api/jobs/clear", { body: {} }).then(pollJobs));

// ── Export view ──────────────────────────────────────────────────────────────
const exp = { root: null };
// Files are copied as they are; these are the ones the deck can't decode.
function deckWarnings(list) {
  if (!list || !list.length) return [];
  return [h("div", {}, h("b", {}, `${list.length} ${list.length === 1 ? "track" : "tracks"} won't play on the deck`), " (they still work on CDJs):"),
    ...list.map((w) => h("div", { class: "muted" }, `${w.title}: ${w.problem}`))];
}
async function loadExport() {
  const drives = await api("/api/drives").catch(() => []);
  $("#drives").replaceChildren(...(drives.length ? drives.map((d) => {
    const c = h("button", { class: "card" + (exp.root === d.root ? " chosen" : "") },
      h("b", {}, `${d.label || "Drive"} (${d.root})`),
      h("div", { class: "muted" }, `${d.removable ? "Removable" : "Fixed disk"} · ${d.filesystem} · ${fmtBytes(d.free)} free of ${fmtBytes(d.total)}`),
      d.has_export ? h("div", { class: "muted" }, "Has a library already: it will be updated") : null);
    c.addEventListener("click", () => { exp.root = d.root; loadExport(); });
    return c;
  }) : [h("p", { class: "muted" }, "No drives found. Insert an SD card or USB stick.")]));
  const lists = state.playlists.filter((p) => !p.is_folder);
  $("#exportPlaylists").replaceChildren(...lists.map((p) => h("label", { class: "check" }, h("input", { type: "checkbox", value: p.id }), `${p.name} (${p.count})`)));
  updatePlan();
}
$$("input[name=scope]").forEach((r) => r.addEventListener("change", () => { $("#exportPlaylists").hidden = $("input[name=scope]:checked").value !== "playlists"; updatePlan(); }));
$("#exportPlaylists").addEventListener("change", updatePlan);
$("#includeStems").addEventListener("change", updatePlan);
$("#refreshDrives").addEventListener("click", loadExport);
function exportBody() {
  const scope = $("input[name=scope]:checked").value;
  const ids = scope === "playlists" ? $$("#exportPlaylists input:checked").map((i) => +i.value) : null;
  return { root: exp.root, playlist_ids: ids, include_stems: $("#includeStems").checked };
}
async function updatePlan() {
  const plan = $("#plan"), btn = $("#doExport");
  btn.disabled = true;
  if (!exp.root) { plan.textContent = "Choose a drive to see what will be copied."; return; }
  const body = exportBody();
  if (body.playlist_ids && !body.playlist_ids.length) { plan.textContent = "Tick at least one playlist."; return; }
  try {
    const p = await api("/api/export/plan", { body });
    plan.classList.toggle("bad", !p.fits);
    plan.replaceChildren(
      h("div", {}, h("b", {}, `${p.tracks} tracks`), `, ${p.playlists} playlists, ${p.with_stems} with stems · up to ${fmtBytes(p.bytes)} (${fmtBytes(p.free)} free)`),
      p.unanalysed ? h("div", { class: "muted" }, `${p.unanalysed} tracks aren't analysed yet; they'll be analysed during the export (slower).`) : null,
      ...deckWarnings(p.warnings),
      p.fits ? null : h("div", {}, h("b", {}, "Doesn't fit. "), "Choose fewer playlists or turn off stems."));
    btn.disabled = !p.fits || !p.tracks;
  } catch (e) { plan.textContent = e.message; }
}
$("#doExport").addEventListener("click", async () => {
  await api("/api/export", { body: exportBody() }).catch(fail);
  pollJobs();
});

// ── Deck view ────────────────────────────────────────────────────────────────
async function loadDeck() {
  const box = $("#deckStatus");
  box.textContent = "Checking…";
  const s = await api("/api/deck/status").catch((e) => ({ reachable: false, error: e.message }));
  if (!s.reachable) {
    box.replaceChildren(h("b", {}, "Deck not reachable"), h("div", { class: "muted" }, `Looked at ${s.address || "the deck"}. Turn on Wi-Fi Remote on the deck and join its network to update or read its status.`));
  } else {
    const f = s.firmware || {};
    box.replaceChildren(h("b", {}, "Deck connected"),
      h("div", {}, `Firmware ${f.version || f.app_version || "?"} · project ${f.project || "?"} · slot ${f.running_partition || f.slot || "?"} · image ${f.running_image_state || "?"}`),
      f.last_error ? h("div", { class: "muted" }, `Last update error: ${f.last_error}`) : null);
  }
  const ports = await api("/api/deck/ports").catch(() => []);
  $("#ports").replaceChildren(...(ports.length ? ports.map((p) => h("option", { value: p.port }, `${p.port} – ${p.description}${p.espressif ? " (ESP32)" : ""}`)) : [h("option", { value: "" }, "No USB serial ports")]));
}
$("#deckRefresh").addEventListener("click", loadDeck);
$("#otaPick").addEventListener("click", () => $("#otaFile").click());
$("#otaFile").addEventListener("change", async () => {
  const f = $("#otaFile").files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  await api("/api/deck/ota", { body: fd }).catch(fail);
  $("#otaFile").value = "";
  pollJobs();
});
$("#flashBtn").addEventListener("click", async () => {
  const port = $("#ports").value, dir = $("#buildDir").value.trim();
  if (!port || !dir) return toast("Choose a port and a build folder", null, true);
  const typed = await ask(`This overwrites the firmware on the device at ${port}. Type FLASH to continue.`, "", "Flash");
  if (typed !== "FLASH") return;
  await api("/api/deck/flash", { body: { port, build_dir: dir } }).catch(fail);
  pollJobs();
});

// ── Settings view ────────────────────────────────────────────────────────────
async function loadSettings() {
  state.settings = await api("/api/settings");
  const f = $("#settingsForm");
  for (const [k, v] of Object.entries(state.settings)) {
    const el = f.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
  }
  state.system = await api("/api/system");
  $("#system").replaceChildren(h("div", {}, `Pocket Prep ${state.system.version}`),
    h("div", { class: "muted" }, `Library and stems are stored in ${state.system.data_dir}`));
  const hist = await api("/api/history");
  $("#history").replaceChildren(...hist.slice(0, 60).map((r) => h("div", { class: "h" },
    h("span", { class: "when" }, new Date(r.at * 1000).toLocaleString()),
    h("span", { style: "flex:1" }, r.action.replaceAll("_", " ")),
    r.can_undo ? h("button", { class: "ghost small", onclick: () => doUndo(r.id).then(loadSettings) }, "Undo") : null)));
}
$("#settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target, v = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    v[el.name] = el.type === "checkbox" ? el.checked : el.type === "number" ? +el.value : el.value;
  }
  await api("/api/settings", { method: "PUT", body: v }).catch(fail);
  toast("Settings saved");
});

// ── Keyboard ─────────────────────────────────────────────────────────────────
addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key === "z" && !typing && state.lastUndo) { e.preventDefault(); doUndo(state.lastUndo); return; }
  if (typing || !$("#view-library").classList.contains("active")) return;
  if (e.key === " " && state.current) { e.preventDefault(); togglePlay(); }
  if (e.key === "Delete" && state.selected.size) removeSelected();
  if ((e.ctrlKey || e.metaKey) && e.key === "a") { e.preventDefault(); state.selected = new Set(state.tracks.map((t) => t.id)); renderTracks(); }
  if (state.current && e.key >= "1" && e.key <= "8") {
    const slot = +e.key, cue = state.current.cues.find((c) => c.slot === slot);
    cue ? (player.currentTime = cue.start_ms / 1000) : setCue(slot, player.currentTime * 1000);
  }
});

async function refreshAll() { await loadPlaylists(); await loadTracks(); if (state.current) openTrack(state.current.id); }
refreshAll().then(pollJobs).catch(fail);
