// DKPRP UI (preparation for the DECK001). Plain JavaScript, no build step: edit and reload.
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
// Like el.replaceChildren(), but skips null/false (the DOM call would print them as text).
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
// An icon from the <symbol> sprite in index.html.
const icon = (name, cls = "ic") => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", cls); s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", `#i-${name}`); s.append(u);
  return s;
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
const fmtMs = (ms) => (ms == null ? "–" : fmtTime(ms / 1000));
const fmtBytes = (b) => b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`;
const fmtDur = (s) => { const m = Math.round(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`; };
const fmtBpm = (b) => (b ? b.toFixed(b % 1 ? 2 : 0) : "");
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
const COLORS = ["", "#e85d9c", "#e5484d", "#f08c3c", "#e6c84a", "#46a758", "#3fb8c4", "#3e7bd6", "#8e4ec6"];
const COLOR_NAMES = ["None", "Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"];
const PAD = "ABCDEFGH";
// Hot cue colours A-H, the DECK001 firmware's slot colours (cues_storage.h). The card itself stores none.
const CUE_COLORS = ["#ff4545", "#ff9f1c", "#ffdf40", "#20d879", "#00c8e8", "#318cff", "#b79bff", "#ff66af"];
// Moods (src/mood.rs), in the server's order, which is also the group order. Colours are the
// deck's slot colours, calm to hard: blue, purple, cyan, orange, red; then green, pink, yellow.
let MOODS = [], MOOD_FEATURES = [];
api("/api/moods").then((r) => { MOODS = r.moods; MOOD_FEATURES = r.features; if (state.loaded) renderTracks(); }).catch(() => {});
const MOOD_COLOR = { deep: CUE_COLORS[5], melodic: CUE_COLORS[6], tech_house: CUE_COLORS[4], peaktime: CUE_COLORS[1],
                     hard_techno: CUE_COLORS[0], dnb: CUE_COLORS[3], pop: CUE_COLORS[7], downtempo: CUE_COLORS[2] };
const moodOf = (id) => MOODS.find((m) => m.id === id);
const moodRank = (id) => { const i = MOODS.findIndex((m) => m.id === id); return i < 0 ? null : i; };
const PLAYHEAD = "#e9b954"; // DECK001 playhead (settings.h kPlayhead)
// How the DJ will use a track (tracks.intent): decides what gets prepared and what "ready" means.
const INTENTS = [["mix", "Mix", "Beat grid, key, cues. No stems."], ["vocal", "Vocal layer", "Stems: the vocal to lay over another track."],
                 ["remix", "Remix", "All four stems for live remixing."], ["section", "Section", "A chosen passage; stems for it first in the queue."],
                 ["backup", "Backup", "Analysed and on the card; nothing more is made."]];
const intentOf = (t) => INTENTS.find((i) => i[0] === (t.intent || "mix")) || INTENTS[0];
const wantsStems = (t) => ["vocal", "remix", "section"].includes(t.intent);

const state = { tracks: [], view: [], selected: new Set(), anchor: null, playlist: 0, playlists: [], open: new Set(),
                current: null, wave: null, jobs: [], lastUndo: null, settings: {}, system: {},
                filter: "all", sort: null, loaded: false, deck: null, zoom: 100 };

// ── Toasts and undo ──────────────────────────────────────────────────────────
function toast(text, undoId = null, alert = false) {
  const t = h("div", { class: "toast" + (alert ? " alert" : ""), role: alert ? "alert" : "status" }, text);
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
    const back = document.activeElement;
    const input = h("input", { value, "aria-label": title });
    const close = (v) => { $("#modal").hidden = true; back?.focus?.(); resolve(v); };
    box.replaceChildren(h("h3", {}, title), input,
      h("div", { class: "buttons" }, h("button", { class: "ghost", onclick: () => close(null) }, "Cancel"),
        h("button", { class: "primary", onclick: () => close(input.value.trim()) }, okLabel)));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") close(input.value.trim()); if (e.key === "Escape") { e.stopPropagation(); close(null); } });
    $("#modal").hidden = false;
    input.focus(); input.select();
  });
}

// ── Views (rail) and theme ───────────────────────────────────────────────────
const tabs = () => $$("[data-view]");
tabs().forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
$$(".tabs").forEach((t) => t.addEventListener("keydown", (e) => {   // arrow keys move between views (roving tabindex)
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
  e.preventDefault();
  const list = tabs(), i = list.indexOf(document.activeElement);
  const n = e.key === "Home" ? 0 : e.key === "End" ? list.length - 1 : (i + (e.key === "ArrowDown" ? 1 : -1) + list.length) % list.length;
  list[n].focus(); showView(list[n].dataset.view);
}));
// Playlists open in a popover under their button; a click outside or Escape closes it.
function setPlaylistsOpen(open) {
  $("#playlistsToggle").setAttribute("aria-expanded", String(open));
  $("#playlistsBox").hidden = !open;
  if (open) ($("#playlistsBox .nav-item.active") || $("#playlistsBox .nav-item"))?.focus();
}
$("#playlistsToggle").addEventListener("click", () => setPlaylistsOpen($("#playlistsBox").hidden));
// Dragging tracks over the Playlists button opens the popover, so they can be dropped on a playlist.
$("#playlistsToggle").addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("application/x-tracks")) { e.preventDefault(); if ($("#playlistsBox").hidden) setPlaylistsOpen(true); } });
$("#playlistsBox").addEventListener("dragleave", (e) => { if (!e.relatedTarget || !e.relatedTarget.closest?.(".pl-wrap")) setPlaylistsOpen(false); });
addEventListener("mousedown", (e) => { if (!$("#playlistsBox").hidden && !e.target.closest(".pl-wrap, #modal")) setPlaylistsOpen(false); });
function showView(v) {
  if (v === "export") { showView("library"); setCardOpen(true); $("#doExport").focus({ preventScroll: true }); return; }   // the card column is part of the library
  $$("[role=tab][data-view]").forEach((b) => { const on = b.dataset.view === v; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; });
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  if (v === "deck") loadDeck();
  if (v === "settings") { loadSettings(); requestAnimationFrame(drawWavePreview); }   // sized once it is visible
}
let cssCache = {};   // colours read from CSS variables; reading them every animation frame is slow
function applyTheme(t) {
  cssCache = {};
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = isDark();   // the switch shows both icons; CSS highlights the current one
  $("#themeBtn").setAttribute("aria-label", dark ? "Dark theme on: switch to light" : "Light theme on: switch to dark");
  $("#themeBtn").title = $("#themeBtn").getAttribute("aria-label");
}
const isDark = () => (document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyTheme(document.documentElement.dataset.theme); if (state.wave) buildWaves(); });
try { applyTheme(localStorage.getItem("theme")); } catch { applyTheme(null); }
$("#themeBtn").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch {}
  if (state.wave) buildWaves();
});

// ── Library: playlists sidebar ───────────────────────────────────────────────
async function loadPlaylists() {
  state.playlists = await api("/api/playlists");
  const kids = (pid) => state.playlists.filter((p) => p.parent_id === pid);
  const build = (pid) => kids(pid).map((p) => {
    const open = state.open.has(p.id);
    const btn = h("button", { class: "nav-item" + (state.playlist === p.id ? " active" : ""), "data-playlist": p.id,
                              "aria-expanded": p.is_folder ? String(open) : null, title: "Right-click to rename or delete" },
      p.is_folder ? h("span", { class: "chev", "aria-hidden": "true" }, open ? "▾" : "▸") : null,
      icon(p.is_folder ? "folder" : "list", "ic sm"),
      h("span", { class: "nm" }, p.name), p.is_folder ? null : h("span", { class: "count" }, String(p.count)));
    btn.addEventListener("click", () => {
      if (p.is_folder) { if (open) { state.open.delete(p.id); state.openFolderTarget = null; } else { state.open.add(p.id); state.openFolderTarget = p.id; } loadPlaylists(); }
      else selectPlaylist(p.id);
    });
    btn.addEventListener("contextmenu", (e) => { e.preventDefault(); playlistMenu(p, e); });
    btn.addEventListener("keydown", (e) => { if (e.key === "F2" || (e.shiftKey && e.key === "F10")) { e.preventDefault(); playlistMenu(p, btn); } });
    if (!p.is_folder) dropTarget(btn, (ids) => addToPlaylist(p.id, ids));
    if (!p.is_folder) return btn;
    return h("div", { class: "folder" + (open ? " open" : "") }, btn, h("div", { class: "children" }, build(p.id)));
  });
  const tree = build(0);
  $("#tree").replaceChildren(...(tree.length ? tree : []));
  $("#allTracks").classList.toggle("active", state.playlist === 0);
  $("#plCurrent").textContent = state.playlists.find((p) => p.id === state.playlist)?.name || "";   // the button names the open playlist
}
function playlistMenu(p, at) {
  const kind = p.is_folder ? "folder" : "playlist";
  openMenu(at, [
    !p.is_folder ? { icon: "library", label: "Open", run: () => selectPlaylist(p.id) } : null,
    !p.is_folder && state.selected.size ? { icon: "plus", label: `Add the ${plural(state.selected.size, "selected track")}`, run: () => addToPlaylist(p.id, [...state.selected]).catch(fail) } : null,
    !p.is_folder ? { icon: "card", label: "Export only this playlist", title: "Sets the card's scope to this playlist", run: () => {
      setCardOpen(true); $('input[name=scope][value="playlists"]').checked = true; $("#exportPlaylists").hidden = false;
      $$("#exportPlaylists input").forEach((i) => { i.checked = +i.value === p.id; }); updatePlan(); } } : null,
    p.is_folder ? { icon: "plus", label: "New playlist in this folder", run: () => { state.openFolderTarget = p.id; state.open.add(p.id); newPlaylist(false).catch(fail); } } : null,
    "-",
    { icon: "pencil", label: "Rename…", hint: "F2", run: async () => { const name = await ask(`Rename "${p.name}"`, p.name, "Save"); if (name) { await api(`/api/playlists/${p.id}`, { method: "PATCH", body: { name } }); loadPlaylists(); } } },
    { icon: "x", label: `Delete ${kind}`, danger: true, title: "Tracks stay in the library", run: async () => {
      await api(`/api/playlists/${p.id}`, { method: "DELETE" });
      toast(`Deleted ${kind} “${p.name}”`);
      if (state.playlist === p.id) selectPlaylist(0); else loadPlaylists(); } },
  ]);
}
async function newPlaylist(isFolder) {
  const inFolder = state.playlists.find((p) => p.id === state.openFolderTarget);
  const name = await ask((isFolder ? "New folder" : "New playlist") + (inFolder ? ` in "${inFolder.name}"` : ""), "", "Create");
  if (!name) return;
  const parent = inFolder ? state.openFolderTarget : 0;
  const p = await api("/api/playlists", { body: { name, parent_id: parent, is_folder: isFolder } });
  if (!isFolder && state.selected.size) await addToPlaylist(p.id, [...state.selected], p.name || name);
  loadPlaylists();
}
$("#newPlaylist").addEventListener("click", () => newPlaylist(false).catch(fail));
$("#newFolder").addEventListener("click", () => newPlaylist(true).catch(fail));
$("#allTracks").addEventListener("click", () => selectPlaylist(0));

function selectPlaylist(id) { state.playlist = id; state.selected.clear(); state.anchor = null; setPlaylistsOpen(false); loadPlaylists(); loadTracks(); }
async function addToPlaylist(pid, ids, name = null) {
  const r = await api(`/api/playlists/${pid}/add`, { body: { track_ids: ids } });
  const p = state.playlists.find((x) => x.id === pid);
  toast(`Added ${ids.length} to "${name || p?.name || "playlist"}"`, r.undo);
  loadPlaylists();
}

// ── Library: track table ─────────────────────────────────────────────────────
async function loadTracks() {
  const q = encodeURIComponent($("#search").value.trim());
  state.tracks = await api(`/api/tracks?q=${q}&playlist=${state.playlist}`);
  state.loaded = true;
  findDuplicates();
  renderTracks();
  if (!state.playlist && !q) $("#allCount").textContent = state.tracks.length || "";
}

// Filters and sorting only change what is shown; playlist order (the # column) stays the data.
const FILTERS = {
  all: () => true,
  analysed: (t) => !!t.analysed_at,
  unanalysed: (t) => !t.analysed_at,
  stems: (t) => t.stems === "done",
  attention: (t) => issues(t).length > 0,
  mixes: (t) => mixesWith(t),
};
// Mixing partners of the track picked when "Mixes with selected" was chosen: the same key, one step either way on the
// Camelot wheel, or the other mode of the same number (relative major/minor), and a tempo within 6 %.
function mixesWith(t) {
  const base = state.tracks.find((x) => x.id === state.mixBase);
  if (!base || base.id === t.id) return false;
  const code = (x) => /^(\d+)([AB])$/.exec(x.shifted_camelot || x.camelot || "");
  const [a, b] = [code(base), code(t)];
  if (!a || !b) return false;
  const d = Math.abs(+a[1] - +b[1]), step = Math.min(d, 12 - d);
  const keyOk = a[2] === b[2] ? step <= 1 : step === 0;
  const tempo = (x) => (x.bpm || 0) * 1;
  return keyOk && tempo(base) > 0 && tempo(t) > 0 && Math.abs(tempo(t) / tempo(base) - 1) <= 0.06;
}
// Tracks that look like the same song twice (same artist and title, ignoring case, spaces and "(Original Mix)").
function findDuplicates() {
  const norm = (s) => (s || "").toLowerCase().replace(/\(original mix\)|\(extended mix\)/g, "").replace(/[^a-z0-9]+/g, " ").trim();
  const groups = new Map();
  for (const t of state.tracks) { if (!t.title) continue; const k = `${norm(t.artist)}|${norm(t.title)}`; groups.set(k, [...(groups.get(k) || []), t.id]); }
  state.dups = new Map();
  for (const ids of groups.values()) if (ids.length > 1) ids.forEach((id) => state.dups.set(id, ids.filter((x) => x !== id)));
}
// What would go wrong at the booth: the server's check of the file (deck_problem: a 24-bit WAV, an M4A...) and what the
// track list already knows. "Convert for the deck" makes a copy the deck can play and clears the first one.
function issues(t) {
  const out = [];
  if (t.lead_s > 2) out.push(`Starts with ${Math.round(t.lead_s)} s of near silence`);
  if (t.tail_s > 6) out.push(`Ends with ${Math.round(t.tail_s)} s of near silence (an autopilot mix would wait for it)`);
  if (t.file_missing) out.push("File not found: it was moved, renamed or its drive is not connected");
  if (t.deck_problem) out.push(t.deck_problem);
  if (state.dups?.has(t.id)) out.push(`Same artist and title as ${state.dups.get(t.id).map((id) => `#${id}`).join(", ")}`);
  if (!t.artist) out.push("No artist");
  if (!t.analysed_at && jobFor(t.id, "analyse")?.status === "failed") out.push("Analysis failed");
  if (t.stems === "failed") out.push("Stems failed");
  return out;
}
function visibleTracks() {
  const list = state.tracks.map((t, i) => ({ t, i })).filter(({ t }) => FILTERS[state.filter](t));
  if (state.sort) {
    const { key, dir } = state.sort;
    const val = (x) => key === "index" ? x.i : key === "mood" ? moodRank(x.t.mood) : x.t[key];
    list.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (va == null || va === "") return vb == null || vb === "" ? 0 : 1;      // empty values last
      if (vb == null || vb === "") return -1;
      return (typeof va === "number" ? va - vb : String(va).localeCompare(String(vb), undefined, { numeric: true, sensitivity: "base" })) * dir;
    });
  }
  return list;
}
const canReorder = () => state.playlist && !state.sort && state.filter === "all";

// Latest job of a kind for a track (the jobs list is newest first).
const jobFor = (id, kind) => state.jobs.find((j) => j.track_id === id && j.kind === kind);
// Energy 1-10 as five bars; the number is in the tooltip.
function energyBars(n) {
  if (n == null) return "";
  const k = Math.round(n / 2);
  return h("span", { class: "ebars", role: "img", "aria-label": `Energy ${n} of 10`, title: `Energy ${n} of 10` },
    Array.from({ length: 5 }, (_, i) => h("i", { class: i < k ? "" : "off", style: `height:${5 + i * 2}px` })));
}
// Readiness: one ring per track, a segment per check, judged against the track's intent. Diagnostic, not a
// score: a Mix track without stems is ready; a Remix track without them isn't. Hover lists what's missing.
function readiness(t) {
  const aj = jobFor(t.id, "analyse"), sj = jobFor(t.id, "stems"), d = exp.root && exp.plan?.details?.find((x) => x.id === t.id);
  const seg = (key, state, text) => ({ key, state, text });
  const out = [];
  out.push(t.file_missing ? seg("file", "bad", "File missing") : t.deck_problem && !t.deck_copy ? seg("file", "bad", "Won't play on the deck: convert it") : seg("file", "ok", "Plays on the deck"));
  out.push(aj?.status === "running" ? seg("analysed", "busy", `Analysing ${Math.round(aj.progress * 100)}%`) : aj?.status === "queued" ? seg("analysed", "busy", "Queued for analysis")
    : t.analysed_at ? seg("analysed", "ok", "Analysed") : seg("analysed", "todo", "Not analysed"));
  const gridChecked = t.grid_ok_at || t.grid_offset_ms || t.downbeat_shift;
  out.push(!t.analysed_at ? seg("grid", "na", "Beat grid: after analysis") : gridChecked ? seg("grid", "ok", "Beat grid confirmed") : seg("grid", "todo", "Beat grid not confirmed by ear (open the track: ✓ or nudge)"));
  out.push(t.hot_cues ? seg("cues", "ok", `${plural(t.hot_cues, "hot cue")}`) : t.analysed_at ? seg("cues", "todo", "No hot cues") : seg("cues", "na", "Cues: after analysis"));
  if (wantsStems(t)) out.push(t.stems === "done" ? seg("stems", "ok", "Stems ready") : sj?.status === "running" ? seg("stems", "busy", `Making stems ${Math.round(sj.progress * 100)}%`)
    : t.stems === "queued" || sj?.status === "queued" ? seg("stems", "busy", "Stems queued") : t.stems === "failed" ? seg("stems", "bad", "Stems failed") : seg("stems", "todo", `No stems yet (${intentOf(t)[1]} needs them)`));
  else out.push(seg("stems", "na", t.stems === "done" ? "Stems ready (not needed for this intent)" : "Stems not needed for this intent"));
  out.push(t.lufs != null ? seg("loud", "ok", `${t.lufs.toFixed(1)} LUFS`) : t.analysed_at ? seg("loud", "todo", "Loudness not measured yet") : seg("loud", "na", "Loudness: after analysis"));
  out.push(!exp.root || !exp.plan ? seg("card", "na", "On the card: choose a drive to know") : !d ? seg("card", "todo", "Not in this export") : d.skipped ? seg("card", "bad", `Skipped by the export: ${d.notes.join(" · ")}`) : seg("card", "ok", "Goes with the next export"));
  return out;
}
const RING_COLOR = { ok: "var(--ok)", todo: "var(--line-strong)", bad: "var(--alert)", busy: "var(--accent)", na: "var(--line)" };
function readinessCell(t) {
  const segs = readiness(t), n = segs.length, R = 9, C = 2 * Math.PI * R, gap = 2.2;
  const bad = segs.some((s) => s.state === "bad"), busy = segs.some((s) => s.state === "busy");
  const live = segs.filter((s) => s.state !== "na"), ready = live.length && live.every((s) => s.state === "ok");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("class", "ready" + (busy ? " busy" : ""));
  segs.forEach((s, i) => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const len = C / n - gap;
    c.setAttribute("cx", "12"); c.setAttribute("cy", "12"); c.setAttribute("r", String(R));
    c.setAttribute("fill", "none"); c.setAttribute("stroke", RING_COLOR[s.state]); c.setAttribute("stroke-width", s.state === "na" ? "1.5" : "3");
    c.setAttribute("stroke-dasharray", `${len.toFixed(2)} ${(C - len).toFixed(2)}`); c.setAttribute("stroke-dashoffset", String((-i * C / n).toFixed(2)));
    c.setAttribute("transform", "rotate(-90 12 12)");
    svg.append(c);
  });
  if (ready || bad) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", ready ? "M8.5 12.2 11 14.7l4.5-5" : "M12 8.5v4M12 15.5h.01");
    p.setAttribute("fill", "none"); p.setAttribute("stroke", ready ? "var(--ok)" : "var(--alert)"); p.setAttribute("stroke-width", "2.2"); p.setAttribute("stroke-linecap", "round"); p.setAttribute("stroke-linejoin", "round");
    svg.append(p);
  }
  const label = ready ? `Ready (${intentOf(t)[1]})` : segs.filter((s) => s.state !== "ok" && s.state !== "na").map((s) => s.text).join(" · ") || "Ready";
  return h("span", { class: "st", role: "img", "aria-label": label, title: `${intentOf(t)[1]}: ${label}` }, svg);
}
function moodPill(t) {
  const m = moodOf(t.mood);
  if (!m) return h("span", { class: "muted" }, t.analysed_at ? "–" : "");
  const sure = Math.round((t.mood_conf ?? 0) * 100), unsure = !t.mood_manual && sure < 50;
  return h("span", { class: "mood" + (unsure ? " unsure" : ""), style: `--mood:${MOOD_COLOR[m.id] || "#8a8a90"}`,
                     title: `${m.about}. ${t.mood_manual ? "Set by you." : `Automatic, ${sure} % sure.`}` },
    h("i"), m.label, unsure ? h("span", { class: "q", "aria-label": "unsure" }, "?") : null);
}
// Sorted by mood, each mood gets a header row with its track count and length.
function withMoodGroups(vis, rows) {
  const out = [];
  vis.forEach(({ t }, k) => {
    if (k === 0 || vis[k - 1].t.mood !== t.mood) {
      const group = vis.filter((x) => x.t.mood === t.mood).map((x) => x.t), m = moodOf(t.mood);
      out.push(h("tr", { class: "group", "aria-hidden": "true" }, h("td", { colspan: "11" },
        h("span", { class: "mood", style: `--mood:${m ? MOOD_COLOR[m.id] : "#8a8a90"}` }, h("i"), m ? m.label : "No mood yet"),
        h("span", { class: "muted" }, `${plural(group.length, "track")} · ${fmtDur(group.reduce((a, x) => a + (x.duration || 0), 0))}`))));
    }
    out.push(rows[k]);
  });
  return out;
}
// Key in the track colours by Camelot number, so neighbouring keys look related.
// Loudness gain the export writes: what brings the track to the target in Settings, within -12..+6 dB (the server decides).
const fmtGain = (t) => { const g = Math.max(-12, Math.min(6, (state.settings?.target_lufs ?? -9) - t.lufs)); return `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(1)} dB`; };
const keyColor = (camelot) => COLORS[1 + ((parseInt(camelot, 10) || 1) - 1) % 8];
function artCell(t) {
  if (t.has_artwork) return h("img", { class: "art", src: `/api/tracks/${t.id}/artwork`, alt: "", loading: "lazy" });
  const c = t.camelot ? keyColor(t.camelot) : "#46464c";
  return h("span", { class: "art", style: `background:linear-gradient(150deg, ${c} 0%, var(--surface-2) 85%)` });
}
function renderTracks() {
  if (state.editing) { state.renderLater = true; return; }      // an inline editor is open; it renders when it closes
  const body = $("#tracks tbody");
  const focusedId = body.contains(document.activeElement) ? document.activeElement.dataset?.id : null;   // keep keyboard focus across re-renders
  const vis = visibleTracks();
  state.view = vis.map((x) => x.t);
  const reorderOk = canReorder();
  const rows = vis.map(({ t, i }, vi) => {
    const tr = h("tr", { "data-id": t.id, draggable: "true", tabindex: "-1", "aria-selected": String(state.selected.has(t.id)),
                         class: (state.selected.has(t.id) ? "sel" : "") + (state.current?.id === t.id ? " playing" : "") },
      h("td", { class: "c-idx num" }, String(i + 1)),
      h("td", { class: "c-art" }, artCell(t)),
      h("td", { class: "c-title", "data-edit": "title", title: t.path }, t.color ? h("span", { class: "swatch", style: `background:${COLORS[t.color]}`, title: COLOR_NAMES[t.color] }) : null, t.title || ""),
      h("td", { class: "c-artist", "data-edit": "artist" }, t.artist || h("span", { class: "muted" }, "Unknown")),
      h("td", { class: "num" + (wasEdited(t, "bpm") ? " edited" : ""), "data-edit": "bpm" }, t.bpm ? fmtBpm(t.bpm) : h("span", { class: "muted" }, t.analysed_at ? "?" : "…")),
      h("td", { class: wasEdited(t, "key") ? "edited" : "", "data-edit": "key" }, t.camelot ? h("span", { class: "key", style: `color:${keyColor(t.shifted_camelot || t.camelot)}`,
          title: t.key_shift ? `${t.camelot} ${t.musical_key || ""} as analysed, shifted ${fmtShift(t.key_shift)} on the deck` : null },
        h("b", {}, t.shifted_camelot || t.camelot), h("span", {}, t.shifted_key || t.musical_key || ""),
        t.key_shift ? h("span", { class: "shift-tag" }, fmtShift(t.key_shift)) : null) : ""),
      h("td", { class: "c-mood" + (t.mood_manual ? " edited" : ""), "data-edit": "mood" }, moodPill(t)),
      h("td", { class: "c-energy" + (wasEdited(t, "energy") ? " edited" : ""), "data-edit": "energy" }, energyBars(t.energy)),
      h("td", { class: "num lufs c-lufs", title: t.lufs != null ? `${t.lufs.toFixed(1)} LUFS. The export writes ${fmtGain(t)} for the deck (target ${state.settings?.target_lufs ?? -9} LUFS).` : null },
        t.lufs != null ? t.lufs.toFixed(1) : ""),
      h("td", { class: "c-an" }, readinessCell(t)),
      h("td", { class: "c-menu" }, h("button", { class: "row-menu", "aria-label": `Actions for ${t.title || "track"}`, title: "Actions",
        onclick: (e) => { e.stopPropagation(); rowMenu(e.currentTarget, t); } }, icon("more", "ic sm"))));
    tr.addEventListener("click", (e) => {
      // A click on a cell of the row that is already the selected one edits it (see "Quick edits" below).
      const cell = e.target.closest("td[data-edit]");
      const alone = state.selected.size === 1 && state.selected.has(t.id) && !e.shiftKey && !e.ctrlKey && !e.metaKey;
      clickRow(e, t.id, vi);
      // The edit waits a moment so a double-click (open the track) can cancel it.
      if (cell && alone) { clearTimeout(state.editT); state.editT = setTimeout(() => beginEdit(t.id, cell.dataset.edit), 260); }
    });
    tr.addEventListener("dblclick", (e) => { clearTimeout(state.editT); if (!e.target.closest("input")) openFocus(t.id); });
    tr.addEventListener("contextmenu", (e) => { if (e.target.closest("input")) return; e.preventDefault(); clearTimeout(state.editT); rowMenu(e, t); });
    tr.addEventListener("dragstart", (e) => {
      if (!state.selected.has(t.id)) { state.selected = new Set([t.id]); renderTracks(); }
      e.dataTransfer.setData("application/x-tracks", JSON.stringify([...state.selected]));
      e.dataTransfer.effectAllowed = "copyMove";
    });
    if (reorderOk) {   // reorder inside a playlist by dragging (only in playlist order, unfiltered)
      tr.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("application/x-tracks")) e.preventDefault(); });
      tr.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); reorder(JSON.parse(e.dataTransfer.getData("application/x-tracks")), t.id); });
    }
    return tr;
  });
  body.replaceChildren(...(state.sort?.key === "mood" ? withMoodGroups(vis, rows) : rows));
  // Keyboard focus lands on the anchor row (or the first), arrow keys move from there.
  (body.querySelector(`tr[data-id="${state.anchor}"]`) || body.firstElementChild)?.setAttribute("tabindex", "0");
  if (focusedId) body.querySelector(`tr[data-id="${focusedId}"]`)?.focus({ preventScroll: true });
  const libEmpty = state.tracks.length === 0 && !$("#search").value && state.playlist === 0;
  $("#empty").hidden = !state.loaded || !libEmpty;
  const note = $("#tableNote");
  note.hidden = state.loaded && (rows.length > 0 || libEmpty);
  note.textContent = !state.loaded ? "Loading library…"
    : state.tracks.length === 0 && state.playlist && !$("#search").value ? "Empty playlist"
    : state.tracks.length === 0 ? "No tracks match the search." : "No tracks match this filter.";
  $$("#tracks thead th[data-sort]").forEach((th) => th.setAttribute("aria-sort", state.sort?.key === th.dataset.sort ? (state.sort.dir > 0 ? "ascending" : "descending") : "none"));
  renderContext();
  renderLibStatus();
  $("#sortSel").value = state.sort ? (state.sort.dir < 0 ? `${state.sort.key}:-1` : state.sort.key) : "index";
  if (!$("#sortSel").value) $("#sortSel").value = "index";
}
// Status cells refresh in place on each jobs poll, so dragging and hovering aren't disturbed.
function updateStatusCells() {
  for (const tr of $$("#tracks tbody tr")) {
    const t = state.tracks.find((x) => x.id === +tr.dataset.id);
    if (!t) continue;
    tr.querySelector(".c-an")?.replaceChildren(readinessCell(t));
  }
}
function renderLibStatus() {
  const n = state.tracks.length, secs = state.tracks.reduce((a, t) => a + (t.duration || 0), 0);
  const pl = state.playlists.find((p) => p.id === state.playlist);
  $("#statusLib").textContent = !state.loaded ? "Loading library…"
    : `${pl ? pl.name + " · " : ""}${plural(n, "track")} · ${fmtDur(secs)}` + (state.view.length !== n ? ` · ${state.view.length} shown` : "");
  const attn = state.tracks.filter((t) => issues(t).length).length;
  $("#attnCount").textContent = attn ? String(attn) : "";
}
$("#sortSel").addEventListener("change", () => {
  const [key, d] = $("#sortSel").value.split(":");
  state.sort = key === "index" ? null : { key, dir: d === "-1" ? -1 : 1 };
  renderTracks();
});

// ── Small menus (row and panel "⋮") ─────────────────────────────────────────
// Items: { label, run, icon?, hint? (right-hand text, e.g. a key), checked? (a tick, for a choice), danger? },
// "-" for a line, { head } for a group title. `at` is a button (the menu hangs under it) or a pointer event
// (the menu opens at the pointer, as a right-click menu does).
function openMenu(at, items) {
  const m = $("#menu");
  fill(m, items.filter(Boolean).map((it) => it === "-" ? h("hr") : it.head ? h("div", { class: "menu-head" }, it.head) :
    h("button", { role: it.checked != null ? "menuitemradio" : "menuitem", "aria-checked": it.checked != null ? String(!!it.checked) : null,
                  class: (it.danger ? "danger-item" : "") + (it.checked ? " checked" : ""), disabled: it.disabled ? true : null, title: it.title || null,
                  onclick: () => { closeMenu(); it.run(); } },
      h("span", { class: "mi-ic" }, it.icon ? icon(it.icon, "ic sm") : null), h("span", { class: "mi-label" }, it.label),
      it.hint ? h("kbd", {}, it.hint) : null, it.checked ? icon("check", "ic sm mi-check") : null)));
  m.hidden = false;
  const mw = m.offsetWidth, mh = m.offsetHeight;
  if (at instanceof Event || at?.clientX != null) {
    m.style.left = `${Math.max(8, Math.min(innerWidth - mw - 8, at.clientX))}px`;
    m.style.top = `${Math.max(8, Math.min(innerHeight - mh - 8, at.clientY))}px`;
    state.menuFrom = at.target?.closest?.("tr, button, .wstack") || null;
  } else {
    const r = at.getBoundingClientRect();
    m.style.left = `${Math.max(8, Math.min(innerWidth - mw - 8, r.right - mw))}px`;
    m.style.top = `${r.bottom + mh + 8 > innerHeight ? r.top - mh - 4 : r.bottom + 4}px`;
    state.menuFrom = at;
  }
  m.querySelector("button:not([disabled])")?.focus();
}
function closeMenu() { const m = $("#menu"); if (!m.hidden) { m.hidden = true; state.menuFrom?.focus?.(); } }
addEventListener("mousedown", (e) => { if (!$("#menu").hidden && !$("#menu").contains(e.target)) closeMenu(); });
$("#menu").addEventListener("keydown", (e) => {
  const b = $$("#menu button"), i = b.indexOf(document.activeElement);
  if (e.key === "Escape") { e.stopPropagation(); closeMenu(); }
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); b[(i + (e.key === "ArrowDown" ? 1 : -1) + b.length) % b.length]?.focus(); }
});
async function bulkEdit(ids, key, label) {
  const raw = await ask(`${label} for ${plural(ids.length, "track")}`, "", "Apply");
  if (raw == null) return;
  const num = key === "year" || key === "key_shift";
  const value = num ? (raw.trim() === "" ? null : +raw) : raw.trim();
  if (num && value != null && !Number.isFinite(value)) return toast("That isn't a number", null, true);
  try {
    for (const id of ids) await api(`/api/tracks/${id}`, { method: "PATCH", body: { [key]: value } });
    toast(`${label} set on ${plural(ids.length, "track")}`);
    await loadTracks();
    if (state.current && ids.includes(state.current.id)) openTrack(state.current.id);
  } catch (e) { fail(e); }
}
async function convertForDeck(ids) {
  try {
    const r = await api("/api/convert", { body: { ids } });
    toast(r.jobs.length ? `Converting ${plural(r.jobs.length, "track")} for the deck` : "Already converting");
    pollJobs();
  } catch (e) { fail(e); }
}
async function removeDeckCopy(ids) {
  try {
    await api("/api/convert", { body: { ids, remove: true } });
    toast(`${plural(ids.length, "track")} will be exported as the original`);
    await loadTracks();
    if (state.current && ids.includes(state.current.id)) openTrack(state.current.id);
  } catch (e) { fail(e); }
}
// The track menu (right-click, the row's ⋮, the open track's ⋮). On a selected row it acts on the whole selection.
function trackMenuItems(t, ids = [t.id]) {
  const many = ids.length > 1, sel = ids.map((id) => state.tracks.find((x) => x.id === id) || t);
  const n = (s) => (many ? ` ${ids.length} ${s}` : "");
  const unplayable = sel.filter((x) => x.deck_problem).map((x) => x.id), copies = sel.filter((x) => x.deck_copy).map((x) => x.id);
  const noStems = sel.filter((x) => !["done", "running", "queued"].includes(x.stems)).map((x) => x.id);
  const intents = new Set(sel.map((x) => x.intent || "mix"));
  const lists = state.playlists.filter((p) => !p.is_folder);
  const inList = state.playlist ? lists.find((p) => p.id === state.playlist) : null;
  return [
    !many ? { icon: "play", label: state.current?.id === t.id && state.focus ? "Playing" : "Open", hint: "↵", run: () => openFocus(t.id) } : null,
    !many && t.analysed_at ? { icon: "shuffle", label: "Mixes with this", title: "Show the tracks in a compatible key within 6 % of its tempo", run: () => { state.mixBase = t.id; $('#filters [data-filter="mixes"]').click(); } } : null,
    { head: many ? `Prepare ${ids.length} as` : "Prepare as" },
    ...INTENTS.map(([id, label, about]) => ({ label, title: about, checked: intents.size === 1 && intents.has(id), run: () => setIntent(ids, id) })),
    "-",
    { icon: "refresh", label: (sel.every((x) => x.analysed_at) ? "Re-analyse" : "Analyse") + n("tracks"), run: () => api("/api/analyse", { body: { ids } }).then(() => { toast(`Analysing${n("tracks") || ""}`); pollJobs(); }).catch(fail) },
    noStems.length ? { icon: "deck", label: "Make stems" + n(""), run: () => api("/api/stems", { body: { ids: noStems } }).then(() => { toast("Making stems"); loadTracks(); pollJobs(); }).catch(fail) } : null,
    unplayable.length ? { icon: "card", label: `Convert${many ? " " + unplayable.length : ""} for the deck`, title: "A 16-bit FLAC copy the deck can play; the original stays", run: () => convertForDeck(unplayable) } : null,
    copies.length ? { icon: "card", label: "Use the original for the deck", run: () => removeDeckCopy(copies) } : null,
    !many && t.analysed_at ? { icon: "check", label: t.grid_ok_at ? "Beat grid confirmed" : "Confirm the beat grid", checked: !!t.grid_ok_at, title: "You listened and the grid is right",
      run: () => api(`/api/tracks/${t.id}`, { method: "PATCH", body: { grid_ok_at: t.grid_ok_at ? null : Date.now() / 1000 } }).then((r) => { toast(t.grid_ok_at ? "Beat grid unconfirmed" : "Beat grid confirmed", r.undo); loadTracks(); if (state.current?.id === t.id) openTrack(t.id); }).catch(fail) } : null,
    lists.length ? { head: "Add to playlist" } : null,
    ...lists.slice(0, 8).map((p) => ({ icon: "list", label: p.name, hint: String(p.count), run: () => addToPlaylist(p.id, ids).catch(fail) })),
    { icon: "plus", label: "New playlist with " + (many ? `these ${ids.length}` : "this track"), run: () => { state.selected = new Set(ids); newPlaylist(false).catch(fail); } },
    inList ? { icon: "x", label: `Remove from “${inList.name}”`, run: async () => {
      const keep = state.tracks.map((x) => x.id).filter((x) => !ids.includes(x));
      const r = await api(`/api/playlists/${state.playlist}/tracks`, { method: "PUT", body: { track_ids: keep } });
      toast(`Removed ${ids.length} from ${inList.name}`, r.undo); state.selected.clear(); loadTracks(); loadPlaylists(); } } : null,
    "-",
    { icon: "x", label: "Remove from library" + n(""), hint: "Del", danger: true, title: "The files stay on disk", run: () => { state.selected = new Set(ids); removeSelected().catch(fail); } },
  ];
}
// The row's ⋮ or a right-click: a selected row acts for the whole selection, any other row for itself.
function rowMenu(at, t) {
  if (!state.selected.has(t.id)) { state.selected = new Set([t.id]); state.anchor = t.id; renderTracks(); }
  openMenu(at, trackMenuItems(t, state.selected.has(t.id) && state.selected.size > 1 ? [...state.selected] : [t.id]));
}
function clickRow(e, id, vi) {
  const view = state.view;
  if (e.shiftKey && state.anchor != null) {
    const a = view.findIndex((t) => t.id === state.anchor);
    const [lo, hi] = [Math.min(a < 0 ? vi : a, vi), Math.max(a < 0 ? vi : a, vi)];
    state.selected = new Set(view.slice(lo, hi + 1).map((t) => t.id));
  } else if (e.ctrlKey || e.metaKey) {
    state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
    state.anchor = id;
  } else {
    state.selected = new Set([id]); state.anchor = id;
    openTrack(id, false);
  }
  renderTracks();
  $(`#tracks tbody tr[data-id="${id}"]`)?.focus({ preventScroll: true });
}
// Arrow keys move the selection (Shift extends it), Enter plays, in the focused table.
$("#tracks tbody").addEventListener("keydown", (e) => {
  if (e.target.closest("input, select")) return;
  if (e.key === "F2") { const id = +document.activeElement?.dataset?.id; if (id) { e.preventDefault(); beginEdit(id, "title"); } return; }
  if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) { const id = +document.activeElement?.dataset?.id, t = state.tracks.find((x) => x.id === id); if (t) { e.preventDefault(); rowMenu(document.activeElement, t); } return; }
  if (!["ArrowDown", "ArrowUp", "Enter", "Home", "End"].includes(e.key) || !state.view.length) return;
  e.preventDefault();
  const view = state.view, cur = view.findIndex((t) => t.id === +document.activeElement?.dataset?.id);
  if (e.key === "Enter") { if (cur >= 0) openFocus(view[cur].id); return; }
  const n = e.key === "Home" ? 0 : e.key === "End" ? view.length - 1 : Math.max(0, Math.min(view.length - 1, cur + (e.key === "ArrowDown" ? 1 : -1)));
  clickRow({ shiftKey: e.shiftKey, ctrlKey: false, metaKey: false }, view[n].id, n);
  $(`#tracks tbody tr[data-id="${view[n].id}"]`)?.scrollIntoView({ block: "nearest" });
});
// ── Quick edits in the list ─────────────────────────────────────────────────
// Click a cell of the selected row (or press F2 for the title): text and tempo become a box (Enter saves, Esc cancels,
// Tab moves to the next cell); key, mood and energy open a small picker. Edits are saved at once and can be undone.
// A tempo, key or energy you set is kept when the track is analysed again; "Analysed value" puts it back.
const EDIT_ORDER = ["title", "artist", "bpm", "key", "mood", "energy"];
const NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const wasEdited = (t, what) => { try { return JSON.parse(t.edited || "[]").includes(what); } catch { return false; } };
function camelotName(n, letter) {                       // 8B = C major, 5A = C minor; one step round the wheel = a fifth
  const major = (((n - 8) * 7) % 12 + 12) % 12;
  return NOTE_NAMES[letter === "B" ? major : (major + 9) % 12] + (letter === "A" ? "m" : "");
}
function endEdit() {
  state.editing = null; $("#picker")?.remove(); document.removeEventListener("mousedown", pickerOutside, true);
  if (state.renderLater) { state.renderLater = false; renderTracks(); }
}
function pickerOutside(e) { if (!e.target.closest("#picker")) endEdit(); }
async function saveTrackEdit(id, fields, label) {
  try {
    const r = await api(`/api/tracks/${id}`, { method: "PATCH", body: fields });
    const i = state.tracks.findIndex((x) => x.id === id);
    if (i >= 0) state.tracks[i] = { ...state.tracks[i], ...r.track };
    toast(label, r.undo);
    if ("mood" in fields || "mood_manual" in fields) await loadTracks();        // a hand-picked mood teaches the others
    if (state.current?.id === id) { state.editing = null; await openTrack(id); }
  } catch (e) { fail(e); } finally { endEdit(); renderTracks(); }
}
function beginEdit(id, kind) {
  if (state.editing) endEdit();
  const t = state.tracks.find((x) => x.id === id);
  const td = $(`#tracks tbody tr[data-id="${id}"] td[data-edit="${kind}"]`);
  if (!t || !td) return;
  state.editing = { id, kind };
  if (kind === "title" || kind === "artist" || kind === "bpm") return editText(t, td, kind);
  const picker = h("div", { id: "picker", class: "picker", role: "dialog", "aria-label": `Edit ${kind}` });
  const pick = (fields, label) => { picker.remove(); state.editing = null; saveTrackEdit(id, fields, label); };
  const analysed = (field, label) => wasEdited(t, label === "key" ? "key" : field)
    ? h("button", { class: "ghost small pick-auto", onclick: () => pick({ [field]: null }, `${label[0].toUpperCase() + label.slice(1)}: analysed value`) }, "Analysed value") : null;
  if (kind === "key") {
    picker.append(h("div", { class: "pick-keys" }, [["A", "Minor"], ["B", "Major"]].map(([letter, name]) =>
      h("div", {}, h("small", {}, name), Array.from({ length: 12 }, (_, i) => {
        const code = `${i + 1}${letter}`;
        return h("button", { class: "pick-key" + (t.camelot === code ? " on" : ""), style: `--k:${keyColor(code)}`,
                             onclick: () => pick({ camelot: code }, `Key: ${code} ${camelotName(i + 1, letter)}`) },
                 h("b", {}, code), h("span", {}, camelotName(i + 1, letter)));
      })))), analysed("camelot", "key"));
  } else if (kind === "energy") {
    picker.append(h("div", { class: "pick-energy" }, Array.from({ length: 10 }, (_, i) =>
      h("button", { class: "pick-num" + (t.energy === i + 1 ? " on" : ""), onclick: () => pick({ energy: i + 1 }, `Energy: ${i + 1}`) }, String(i + 1)))),
      analysed("energy", "energy"));
  } else if (kind === "mood") {
    picker.append(h("div", { class: "pick-moods" }, MOODS.map((m) =>
      h("button", { class: "pick-mood" + (t.mood_manual && t.mood === m.id ? " on" : ""), title: m.about,
                    onclick: () => pick({ mood: m.id, mood_manual: 1 }, `Mood: ${m.label}`) },
        h("span", { class: "mood", style: `--mood:${MOOD_COLOR[m.id]}` }, h("i"), m.label)))),
      t.mood_manual ? h("button", { class: "ghost small pick-auto", onclick: () => pick({ mood_manual: 0 }, "Mood: automatic") }, "Automatic") : null);
  }
  document.body.append(picker);
  const r = td.getBoundingClientRect(), w = picker.offsetWidth, ph = picker.offsetHeight;
  picker.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left))}px`;
  picker.style.top = `${r.bottom + ph + 8 > innerHeight ? Math.max(8, r.top - ph - 4) : r.bottom + 4}px`;
  picker.querySelector("button")?.focus();
  picker.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); endEdit(); renderTracks(); } });
  document.addEventListener("mousedown", pickerOutside, true);
}
function editText(t, td, kind) {
  const num = kind === "bpm";
  const input = h("input", { class: "cell-edit", type: num ? "number" : "text", step: num ? "0.01" : null, min: num ? "1" : null,
                             value: num ? (t.bpm ?? "") : (t[kind] ?? ""), "aria-label": `Edit ${kind}`, placeholder: num ? "analysed" : "" });
  const tr = td.closest("tr"); tr.draggable = false;
  td.replaceChildren(input);
  input.focus(); input.select();
  let done = false;
  const finish = (save, next = 0) => {
    if (done) return;
    done = true;
    const v = num ? (input.value === "" ? null : +input.value) : input.value.trim();
    const was = num ? (t.bpm ?? null) : (t[kind] ?? "");
    state.editing = null;
    if (save && v !== was && !(num && v === was)) {
      saveTrackEdit(t.id, { [kind]: v }, num ? (v == null ? "Tempo: analysed value" : `Tempo: ${v} BPM`) : `${kind === "title" ? "Title" : "Artist"} saved`)
        .then(() => { if (next) beginEdit(t.id, EDIT_ORDER[(EDIT_ORDER.indexOf(kind) + next + EDIT_ORDER.length) % EDIT_ORDER.length]); });
    } else {
      renderTracks();
      if (next) beginEdit(t.id, EDIT_ORDER[(EDIT_ORDER.indexOf(kind) + next + EDIT_ORDER.length) % EDIT_ORDER.length]);
    }
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { e.preventDefault(); finish(true); $(`#tracks tbody tr[data-id="${t.id}"]`)?.focus({ preventScroll: true }); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); $(`#tracks tbody tr[data-id="${t.id}"]`)?.focus({ preventScroll: true }); }
    else if (e.key === "Tab") { e.preventDefault(); finish(true, e.shiftKey ? -1 : 1); }
  });
  input.addEventListener("blur", () => finish(true));
  for (const ev of ["click", "dblclick", "mousedown"]) input.addEventListener(ev, (e) => e.stopPropagation());
}
// Sort by clicking a column header: ascending, descending, then back to playlist order.
$$("#tracks thead th[data-sort]").forEach((th) => {
  th.tabIndex = 0;
  const go = () => {
    const key = th.dataset.sort, s = state.sort;
    state.sort = !s || s.key !== key ? { key, dir: 1 } : s.dir > 0 ? { key, dir: -1 } : null;
    renderTracks();
  };
  th.addEventListener("click", go);
  th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
});
$$("#filters button").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.filter === "mixes") {
    const one = state.selected.size === 1 ? state.tracks.find((t) => state.selected.has(t.id)) : null;
    if (!one && state.mixBase == null) return toast("Select one track first, then show what mixes with it");
    if (one) state.mixBase = one.id;
    const base = state.tracks.find((t) => t.id === state.mixBase);
    b.textContent = base ? `Mixes with ${(base.title || "track").slice(0, 22)}` : "Mixes with selected";
  }
  state.filter = b.dataset.filter;
  $$("#filters button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  renderTracks();
}));

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
  if (ids.length > 1) parts.push(h("button", { class: "ghost small", onclick: (e) => openMenu(e.currentTarget, [
    ...[["artist", "Artist"], ["album", "Album"], ["genre", "Genre"], ["year", "Year"], ["comment", "Comment"]].map(([k, label]) => ({ label: `Set ${label.toLowerCase()} for ${ids.length}…`, run: () => bulkEdit(ids, k, label) })),
    "-", { label: `Set key shift for ${ids.length}…`, run: () => bulkEdit(ids, "key_shift", "Key shift (semitones, -12 to 12)") },
  ]), title: "Change one field on all selected tracks (each can be undone)" }, "Edit selected…"));
  parts.push(h("button", { class: "ghost small", title: "How you'll use these tracks: decides what gets prepared", onclick: (e) => openMenu(e.currentTarget, [
    ...INTENTS.map(([id, label, about]) => ({ label: `${label}: ${about}`, run: () => setIntent(ids, id) })),
    "-", { label: `Prepare ${ids.length} as set`, run: () => api("/api/prepare", { body: { ids } }).then((r) => { toast(`Analysing ${r.analyse}, stems for ${r.stems}, converting ${r.convert}`); loadTracks(); pollJobs(); }).catch(fail) },
  ]) }, "Prepare…"));
  const unplayable = sel.filter((t) => t.deck_problem).map((t) => t.id);
  if (unplayable.length) parts.push(h("button", { class: "ghost small", onclick: () => convertForDeck(unplayable),
    title: "16-bit FLAC copies the deck can play; the originals stay as they are" }, `Convert ${unplayable.length} for the deck`));
  const noStems = sel.filter((t) => t.stems !== "done" && t.stems !== "running" && t.stems !== "queued").map((t) => t.id);
  if (noStems.length) parts.push(h("button", { class: "ghost small", onclick: () => api("/api/stems", { body: { ids: noStems } }).then(() => { toast(`Making stems for ${noStems.length}`); loadTracks(); pollJobs(); }).catch(fail) }, "Make stems"));
  const lists = state.playlists.filter((p) => !p.is_folder);
  if (lists.length) {
    const s = h("select", { class: "small", "aria-label": "Add to playlist" }, h("option", { value: "" }, "Add to playlist…"), lists.map((p) => h("option", { value: p.id }, p.name)));
    s.addEventListener("change", () => { if (s.value) addToPlaylist(+s.value, ids).catch(fail); s.value = ""; });
    parts.push(s);
  }
  if (state.playlist) parts.push(h("button", { class: "ghost small", onclick: async () => {
    const keep = state.tracks.map((t) => t.id).filter((x) => !state.selected.has(x));
    const r = await api(`/api/playlists/${state.playlist}/tracks`, { method: "PUT", body: { track_ids: keep } });
    toast(`Removed ${ids.length} from playlist`, r.undo); state.selected.clear(); loadTracks(); loadPlaylists();
  } }, "Remove from playlist"));
  parts.push(h("button", { class: "danger small", onclick: () => removeSelected().catch(fail), title: "Files on disk are kept (Delete)" }, "Remove from library"));
  parts.push(h("button", { class: "ghost small icon-btn", "aria-label": "Clear selection", title: "Clear selection",
                           onclick: () => { state.selected.clear(); renderTracks(); } }, icon("x", "ic sm")));
  box.replaceChildren(...parts);
}
// Intent for one or many tracks, then prepare what it asks for.
async function setIntent(ids, intent) {
  try {
    for (const id of ids) await api(`/api/tracks/${id}`, { method: "PATCH", body: { intent } });
    const r = await api("/api/prepare", { body: { ids } });
    const made = [r.analyse ? `analysing ${r.analyse}` : null, r.stems ? `stems for ${r.stems}` : null, r.convert ? `converting ${r.convert}` : null].filter(Boolean).join(", ");
    toast(`${INTENTS.find((i) => i[0] === intent)[1]} for ${plural(ids.length, "track")}${made ? ": " + made : ""}`);
    await loadTracks(); pollJobs();
    if (state.current && ids.includes(state.current.id)) openTrack(state.current.id);
  } catch (e) { fail(e); }
}
async function removeSelected() {
  const ids = [...state.selected];
  if (!ids.length) return;
  const r = await api("/api/tracks/remove", { body: { ids } });
  toast(`Removed ${ids.length} from the library`, r.undo);
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

// ── Track preparation: player, waveform, cues, tags, stems ───────────────────
const player = $("#player");
async function openTrack(id, play) {
  if (state.gridPending && state.gridPending.id !== id) await saveGrid().catch(fail);
  const t = await api(`/api/tracks/${id}`);
  const same = state.current?.id === id;
  state.current = t;
  if (!same) {
    state.stemSolo = null;
    state.stemPeaks = null;
    player.src = `/api/tracks/${id}/audio`;
    state.wave = null;
    if (state.focus) buildWaves(); else destroyWaves();      // the waveforms are built when the track is opened full screen
    api(`/api/tracks/${id}/waveform`).then((w) => { if (state.current?.id === id) { state.wave = decodeWave(w); if (state.focus) buildWaves(); } }).catch(() => {});
    if (t.stem_files.length) loadStemPeaks(id);
  } else if (t.stem_files.length && !state.stemPeaks) loadStemPeaks(id);
  if (state.focus) renderDetail();
  updatePreviewBar();
  if (play) player.play().catch(() => {});
  renderTracks();
}
function loadStemPeaks(id) {
  api(`/api/tracks/${id}/stems/peaks`).then((p) => {
    if (state.current?.id !== id) return;
    const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
    state.stemPeaks = Object.fromEntries(Object.entries(p.peaks).map(([k, v]) => [k, Float32Array.from(b64(v), (x) => x / 255)]));
    if (state.wave && state.focus) buildWaves();
  }).catch(() => {});
}
// The open track covers the list and the card; Esc, the back glyph or the preview bar's chevron brings the list back.
function setFocus(on) {
  on = on && !!state.current;
  state.focus = on;
  $(".app").classList.toggle("focus", on);
  $("#detail").hidden = !on;
  $("#focusBtn").setAttribute("aria-expanded", String(on));
  if (on) { renderDetail(); if (state.wave) buildWaves(); requestAnimationFrame(() => waveStack.focus({ preventScroll: true })); }
  else { destroyWaves(); $(`#tracks tbody tr[data-id="${state.current?.id}"]`)?.focus({ preventScroll: true }); }
}
const openFocus = (id) => openTrack(id, true).then(() => setFocus(true)).catch(fail);
$("#focusBtn").addEventListener("click", () => setFocus(!state.focus));
function closeDetail() {
  player.pause(); destroyWaves();
  state.current = null; state.wave = null; state.stemPeaks = null; state.focus = false;
  $(".app").classList.remove("focus"); $("#detail").hidden = true; $("#focusBtn").setAttribute("aria-expanded", "false");
  renderTracks(); updatePreviewBar();
}

function decodeWave(w) {
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  return { detail: b64(w.detail), duration: w.duration_ms / 1000 };
}

// "Artist - Title" in the title or filename (spaced dash), or "Artist-Title" when the part
// before the dash has no spaces (e.g. "FISHER-Atmosphere"). Only offered when the artist is empty.
function splitGuess(t) {
  if (t.artist || state.noSplit?.has(t.id)) return null;
  const file = (t.path || "").split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  for (const s of [t.title, file]) {
    const m = s && (s.match(/^(.+?)\s+[-–—]\s+(.+)$/) || s.match(/^([^\s\-–—]+)[-–—](\S.*)$/));
    if (m) return { artist: m[1].trim(), title: m[2].trim() };
  }
  return null;
}
function splitSuggestion(t) {
  const g = splitGuess(t);
  if (!g) return null;
  const apply = async () => {
    try {
      const r = await api(`/api/tracks/${t.id}`, { method: "PATCH", body: g });
      Object.assign(state.current, r.track);
      toast("Artist and title split", r.undo);
      loadTracks(); renderDetail();
    } catch (e) { fail(e); }
  };
  const no = () => { (state.noSplit ??= new Set()).add(t.id); renderDetail(); };
  return h("div", { class: "suggest" },
    h("span", {}, "No artist. Split into ", h("b", {}, g.artist), " – ", h("b", {}, g.title), "?"),
    h("button", { class: "ghost small", onclick: apply }, "Split"),
    h("button", { class: "ghost small", onclick: no }, "No thanks"));
}

// ── Track panel ──────────────────────────────────────────────────────────────
// Phrase colours: drops in the accent, builds in the cue orange, breaks in the cue blue.
function phraseColor(kind) {
  const k = String(kind || "").toLowerCase();
  if (k.includes("drop") || k.includes("chorus")) return "var(--accent)";
  if (k.includes("build") || k === "up") return withAlpha(CUE_COLORS[1], 0.7);
  if (k.includes("break") || k.includes("bridge")) return withAlpha(CUE_COLORS[5], 0.7);
  return "var(--line-strong)";
}
const secH = (title, ...rest) => h("div", { class: "sec-h" }, title, ...rest);
function renderDetail() {
  const t = state.current;
  const d = $("#detail");
  if (!t) { fill(d); return; }
  const field = (label, key, type = "text") => {
    const inp = h("input", { type, value: t[key] ?? "", step: key === "bpm" ? "0.01" : null });
    inp.addEventListener("change", () => saveField(key, type === "number" ? (inp.value === "" ? null : +inp.value) : inp.value));
    return h("label", {}, label, inp);
  };
  const color = h("select", {}, COLOR_NAMES.map((n, i) => h("option", { value: i, selected: (t.color || 0) === i }, n)));
  color.addEventListener("change", () => saveField("color", +color.value));
  const rating = h("select", {}, [0, 1, 2, 3, 4, 5].map((r) => h("option", { value: r, selected: (t.rating || 0) === r }, r ? "★".repeat(r) : "No rating")));
  rating.addEventListener("change", () => saveField("rating", +rating.value));

  // Hot cues A-H in the DECK001 slot colours: click jumps (or sets at the playhead when empty);
  // right-click, or × in Edit, clears.
  const pads = h("div", { class: "pads", role: "group", "aria-label": "Hot cues (keys 1 to 8)" }, [...PAD].map((letter, i) => {
    const cue = t.cues.find((c) => c.slot === i + 1);
    const pad = h("button", { class: "pad" + (cue ? " set" : ""), style: `--cue:${CUE_COLORS[i]}`,
                              title: cue ? `Jump to ${letter} (key ${i + 1}; right-click to clear)` : `Set ${letter} at the playhead (key ${i + 1})`,
                              "aria-label": cue ? `Hot cue ${letter} at ${fmtMs(cue.start_ms)}` : `Set hot cue ${letter}` }, letter);
    pad.addEventListener("click", () => { if (cue) player.currentTime = cue.start_ms / 1000; else setCue(i + 1, player.currentTime * 1000).catch(fail); });
    pad.addEventListener("contextmenu", (e) => { e.preventDefault(); if (cue) setCue(i + 1, null).catch(fail); });
    return h("div", { class: "pad-w" }, pad, h("span", { class: "pad-t" }, cue ? (cue.label || fmtMs(cue.start_ms)) : "–"),
      cue && state.cueEdit ? h("button", { class: "pad-x", "aria-label": `Clear hot cue ${letter}`, title: "Clear", onclick: () => setCue(i + 1, null).catch(fail) }, "×") : null,
      cue && state.cueEdit ? h("button", { class: "pad-n", "aria-label": `Name hot cue ${letter}`, title: "Name this cue", onclick: () => renameCue(cue) }, "✎") : null);
  }));
  const mems = t.cues.filter((c) => !c.slot).sort((a, b) => a.start_ms - b.start_ms);
  const memList = h("div", { class: "mem" },
    secH("Memory cues", h("span", { class: "spacer" }),
      h("button", { class: "ghost small", onclick: () => addMemory(player.currentTime * 1000).catch(fail), title: "Add at the playhead (M)" }, "+ At playhead")),
    mems.length ? mems.map((c) => h("div", { class: "mem-row" },
      h("span", { class: "m" }, "M"),
      h("span", { class: "t" }, fmtMs(c.start_ms), c.label ? h("span", { class: "muted" }, ` · ${c.label}`) : null),
      h("button", { class: "ghost small", onclick: () => { player.currentTime = c.start_ms / 1000; } }, "Jump"),
      h("button", { class: "ghost small", title: "Name this cue", onclick: () => renameCue(c) }, "Name"),
      h("button", { class: "ghost small icon-btn", "aria-label": `Delete memory cue at ${fmtMs(c.start_ms)}`, title: "Delete",
                    onclick: () => removeMemory(c).catch(fail) }, icon("x", "ic sm"))))
    : h("div", { class: "muted", style: "font-size:12px" }, "None"));

  // Stems: listen to one at a time through the same player (a dot per stem), or make them.
  const sj = jobFor(t.id, "stems");
  let stemsCtl;
  if (t.stem_files.length) {
    const solo = (name, label, c) => h("button", { class: "solo", style: `--c:${c}`, "aria-pressed": String((state.stemSolo || null) === name), "aria-label": `Listen to ${label}`, title: label, onclick: () => soloStem(name) }, h("i"));
    stemsCtl = h("div", { class: "solos", role: "group", "aria-label": "Listen to" }, solo(null, "Full mix", "var(--ink)"),
      STEM_ORDER.filter((s) => t.stem_files.includes(s)).map((s) => solo(s, STEM_LABEL[s][0] + STEM_LABEL[s].slice(1).toLowerCase(), STEM_COLORS[s])));
  } else if (sj?.status === "running") {
    stemsCtl = h("span", { class: "tp-stems" }, h("span", { class: "muted stem-state" }, `Stems ${Math.round(sj.progress * 100)}%`), h("span", { class: "progress" }, h("i", { style: `width:${sj.progress * 100}%` })));
  } else {
    const label = { queued: "Stems queued", failed: "Stems failed. Make them again" }[t.stems] || "Make stems";
    stemsCtl = h("span", { class: "tp-stems" }, h("button", { class: "tb", "aria-label": label, title: label, disabled: t.stems === "queued",
      onclick: () => api("/api/stems", { body: { ids: [t.id] } }).then(() => { pollJobs(); loadTracks(); openTrack(t.id); }).catch(fail) }, icon("deck")),
      t.stems === "failed" ? h("span", { class: "alert-text stem-state" }, "Stems failed") : null);
  }
  const ph = t.phrases, why = issues(t);
  const keyEl = t.camelot ? h("span", { class: "key", style: `color:${keyColor(t.shifted_camelot || t.camelot)}`,
      title: `Key ${t.camelot} ${t.musical_key || ""}${t.key_shift ? `, shifted ${fmtShift(t.key_shift)} on the deck` : ""}` },
    h("b", {}, t.shifted_camelot || t.camelot), h("span", {}, t.shifted_key || t.musical_key || ""), t.key_shift ? h("span", { class: "shift-tag" }, fmtShift(t.key_shift)) : null) : null;
  const gridPill = t.grid ? h("div", { class: "pill-ctl", role: "group", "aria-label": "Beat grid nudge" }, icon("nudge", "ic sm"),
      h("button", { "aria-label": "Earlier by 1 ms ([, Shift for 10 ms)", title: "Earlier by 1 ms ([, Shift+[ for 10 ms)", onclick: (e) => gridNudge(e.shiftKey ? -10 : -1) }, "‹"),
      h("span", { id: "gridReadout", "aria-live": "polite" }, fmtNudge(gridNow(t))),
      h("button", { "aria-label": "Later by 1 ms (], Shift for 10 ms)", title: "Later by 1 ms (], Shift+] for 10 ms)", onclick: (e) => gridNudge(e.shiftKey ? 10 : 1) }, "›"),
      h("button", { class: "tb small", "aria-label": "Move the bar start one beat later", title: "Downbeat +1", onclick: () => gridNudge(0, 1) }, "+1"),
      h("button", { class: "tb small grid-ok", "aria-pressed": String(!!t.grid_ok_at), "aria-label": t.grid_ok_at ? "Beat grid confirmed: click to unconfirm" : "Confirm the beat grid by ear",
        title: t.grid_ok_at ? "Beat grid confirmed by you. Click to take that back" : "I listened: the grid is right", onclick: () => saveField("grid_ok_at", t.grid_ok_at ? null : Date.now() / 1000) }, icon("check", "ic sm")),
      h("button", { class: "tb small", "aria-label": "Back to the analysis grid", title: "Back to the analysis grid", onclick: gridReset }, icon("refresh", "ic sm")))
    : h("span", { class: "muted", title: "The beat grid appears after analysis" }, icon("nudge", "ic sm"));
  const keyPill = h("div", { class: "pill-ctl key", role: "group", "aria-label": "Key shift in semitones: the deck plays the track shifted, the file stays as it is" }, icon("key", "ic sm"),
    h("button", { "aria-label": "Key down one semitone", title: "One semitone down", onclick: () => keyShiftBy(-1) }, "‹"),
    h("span", { id: "shiftReadout", "aria-live": "polite", title: "Key shift on the deck (semitones)" }, keyShiftText(t)),
    h("button", { "aria-label": "Key up one semitone", title: "One semitone up", onclick: () => keyShiftBy(1) }, "›"),
    keyShiftNow(t) ? h("button", { class: "tb small", "aria-label": "No key shift", title: "No key shift", onclick: () => keyShiftBy(-keyShiftNow(t)) }, icon("x", "ic sm")) : null);
  // Mix in / out: where this track enters and leaves a mix. Analysed by default (hollow), yours once you set them
  // (I / O at the playhead, or a click on the readout to jump there). The card gets them for the deck's autopilot.
  const mixBtn = (k, name, key) => {
    const own = !t[`${k}_auto`], v = t[k];
    return h("button", { class: "mix-pt" + (own ? " own" : ""), title: v == null ? `No ${name.toLowerCase()} point yet. Press ${key} to set it at the playhead` : `Mix ${name.toLowerCase()} at ${fmtMs(v)} (${own ? "yours" : "analysed"}). Click to jump; ${key} sets it at the playhead`,
      "aria-label": `Mix ${name.toLowerCase()} ${v == null ? "not set" : "at " + fmtMs(v)}${own ? ", set by you" : ""}`, onclick: () => { if (v != null) player.currentTime = v / 1000; } },
      h("b", {}, name), v == null ? "–" : fmtMs(v));
  };
  const mixPill = h("div", { class: "pill-ctl mix", role: "group", "aria-label": "Mix in and mix out points" }, icon("arrow-right", "ic sm"),
    mixBtn("mix_in_ms", "IN", "I"), mixBtn("mix_out_ms", "OUT", "O"),
    (!t.mix_in_ms_auto || !t.mix_out_ms_auto) ? h("button", { class: "tb small", "aria-label": "Back to the analysed mix points", title: "Back to the analysed mix points", onclick: () => setMixPoints({ mix_in_ms: null, mix_out_ms: null }) }, icon("refresh", "ic sm")) : null);
  const lines = [splitSuggestion(t),
    why.length ? h("div", { class: "warn-line" }, icon("alert", "ic sm"), h("span", {}, why.join(" · "), t.deck_problem ? ". The original is still exported (CDJs play it)." : ""),
      t.deck_problem ? h("button", { class: "small", onclick: () => convertForDeck([t.id]), title: "Makes a 16-bit FLAC copy for the deck. The original stays as it is." }, "Convert for the deck") : null) : null,
    t.deck_copy ? h("div", { class: "ok-line" }, icon("check", "ic sm"),
      h("span", {}, `The deck gets a 16-bit FLAC copy${t.deck_copy_bytes ? ` (${fmtBytes(t.deck_copy_bytes)})` : ""} instead of the original.`),
      h("button", { class: "ghost small", onclick: () => removeDeckCopy([t.id]) }, "Use the original")) : null].filter(Boolean);
  fill(d,
    h("div", { class: "tp-head" },
      h("button", { class: "tb", "aria-label": "Back to the library (Esc)", title: "Back (Esc)", onclick: () => setFocus(false) }, icon("arrow-left")),
      t.has_artwork ? h("img", { class: "tp-art", src: `/api/tracks/${t.id}/artwork`, alt: "" }) : h("div", { class: "tp-art", "aria-hidden": "true" }, icon("library")),
      h("div", { class: "tp-id" },
        h("div", { class: "tp-title", title: t.path }, t.title || ""),
        h("div", { class: "tp-artist" }, t.artist || "Unknown artist")),
      h("div", { class: "tp-meta" },
        t.bpm ? h("b", { title: "BPM" }, fmtBpm(t.bpm)) : null, keyEl, moodPicker(t), energyBars(t.energy),
        intentPicker(t),
        t.lufs != null ? h("span", { class: "muted", title: `${t.lufs.toFixed(1)} LUFS. The export writes ${fmtGain(t)} for the deck (target ${state.settings?.target_lufs ?? -9} LUFS).` }, `${t.lufs.toFixed(1)} LUFS`) : null,
        h("span", { class: "muted", title: "Length" }, fmtTime(t.duration))),
      h("button", { class: "tb", "aria-label": "Track actions", title: "Actions",
        onclick: (e) => openMenu(e.currentTarget, [...trackMenuItems(t), { icon: "arrow-left", label: "Close track", hint: "Esc", run: closeDetail }]) }, icon("more"))),
    lines.length ? h("div", { class: "tp-lines" }, lines) : null,
    h("section", { "aria-label": "Waveforms", class: "tp-waves" }, waveStack),
    t.sections?.length ? h("div", { class: "phrases", role: "group", "aria-label": "Sections and phrases" }, t.sections.map((s) => h("button", { class: "phrase", style: `flex:${Math.max(1, s.end_ms - s.start_ms)} 1 0`,
        title: `${s.kind} at ${fmtMs(s.start_ms)}${s.bars ? ` · ${s.bars} bars` : ""}`, onclick: () => { player.currentTime = s.start_ms / 1000; } },
      h("i", { style: `background:${phraseColor(s.kind)}` }), h("b", {}, s.kind)))) : null,
    h("div", { class: "tp-tools" },
      h("span", { class: "wave-time", id: "waveTime", "aria-hidden": "true" }, waveTimeText()),
      pads,
      h("button", { class: "tb", "aria-pressed": String(!!state.cueEdit), "aria-label": "Edit cues: clear or name them, memory cues", title: "Edit cues",
        onclick: () => { state.cueEdit = !state.cueEdit; renderDetail(); } }, icon("pencil")),
      mixPill, gridPill, keyPill, stemsCtl,
      h("span", { class: "spacer" }),
      h("span", { class: "zoom", role: "group", "aria-label": "Waveform zoom" },
        h("button", { "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoomWave(0.5) }, "−"),
        h("button", { "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoomWave(2) }, "+")),
      h("button", { class: "tb", "aria-pressed": String(!!state.detailsOpen), "aria-label": "Track details: tags, rating, phrases, why this mood", title: "Track details",
        onclick: () => { state.detailsOpen = !state.detailsOpen; renderDetail(); } }, icon("info"))),
    state.cueEdit ? memList : null,
    state.detailsOpen ? h("div", { class: "details" },
      h("div", { class: "fields" },
        field("Title", "title"), field("Artist", "artist"), field("Album", "album"),
        field("Genre", "genre"), field("BPM", "bpm", "number"), field("Year", "year", "number"),
        h("label", {}, "Colour", color), h("label", {}, "Rating", rating), field("Comment", "comment")),
      moodWhy(t),
      ph ? h("dl", { class: "kv", style: "margin-top:10px" },
        h("dt", {}, "First downbeat"), h("dd", {}, fmtMs(ph.first_downbeat_ms)),
        h("dt", {}, "Intro ends"), h("dd", {}, fmtMs(ph.intro_end_ms)),
        h("dt", {}, "Outro starts"), h("dd", {}, fmtMs(ph.outro_start_ms)),
        h("dt", {}, plural(ph.drops.length, "drop")), h("dd", {}, ph.drops.map(fmtMs).join(", ") || "–")) : null) : null);
  syncMarkers();
}

// Intent: how the track will be used, with what that prepares in the tooltip.
function intentPicker(t) {
  const sel = h("select", { class: "intent-sel", "aria-label": "How you'll use this track", title: INTENTS.map(([, l, a]) => `${l}: ${a}`).join("\n") },
    INTENTS.map(([id, label]) => h("option", { value: id, selected: (t.intent || "mix") === id }, label)));
  sel.addEventListener("change", () => setIntent([t.id], sel.value));
  return sel;
}
// Mood in the track panel: the label and a menu to set it. Picking one teaches the automatic
// moods (mood.rs); "Automatic" hands the track back to the analysis.
function moodPicker(t) {
  if (!t.analysed_at || !MOODS.length) return null;
  const sel = h("select", { "aria-label": "Mood", class: "mood-sel" },
    h("option", { value: "", selected: !t.mood_manual }, "Auto"),
    MOODS.map((m) => h("option", { value: m.id, selected: !!t.mood_manual && t.mood === m.id }, m.label)));
  sel.addEventListener("change", () => saveMood(sel.value || null));
  return h("div", { class: "tp-mood" }, moodPill(t), sel);
}
async function saveMood(id) {
  try {
    const body = id ? { mood: id, mood_manual: 1 } : { mood_manual: 0 };
    const r = await api(`/api/tracks/${state.current.id}`, { method: "PATCH", body });
    const m = moodOf(r.track.mood);
    toast(id ? `Mood: ${m?.label}` : `Mood: automatic (${m?.label || "none"})`, r.undo);
    await openTrack(state.current.id);
    loadTracks();
  } catch (e) { fail(e); }
}
// Why: the three nearest moods and the measured numbers they were compared on.
const FEATURE_LABEL = { bpm: "Tempo", lufs: "Loudness", bass: "Bass", bright: "Treble", drums: "Drums", melody: "Melody",
                        vocals: "Vocals", breaks: "Breakdowns", minor: "Key" };
const FEATURE_FMT = {
  bpm: (v) => `${fmtBpm(v)} BPM`, lufs: (v) => `${v.toFixed(1)} LUFS`, minor: (v) => (v >= 0.5 ? "minor" : "major"),
};
function moodWhy(t) {
  const w = t.mood_why;
  if (!w?.ranking) return null;
  const pct = (v) => `${Math.round(v * 100)} %`;
  return h("div", { class: "mood-why" },
    h("div", { class: "sec-h" }, "Why this mood"),
    h("div", { class: "why-rank" }, w.ranking.map((r) => {
      const m = moodOf(r.id);
      return h("span", { class: "mood", style: `--mood:${MOOD_COLOR[r.id]}`, title: r.distance == null ? "Chance from the Needle Tags model (an embedding of the sound)" : `Distance ${r.distance} (0 = exactly typical)` },
        h("i"), m?.label || r.id, h("span", { class: "muted" }, ` ${pct(r.confidence)}`));
    })),
    h("dl", { class: "kv" }, MOOD_FEATURES.map((f) => {
      const v = w.features?.[f.id];
      return [h("dt", { title: f.about[0].toUpperCase() + f.about.slice(1) }, FEATURE_LABEL[f.id] || f.id),
              h("dd", {}, v == null ? (f.id === "vocals" ? "needs stems" : "–") : (FEATURE_FMT[f.id] || pct)(v))];
    }).flat()),
  );
}

// ── Preview bar: listen to check a track; it never changes anything ─────────
try { state.volume = Math.max(0, Math.min(1, +(localStorage.getItem("volume") ?? 1))); } catch { state.volume = 1; }
if (!Number.isFinite(state.volume)) state.volume = 1;
function applyRate() { player.playbackRate = player.defaultPlaybackRate = 1; player.volume = state.volume; }
player.addEventListener("loadedmetadata", applyRate);
applyRate();
$("#vol").value = state.volume;
$("#vol").addEventListener("input", () => { state.volume = +$("#vol").value; player.volume = state.volume; try { localStorage.setItem("volume", String(state.volume)); } catch {} });
$("#seek").addEventListener("input", () => { state.seeking = true; if (state.current) player.currentTime = +$("#seek").value; });
$("#seek").addEventListener("change", () => { state.seeking = false; });
$("#playBtn").addEventListener("click", () => { if (state.current) togglePlay(); });
$("#pvPrev").addEventListener("click", () => stepTrack(-1));
$("#pvNext").addEventListener("click", () => stepTrack(1));
function updatePreviewBar() {
  const t = state.current;
  $(".preview-bar").classList.toggle("idle", !t);
  $("#pvTitle").textContent = t ? (t.title || "Untitled") : "No track selected";
  $("#pvArtist").textContent = t ? (t.artist || "Unknown artist") : "";
  $("#pvArt").style.backgroundImage = t?.has_artwork ? `url(/api/tracks/${t.id}/artwork)` : "";
  $("#durLabel").textContent = fmtTime(t?.duration || 0);
  $("#seek").max = t?.duration || 0;
  if (!t) { $("#seek").value = 0; $("#timeLabel").textContent = "0:00"; }
}
// Previous goes back to the start first (after 3 s), like a deck; both follow the table's order.
function stepTrack(dir) {
  if (dir < 0 && player.currentTime > 3) { player.currentTime = 0; return; }
  const v = state.view, i = v.findIndex((x) => x.id === state.current?.id), n = v[i + dir];
  if (!n) return;
  state.selected = new Set([n.id]); state.anchor = n.id;
  openTrack(n.id, !player.paused).catch(fail);
}
async function saveField(key, value) {
  try {
    const r = await api(`/api/tracks/${state.current.id}`, { method: "PATCH", body: { [key]: value } });
    Object.assign(state.current, r.track);
    toast(key === "grid_ok_at" ? (value ? "Beat grid confirmed" : "Beat grid unconfirmed") : "Saved", r.undo);
    loadTracks();
    if (state.focus && key === "grid_ok_at") renderDetail();
  } catch (e) { fail(e); }
}
// All cue edits go through one PUT of the whole list. A user edit makes every remaining
// cue a "user" cue, so re-analysis keeps them.
async function putCues(cues, message) {
  cues.forEach((c) => (c.auto = 0));
  const r = await api(`/api/tracks/${state.current.id}/cues`, { method: "PUT", body: cues });
  state.current.cues = r.cues;
  toast(message, r.undo);
  renderDetail();
}
// Mix points: saved per track (NULL = the analysed point), beat-snapped like a cue, one undo.
const MIX = [["mix_in_ms", "IN"], ["mix_out_ms", "OUT"]];
async function setMixPoints(fields) {
  const t = state.current;
  if (!t) return;
  try {
    const r = await api(`/api/tracks/${t.id}`, { method: "PATCH", body: fields });
    const what = Object.entries(fields).map(([k, v]) => `${k === "mix_in_ms" ? "Mix in" : "Mix out"} ${v == null ? "analysed" : "at " + fmtMs(v)}`).join(", ");
    toast(what, r.undo);
    await openTrack(t.id);
  } catch (e) { fail(e); }
}
// Cue names live in DKPRP (the pads, the waveform) and go to the deck as hints in POCKETDECK/analysis; the rekordbox files
// have no field for them.
async function renameCue(cue) {
  const name = await ask(cue.slot ? `Name for hot cue ${PAD[cue.slot - 1]}` : "Name for this memory cue", cue.label || "", "Save");
  if (name == null) return;
  const cues = state.current.cues.map((c) => ({ ...c, label: c.slot === cue.slot && c.start_ms === cue.start_ms ? (name || null) : c.label }));
  putCues(cues, name ? `Cue named “${name}”` : "Cue name cleared").catch(fail);
}
function setCue(slot, ms) {
  const cues = state.current.cues.filter((c) => c.slot !== slot).map((c) => ({ ...c }));
  if (ms != null) cues.push({ slot, start_ms: Math.round(snapToBeat(ms)), label: null, auto: 0 });
  return putCues(cues, ms == null ? `Cleared hot cue ${PAD[slot - 1]}` : `Hot cue ${PAD[slot - 1]} set`);
}
function addMemory(ms) {
  const cues = state.current.cues.map((c) => ({ ...c }));
  cues.push({ slot: 0, start_ms: Math.round(snapToBeat(ms)), label: null, auto: 0 });
  return putCues(cues, `Memory cue added at ${fmtMs(snapToBeat(ms))}`);
}
function removeMemory(cue) {
  return putCues(state.current.cues.filter((c) => c !== cue).map((c) => ({ ...c })), "Memory cue deleted");
}
function snapToBeat(ms) {
  const g = effectiveGrid(state.current);
  if (!g) return ms;
  const n = Math.round((ms - g.first_beat_ms) / g.period_ms);
  const snapped = g.first_beat_ms + n * g.period_ms;
  return Math.abs(snapped - ms) < g.period_ms / 4 ? snapped : ms;
}
// ── Beat grid nudge ──────────────────────────────────────────────────────────
// The saved nudge (grid_offset_ms, downbeat_shift) is already in t.grid from the server. While
// the user is nudging, the change shows at once and is saved 0.6 s after the last step, as one
// edit with one undo. The export writes the same grid to the card.
function shiftGrid(g, dOff, dShift) {
  if (!g || !g.period_ms || (!dOff && !dShift)) return g;
  const moved = g.first_beat_ms + dOff, k = Math.floor(moved / g.period_ms);
  return { ...g, first_beat_ms: moved - k * g.period_ms, downbeat_index: (((g.downbeat_index + k + dShift) % 4) + 4) % 4 };
}
function effectiveGrid(t) {
  const p = state.gridPending;
  if (!t?.grid || !p || p.id !== t.id) return t?.grid;
  return shiftGrid(t.grid, p.off - (t.grid_offset_ms || 0), p.shift - (t.downbeat_shift || 0));
}
const gridNow = (t) => (state.gridPending?.id === t.id ? state.gridPending : { off: t.grid_offset_ms || 0, shift: t.downbeat_shift || 0 });
const fmtNudge = (n) => `${n.off > 0 ? "+" : n.off < 0 ? "−" : "±"}${Math.abs(n.off).toFixed(1)} ms` + (n.shift ? ` · downbeat +${n.shift}` : "");
function gridNudge(dMs, dBeat = 0) {
  const t = state.current;
  if (!t?.grid) return toast("Analyse the track first: there's no beat grid to nudge", null, true);
  const p = state.gridPending?.id === t.id ? state.gridPending : (state.gridPending = { id: t.id, ...gridNow(t) });
  p.off = Math.max(-1000, Math.min(1000, Math.round((p.off + dMs) * 10) / 10));
  p.shift = (((p.shift + dBeat) % 4) + 4) % 4;
  syncMarkers();
  const r = $("#gridReadout"); if (r) r.textContent = fmtNudge(p);
  clearTimeout(state.gridT); state.gridT = setTimeout(() => saveGrid().catch(fail), 600);
}
function gridReset() {
  const t = state.current;
  if (!t) return;
  const n = gridNow(t);
  gridNudge(-n.off, -n.shift);
}
async function saveGrid() {
  clearTimeout(state.gridT);
  const p = state.gridPending;
  if (!p) return;
  state.gridPending = null;
  const r = await api(`/api/tracks/${p.id}`, { method: "PATCH", body: { grid_offset_ms: p.off, downbeat_shift: p.shift } });
  toast(`Beat grid ${fmtNudge(p)}`, r.undo);
  if (state.current?.id === p.id) await openTrack(p.id);   // fetch the grid as saved (and exported)
}
// Key shift: semitones (−12..+12) the deck shifts the track by while it plays it. Saved per track, written to the
// drive in POCKETDECK/analysis (docs/DECK_KEY_SHIFT.md); the audio file is not changed and nothing plays here.
const fmtShift = (n) => `${n > 0 ? "+" : n < 0 ? "−" : "±"}${Math.abs(n)}`;
const keyShiftNow = (t) => (state.keyPending?.id === t.id ? state.keyPending.n : t.key_shift || 0);
const keyShiftText = (t) => {
  const n = keyShiftNow(t);
  const to = state.keyPending ? "" : t.shifted_camelot ? ` → ${t.shifted_camelot} ${t.shifted_key}` : "";
  return `${fmtShift(n)}${to}`;
};
function keyShiftBy(d) {
  const t = state.current;
  if (!t) return;
  const n = Math.max(-12, Math.min(12, keyShiftNow(t) + d));
  state.keyPending = { id: t.id, n };
  const r = $("#shiftReadout"); if (r) r.textContent = keyShiftText(t);
  clearTimeout(state.keyT); state.keyT = setTimeout(() => saveKeyShift().catch(fail), 600);   // one saved edit per burst of clicks
}
async function saveKeyShift() {
  clearTimeout(state.keyT);
  const p = state.keyPending;
  if (!p) return;
  const r = await api(`/api/tracks/${p.id}`, { method: "PATCH", body: { key_shift: p.n } });
  state.keyPending = null;
  toast(`Key shift ${fmtShift(p.n)}`, r.undo);
  loadTracks();
  if (state.current?.id === p.id) await openTrack(p.id);   // the shifted key as the server worked it out
}
function togglePlay() { player.paused ? player.play().catch(() => {}) : player.pause(); }
function soloStem(name) {
  const t = state.current;
  const pos = player.currentTime, playing = !player.paused;
  state.stemSolo = name;
  player.src = name ? `/api/tracks/${t.id}/stem/${name}` : `/api/tracks/${t.id}/audio`;
  player.addEventListener("loadedmetadata", () => { player.currentTime = pos; if (playing) player.play(); }, { once: true });
  const focused = document.activeElement === waveStack;
  renderDetail();
  markActiveLane();
  if (focused) waveStack.focus({ preventScroll: true });
}
function setPlayIcon() {
  const b = $("#playBtn");
  b.querySelector("use").setAttribute("href", player.paused ? "#i-play" : "#i-pause");
  b.setAttribute("aria-label", player.paused ? "Play (Space)" : "Pause (Space)");
}
player.addEventListener("play", setPlayIcon);
player.addEventListener("pause", setPlayIcon);
player.addEventListener("timeupdate", () => {
  if (!state.current) return;
  $("#timeLabel").textContent = fmtTime(player.currentTime);
  const wt = $("#waveTime"); if (wt) wt.textContent = waveTimeText();
  if (!state.seeking) $("#seek").value = player.currentTime;
});

// ── Waveforms ────────────────────────────────────────────────────────────────
// wavesurfer.js (ui/vendor/wavesurfer, BSD-3-Clause) scrolls the detail view and the overview
// (Minimap plugin) and follows the shared <audio> player; beat grid and cues are Regions markers.
// The look follows the DECK001 waveform settings: a colour theme and a style, the same names
// and colours as the deck ("DECK001 Waveform Themes"), so a track looks the same on both.
const WAVE_THEMES = {
  coral_teal: { name: "Coral / Teal", solid: "#f0574c", frequency: ["#19b5bd", "#f0574c", "#fff4ef"], rainbow: ["#f0574c", "#19b5bd", "#ffffff"] },
  graphite:   { name: "Graphite",     solid: "#c9c6c0", frequency: ["#4a4845", "#8f8b85", "#f4f1ea"], rainbow: ["#6b6864", "#b4b0a9", "#ffffff"] },
  valve:      { name: "Valve",        solid: "#e8a33d", frequency: ["#a8452a", "#d9762b", "#f6d58e"], rainbow: ["#b4432a", "#f0a03a", "#fbeac0"] },
};
// Contour styles draw separate bars, the others one column per pixel. Band palettes are
// [bass, mids, highs]; "Contour · frequency" uses the rainbow colours, as on the deck.
const WAVE_STYLES = {
  contour_solid:     { name: "Contour · solid",     bars: true,  palette: "solid" },
  contour_frequency: { name: "Contour · frequency", bars: true,  palette: "rainbow" },
  frequency:         { name: "Frequency",           bars: false, palette: "frequency" },
  rainbow:           { name: "Rainbow",             bars: false, palette: "rainbow" },
};
let waveTheme = "coral_teal", waveStyle = "contour_frequency";
try {
  if (WAVE_THEMES[localStorage.getItem("waveTheme")]) waveTheme = localStorage.getItem("waveTheme");
  if (WAVE_STYLES[localStorage.getItem("waveStyle")]) waveStyle = localStorage.getItem("waveStyle");
} catch {}
function wavePalette() {
  const th = WAVE_THEMES[waveTheme], st = WAVE_STYLES[waveStyle];
  return { bars: st.bars, solid: st.palette === "solid" ? th.solid : null, bands: st.palette === "solid" ? null : th[st.palette] };
}
// Stem rows: each stem in its own colour (the DECK001 cue colours), in the chosen style.
const STEM_ORDER = ["drums", "bass", "other", "vocals"];
const STEM_LABEL = { drums: "DRUMS", bass: "BASS", other: "MELODY", vocals: "VOCALS" };
const STEM_COLORS = { drums: CUE_COLORS[1], bass: CUE_COLORS[2], other: CUE_COLORS[5], vocals: CUE_COLORS[6] };

function cssVar(n) { return cssCache[n] ??= getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const waveStack = h("div", { class: "wstack", tabindex: "0", role: "group",
  "aria-label": "Waveforms. Up and Down choose the full mix or a stem to listen to, Left and Right move the playhead (Shift: 10 s, Alt: a beat, Ctrl: a bar), Page Up and Page Down jump between cues, I and O set the mix in and out points, double-click sets the next hot cue." });
const fmtClock = (s) => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`; };
function waveTimeText() { const t = state.current; return t ? `${fmtClock(player.currentTime)} / ${fmtClock(t.duration)}` : ""; }
const waveBox = h("div", { class: "waves", id: "waves" });
const overviewBox = h("div", { class: "waves overview" });
const withAlpha = (hex, a) => { const m = hex.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i); return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : hex; };

// PWV5 (150 columns per second: height plus a 3-bit bass/mid/high mix) as three bands.
function wavePeaks(w) {
  const n = w.detail.length / 2, lo = new Float32Array(n), md = new Float32Array(n), hi = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = (w.detail[i * 2] << 8) | w.detail[i * 2 + 1];
    // Bits: bass 13-15, mids 10-12, highs 7-9 (each 0-7, the loudest band is 7), height 2-6.
    const r = (v >> 13) & 7, g = (v >> 10) & 7, b = (v >> 7) & 7, ht = ((v >> 2) & 31) / 31;
    lo[i] = (ht * r) / 7; md[i] = (ht * g) / 7; hi[i] = (ht * b) / 7;
  }
  return [lo, md, hi];
}

// Draw `bands` (one array per band, 0-1) into a canvas. In each column or bar the tallest band
// is drawn first and the smallest last, so all bands stay visible (the deck draws the same way).
// WaveSurfer passes a custom renderer at most two channels, so it gets one channel of column
// positions (index / (N-1); WaveSurfer keeps peaks in 0-1) and the bands are looked up by index.
// That keeps WaveSurfer's slicing for scrolling, zoom and the minimap.
function drawWave(idx, ctx, bands, pal) {
  const W = ctx.canvas.width, H = ctx.canvas.height, c = H / 2, n = idx?.length || 0;
  if (!n || !W || !bands?.[0]?.length) return;
  const last = bands[0].length - 1, dpr = Math.max(1, Math.round(devicePixelRatio || 1));
  const pitch = pal.bars ? 5 * dpr : 1, bw = pal.bars ? 3 * dpr : 1, scale = pal.bars ? 0.9 : 1;
  const k = bands.length, colors = pal.solid ? [pal.solid] : pal.bands;
  const paths = Array.from({ length: k }, () => Array.from({ length: k }, () => new Path2D()));
  const v = new Float32Array(k);
  for (let x = 0; x < W; x += pitch) {
    const a = Math.floor((x * n) / W), b = Math.max(a + 1, Math.floor((Math.min(W, x + pitch) * n) / W));
    v.fill(0);
    for (let j = a; j < b && j < n; j++) { const i = Math.round(idx[j] * last); for (let q = 0; q < k; q++) if (bands[q][i] > v[q]) v[q] = bands[q][i]; }
    if (pal.solid) {
      const hh = Math.max(...v) * c * scale;
      if (hh > 0.5) paths[0][0].rect(x, c - hh, bw, hh * 2);
      continue;
    }
    const order = [...Array(k).keys()].sort((p, q) => v[q] - v[p]);
    order.forEach((band, rank) => { const hh = v[band] * c * scale; if (hh > 0.5) paths[rank][band].rect(x, c - hh, bw, hh * 2); });
  }
  if (pal.solid) { ctx.fillStyle = pal.solid; ctx.fill(paths[0][0]); return; }
  for (let rank = 0; rank < k; rank++) for (let band = 0; band < k; band++) { ctx.fillStyle = colors[band]; ctx.fill(paths[rank][band]); }
}

function destroyWaves() {
  state.ws?.destroy(); (state.stemWs || []).forEach((w) => w.destroy());
  state.ws = null; state.stemWs = []; state.regions = null; state.stemRegions = [];
}
// Which lane is the one you listen to: the full mix, or a stem (the same as the "Listen to" buttons).
const waveLanes = () => [null, ...STEM_ORDER.filter((s) => state.stemPeaks?.[s])];
function markActiveLane() {
  const act = state.stemSolo || "full";
  $$(".wlabel", waveStack).forEach((l) => { const on = (l.dataset.lane || "") === act; l.classList.toggle("active", on); l.setAttribute("aria-current", String(on)); l.nextElementSibling?.classList.toggle("active", on); });
}
// Double-click a waveform: the next free hot cue goes there (snapped to the beat grid like every cue).
function nextHotCueAt(seconds) {
  const t = state.current;
  const slot = [1, 2, 3, 4, 5, 6, 7, 8].find((s) => !t.cues.some((c) => c.slot === s));
  if (!slot) return toast("All eight hot cues are in use. Clear one first.", null, true);
  setCue(slot, seconds * 1000).catch(fail);
}
// Keys in the waveforms: Up/Down choose the lane you listen to; Left/Right move the playhead (1 s, Shift 10 s, Alt one beat,
// Ctrl one bar); Page Up / Page Down jump to the previous / next cue.
waveStack.addEventListener("keydown", (e) => {
  const t = state.current;
  if (!t || e.target !== waveStack || ["Tab"].includes(e.key)) return;
  const lanes = waveLanes(), i = Math.max(0, lanes.indexOf(state.stemSolo ?? null));
  const go = (ms) => { player.currentTime = Math.max(0, Math.min(t.duration || 0, ms / 1000)); };
  const beat = effectiveGrid(t)?.period_ms || 500, now = player.currentTime * 1000;
  if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    e.preventDefault();
    const n = Math.max(0, Math.min(lanes.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
    if (n !== i) soloStem(lanes[n]);
  } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    const step = e.ctrlKey || e.metaKey ? beat * 4 : e.altKey ? beat : e.shiftKey ? 10000 : 1000;
    go(now + (e.key === "ArrowRight" ? step : -step));
  } else if (e.key === "PageUp" || e.key === "PageDown") {
    e.preventDefault();
    const at = t.cues.map((c) => c.start_ms).sort((a, b) => a - b);
    const target = e.key === "PageDown" ? at.find((ms) => ms > now + 50) : [...at].reverse().find((ms) => ms < now - 500);
    if (target != null) go(target); else toast(e.key === "PageDown" ? "No later cue" : "No earlier cue");
  }
});
const hoverPlugin = () => WaveSurfer.Hover.create({ lineColor: "rgba(242,242,242,.7)", lineWidth: 1, labelBackground: "rgba(0,0,0,.78)", labelColor: "#fff", labelSize: 11,
  formatTimeCallback: fmtClock });
function buildWaves() {
  destroyWaves();
  const t = state.current, w = state.wave, pal = wavePalette();
  fill(waveStack, h("span", { class: "wlabel", "data-lane": "full", style: `color:${pal.solid || pal.bands[0]}`, title: "Full mix (click to listen to it)", onclick: () => soloStem(null) }, h("i")), waveBox);
  if (!w) { waveBox.replaceChildren(h("div", { class: "wave-empty" }, t?.analysed_at ? "Loading waveform…" : "Waveform appears after analysis")); return; }
  waveBox.replaceChildren(); overviewBox.replaceChildren();
  state.bands = wavePeaks(w);
  const positions = (n) => Float32Array.from({ length: n }, (_, i) => i / Math.max(1, n - 1));
  // The played part is dimmed, like the deck (the progress layer is the drawing filled in 50 % black).
  const base = (renderFunction, height) => ({ media: player, duration: w.duration, renderFunction, waveColor: "#888888", progressColor: "rgba(0,0,0,0.5)",
    height, minPxPerSec: state.zoom, autoScroll: true, autoCenter: true, hideScrollbar: true, dragToSeek: true, normalize: false,
    cursorColor: PLAYHEAD, cursorWidth: 2 });
  const full = (ch, ctx) => drawWave(ch[0], ctx, state.bands, pal);
  state.regions = WaveSurfer.Regions.create();
  state.stemRegions = [];
  const clockLabel = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  // The full mix takes the height the stack has left after the stem lanes and the overview (120-320 px).
  const stemLanes = STEM_ORDER.filter((n) => state.stemPeaks?.[n]).length;
  const fullH = Math.max(120, Math.min(320, (waveStack.clientHeight || 0) - 20 - stemLanes * 47 - 29 - 18 - 4));
  state.ws = WaveSurfer.create({ container: waveBox, peaks: [positions(state.bands[0].length)], ...base(full, fullH),
    plugins: [state.regions, hoverPlugin(), WaveSurfer.Timeline.create({ height: 18, formatTimeCallback: clockLabel, secondaryLabelOpacity: 0.35,
      style: { fontSize: "10px", color: "#b6b6bb", fontFamily: "inherit" } }), WaveSurfer.Minimap.create({ container: overviewBox, height: 26, renderFunction: full, waveColor: "#888888", progressColor: "rgba(0,0,0,0.5)",
      dragToSeek: true, cursorColor: PLAYHEAD, overlayColor: "rgba(255,255,255,.12)" })] });
  state.ws.on("ready", syncMarkers);
  state.ws.on("dblclick", (rx) => nextHotCueAt(rx * w.duration));
  waveBox.append(h("span", { class: "lane-tip" }, "FULL MIX"));
  const overviewRow = () => { waveStack.append(h("span", { class: "wlabel", style: "color:#8a8a90", title: "Overview" }, h("i")), overviewBox); overviewBox.append(h("span", { class: "lane-tip" }, "OVERVIEW")); markActiveLane(); };
  if (!state.stemPeaks) return overviewRow();
  for (const name of STEM_ORDER) {
    const peaks = state.stemPeaks[name];
    if (!peaks) continue;
    const box = h("div", { class: "waves" });
    const label = h("span", { class: "wlabel", "data-lane": name, style: `color:${STEM_COLORS[name]}`, title: `${STEM_LABEL[name][0]}${STEM_LABEL[name].slice(1).toLowerCase()} (click to listen to it alone)`, onclick: () => soloStem(name) }, h("i"));
    waveStack.append(label, box);
    const sp = { bars: pal.bars, solid: STEM_COLORS[name] };
    const reg = WaveSurfer.Regions.create();
    state.stemRegions.push(reg);
    const sws = WaveSurfer.create({ container: box, peaks: [positions(peaks.length)], ...base((ch, ctx) => drawWave(ch[0], ctx, [peaks], sp), 44), plugins: [reg, hoverPlugin()] });
    sws.on("dblclick", (rx) => nextHotCueAt(rx * w.duration));
    sws.on("ready", syncMarkers);
    box.append(h("span", { class: "lane-tip" }, STEM_LABEL[name]));
    state.stemWs.push(sws);
  }
  overviewRow();
}
function zoomWave(f) {
  state.zoom = Math.max(25, Math.min(800, state.zoom * f));
  for (const w of [state.ws, ...(state.stemWs || [])]) { try { w?.zoom(state.zoom); } catch {} }
}

// Beat grid (faint beats, solid downbeats, bar numbers every 8 bars), then hot cues (A-H) and
// memory cues (M). The waveform panel is always dark, so the lines are light.
function syncMarkers() {
  const t = state.current, g = effectiveGrid(t), reg = state.regions;
  if (!reg || !state.wave) return;
  reg.clearRegions();
  // The stem lanes carry the hot and memory cues and the downbeats, so a cue can be placed by what you see in a stem.
  for (const sr of state.stemRegions || []) {
    sr.clearRegions();
    if (g && g.period_ms > 0 && (g.duration_ms - g.first_beat_ms) / g.period_ms < 3000) {
      const beats = Math.floor((g.duration_ms - g.first_beat_ms) / g.period_ms) + 1;
      for (let k = 0; k < beats; k++) if (((k - g.downbeat_index) % 4 + 4) % 4 === 0) sr.addRegion({ start: (g.first_beat_ms + k * g.period_ms) / 1000, color: "rgba(242,242,242,0.28)", drag: false, resize: false });
    }
    for (const c of t.cues) sr.addRegion({ start: c.start_ms / 1000, color: c.slot ? CUE_COLORS[c.slot - 1] : "#f2f2f2", drag: false, resize: false });
    for (const [k] of MIX) if (t[k] != null) sr.addRegion({ start: t[k] / 1000, color: "rgba(242,242,242,0.6)", drag: false, resize: false });
  }
  if (g && g.period_ms > 0) {
    const down = "rgba(242,242,242,0.55)", beat = "rgba(242,242,242,0.16)";
    const beats = Math.floor((g.duration_ms - g.first_beat_ms) / g.period_ms) + 1;
    const everyBeat = beats < 3000;           // very long files (mixes): downbeats only
    let bar = 0;
    for (let k = 0; k < beats; k++) {
      const isDown = ((k - g.downbeat_index) % 4 + 4) % 4 === 0;
      if (isDown) bar++;
      if (!isDown && !everyBeat) continue;
      const label = isDown && bar % 8 === 1 ? h("span", { style: "font:600 10px var(--f-text);color:#b6b6bb;padding:0 3px" }, String(bar)) : undefined;
      reg.addRegion({ start: (g.first_beat_ms + k * g.period_ms) / 1000, color: isDown ? down : beat, content: label, drag: false, resize: false });
    }
  }
  for (const c of t.cues) {
    // The waveform lives in a shadow root, so the label is styled inline.
    const col = c.slot ? CUE_COLORS[c.slot - 1] : "#f2f2f2";
    const label = h("span", { style: `display:inline-block;font:600 11px var(--f-text);padding:0 4px;border-radius:0 2px 2px 0;background:${col};color:#111112` },
      c.slot ? PAD[c.slot - 1] : "M");
    reg.addRegion({ start: c.start_ms / 1000, color: col, content: label, drag: false, resize: false });
  }
  // Mix in / out: white flags, solid when they are yours, hollow when analysed.
  for (const [k, name] of MIX) {
    if (t[k] == null) continue;
    const own = !t[`${k}_auto`];
    const label = h("span", { style: `display:inline-block;font:600 10px var(--f-text);letter-spacing:.04em;padding:0 4px;border-radius:0 2px 2px 0;${own ? "background:#f2f2f2;color:#111112" : "border:1px solid #f2f2f2;color:#f2f2f2;background:rgba(0,0,0,.5)"}` }, name);
    reg.addRegion({ start: t[k] / 1000, color: own ? "#f2f2f2" : "rgba(242,242,242,0.6)", content: label, drag: false, resize: false });
  }
}

// Settings: waveform theme and style, with a preview drawn the same way as the real waveform.
function drawWavePreview() {
  const cv = $("#wavePreview"), N = 420, dpr = devicePixelRatio || 1;
  if (cv.clientWidth) { cv.width = Math.round(cv.clientWidth * dpr); cv.height = Math.round(cv.clientHeight * dpr); }  // real pixels, so bars match the waveform
  const ctx = cv.getContext("2d");
  const env = (t) => (t < 0.15 ? 0.4 : t < 0.3 ? 0.62 : t < 0.55 ? 0.95 : t < 0.68 ? 0.38 : t < 0.9 ? 0.97 : 0.45);
  const wob = (i, s) => 0.55 + 0.45 * Math.abs(Math.sin(i * 0.9 + s) * Math.cos(i * 0.23 + s * 2));
  const bands = [0, 1, 2].map((q) => Float32Array.from({ length: N }, (_, i) => env(i / N) * wob(i, q * 1.7) * [1, 0.7, 0.35][q]));
  ctx.fillStyle = "#0b0b0d"; ctx.fillRect(0, 0, cv.width, cv.height);
  drawWave(Float32Array.from({ length: N }, (_, i) => i / (N - 1)), ctx, bands, wavePalette());
}
function initWaveSettings() {
  fill($("#waveTheme"), Object.entries(WAVE_THEMES).map(([k, v]) => h("option", { value: k, selected: k === waveTheme }, v.name)));
  fill($("#waveStyle"), Object.entries(WAVE_STYLES).map(([k, v]) => h("option", { value: k, selected: k === waveStyle }, v.name)));
  const apply = () => {
    waveTheme = $("#waveTheme").value; waveStyle = $("#waveStyle").value;
    try { localStorage.setItem("waveTheme", waveTheme); localStorage.setItem("waveStyle", waveStyle); } catch {}
    drawWavePreview();
    if (state.current) { buildWaves(); renderDetail(); }
  };
  $("#waveTheme").addEventListener("change", apply);
  $("#waveStyle").addEventListener("change", apply);
  drawWavePreview();
  addEventListener("resize", () => { if ($("#view-settings").classList.contains("active")) drawWavePreview(); });
}
initWaveSettings();

// ── Jobs ─────────────────────────────────────────────────────────────────────
let pollTimer = null, lastActive = 0;
const isActive = (j) => j.status === "running" || j.status === "queued";
async function pollJobs() {
  clearTimeout(pollTimer);
  try { state.jobs = await api("/api/jobs"); } catch { state.jobs = []; }
  const active = state.jobs.filter(isActive);
  const running = active.filter((j) => j.status === "running");
  $("#jobsDot").classList.toggle("busy", active.length > 0);
  $("#jobsLabel").textContent = active.length ? String(active.length) : "";
  const prog = running.length ? running.reduce((a, j) => a + (j.progress || 0), 0) / running.length : 0;
  $("#jobsRing").style.strokeDasharray = `${(prog * 56.6).toFixed(1)} 56.6`;
  $("#jobsBtn").title = !active.length ? "No jobs running"
    : [running.length ? `${running.length} running` : null, active.length - running.length ? `${active.length - running.length} queued` : null].filter(Boolean).join(" · ");
  if (!$("#jobsPanel").hidden) renderJobs();
  renderInlineJobs();
  updateStatusCells();
  if (lastActive && active.length < lastActive) {     // something finished
    loadTracks();
    if (state.models) loadModels();
    if (state.current) openTrack(state.current.id);
    if (exp.root) updatePlan();
  } else if (state.current && active.some((j) => j.track_id === state.current.id && j.kind === "stems")) {
    renderStemProgress();
  }
  if (state.models?.stems_model?.job && $("#view-settings").classList.contains("active")) loadModels();
  lastActive = active.length;
  pollTimer = setTimeout(pollJobs, active.length ? 1000 : 4000);
}
// Stem progress of the open track, without re-rendering the whole workspace (keeps edits in fields).
function renderStemProgress() {
  const j = jobFor(state.current.id, "stems"), bar = $("#detail .tp-stems .progress > i"), lbl = $("#detail .tp-stems .stem-state");
  if (!j || j.status !== "running") return;
  if (bar) bar.style.width = `${j.progress * 100}%`;
  if (lbl) lbl.textContent = `Making… ${Math.round(j.progress * 100)}%`;
}
function jobEl(j) {
  const label = { queued: "Queued", running: `${Math.round(j.progress * 100)}%`, done: "Done", failed: "Failed", cancelled: "Cancelled" }[j.status];
  return h("div", { class: `job ${j.status}` },
    h("div", { class: "row" }, h("b", { title: j.title }, j.title), h("span", { class: `pill ${j.status}` }, label),
      isActive(j) ? h("button", { class: "ghost small", onclick: () => api(`/api/jobs/${j.id}/cancel`, { body: {} }).then(pollJobs) }, "Cancel") : null),
    j.status === "running" ? h("div", { class: "progress" }, h("i", { style: `width:${j.progress * 100}%` })) : null,
    j.message ? h("div", { class: "msg" }, j.message) : null);
}
function renderJobs() {
  const act = state.jobs.filter(isActive), fin = state.jobs.filter((j) => !isActive(j));
  $("#jobsList").replaceChildren(...(state.jobs.length
    ? [act.length ? h("div", { class: "job-group" }, `Active · ${act.length}`) : null, ...act.map(jobEl),
       fin.length ? h("div", { class: "job-group" }, `Finished · ${fin.length}`) : null, ...fin.map(jobEl)].filter(Boolean)
    : [h("p", { class: "muted", style: "padding:12px" }, "Nothing running.")]));
}
function renderInlineJobs() {
  for (const [slot, kind] of [["#exportJob", "export"], ["#otaJob", "ota"], ["#flashJob", "flash"]]) {
    const j = state.jobs.find((x) => x.kind === kind);
    const el = $(slot);
    if (!el) continue;
    if (!j) { el.replaceChildren(); continue; }
    el.replaceChildren(jobEl(j));
    if (kind === "export" && j.status === "done" && j.result) {
      el.append(h("div", { class: "plan" }, h("b", {}, "Export complete"),
        h("div", {}, `Exported ${j.result.tracks} tracks (${j.result.copied} files copied), ${j.result.playlists} playlists, ${j.result.stems} with stems.` +
          (j.result.missing_files ? ` ${j.result.missing_files} files were missing and skipped.` : "")),
        ...deckWarnings(j.result.warnings)));
    }
  }
  // Only one export at a time: the button waits while one is queued or running, and the plan
  // is worked out again once it ends (a small export can finish between two polls).
  const ej = state.jobs.find((x) => x.kind === "export"), ejKey = ej ? `${ej.id}:${isActive(ej)}` : "";
  if (exp.root && ej && isActive(ej)) $("#doExport").disabled = true;
  if (exp.jobKey !== undefined && ejKey !== exp.jobKey && ej && !isActive(ej) && exp.root) updatePlan();
  exp.jobKey = ejKey;
}
function toggleJobs(show = $("#jobsPanel").hidden) {
  $("#jobsPanel").hidden = !show;
  $("#jobsBtn").setAttribute("aria-expanded", String(show));
  if (show) { renderJobs(); $("#closeJobs").focus(); } else $("#jobsBtn").focus();
}
$("#jobsBtn").addEventListener("click", () => toggleJobs());
$("#closeJobs").addEventListener("click", () => toggleJobs(false));
$("#clearJobs").addEventListener("click", () => api("/api/jobs/clear", { body: {} }).then(pollJobs));
$("#prepareAll").addEventListener("click", () => api("/api/prepare", { body: {} }).then((r) => { toast(`Analysing ${r.analyse}, stems for ${r.stems}, converting ${r.convert}`); loadTracks(); pollJobs(); }).catch(fail));

// ── Export view ──────────────────────────────────────────────────────────────
const exp = { root: null, drives: [], plan: null };
// The card column can be folded away (the card glyph in the top bar); it opens by itself on wide windows.
function setCardOpen(on) {
  $(".app").classList.toggle("no-card", !on);
  $("#cardBtn").setAttribute("aria-pressed", String(on));
  try { localStorage.setItem("cardOpen", on ? "1" : "0"); } catch {}
  if (on) loadExport().catch(() => {});
  window.dispatchEvent(new Event("resize"));
}
$("#cardBtn").addEventListener("click", () => setCardOpen($(".app").classList.contains("no-card")));
function renderCardCount() { $("#cardCount").textContent = exp.root && exp.plan?.tracks ? String(exp.plan.tracks) : ""; }
// A rekordbox library we didn't write is never overwritten. Merging into it isn't built yet.
function foreignLibrary(why) {
  return h("div", { class: "suggest" },
    h("span", {}, why, " Exporting would replace it, so DKPRP won't."),
    h("button", { class: "ghost small", onclick: () => { exp.root = null; loadExport(); } }, "Choose another drive"));
}

// Files are copied as they are; these are the ones the deck can't decode.
function deckWarnings(list) {
  if (!list || !list.length) return [];
  return [h("div", { class: "warn-line" }, h("b", {}, `${plural(list.length, "track")} won't play on the DECK001`), " (they're still copied, CDJs play them):"),
    h("div", { class: "warns" }, list.map((w) => h("div", { class: "muted" }, `${w.title}: ${w.problem}`)))];
}
const driveName = (d) => `${d.label || "Drive"} (${d.root})`;
async function loadExport() {
  const drives = await api(`/api/drives?all=${$("#allDrives").checked}`).catch(() => []);
  exp.drives = drives;
  if (!drives.some((d) => d.root === exp.root && !d.problem)) exp.root = null;
  $("#drives").replaceChildren(...(drives.length ? drives.map((d) => {
    // Drives the deck can't read (NTFS, APFS...) are shown, with the reason, but can't be chosen.
    const chosen = exp.root === d.root;
    const used = d.total ? (d.total - d.free) / d.total : 0;
    const c = h("button", { class: "card" + (chosen ? " chosen" : ""), disabled: !!d.problem, role: "radio", "aria-checked": String(chosen) },
      h("div", { class: "c-top" }, h("b", {}, driveName(d)), chosen ? h("span", { class: "c-sel" }, "Selected") : null),
      h("div", { class: "muted" }, `${d.removable ? "Removable" : "Fixed disk"} · ${d.filesystem}`),
      h("div", { class: "bar", title: `${Math.round(used * 100)}% used` }, h("i", { style: `width:${used * 100}%` })),
      h("div", { class: "muted tnum" }, `${fmtBytes(d.free)} free of ${fmtBytes(d.total)}`),
      d.problem ? h("div", { class: "alert-text" }, d.problem) : null,
      d.has_export ? h("div", { class: "muted" }, "Has a rekordbox library") : null);
    c.addEventListener("click", () => { exp.root = d.root; loadExport(); });
    return c;
  }) : [h("p", { class: "muted" }, "No removable drives")]));
  const lists = state.playlists.filter((p) => !p.is_folder);
  const ticked = new Set($$("#exportPlaylists input:checked").map((i) => +i.value));
  $("#exportPlaylists").replaceChildren(...(lists.length ? lists.map((p) => h("label", { class: "check" }, h("input", { type: "checkbox", value: p.id, checked: ticked.has(p.id) }), `${p.name} (${p.count})`))
    : [h("span", { class: "muted" }, "No playlists")]));
  updatePlan();
}
$$("input[name=scope]").forEach((r) => r.addEventListener("change", () => { $("#exportPlaylists").hidden = $("input[name=scope]:checked").value !== "playlists"; updatePlan(); }));
$("#exportPlaylists").addEventListener("change", updatePlan);
$("#includeStems").addEventListener("change", updatePlan);
$("#refreshDrives").addEventListener("click", loadExport);
$("#allDrives").addEventListener("change", loadExport);
function exportBody() {
  const scope = $("input[name=scope]:checked").value;
  const ids = scope === "playlists" ? $$("#exportPlaylists input:checked").map((i) => +i.value) : null;
  return { root: exp.root, playlist_ids: ids, include_stems: $("#includeStems").checked };
}
// The target box: which drive, and how much of it the export will fill.
function renderTarget(p) {
  const box = $("#exportTarget"), d = exp.drives.find((x) => x.root === exp.root);
  box.classList.toggle("set", !!d);
  if (!d) { box.replaceChildren(h("b", {}, "No drive"), h("span", { class: "muted" }, exp.drives.length ? "Choose one below" : "Insert a card or a USB stick")); return; }
  const used = d.total ? (d.total - d.free) / d.total : 0, need = p && d.total ? Math.min(1 - used, p.bytes / d.total) : 0;
  fill(box, h("span", { class: "t-label" }, "Target"), h("b", {}, driveName(d)),
    h("span", { class: "muted" }, `${d.filesystem} · ${fmtBytes(d.free)} free of ${fmtBytes(d.total)}`),
    h("div", { class: "bar", "aria-hidden": "true" }, h("i", { style: `width:${used * 100}%` }),
      p ? h("i", { class: p.fits ? "need" : "over", style: `left:${used * 100}%;width:${(p.fits ? need : 1 - used) * 100}%` }) : null),
    p ? h("span", { class: "muted tnum" }, p.fits ? `Uses up to ${fmtBytes(p.bytes)}, leaving about ${fmtBytes(Math.max(0, p.free - p.bytes))}` : `Needs ${fmtBytes(p.bytes)}, only ${fmtBytes(p.free)} free`) : null);
}
let planSeq = 0;
// The dry run: what each track needs before you export. Nothing is written until you press Export.
function dryRun(p) {
  const rows = (p.details || []).filter((d) => d.notes.length);
  if (!p.details?.length) return null;
  const skipped = p.details.filter((d) => d.skipped).length;
  return h("details", { class: "dry-run", open: rows.some((d) => d.skipped || d.notes.some((n) => /can't be played|missing/.test(n))) ? true : null },
    h("summary", {}, rows.length ? `Check before exporting: ${plural(rows.length, "track")} with notes${skipped ? `, ${skipped} skipped` : ""}` : "Check before exporting: nothing to flag"),
    rows.length ? h("ul", {}, rows.map((d) => h("li", {}, h("b", {}, `${d.artist ? d.artist + " – " : ""}${d.title || "Untitled"}`), h("ul", {}, d.notes.map((n) => h("li", { class: d.skipped ? "alert-text" : "" }, n)))))) : null);
}
async function updatePlan() {
  const plan = $("#plan"), btn = $("#doExport"), seq = ++planSeq;
  btn.disabled = true; $("#doExport .eb-n").textContent = ""; btn.title = "Export"; btn.setAttribute("aria-label", "Export");
  plan.classList.remove("bad");
  renderTarget(null);
  if (!exp.root) { plan.textContent = ""; exp.plan = null; renderCardCount(); updateStatusCells(); return; }
  const body = exportBody();
  if (body.playlist_ids && !body.playlist_ids.length) { plan.textContent = "Tick at least one playlist."; return; }
  plan.textContent = "Working out the plan…";
  try {
    const p = await api("/api/export/plan", { body });
    if (seq !== planSeq) return;           // a newer choice is already being planned
    exp.plan = p;
    renderTarget(p);
    plan.classList.toggle("bad", !p.fits || !!p.target_problem || !!p.foreign_library);
    fill(plan,
      h("dl", { class: "kv" },
        h("dt", {}, "Tracks"), h("dd", {}, String(p.tracks)),
        h("dt", {}, "Playlists"), h("dd", {}, String(p.playlists)),
        h("dt", {}, "With stems"), h("dd", {}, body.include_stems ? String(p.with_stems) : "Stems off"),
        h("dt", {}, "Size"), h("dd", {}, `up to ${fmtBytes(p.bytes)} (${fmtBytes(p.free)} free)`)),
      p.unanalysed ? h("div", { class: "muted" }, `${p.unanalysed} tracks aren't analysed yet; they'll be analysed during the export (slower).`) : null,
      dryRun(p),
      ...deckWarnings(p.warnings),
      p.fits ? null : h("div", { class: "alert-text" }, h("b", {}, "Doesn't fit. "), "Choose fewer playlists or turn off stems."),
      p.target_problem ? h("div", { class: "alert-text" }, p.target_problem) : null,
      p.foreign_library ? foreignLibrary(p.foreign_library) : null,
      !p.tracks ? h("div", { class: "muted" }, "Nothing to export.") : null);
    const busy = state.jobs.some((j) => j.kind === "export" && isActive(j));
    btn.disabled = !p.fits || !p.tracks || !!p.target_problem || !!p.foreign_library || busy;
    const d = exp.drives.find((x) => x.root === exp.root);
    if (!btn.disabled) { $("#doExport .eb-n").textContent = String(p.tracks); btn.title = `Export ${plural(p.tracks, "track")} to ${d?.label || d?.root || "drive"}`; btn.setAttribute("aria-label", btn.title); }
    renderCardCount(); updateStatusCells();
  } catch (e) { if (seq === planSeq) plan.textContent = e.message; }
}
$("#doExport").addEventListener("click", async () => {
  $("#doExport").disabled = true;
  await api("/api/export", { body: exportBody() }).catch(fail);
  pollJobs();
});

// ── Deck view (DECK001) ──────────────────────────────────────────────────────
// Status is checked at start-up, on the Deck view and on request: never on a timer.
async function checkDeck() {
  const s = await api("/api/deck/status").catch((e) => ({ reachable: false, error: e.message }));
  state.deck = s;
  renderDeckPill();
  return s;
}
function renderDeckPill() {
  const s = state.deck, ok = !!s?.reachable;
  const f = s?.firmware || {}, ver = f.version || f.app_version;
  $("#deckDot").className = "dot" + (s ? (ok ? " ok" : "") : " busy");
  $("#deckPillLabel").textContent = !s ? "Checking" : ok ? `Connected${ver ? " · " + ver : ""}` : "Not connected";
  $("#deckPill").classList.toggle("ok", ok);
  $("#deckPill").title = `DECK001 ${!s ? "checking" : ok ? "connected" : "not connected"}${s?.address ? " at " + s.address : ""} (opens Deck)`;
}
async function loadDeck() {
  const box = $("#deckStatus");
  box.replaceChildren(h("div", { class: "dev-state" }, h("span", { class: "dot busy" }), "Checking…"));
  const s = await checkDeck();
  const f = s.firmware || {};
  const row = (k, v) => [h("dt", {}, k), h("dd", {}, v ?? "–")];
  fill(box,
    h("div", { class: "dev-id" + (s.reachable ? " ok" : "") }, h("span", { class: "glyph", "aria-hidden": "true" }),
      h("div", {}, h("b", {}, "DECK001"),
        h("span", { class: "dev-state" }, h("span", { class: "dot" + (s.reachable ? " ok" : "") }), s.reachable ? "Connected" : "Not reachable"))),
    h("dl", { class: "kv" },
      row("Address", s.address),
      ...(s.reachable ? [row("Firmware", f.version || f.app_version || "?"), row("Project", f.project), row("Running partition", f.running_partition || f.slot),
                         row("Image state", f.running_image_state)] : [])),
    s.reachable && f.last_error ? h("div", { class: "alert-text" }, `Last update error: ${f.last_error}`) : null,
  );
  const ports = await api("/api/deck/ports").catch(() => []);
  $("#ports").replaceChildren(...(ports.length ? ports.map((p) => h("option", { value: p.port }, `${p.port} – ${p.description}${p.espressif ? " (ESP32)" : ""}`)) : [h("option", { value: "" }, "No USB serial ports")]));
}
$("#deckRefresh").addEventListener("click", loadDeck);
$("#deckPill").addEventListener("click", () => showView("deck"));
$("#otaPick").addEventListener("click", () => $("#otaFile").click());
$("#otaFile").addEventListener("change", async () => {
  const f = $("#otaFile").files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  await api("/api/deck/ota", { body: fd }).catch(fail);
  $("#otaFile").value = "";
  pollJobs();
});
// Wired flashing overwrites the firmware, so it is the one action behind a typed confirmation.
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
  await loadModels();
  const hist = await api("/api/history");
  $("#history").replaceChildren(...(hist.length ? hist.slice(0, 60).map((r) => h("div", { class: "h" },
    h("span", { class: "when" }, new Date(r.at * 1000).toLocaleString()),
    h("span", { class: "what" }, r.action.replaceAll("_", " ")),
    r.can_undo ? h("button", { class: "ghost small", onclick: () => doUndo(r.id).then(loadSettings) }, "Undo") : null))
    : [h("p", { class: "muted" }, "No changes yet.")]));
}
function renderSystem() {
  const s = state.system, m = state.models?.stems_model;
  // The stems model (174 MB) isn't in the installer on Windows and Linux: fetched once, checked against the build's hash.
  const model = !m ? "…" : m.present ? h("span", { class: "ok-text" }, icon("check", "ic sm"), "Ready")
    : m.job ? h("span", { class: "muted" }, `Downloading ${Math.round(m.job.progress * 100)}% · ${m.job.message}`)
    : h("span", { class: "row-btns" }, h("span", { class: "muted" }, m.fetchable ? `Not downloaded (${m.bytes ? Math.round(m.bytes / 1e6) + " MB" : "174 MB"}, once)` : "Not found"),
        m.fetchable ? h("button", { class: "small primary", onclick: () => api("/api/models", { body: {} }).then(() => { toast("Downloading the stems model"); pollJobs(); loadModels(); }).catch(fail) }, "Download") : null,
        !m.fetchable ? h("span", { class: "muted" }, `Put htdemucs.onnx in ${m.dir || "the models folder"}`) : null);
  $("#system").replaceChildren(h("span", { class: "k" }, "Application"), h("span", {}, `DKPRP ${s.version || ""}`),
    h("span", { class: "k" }, "Device"), h("span", {}, "DECK001"),
    h("span", { class: "k" }, "Stems model"), model,
    h("span", { class: "k" }, "Library and stems"), h("span", {}, s.data_dir || ""));
}
function loadModels() { return api("/api/models").then((m) => { state.models = m; renderSystem(); }).catch(() => {}); }
$("#settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target, v = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    v[el.name] = el.type === "checkbox" ? el.checked : el.type === "number" ? +el.value : el.value;
  }
  try {
    await api("/api/settings", { method: "PUT", body: v });
    toast("Settings saved");
    if (v.deck_address !== state.settings.deck_address) { state.settings.deck_address = v.deck_address; checkDeck(); }
  } catch (err) { fail(err); }
});

// ── Keyboard ─────────────────────────────────────────────────────────────────
addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (!$("#modal").hidden) return;                   // the prompt handles its own keys
  if ((e.ctrlKey || e.metaKey) && e.key === "z" && !typing && state.lastUndo) { e.preventDefault(); doUndo(state.lastUndo); return; }
  if (e.altKey && e.key >= "1" && e.key <= "4") { e.preventDefault(); showView(["library", "export", "deck", "settings"][+e.key - 1]); return; }
  if (e.key === "Escape" && !typing) {
    if (!$("#jobsPanel").hidden) return toggleJobs(false);
    if (!$("#playlistsBox").hidden) return setPlaylistsOpen(false);
    if (state.focus) return setFocus(false);
    if (!$("#view-library").classList.contains("active")) return showView("library");
  }
  if (typing || !$("#view-library").classList.contains("active")) return;
  if (e.key === "/") { e.preventDefault(); $("#search").focus(); return; }
  if (e.key === " " && state.current) { e.preventDefault(); togglePlay(); }
  if (e.key === "Delete" && state.selected.size) removeSelected().catch(fail);
  if ((e.ctrlKey || e.metaKey) && e.key === "a") { e.preventDefault(); state.selected = new Set(state.view.map((t) => t.id)); renderTracks(); }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (state.current && e.key >= "1" && e.key <= "8") {
    const slot = +e.key, cue = state.current.cues.find((c) => c.slot === slot);
    cue ? (player.currentTime = cue.start_ms / 1000) : setCue(slot, player.currentTime * 1000).catch(fail);
  }
  if (state.current && (e.key === "m" || e.key === "M")) addMemory(player.currentTime * 1000).catch(fail);
  if (state.current && (e.key === "i" || e.key === "I")) setMixPoints({ mix_in_ms: Math.round(snapToBeat(player.currentTime * 1000)) });
  if (state.current && (e.key === "o" || e.key === "O")) setMixPoints({ mix_out_ms: Math.round(snapToBeat(player.currentTime * 1000)) });
  if (state.current && (e.code === "BracketLeft" || e.code === "BracketRight")) {
    e.preventDefault();
    gridNudge((e.code === "BracketLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1));
  }
});

// ── Drag to resize: the sidebar and the track panel ─────────────────────────
// A thin handle on the edge: drag it, or focus it and use the arrow keys (Shift = bigger steps); double-click resets.
// The width is kept in the browser (localStorage) and set as a CSS variable, so everything that follows it moves with it.
function makeResizer(handle, { name, label, cssVar, min, max, edge, defaultPx }) {
  const root = document.documentElement;
  const current = () => parseFloat(getComputedStyle(root).getPropertyValue(cssVar)) || defaultPx;
  const clamp = (v) => Math.round(Math.max(min, Math.min(max(), v)));
  const set = (px, save = false) => {
    px = clamp(px);
    root.style.setProperty(cssVar, `${px}px`);
    handle.setAttribute("aria-valuenow", String(px));
    if (save) { try { localStorage.setItem(`size-${name}`, String(px)); } catch {} }
  };
  handle.setAttribute("role", "separator"); handle.setAttribute("aria-orientation", "vertical");
  handle.setAttribute("aria-label", label); handle.tabIndex = 0;
  handle.setAttribute("aria-valuemin", String(min));
  handle.title = `${label} (drag, or arrow keys; double-click to reset)`;
  try { const saved = parseFloat(localStorage.getItem(`size-${name}`)); if (saved) set(saved); } catch {}
  handle.setAttribute("aria-valuenow", String(clamp(current())));
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const x0 = e.clientX, w0 = current();
    handle.classList.add("drag"); document.body.classList.add("resizing");
    const move = (m) => set(w0 + (edge === "right" ? m.clientX - x0 : x0 - m.clientX));
    const up = () => {
      handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); handle.removeEventListener("pointercancel", up);
      handle.classList.remove("drag"); document.body.classList.remove("resizing");
      set(current(), true); window.dispatchEvent(new Event("resize"));      // the waveforms follow the new width
    };
    handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up); handle.addEventListener("pointercancel", up);
  });
  handle.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (!d) return;
    e.preventDefault();
    set(current() + d * (edge === "right" ? 1 : -1) * (e.shiftKey ? 64 : 16), true);
    window.dispatchEvent(new Event("resize"));
  });
  handle.addEventListener("dblclick", () => {
    root.style.removeProperty(cssVar);
    try { localStorage.removeItem(`size-${name}`); } catch {}
    handle.setAttribute("aria-valuenow", String(clamp(current())));
    window.dispatchEvent(new Event("resize"));
  });
}
{
  const panelHandle = h("div", { class: "resizer resizer-panel" });
  $("#view-export").before(panelHandle);
  makeResizer(panelHandle, { name: "panel", label: "Resize the card", cssVar: "--panel-w", min: 280, max: () => Math.max(380, innerWidth - 640), edge: "left", defaultPx: 380 });
}

async function refreshAll() { await loadPlaylists(); await loadTracks(); if (state.current) openTrack(state.current.id); }
renderTracks();
renderDeckPill();
refreshAll().then(() => {
  pollJobs();
  let open = innerWidth >= 1100;
  try { if (localStorage.getItem("cardOpen") === "0") open = false; } catch {}
  setCardOpen(open);
}).catch(fail);
api("/api/system").then((s) => { state.system = s; renderSystem(); }).catch(() => {});
api("/api/settings").then((s) => { state.settings = s; renderTracks(); }).catch(() => {});
checkDeck();
