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
// Hot cue colours A-H (pads and waveform markers). Display only: the card doesn't store cue colours.
const CUE_COLORS = ["#f0574c", "#f08c3c", "#e6c84a", "#46a758", "#3fb8c4", "#3e7bd6", "#8e4ec6", "#e85d9c"];

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
const tabs = () => $$(".tabs button");
tabs().forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
$(".tabs").addEventListener("keydown", (e) => {       // arrow keys move between views (roving tabindex)
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
  e.preventDefault();
  const list = tabs(), i = list.indexOf(document.activeElement);
  const n = e.key === "Home" ? 0 : e.key === "End" ? list.length - 1 : (i + (e.key === "ArrowDown" ? 1 : -1) + list.length) % list.length;
  list[n].focus(); showView(list[n].dataset.view);
});
function showView(v) {
  tabs().forEach((b) => { const on = b.dataset.view === v; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; });
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  if (v === "export") loadExport();
  if (v === "deck") loadDeck();
  if (v === "settings") loadSettings();
}
let cssCache = {};   // colours read from CSS variables; reading them every animation frame is slow
function applyTheme(t) {
  cssCache = {};
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = isDark();
  $("#themeBtn use").setAttribute("href", dark ? "#i-sun" : "#i-moon");
  $("#themeBtn").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
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
    btn.addEventListener("contextmenu", (e) => { e.preventDefault(); playlistMenu(p); });
    btn.addEventListener("keydown", (e) => { if (e.key === "F2" || (e.shiftKey && e.key === "F10")) { e.preventDefault(); playlistMenu(p); } });
    if (!p.is_folder) dropTarget(btn, (ids) => addToPlaylist(p.id, ids));
    if (!p.is_folder) return btn;
    return h("div", { class: "folder" + (open ? " open" : "") }, btn, h("div", { class: "children" }, build(p.id)));
  });
  const tree = build(0);
  $("#tree").replaceChildren(...(tree.length ? tree : [h("div", { class: "tree-empty" }, "No playlists yet.")]));
  $("#allTracks").classList.toggle("active", state.playlist === 0);
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
  const parent = inFolder ? state.openFolderTarget : 0;
  const p = await api("/api/playlists", { body: { name, parent_id: parent, is_folder: isFolder } });
  if (!isFolder && state.selected.size) await addToPlaylist(p.id, [...state.selected], p.name || name);
  loadPlaylists();
}
$("#newPlaylist").addEventListener("click", () => newPlaylist(false).catch(fail));
$("#newFolder").addEventListener("click", () => newPlaylist(true).catch(fail));
$("#allTracks").addEventListener("click", () => selectPlaylist(0));

function selectPlaylist(id) { state.playlist = id; state.selected.clear(); state.anchor = null; loadPlaylists(); loadTracks(); }
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
  renderTracks();
  if (!state.playlist && !q) $("#allCount").textContent = state.tracks.length || "";
}

// Filters and sorting only change what is shown; playlist order (the # column) stays the data.
const FILTERS = {
  all: () => true,
  unanalysed: (t) => !t.analysed_at,
  nostems: (t) => t.stems !== "done",
  stems: (t) => t.stems === "done",
};
function visibleTracks() {
  const list = state.tracks.map((t, i) => ({ t, i })).filter(({ t }) => FILTERS[state.filter](t));
  if (state.sort) {
    const { key, dir } = state.sort;
    const val = (x) => key === "index" ? x.i : x.t[key];
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
function analysisCell(t) {
  const j = jobFor(t.id, "analyse");
  if (j?.status === "running") return h("span", { class: "pill running" }, `Analysing ${Math.round(j.progress * 100)}%`);
  if (j?.status === "queued") return h("span", { class: "pill queued" }, "Queued");
  if (t.analysed_at) return h("span", { class: "muted" }, "Analysed");
  if (j?.status === "failed") return h("span", { class: "pill failed", title: j.message }, "Failed");
  return h("span", { class: "muted" }, "Not analysed");
}
function stemPill(t) {
  const j = jobFor(t.id, "stems");
  if (t.stems === "running" || j?.status === "running") return h("span", { class: "pill running" }, j?.status === "running" ? `Making ${Math.round(j.progress * 100)}%` : "Making");
  const label = { queued: "Queued", done: "Ready", failed: "Failed" }[t.stems];
  return label ? h("span", { class: `pill ${t.stems}` }, label) : h("span", { class: "muted" }, "–");
}
function renderTracks() {
  const body = $("#tracks tbody");
  const focusedId = body.contains(document.activeElement) ? document.activeElement.dataset?.id : null;   // keep keyboard focus across re-renders
  const vis = visibleTracks();
  state.view = vis.map((x) => x.t);
  const reorderOk = canReorder();
  const rows = vis.map(({ t, i }, vi) => {
    const tr = h("tr", { "data-id": t.id, draggable: "true", tabindex: "-1", "aria-selected": String(state.selected.has(t.id)),
                         class: (state.selected.has(t.id) ? "sel" : "") + (state.current?.id === t.id ? " playing" : "") },
      h("td", { class: "c-col" }, t.color ? h("span", { class: "swatch", style: `background:${COLORS[t.color]}`, title: COLOR_NAMES[t.color] }) : null),
      h("td", { class: "c-idx num" }, String(i + 1)),
      h("td", { class: "c-title", title: t.path }, t.title || ""),
      h("td", {}, t.artist || h("span", { class: "muted" }, "Unknown")),
      h("td", {}, t.album || ""),
      h("td", {}, t.genre || ""),
      h("td", { class: "num" }, t.bpm ? fmtBpm(t.bpm) : h("span", { class: "muted" }, t.analysed_at ? "?" : "…")),
      h("td", {}, t.camelot ? h("span", { class: "key" }, h("b", {}, t.camelot), h("span", {}, t.musical_key || "")) : ""),
      h("td", {}, t.energy != null ? h("span", { class: "meter", title: `Energy ${t.energy}/10` }, h("i", { style: `--v:${Math.min(100, t.energy * 10)}%` }), h("span", {}, String(t.energy))) : ""),
      h("td", { class: "num" }, fmtTime(t.duration)),
      h("td", { class: "c-an" }, analysisCell(t)),
      h("td", { class: "c-st" }, stemPill(t)));
    tr.addEventListener("click", (e) => clickRow(e, t.id, vi));
    tr.addEventListener("dblclick", () => openTrack(t.id, true));
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
  body.replaceChildren(...rows);
  // Keyboard focus lands on the anchor row (or the first), arrow keys move from there.
  (body.querySelector(`tr[data-id="${state.anchor}"]`) || body.firstElementChild)?.setAttribute("tabindex", "0");
  if (focusedId) body.querySelector(`tr[data-id="${focusedId}"]`)?.focus({ preventScroll: true });
  const libEmpty = state.tracks.length === 0 && !$("#search").value && state.playlist === 0;
  $("#empty").hidden = !state.loaded || !libEmpty;
  const note = $("#tableNote");
  note.hidden = state.loaded && (rows.length > 0 || libEmpty);
  note.textContent = !state.loaded ? "Loading library…"
    : state.tracks.length === 0 && state.playlist && !$("#search").value ? "This playlist is empty. Drag tracks onto it in the sidebar."
    : state.tracks.length === 0 ? "No tracks match the search." : "No tracks match this filter.";
  $$("#tracks thead th[data-sort]").forEach((th) => th.setAttribute("aria-sort", state.sort?.key === th.dataset.sort ? (state.sort.dir > 0 ? "ascending" : "descending") : "none"));
  renderContext();
  renderLibStatus();
  renderDockExport();
}
// Status cells refresh in place on each jobs poll, so dragging and hovering aren't disturbed.
function updateStatusCells() {
  for (const tr of $$("#tracks tbody tr")) {
    const t = state.tracks.find((x) => x.id === +tr.dataset.id);
    if (!t) continue;
    tr.querySelector(".c-an")?.replaceChildren(analysisCell(t));
    tr.querySelector(".c-st")?.replaceChildren(stemPill(t));
  }
}
function renderLibStatus() {
  const n = state.tracks.length, secs = state.tracks.reduce((a, t) => a + (t.duration || 0), 0);
  const pl = state.playlists.find((p) => p.id === state.playlist);
  $("#statusLib").textContent = !state.loaded ? "Loading library…"
    : `${pl ? pl.name + ": " : ""}${plural(n, "track")} · ${fmtDur(secs)}` + (state.view.length !== n ? ` · ${state.view.length} shown` : "");
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
  if (!["ArrowDown", "ArrowUp", "Enter", "Home", "End"].includes(e.key) || !state.view.length) return;
  e.preventDefault();
  const view = state.view, cur = view.findIndex((t) => t.id === +document.activeElement?.dataset?.id);
  if (e.key === "Enter") { if (cur >= 0) openTrack(view[cur].id, true); return; }
  const n = e.key === "Home" ? 0 : e.key === "End" ? view.length - 1 : Math.max(0, Math.min(view.length - 1, cur + (e.key === "ArrowDown" ? 1 : -1)));
  clickRow({ shiftKey: e.shiftKey, ctrlKey: false, metaKey: false }, view[n].id, n);
  $(`#tracks tbody tr[data-id="${view[n].id}"]`)?.scrollIntoView({ block: "nearest" });
});
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

// ── Track preparation: player, waveform, cues, tags, stems ───────────────────
const player = $("#player");
async function openTrack(id, play) {
  if (state.gridPending && state.gridPending.id !== id) await saveGrid().catch(fail);
  const t = await api(`/api/tracks/${id}`);
  const same = state.current?.id === id;
  state.current = t;
  if (!same) {
    state.stemSolo = null;
    player.src = `/api/tracks/${id}/audio`;
    state.wave = null;
    buildWaves();
    api(`/api/tracks/${id}/waveform`).then((w) => { if (state.current?.id === id) { state.wave = decodeWave(w); buildWaves(); } }).catch(() => {});
  }
  renderDetail();
  if (play) player.play().catch(() => {});
  renderTracks();
}
function closeDetail() { player.pause(); state.ws?.destroy(); state.ws = null; state.current = null; state.wave = null; $("#detail").hidden = true; renderTracks(); }

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

const stat = (label, value) => h("div", { class: "stat" }, h("small", {}, label), h("b", {}, value || "–"));
function renderDetail() {
  const t = state.current;
  const d = $("#detail");
  if (!t) { d.hidden = true; return; }
  d.hidden = false;
  const field = (label, key, type = "text") => {
    const inp = h("input", { type, value: t[key] ?? "", step: key === "bpm" ? "0.01" : null });
    inp.addEventListener("change", () => saveField(key, type === "number" ? (inp.value === "" ? null : +inp.value) : inp.value));
    return h("label", {}, label, inp);
  };
  const color = h("select", {}, COLOR_NAMES.map((n, i) => h("option", { value: i, selected: (t.color || 0) === i }, n)));
  color.addEventListener("change", () => saveField("color", +color.value));
  const rating = h("select", {}, [0, 1, 2, 3, 4, 5].map((r) => h("option", { value: r, selected: (t.rating || 0) === r }, r ? "★".repeat(r) : "No rating")));
  rating.addEventListener("change", () => saveField("rating", +rating.value));

  // Hot cues A-H, each with its own colour (as on the waveform): click jumps, or sets at the
  // playhead when empty; right-click or × clears. Colours are a DKPRP display aid only.
  const pads = h("div", { class: "pads", role: "group", "aria-label": "Hot cues (keys 1 to 8)" }, [...PAD].map((letter, i) => {
    const cue = t.cues.find((c) => c.slot === i + 1);
    const pad = h("button", { class: "pad" + (cue ? " set" : ""), style: `--cue:${CUE_COLORS[i]}`,
                              title: cue ? `Jump to ${letter} (key ${i + 1}; right-click to clear)` : `Set ${letter} at the playhead (key ${i + 1})`,
                              "aria-label": cue ? `Hot cue ${letter} at ${fmtMs(cue.start_ms)}` : `Set hot cue ${letter}` },
      h("b", {}, letter), h("small", {}, cue ? (cue.label || fmtMs(cue.start_ms)) : "–"), cue ? h("span", { class: "x", title: "Clear" }, "×") : null);
    pad.addEventListener("click", (e) => {
      if (cue && !e.target.classList.contains("x")) { player.currentTime = cue.start_ms / 1000; return; }
      setCue(i + 1, cue && e.target.classList.contains("x") ? null : player.currentTime * 1000).catch(fail);
    });
    pad.addEventListener("contextmenu", (e) => { e.preventDefault(); if (cue) setCue(i + 1, null).catch(fail); });
    return pad;
  }));
  const mems = t.cues.filter((c) => !c.slot).sort((a, b) => a.start_ms - b.start_ms);
  const memList = h("div", { class: "mem" }, mems.length ? mems.map((c) => h("div", { class: "mem-row" },
      h("span", { class: "m" }, "M"),
      h("span", { class: "t" }, fmtMs(c.start_ms), c.label ? h("span", { class: "muted" }, ` · ${c.label}`) : null),
      h("button", { class: "ghost small", onclick: () => { player.currentTime = c.start_ms / 1000; } }, "Jump"),
      h("button", { class: "ghost small icon-btn", "aria-label": `Delete memory cue at ${fmtMs(c.start_ms)}`, title: "Delete",
                    onclick: () => removeMemory(c).catch(fail) }, icon("x", "ic sm"))))
    : h("div", { class: "muted", style: "font-size:12px" }, "None. Press M to add one at the playhead."));

  // Stems: solo one at a time through the same player, or make them.
  const sj = jobFor(t.id, "stems");
  const stemRow = h("div", { class: "stems-row" });
  if (t.stem_files.length) {
    stemRow.append(h("button", { class: "ghost small" + (!state.stemSolo ? " on" : ""), "aria-pressed": String(!state.stemSolo), onclick: () => soloStem(null) }, "Full mix"));
    t.stem_files.forEach((s) => stemRow.append(h("button", { class: "ghost small stem" + (state.stemSolo === s ? " on" : ""), "aria-pressed": String(state.stemSolo === s),
                                                             onclick: () => soloStem(s) }, s === "other" ? "melody" : s)));
  } else {
    const running = sj?.status === "running";
    stemRow.append(h("span", { class: "muted" }, running ? `Making… ${Math.round(sj.progress * 100)}%` : { none: "Not made", queued: "Queued", running: "Making…", failed: "Failed" }[t.stems] || ""),
      t.stems === "none" || t.stems === "failed" ? h("button", { class: "ghost small", onclick: () => api("/api/stems", { body: { ids: [t.id] } }).then(() => { pollJobs(); loadTracks(); openTrack(t.id); }).catch(fail) }, "Make stems") : null);
    if (running) stemRow.append(h("div", { class: "progress", style: "width:100%" }, h("i", { style: `width:${sj.progress * 100}%` })));
  }

  // Transport: play, previous/next track, position, preview tempo (±8 %, key lock), volume, zoom.
  const playing = !player.paused;
  const seek = h("input", { type: "range", class: "seek", id: "seek", min: 0, max: t.duration || 0, step: 0.1, value: player.currentTime || 0, "aria-label": "Position" });
  seek.addEventListener("input", () => { state.seeking = true; player.currentTime = +seek.value; });
  seek.addEventListener("change", () => { state.seeking = false; });
  const vol = h("input", { type: "range", class: "vol", min: 0, max: 1, step: 0.01, value: state.volume, "aria-label": "Preview volume" });
  vol.addEventListener("input", () => { state.volume = +vol.value; player.volume = state.volume; try { localStorage.setItem("volume", vol.value); } catch {} });
  const tbtn = (label, ic, onclick, extra = {}) => h("button", { class: "ghost icon-btn t-btn", "aria-label": label, title: label, onclick, ...extra }, icon(ic));
  const transport = h("div", { class: "transport-bar" },
    h("button", { class: "play-btn", id: "playBtn", onclick: togglePlay, "aria-label": playing ? "Pause (Space)" : "Play (Space)", title: "Play or pause (Space)" }, icon(playing ? "pause" : "play")),
    tbtn("Previous track (or back to the start)", "prev", () => stepTrack(-1)),
    tbtn("Next track", "next", () => stepTrack(1)),
    h("span", { class: "time", id: "timeLabel" }, `${fmtTime(player.currentTime || 0)} / ${fmtTime(t.duration)}`),
    seek,
    h("span", { class: "t-group", role: "group", "aria-label": "Preview tempo" },
      h("button", { class: "ghost small", "aria-label": "Tempo down 0.5%", title: "Tempo down", onclick: () => setTempo((state.tempo || 0) - 0.5) }, "−"),
      h("button", { class: "ghost small tempo", id: "tempoLabel", title: "Preview tempo (click to reset). Playback only: nothing is saved.", onclick: () => setTempo(0) }, fmtTempo()),
      h("button", { class: "ghost small", "aria-label": "Tempo up 0.5%", title: "Tempo up", onclick: () => setTempo((state.tempo || 0) + 0.5) }, "+")),
    tbtn(state.keyLock ? "Key lock on" : "Key lock off", "lock", () => { state.keyLock = !state.keyLock; applyRate(); renderDetail(); },
         { "aria-pressed": String(!!state.keyLock), class: "ghost icon-btn t-btn" + (state.keyLock ? " on" : "") }),
    h("span", { class: "t-vol" }, icon("volume", "ic sm"), vol),
    h("span", { class: "spacer" }),
    h("span", { class: "legend", "aria-label": "Waveform colours" },
      h("i", { style: "background:var(--wave-low)" }), "Low", h("i", { style: "background:var(--wave-mid)" }), "Mid", h("i", { style: "background:var(--wave-high)" }), "High"),
    h("span", { class: "zoom", role: "group", "aria-label": "Waveform zoom" },
      h("button", { class: "ghost small", "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoomWave(0.5) }, "−"),
      h("button", { class: "ghost small", "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoomWave(2) }, "+")),
    h("button", { class: "ghost small", "aria-expanded": String(!state.editorHidden), "aria-controls": "dRow", onclick: toggleEditor,
                  title: "Show or hide memory cues, tags and stems (E)" }, state.editorHidden ? "Show editor" : "Hide editor"));

  const ph = t.phrases;
  fill(d,
    h("div", { class: "d-head" },
      t.has_artwork ? h("img", { class: "d-art", src: `/api/tracks/${t.id}/artwork`, alt: "" }) : h("div", { class: "d-art", "aria-hidden": "true" }, icon("library")),
      h("div", { class: "d-id" },
        h("div", { class: "d-title", title: t.path }, t.title || ""),
        h("div", { class: "d-meta" }, [t.artist || "Unknown artist", t.album, t.genre].filter(Boolean).join(" · ")),
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("small", {}, "BPM"), h("b", { id: "bpmStat" }, bpmText(t))),
          stat("Key", t.camelot ? `${t.camelot} ${t.musical_key || ""}` : null),
          stat("Energy", t.energy != null ? `${t.energy}/10` : null), stat("Length", fmtTime(t.duration)))),
      h("div", { class: "spacer" }),
      h("div", { class: "d-cues" },
        h("div", { class: "d-cues-head" }, h("span", {}, "Hot cues"), h("span", { class: "muted" }, "keys 1–8"), h("span", { class: "spacer" }),
          h("button", { class: "ghost small", onclick: () => addMemory(player.currentTime * 1000).catch(fail), title: "Add a memory cue at the playhead (M)" }, "+ Memory cue")),
        pads),
      h("button", { class: "ghost small icon-btn d-close", onclick: closeDetail, "aria-label": "Close track", title: "Close" }, icon("x", "ic sm"))),
    splitSuggestion(t),
    t.deck_problem ? h("div", { class: "warn-line" }, `Won't play on the DECK001: ${t.deck_problem}. It's still exported (CDJs play it).`) : null,
    waveBox,
    phraseStrip(t),
    transport,
    h("div", { class: "d-row", id: "dRow", hidden: !!state.editorHidden },
      h("div", { class: "d-sec" },
        h("h3", {}, "Memory cues", h("span", { class: "spacer" }),
          h("button", { class: "ghost small", onclick: () => addMemory(player.currentTime * 1000).catch(fail), title: "Add at the playhead (M)" }, "+ At playhead")),
        memList),
      h("div", { class: "d-sec" }, h("h3", {}, "Track"),
        h("div", { class: "fields" },
          field("Title", "title"), field("Artist", "artist"), field("Album", "album"),
          field("Genre", "genre"), field("BPM", "bpm", "number"), field("Year", "year", "number"),
          h("label", {}, "Colour", color), h("label", {}, "Rating", rating), field("Comment", "comment"))),
      h("div", { class: "d-sec d-stems" },
        h("div", {}, h("h3", {}, "Beat grid", h("span", { class: "muted", style: "text-transform:none;letter-spacing:0;font-weight:400" }, "[ ] 1 ms, Shift 10 ms")),
          t.grid ? h("div", { class: "grid-row" },
            h("span", { class: "t-group", role: "group", "aria-label": "Nudge the beat grid" },
              h("button", { class: "ghost small", title: "Earlier by 10 ms (Shift+[)", onclick: () => gridNudge(-10) }, "«"),
              h("button", { class: "ghost small", title: "Earlier by 1 ms ([)", onclick: () => gridNudge(-1) }, "‹"),
              h("button", { class: "ghost small", title: "Later by 1 ms (])", onclick: () => gridNudge(1) }, "›"),
              h("button", { class: "ghost small", title: "Later by 10 ms (Shift+])", onclick: () => gridNudge(10) }, "»")),
            h("span", { class: "grid-val", id: "gridReadout", "aria-live": "polite" }, fmtNudge(gridNow(t))),
            h("button", { class: "ghost small", title: "Move the bar start one beat later", onclick: () => gridNudge(0, 1) }, "Downbeat +1"),
            h("button", { class: "ghost small", title: "Back to the analysis grid", onclick: gridReset }, "Reset"))
          : h("div", { class: "muted" }, "Appears after analysis")),
        h("div", {}, h("h3", { style: "margin-top:10px" }, "Stems"), stemRow),
        h("div", {}, h("h3", { style: "margin-top:10px" }, "Phrases"),
          ph ? h("dl", { class: "kv" },
            h("dt", {}, "First downbeat"), h("dd", {}, fmtMs(ph.first_downbeat_ms)),
            h("dt", {}, "Intro ends"), h("dd", {}, fmtMs(ph.intro_end_ms)),
            h("dt", {}, "Outro starts"), h("dd", {}, fmtMs(ph.outro_start_ms)),
            h("dt", {}, plural(ph.drops.length, "drop")), h("dd", {}, ph.drops.map(fmtMs).join(", ") || "–"))
          : h("div", { class: "muted" }, "Not analysed yet")))));
  syncMarkers();
}
// The editor row (cues, tags, stems) can be folded away to give the track table more room.
try { state.editorHidden = localStorage.getItem("editorHidden") === "1"; } catch {}
function toggleEditor() {
  state.editorHidden = !state.editorHidden;
  try { localStorage.setItem("editorHidden", state.editorHidden ? "1" : "0"); } catch {}
  renderDetail();
}
// Preview tempo (playback only, nothing saved): ±8 % like a deck's pitch fader, optional key lock.
try { state.volume = Math.max(0, Math.min(1, +(localStorage.getItem("volume") ?? 1))); } catch { state.volume = 1; }
if (!Number.isFinite(state.volume)) state.volume = 1;
state.keyLock = true; state.tempo = 0;
const fmtTempo = () => `${state.tempo > 0 ? "+" : state.tempo < 0 ? "−" : "±"}${Math.abs(state.tempo).toFixed(1)}%`;
const bpmText = (t) => (t.bpm ? fmtBpm(Math.round(t.bpm * (1 + state.tempo / 100) * 100) / 100) : "–");
function applyRate() {
  const r = 1 + state.tempo / 100;
  player.defaultPlaybackRate = r; player.playbackRate = r;
  player.preservesPitch = !!state.keyLock;
  player.volume = state.volume;
}
function setTempo(v) {
  state.tempo = Math.max(-8, Math.min(8, Math.round(v * 2) / 2));
  applyRate();
  const l = $("#tempoLabel"); if (l) l.textContent = fmtTempo();
  const b = $("#bpmStat"); if (b && state.current) b.textContent = bpmText(state.current);
}
player.addEventListener("loadedmetadata", applyRate);   // a new source resets the rate
applyRate();
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
    toast("Saved", r.undo);
    loadTracks();
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
function togglePlay() { player.paused ? player.play().catch(() => {}) : player.pause(); }
function soloStem(name) {
  const t = state.current;
  const pos = player.currentTime, playing = !player.paused;
  state.stemSolo = name;
  player.src = name ? `/api/tracks/${t.id}/stem/${name}` : `/api/tracks/${t.id}/audio`;
  player.addEventListener("loadedmetadata", () => { player.currentTime = pos; if (playing) player.play(); }, { once: true });
  renderDetail();
}
function setPlayIcon() {
  const b = $("#playBtn");
  if (!b) return;
  b.querySelector("use").setAttribute("href", player.paused ? "#i-play" : "#i-pause");
  b.setAttribute("aria-label", player.paused ? "Play (Space)" : "Pause (Space)");
}
player.addEventListener("play", setPlayIcon);
player.addEventListener("pause", setPlayIcon);
player.addEventListener("timeupdate", () => {
  const l = $("#timeLabel");
  if (l && state.current) l.textContent = `${fmtTime(player.currentTime)} / ${fmtTime(state.current.duration)}`;
  const sk = $("#seek"); if (sk && !state.seeking) sk.value = player.currentTime;
  const c = $("#phraseCursor"), dur = state.current?.duration;
  if (c && dur) c.style.left = `${Math.min(100, (player.currentTime / dur) * 100)}%`;
});

// Phrase strip: the analysis sections across the whole track; click one to jump to it.
function phraseStrip(t) {
  const dur = (t.grid?.duration_ms || (t.duration || 0) * 1000);
  if (!t.sections?.length || !dur) return null;
  const strip = h("div", { class: "phrases", role: "group", "aria-label": "Phrases" },
    t.sections.map((s) => {
      const kind = String(s.kind || "").toLowerCase();
      return h("button", { class: `k-${kind.replace(/[^a-z]/g, "")}`, style: `left:${(s.start_ms / dur) * 100}%;width:${((s.end_ms - s.start_ms) / dur) * 100}%`,
                           title: `${s.kind} · ${fmtMs(s.start_ms)}${s.bars ? ` · ${s.bars} bars` : ""}`,
                           onclick: () => { player.currentTime = s.start_ms / 1000; } }, s.kind);
    }),
    h("i", { id: "phraseCursor", "aria-hidden": "true", style: "position:absolute;top:0;bottom:0;width:2px;background:var(--ink);pointer-events:none;left:0" }));
  return strip;
}

// Waveforms: wavesurfer.js (ui/vendor/wavesurfer, BSD-3-Clause) draws the scrolling detail
// view and the whole-track overview (Minimap plugin) from the analysis peaks, and follows
// the shared <audio> player. Beat grid and cues are Regions-plugin markers.
function cssVar(n) { return cssCache[n] ??= getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const waveBox = h("div", { class: "waves", id: "waves" });
const withAlpha = (hex, a) => { const m = hex.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i); return m ? `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})` : hex; };

// PWV5 (150 columns per second: height plus a 3-bit bass/mid/high mix) as three coloured
// layers drawn over each other, rekordbox "3 band" style: lows, then mids, then highs on top.
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

// rekordbox "3 band" drawing: per pixel column, the tallest band is drawn first (behind) and the
// smallest last, so lows, mids and highs all stay visible wherever they are present.
// WaveSurfer passes a custom renderer at most two channels (top and bottom halves), so it gets
// one "channel" holding each column's position (index / (N-1), as WaveSurfer keeps peaks in 0-1),
// and the bands are read from state.bands at that index.
// That keeps WaveSurfer's slicing for scrolling, zoom and the minimap.
function drawBands(channels, ctx) {
  const idx = channels[0], W = ctx.canvas.width, H = ctx.canvas.height, c = H / 2, n = idx?.length || 0;
  const [lo, md, hi] = state.bands || [];
  if (!n || !W || !lo) return;
  const last = lo.length - 1;
  const colors = [cssVar("--wave-low"), cssVar("--wave-mid"), cssVar("--wave-high")];
  const paths = [[new Path2D(), new Path2D(), new Path2D()], [new Path2D(), new Path2D(), new Path2D()], [new Path2D(), new Path2D(), new Path2D()]];
  for (let x = 0; x < W; x++) {
    const a = Math.floor((x * n) / W), b = Math.max(a + 1, Math.floor(((x + 1) * n) / W));
    let v0 = 0, v1 = 0, v2 = 0;
    for (let k = a; k < b && k < n; k++) { const i = Math.round(idx[k] * last); if (lo[i] > v0) v0 = lo[i]; if (md[i] > v1) v1 = md[i]; if (hi[i] > v2) v2 = hi[i]; }
    const v = [v0, v1, v2], order = [0, 1, 2].sort((p, q) => v[q] - v[p]);
    order.forEach((band, rank) => { const hh = Math.max(v[band] * c, 0); if (hh > 0.5) paths[rank][band].rect(x, c - hh, 1, hh * 2); });
  }
  for (const rank of [0, 1, 2]) for (const band of [0, 1, 2]) { ctx.fillStyle = colors[band]; ctx.fill(paths[rank][band]); }
}

function buildWaves() {
  state.ws?.destroy(); state.ws = null; state.regions = null;
  const t = state.current, w = state.wave;
  if (!w) { waveBox.replaceChildren(h("div", { class: "wave-empty muted" }, t?.analysed_at ? "Loading waveform…" : "Waveform appears after analysis")); return; }
  waveBox.replaceChildren();
  const compact = innerHeight < 860;
  // The three bands go in as three "channels" and drawBands paints them. The played part keeps
  // its colours (a transparent progress colour leaves the drawing as it is).
  state.bands = wavePeaks(w);
  const N = state.bands[0].length;
  const index = Float32Array.from({ length: N }, (_, i) => i / Math.max(1, N - 1));
  const bands = { renderFunction: drawBands, waveColor: cssVar("--wave-low"), progressColor: "rgba(0,0,0,0)" };
  state.regions = WaveSurfer.Regions.create();
  state.ws = WaveSurfer.create({
    container: waveBox, media: player, peaks: [index], duration: w.duration, ...bands,
    height: compact ? 96 : 128, minPxPerSec: state.zoom, autoScroll: true, autoCenter: true, hideScrollbar: true,
    dragToSeek: true, normalize: false, cursorColor: cssVar("--ink"), cursorWidth: 2,
    plugins: [state.regions, WaveSurfer.Minimap.create({ height: compact ? 34 : 42, ...bands, dragToSeek: true,
      cursorColor: cssVar("--ink"), overlayColor: "rgba(128,128,128,.22)" })],
  });
  state.ws.on("ready", syncMarkers);
}
function zoomWave(f) {
  state.zoom = Math.max(25, Math.min(800, state.zoom * f));
  try { state.ws?.zoom(state.zoom); } catch {}
}

// Beat grid (thin beats, solid downbeats), then hot cues (A-H) and memory cues (M) on top.
function syncMarkers() {
  const t = state.current, g = effectiveGrid(t), reg = state.regions;
  if (!reg || !state.wave) return;
  reg.clearRegions();
  if (g && g.period_ms > 0) {
    // Lines in the text colour (light on dark, dark on light), over the coloured waveform:
    // downbeats strong, other beats faint, so a nudge can be lined up with the kicks.
    const ink = cssVar("--ink"), bg = withAlpha(ink, 0.7), beat = withAlpha(ink, 0.22);
    const beats = Math.floor((g.duration_ms - g.first_beat_ms) / g.period_ms) + 1;
    const everyBeat = beats < 3000;           // very long files (mixes): downbeats only
    for (let k = 0; k < beats; k++) {
      const isDown = ((k - g.downbeat_index) % 4 + 4) % 4 === 0;
      if (!isDown && !everyBeat) continue;
      reg.addRegion({ start: (g.first_beat_ms + k * g.period_ms) / 1000, color: isDown ? bg : beat, drag: false, resize: false });
    }
  }
  for (const c of t.cues) {
    // The waveform lives in a shadow root, so the label is styled inline.
    const col = c.slot ? CUE_COLORS[c.slot - 1] : cssVar("--ink");
    const label = h("span", { style: `display:inline-block;font:600 11px var(--f-text);padding:0 4px;border-radius:0 2px 2px 0;` +
      `background:${col};color:${c.slot ? "#111112" : cssVar("--panel")}` }, c.slot ? PAD[c.slot - 1] : "M");
    reg.addRegion({ start: c.start_ms / 1000, color: col, content: label, drag: false, resize: false });
  }
}

// ── Jobs ─────────────────────────────────────────────────────────────────────
let pollTimer = null, lastActive = 0;
const isActive = (j) => j.status === "running" || j.status === "queued";
async function pollJobs() {
  clearTimeout(pollTimer);
  try { state.jobs = await api("/api/jobs"); } catch { state.jobs = []; }
  const active = state.jobs.filter(isActive);
  const running = active.filter((j) => j.status === "running");
  $("#jobsDot").classList.toggle("busy", active.length > 0);
  $("#jobsLabel").textContent = !active.length ? "No jobs"
    : [running.length ? `${running.length} running` : null, active.length - running.length ? `${active.length - running.length} queued` : null].filter(Boolean).join(" · ");
  const r0 = running[0];
  $("#statusJobs").textContent = r0 ? `${r0.title} · ${Math.round(r0.progress * 100)}%${active.length > 1 ? ` (+${active.length - 1} more)` : ""}`
    : active.length ? `${plural(active.length, "job")} queued` : "No background jobs";
  if (!$("#jobsPanel").hidden) renderJobs();
  renderInlineJobs();
  renderDockJobs();
  updateStatusCells();
  if (lastActive && active.length < lastActive) {     // something finished
    loadTracks();
    if (state.current) openTrack(state.current.id);
  } else if (state.current && active.some((j) => j.track_id === state.current.id && j.kind === "stems")) {
    renderStemProgress();
  }
  lastActive = active.length;
  pollTimer = setTimeout(pollJobs, active.length ? 1000 : 4000);
}
// Stem progress of the open track, without re-rendering the whole workspace (keeps edits in fields).
function renderStemProgress() {
  const j = jobFor(state.current.id, "stems"), bar = $("#detail .d-stems .progress > i"), lbl = $("#detail .d-stems .stems-row .muted");
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
        h("div", { class: "muted" }, "Eject the drive before unplugging it."),
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

// ── Dock (Library view): device, export summary, jobs ──────────────────────────
const jobLabel = (j) => ({ queued: "Queued", running: `${Math.round(j.progress * 100)}%`, done: "Done", failed: "Failed", cancelled: "Cancelled" }[j.status]);
function renderDockJobs() {
  const box = $("#dockJobs");
  const list = [...state.jobs.filter(isActive), ...state.jobs.filter((j) => !isActive(j))].slice(0, 6);
  fill(box, list.length ? list.map((j) => h("div", { class: `dj ${j.status}`, title: j.message || j.title },
      h("div", { class: "dj-row" }, h("span", { class: `dot ${j.status === "running" ? "busy" : j.status === "done" ? "ok" : j.status === "failed" ? "bad" : ""}` }),
        h("span", { class: "dj-t" }, j.title), h("span", { class: `pill ${j.status}` }, jobLabel(j))),
      j.status === "running" ? h("div", { class: "progress" }, h("i", { style: `width:${j.progress * 100}%` })) : null,
      j.status === "failed" && j.message ? h("div", { class: "dj-msg" }, j.message) : null))
    : h("div", { class: "dock-note" }, "Nothing running."));
}
function renderDockExport() {
  const d = exp.drives.find((x) => x.root === exp.root);
  const n = +$("#allCount").textContent || 0, stems = state.playlist === 0 ? state.tracks.filter((t) => t.stems === "done").length : null;
  fill($("#dockExport"), h("div", {}, `Library: ${plural(n, "track")}` + (stems != null ? `, ${stems} with stems` : "")),
    h("div", {}, d ? `Card: ${driveName(d)}` : "Card: choose it in Export"));
}
$("#dockStems").addEventListener("change", () => { $("#includeStems").checked = $("#dockStems").checked; updatePlan(); });
$("#includeStems").addEventListener("change", () => { $("#dockStems").checked = $("#includeStems").checked; });
$("#dockExportBtn").addEventListener("click", () => showView("export"));
$("#dockFirmware").addEventListener("click", () => showView("deck"));
$("#dockDeckCheck").addEventListener("click", () => checkDeck());
$("#dockClear").addEventListener("click", () => api("/api/jobs/clear", { body: {} }).then(pollJobs));

// ── Export view ──────────────────────────────────────────────────────────────
const exp = { root: null, drives: [], plan: null };
// A rekordbox library we didn't write is never overwritten. Merging into it isn't built yet.
function foreignLibrary(why) {
  return h("div", { class: "suggest" },
    h("span", {}, why, " Exporting would replace it, so DKPRP won't."),
    h("button", { class: "ghost small", disabled: true, title: "Coming soon: keeps every existing track, cue and playlist" }, "Add to the existing library"),
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
      d.has_export ? h("div", { class: "muted" }, "Has a rekordbox library: checked in the plan before anything is written") : null);
    c.addEventListener("click", () => { exp.root = d.root; loadExport(); });
    return c;
  }) : [h("p", { class: "muted" }, "No removable drives found. Insert an SD card or USB stick, then Refresh.")]));
  const lists = state.playlists.filter((p) => !p.is_folder);
  const ticked = new Set($$("#exportPlaylists input:checked").map((i) => +i.value));
  $("#exportPlaylists").replaceChildren(...(lists.length ? lists.map((p) => h("label", { class: "check" }, h("input", { type: "checkbox", value: p.id, checked: ticked.has(p.id) }), `${p.name} (${p.count})`))
    : [h("span", { class: "muted" }, "No playlists yet.")]));
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
  if (!d) { box.replaceChildren(h("span", { class: "t-label" }, "Target"), h("span", { class: "muted" }, "No drive selected")); return; }
  const used = d.total ? (d.total - d.free) / d.total : 0, need = p && d.total ? Math.min(1 - used, p.bytes / d.total) : 0;
  fill(box, h("span", { class: "t-label" }, "Target"), h("b", {}, driveName(d)),
    h("span", { class: "muted" }, `${d.filesystem} · ${fmtBytes(d.free)} free of ${fmtBytes(d.total)}`),
    h("div", { class: "bar", "aria-hidden": "true" }, h("i", { style: `width:${used * 100}%` }),
      p ? h("i", { class: p.fits ? "need" : "over", style: `left:${used * 100}%;width:${(p.fits ? need : 1 - used) * 100}%` }) : null),
    p ? h("span", { class: "muted tnum" }, p.fits ? `Uses up to ${fmtBytes(p.bytes)}, leaving about ${fmtBytes(Math.max(0, p.free - p.bytes))}` : `Needs ${fmtBytes(p.bytes)}, only ${fmtBytes(p.free)} free`) : null);
}
let planSeq = 0;
async function updatePlan() {
  const plan = $("#plan"), btn = $("#doExport"), seq = ++planSeq;
  btn.disabled = true; btn.textContent = "Export";
  plan.classList.remove("bad");
  renderTarget(null);
  if (!exp.root) { plan.textContent = "Choose a drive to see what will be copied."; return; }
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
      ...deckWarnings(p.warnings),
      p.fits ? null : h("div", { class: "alert-text" }, h("b", {}, "Doesn't fit. "), "Choose fewer playlists or turn off stems."),
      p.target_problem ? h("div", { class: "alert-text" }, p.target_problem) : null,
      p.foreign_library ? foreignLibrary(p.foreign_library) : null,
      !p.tracks ? h("div", { class: "muted" }, "Nothing to export.") : null);
    const busy = state.jobs.some((j) => j.kind === "export" && isActive(j));
    btn.disabled = !p.fits || !p.tracks || !!p.target_problem || !!p.foreign_library || busy;
    const d = exp.drives.find((x) => x.root === exp.root);
    if (!btn.disabled) btn.textContent = `Export ${plural(p.tracks, "track")} to ${d?.label || d?.root || "drive"}`;
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
  $("#deckPillLabel").textContent = !s ? "DECK001 · checking" : ok ? `DECK001 · connected${ver ? " · " + ver : ""}` : "DECK001 · not connected";
  $("#statusDeck").replaceChildren(h("span", { class: "dot" + (ok ? " ok" : "") }), !s ? "DECK001 …" : ok ? "DECK001 connected" : "DECK001 offline");
  fill($("#dockDeckState"), h("span", { class: "dot" + (s ? (ok ? " ok" : "") : " busy") }), !s ? "Checking" : ok ? "Connected" : "Not connected");
  fill($("#dockDeck"),
    h("div", { class: "dev-id" + (ok ? " ok" : "") }, h("span", { class: "glyph", "aria-hidden": "true" }),
      h("div", {}, h("b", {}, "DECK001"), h("div", { class: "muted" }, ok ? `Firmware ${ver || "?"}` : "Firmware unknown until connected"))),
    h("div", { class: "dock-note" }, s?.address || ""));
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
    s.reachable ? null : h("p", { class: "hint", style: "margin:0" }, `Turn on Wi-Fi Remote on the deck and join its network to update or read its status. The address is set in Settings.`));
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
  renderSystem();
  const hist = await api("/api/history");
  $("#history").replaceChildren(...(hist.length ? hist.slice(0, 60).map((r) => h("div", { class: "h" },
    h("span", { class: "when" }, new Date(r.at * 1000).toLocaleString()),
    h("span", { class: "what" }, r.action.replaceAll("_", " ")),
    r.can_undo ? h("button", { class: "ghost small", onclick: () => doUndo(r.id).then(loadSettings) }, "Undo") : null))
    : [h("p", { class: "muted" }, "No changes yet.")]));
}
function renderSystem() {
  const s = state.system;
  $("#system").replaceChildren(h("span", { class: "k" }, "Application"), h("span", {}, `DKPRP ${s.version || ""}`),
    h("span", { class: "k" }, "Device"), h("span", {}, "DECK001"),
    h("span", { class: "k" }, "Library and stems"), h("span", {}, s.data_dir || ""));
  $("#statusVersion").textContent = s.version ? `DKPRP ${s.version}` : "";
}
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
  if (e.key === "Escape" && !$("#jobsPanel").hidden) { toggleJobs(false); return; }
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
  if (state.current && (e.code === "BracketLeft" || e.code === "BracketRight")) {
    e.preventDefault();
    gridNudge((e.code === "BracketLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1));
  }
  if (state.current && (e.key === "e" || e.key === "E")) toggleEditor();
});

async function refreshAll() { await loadPlaylists(); await loadTracks(); if (state.current) openTrack(state.current.id); }
renderTracks();
renderDeckPill();
refreshAll().then(pollJobs).catch(fail);
api("/api/system").then((s) => { state.system = s; renderSystem(); }).catch(() => {});
checkDeck();
