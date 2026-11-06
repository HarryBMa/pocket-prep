# DKPRP redesign: audit and notes

The UI was rebranded from "Pocket Prep" to **DKPRP** (the app) managing **DECK001** (the deck).
This file records what the redesign had to keep working, and what it deliberately did not touch.

## Stage 1 audit (before the redesign)

**Structure.** `ui/index.html` (static shell, four `<section class="view">`), `ui/style.css`
(tokens on `:root`, light/dark), `ui/app.js` (vanilla JS, ~670 lines, no build step). The Rust
server (`src/server.rs`, axum) embeds `ui/` with rust-embed and serves `/api/*` on 127.0.0.1.

**DOM integration points used by app.js** (all kept):
`#tracks` (+ `tbody`), `#contextActions`, `#detail`, `#empty`, `#emptyImport`, `#search`,
`#importFiles`, `#importFolder`, `#dropZone`, `#tree`, `#allCount`, `#newPlaylist`, `#newFolder`,
`#drives`, `#refreshDrives`, `#allDrives`, `input[name=scope]`, `#exportPlaylists`, `#includeStems`,
`#plan`, `#doExport`, `#exportJob`, `#deckStatus`, `#deckRefresh`, `#otaPick`, `#otaFile`, `#otaJob`,
`#ports`, `#buildDir`, `#flashBtn`, `#flashJob`, `#settingsForm` (named fields `auto_analyse`,
`bpm_min`, `bpm_max`, `deck_address`), `#system`, `#history`, `#jobsPanel`, `#jobsList`, `#jobsBtn`,
`#jobsDot`, `#jobsLabel`, `#clearJobs`, `#closeJobs`, `#themeBtn`, `#modal .modal-box`, `#toasts`,
`#player`, `.tabs button[data-view]`, `.view#view-*`.

**API used by the UI** (unchanged): `GET /api/tracks?q=&playlist=`, `GET|PATCH /api/tracks/{id}`,
`/waveform`, `/audio`, `/artwork`, `/stem/{name}`, `PUT /cues`, `POST /api/import`, `/api/pick`,
`/api/tracks/remove`, `/api/analyse`, `/api/stems`, `GET|POST /api/playlists`,
`PATCH|DELETE /api/playlists/{id}`, `PUT /tracks`, `POST /add`, `POST /api/undo/{id}`,
`GET /api/history`, `GET /api/jobs`, `POST /api/jobs/{id}/cancel`, `/api/jobs/clear`,
`GET /api/drives?all=`, `POST /api/export/plan`, `/api/export`, `GET /api/deck/status`,
`/api/deck/ports`, `POST /api/deck/ota` (multipart), `/api/deck/flash`, `GET|PUT /api/settings`,
`GET /api/system`.

**Tokens.** Tallgeese greys (`--bg --surface --surface-2 --line --line-strong --ink --ink-2 --ink-3`),
Hot Coral `--accent`, `--alert #ff5454`, `--oxblood`, `--ok`, three waveform band colours.
Work Sans 400/500/600 and Fugaz One are bundled in `ui/fonts`.

**Limits that shaped the redesign.**
- No build step, no framework: everything is hand-written DOM via the `h()` helper.
- wavesurfer.js 7.12 Regions draws markers as absolutely-positioned divs. Every beat is a marker
  (thin, half-transparent; downbeats solid) for normal tracks; above 3000 beats (long mixes) only
  downbeats are drawn. Cue labels are styled inline because the waveform lives in a shadow root.
- `/api/tracks` (list) has no per-track analysis job id; analysis state comes from `analysed_at`
  plus the jobs list (`track_id`, `kind`).
- `/api/deck/status` is a blocking request with a 2.5 s timeout; the app checks it at start-up and
  on the Deck view, never on a timer.
- Internal identifiers stay: crate `pocket-prep`, bundle id `com.pocketdeck.pocketprep`, data folder
  `PocketPrep`, env vars `POCKET_PREP_*`, and the `"app": "Pocket Prep"` marker in `export.json`
  on the card. That marker is how foreign rekordbox libraries are told apart from ours; renaming it
  would make every existing stick look foreign.

## What changed (additive only)
- `GET /api/tracks/{id}` also returns `sections` (kind, start/end, bars, energy) from the analysis,
  so the prep view can draw a phrase strip. No existing field changed.
- Window title and Tauri `productName` are DKPRP. User-facing server messages say DKPRP.

## Colour beyond Hot Coral (asked for after the first redesign)
- **Waveform:** a rekordbox-style 3-band view from the PWV5 data (lows coral, mids blue, highs
  white in dark / navy in light). Each pixel column draws its tallest band first, so all three stay
  visible. WaveSurfer hands a custom renderer only two channels, so it gets one channel of column
  positions and `drawBands` reads the bands from `state.bands`.
- **Hot cues A–H** each have a colour (pads and waveform markers). DKPRP only: the card's PCOB
  cues carry no colour, so the deck and CDJs use their own.
- **Transport bar:** previous/next track, seek, preview tempo ±8 % with key lock, and volume.
  Playback only; nothing is saved.
- **Dock** (Library, windows 1480 px and wider): DECK001 status, what an export writes, the stems
  switch (the same setting as in Export), and recent jobs. Export itself still happens in the Export
  view, behind the drive checks.
