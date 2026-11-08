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
- **Hot cues A–H** use the DECK001 firmware's fixed slot colours (`cues_storage.h`) on the pads
  and waveform markers, and the playhead is the deck's amber (`kPlayhead`). The card's PCOB cues
  carry no colour; the deck colours by slot, so the two always agree.
- **Icons** are the deck's Lucide set (`assets/icons/`) wherever the deck has the same control,
  and plain Lucide elsewhere; mapping in `ui/icons/README.md`.
- **Transport bar:** previous/next track, seek, preview tempo ±8 % with key lock, and volume.
  Playback only; nothing is saved.
- **Dock** (Library, windows 1480 px and wider): DECK001 status, what an export writes, the stems
  switch (the same setting as in Export), and recent jobs. Export itself still happens in the Export
  view, behind the drive checks.
- **Beat grid nudge** (editor, "Beat grid"; keys `[` `]` 1 ms, Shift 10 ms): moves every beat,
  and "Downbeat +1" moves the bar start one beat. Saved per track (`grid_offset_ms`,
  `downbeat_shift`), one undo per burst of nudges, written to the card's PQTZ by the next export.
  A re-analysis that moves the grid clears the nudge (logged in History). Cues stay where they
  are in the audio; phrase sections in POCKETDECK/analysis are not shifted.

## v3: one prep screen (replaces the transport bar and dock above)
DKPRP is for preparing tracks, not for playing or mixing, so the Library became one screen:
- **Sidebar:** the logo, Library, Playlists (folds away), Analysis queue (the jobs drawer), Export,
  Deck, Settings. **Top bar:** search, Import, the DECK001 status pill, the theme switch.
- **Track table:** artwork, title, artist, BPM, key (Camelot colour), energy, stems and analysis
  state, a ⋮ menu per row. Filters: All, Analysed, Unanalysed, With stems, Needs attention, the last
  counted in the browser: a format the deck can't play, no artist, failed analysis or stems.
- **Track panel** (right): waveform with stem rows (FULL, DRUMS, BASS, MELODY, VOCALS) and an
  OVERVIEW strip, hot cues A–H (Edit shows memory cues), phrases, beat grid nudge, stems, tag
  fields under "Track details", and what an export writes. Stem rows read
  `GET /api/tracks/{id}/stems/peaks` (new; per-stem peaks from the stem file, 150 columns/s).
- **Preview bar** (bottom): previous/next, play, seek, volume. The tempo/key lock preview, the dock
  and the status bar are gone; their information is in the sidebar, top bar and panel.
- **Waveform themes** (Settings → Waveforms): the DECK001 themes Coral/Teal, Graphite and Valve
  times the styles Contour solid, Contour frequency (default), Frequency and Rainbow, with a live
  preview. Saved per computer (browser storage); the deck keeps its own setting.
- **Logo:** five solid coral bars, the deck's solid contour style. `tools/make_icons.py` (Pillow)
  draws the app icons from the same geometry.
- **Mood** (next to Key and Energy): one label per track (Deep, Melodic, Tech house, Peak time,
  Hard techno, Drum & bass, Mainstream pop, Downtempo). It is repeatable and learns from the
  moods you set by hand. The library can be grouped by it. Method and limits: `docs/MOOD.md`.

## Working in the list and the waveforms (0.5)
- **Resize**: drag the edge of the sidebar or the track panel (or focus it and use the arrow keys, Shift for bigger
  steps; double-click resets). The widths are remembered.
- **Quick edits**: click a cell of the selected row (F2 for the title). Title, artist and tempo become a box
  (Enter saves, Esc cancels, Tab moves on); key (a Camelot grid), mood and energy open a picker. A tempo, key or energy
  you set is kept by a re-analysis (`tracks.edited`); "Analysed value" puts the analysed one back. Every edit can be undone.
- **Convert for the deck**: the deck plays MP3, FLAC and 16-bit WAV. For a 24-bit WAV, an AIFF or an M4A the panel, the
  row menu and the selection bar offer a 16-bit FLAC copy (dithered, same sample rate) in `converted/` in the data folder;
  the export writes the copy instead of the original, "Use the original" removes it.
- **Waveforms**: a time ruler under the main waveform, the time under the pointer, and the current time in the header.
  Click the lane labels, or focus the waveforms and press Up / Down, to choose the full mix or a stem to listen to;
  Left / Right move the playhead (1 s, Shift 10 s, Alt a beat, Ctrl a bar); Page Up / Page Down jump between cues.
  Hot and memory cues and the downbeats are drawn on every stem lane, and a double-click on any lane sets the next free
  hot cue there.
- **Mixes with selected**: select one track and choose the chip; the list shows the tracks in the same key, one step
  either way on the Camelot wheel, or the other mode of the same number, within 6 % of its tempo (the shifted key counts).
- **Needs attention** also flags a file that can't be found (moved, renamed, drive not connected) and tracks that look
  like the same song twice (same artist and title, ignoring case and "(Original Mix)").
- **Edit selected…** (selection bar, two or more tracks): set the artist, album, genre, year, comment or key shift on all
  of them at once; each change can be undone.
- **Loudness**: a Loudness column (integrated LUFS) and a target in Settings (default -9 LUFS). The export writes
  `loudness_lufs` and the `gain_db` that reaches the target (limited to -12..+6 dB) into POCKETDECK/analysis for the deck.
- **Near silence at the start or end**: tracks that start with more than 2 s or end with more than 6 s of near silence
  (30 dB under the loud level) are flagged in "Needs attention", and the export writes `audible_start_ms` and
  `audible_end_ms` so an autopilot mix doesn't wait for dead air. Older tracks are measured by a "Loudness and silence
  check" job after the update.
- **Cue names**: in Edit mode a hot cue (and a memory cue, "Name") can be named; the name shows on the pad and goes to the
  deck as `cue_labels` in POCKETDECK/analysis, because the rekordbox files have no field for it.
- **Check before exporting**: the plan lists, per track, what the export will do and what to know first (unplayable file,
  missing file, converted copy used, no stems, near silence, key shift, loudness gain). Nothing is written until Export.

## v4: one bar, the list with the card beside it, the open track full screen
- **Top bar only**: logo (= Library), search, Playlists (a popover), then the glyphs: analysis queue
  (a ring), deck (dot), card (count), add files, `+` import, theme, settings. No sidebar, no titles.
- **Library**: filters are glyphs; energy is five bars; state is three glyphs per row (stems,
  analysis, on the card: from the export plan). Double-click or Enter opens the track, a click loads
  it into the preview bar.
- **Card column** (`#view-export`, always beside the list): drive, free space, scope and stems
  toggles, the plan and its check, one `N →` button. Folds with the card glyph, resizable.
- **Open track** (`#detail.track-focus` over the list): waveform stack filling the height, dot lane
  labels with the name on hover, cue pads, edit-cues pencil, grid-nudge and key-shift pills,
  listen-to dots, zoom, details (ⓘ). Esc or the chevron in the preview bar goes back.
- Toasts show under the top bar. Deck and Settings have a back arrow.
- Product direction and the lean-installer plan: `docs/DECK001_PRODUCT_SPEC.md`.
- **Mix in / out points** (0.6): the analysis places them (ClubCraft's planner: the first full-groove phrase, the
  bar before the energy falls for good). White flags on every lane, hollow while analysed, solid once yours; the
  `IN 0:16 · OUT 5:12` pill in the tool row jumps there, `I` / `O` set them at the playhead (beat-snapped), the
  refresh glyph goes back to the analysed ones. Saved per track, undoable, written to POCKETDECK/analysis as
  `mix_in_ms` / `mix_out_ms` and named in the dry run.
- **Intent and readiness** (0.6): each track has an intent (Mix · Vocal layer · Remix · Section · Backup: the select
  in the open track, "Prepare as" in the row menu, "Prepare…" on a selection) that decides what gets made and what
  "ready" means. `POST /api/prepare` queues analysis, the stems the intents ask for (sections first) and deck
  copies; "Prepare everything" sits in the jobs drawer. The Ready column is one ring per track: plays on the deck ·
  analysed · grid confirmed (✓ in the grid pill, or a nudge) · hot cues · stems (only for intents that want them)
  · loudness · on the card (from the plan). Green = done, grey = to do, red = blocked, coral = running; the
  tooltip lists what's missing. The stems and card glyph columns are gone: the ring carries them.
- **Right-click menu** (0.6): on a row (a selected row acts for the whole selection), via the row's ⋮, Shift+F10
  or the menu key, and in the open track's ⋮: Open, Mixes with this, Prepare as (the intent, ticked), Re-analyse,
  Make stems, Convert for the deck, Confirm the beat grid, Add to playlist (counts), New playlist with these, Remove
  from the playlist, Remove from library. Playlists get their own: Open, Add the selected, Export only this
  playlist, New playlist in this folder, Rename, Delete. Grouped, with glyphs and key hints, in the app's style.
