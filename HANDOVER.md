# Pocket Prep: handover

Companion prep app for the Pocket Deck: library, analysis, Demucs stems, rekordbox-format
export to SD/USB, deck firmware updates. Rewritten from Python to Rust + Tauri on 2026-11-04
so it ships as a normal app on Windows and macOS. Keep the code readable and commented,
and explain choices.

## Run it
    cargo run --release                 # desktop window (Tauri)
    cargo run --release -- --browser    # same UI in the browser, http://127.0.0.1:8765
    cargo test --release                # unit tests + full export read back with rekordcrate
    POCKET_PREP_KNOWN_TRACKS=<dir> cargo test --release -- --ignored known_tracks   # real tracks
    tests/e2e.sh <export dir> <audio files...>    # import, analyse, stems, export, verify card
    npx @tauri-apps/cli@2 build         # installers (CI: .github/workflows/build.yml, Win + both Macs)

One-time: the Demucs model isn't in git (174 MB). Make it with uv (see the script's docstring):
`uv run --with ... tools/export_demucs_onnx.py models` -> models/htdemucs.onnx + htdemucs.json.
Building needs cmake (ClubCraft's decoder builds libopus). The repo is in git: commit after every change.
Data lives in %APPDATA%\PocketPrep or ~/Library/Application Support/PocketPrep (override with
POCKET_PREP_DATA). The database and analysis JSON are the same format as the Python version, so an
existing library carries over. Old MP3 stems are marked "not made" and need remaking (new format).

## Layout (src/)
- `server.rs`: axum routes, same /api/* as before, so `ui/` (vanilla JS) barely changed. Listens on
  127.0.0.1 only and rejects foreign Host headers (DNS rebinding).
- `main.rs`: starts the server, opens the Tauri window. `--browser`, `--headless`.
- `library.rs`, `db.rs`: import (lofty tags, artwork), playlists, cues, undo history.
- `analysis.rs`: ClubCraft's `clubcraft-analysis` (MIT/Apache, pinned commit) at the file's own rate:
  beat grid, key (EDM profiles), loudness, energy, sections. We add the rekordbox waveforms and our
  auto cues (A start, B first drop, C breakdown, D outro; phrase lines, <= 1 per 16-bar phrase).
  Analysis VERSION 3; older analyses are redone at startup.
- `stems.rs`: ClubCraft's `clubcraft-stems` (htdemucs, Quality::Fast): CoreML on Mac if the
  .mlmodelc exists, DirectML on Windows only if its self-check vs CPU passes, else CPU (~41 s per
  4:38 track). We write one 8-channel 16-bit 44.1 kHz FLAC per track (docs/STEMS_FORMAT.md v3).
- `export/`: pdb.rs, anlz.rs (our writers; rekordcrate can't create a PDB yet, but the tests parse
  our output with it), usb.rs (card layout; removable drives only unless "Show all"; never the
  system drive; FAT32/exFAT only; deck-decodability warnings).
- `ui/`: vanilla JS; waveform = wavesurfer.js 7 (vendored, BSD-3).
- `deck.rs`: OTA over Wi-Fi (reqwest), wired flash with the espflash library (no Python/esptool).
- `jobs.rs`: cpu / gpu / io worker lanes.

## Card format decisions
- Audio files are copied **byte for byte**, never transcoded or resampled. The PDB sample_rate,
  bitrate and bit depth come from the file itself. The deck resamples everything to its 48 kHz output.
- The deck decodes MP3, FLAC and 16-bit PCM WAV only. Anything else (M4A, AIFF, 24-bit WAV) is
  still copied (CDJs play it) but listed as a warning in the export plan, the result and export.json.
- Stems: see docs/STEMS_FORMAT.md (8-ch interleaved, data at byte 4096, timed in seconds).

## Not verified / known gaps (priority order)
1. **Keys unconfirmed** on real tracks: tests/known_tracks.json holds ClubCraft's keys; confirm by ear
   or Mixed In Key and update the file.
2. **Mac window never opened by a person** (built and tested over SSH on `brain`). Unsigned: right-click
   > Open. No CoreML model yet (needs coremltools on a Mac), so Mac stems run on the CPU (~100 s/track).
3. **Dragging files from Explorer/Finder doesn't import** (HTML drag and drop is kept for playlists).
4. CDJ compatibility of export.pdb untested on hardware (data pages only, no index pages).
   rekordcrate and Pajoniiir both parse it.
5. Stems are hard-clipped at 16-bit: on a brickwalled master ~0.01 % of drum samples clip.
6. GPU stems: DirectML fails its self-check because ONNX Runtime's DML graph fusion miscomputes this
   model (-82 dB vs CPU); with session config `ep.dml.disable_graph_fusion=1` it is correct (72/91 dB)
   and ~0.087 s per 7.8 s segment (CPU 0.77 s). Needs a one-line change in ClubCraft's
   crates/stems/src/backend/onnx.rs (fork or upstream), then delete data/clubcraft/directml-htdemucs.txt.
7. Export rewrites all ANLZ files and the PDB each time; doesn't remove deleted tracks from the stick.
8. Firmware doesn't read POCKETDECK/ yet, and has no file-upload API (see docs/SYNC_OPTIONS.md).

## Format notes worth keeping
- PDB: little-endian, 4096-byte pages, 0x28-byte page header. Row slots grow backwards from the page
  end in groups of 16: [offsets...][present bitmask u16][u16 0]. Track row: 0x5E fixed bytes + 21
  string offsets (14 anlz path, 17 title, 19 filename, 20 file path). Artist row: subtype 0x60, name
  offset byte at +9. Album: subtype 0x80, name offset byte at +21. Key row: id, id, name.
- DeviceSQL strings: short ASCII = ((len+1)<<1)|1, then the bytes; long ASCII = 0x40 + u16 len + pad;
  UTF-16LE = 0x90 + u16 len + pad.
- ANLZ: big-endian. PMAI header size 28, and its segment size = whole file. Pajoniiir walks sections and
  rejects any trailing bytes. PWV3 needs entry size 1, count = data length, marker 0x00960000.
  PWV4 = 6-byte entries; Pajoniiir reads [0] as height and [3..5] as colour.
- Demucs ONNX: inputs/outputs are normalised outside the graph (DirectML bug, and it keeps the graph
  simple). Segment 343980 samples, 25 % overlap, triangular weights, no random shift.

## Conventions
Follow Harry's design notes: utilitarian (gov.uk-like), Tallgeese greys, Hot Coral accent only,
#ff5454 for alerts only, destructive actions in outlined oxblood, light and dark modes, actions
appear in context, undo instead of confirmation dialogs (except flashing firmware), every job shows
queued, then running, then done, and the app must work offline.
