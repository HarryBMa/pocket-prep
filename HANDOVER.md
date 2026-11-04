# Pocket Prep: handover

Companion prep app for the Pocket Deck: library, analysis, Demucs stems, rekordbox-format
export to SD/USB, deck firmware updates. Rewritten from Python to Rust + Tauri on 2026-11-04
so it ships as a normal app on Windows and macOS. Keep the code readable and commented,
and explain choices.

## Run it
    cargo run --release                 # desktop window (Tauri)
    cargo run --release -- --browser    # same UI in the browser, http://127.0.0.1:8765
    cargo test --release -- --include-ignored     # includes Demucs vs PyTorch check
    tests/e2e.sh <export dir> <audio files...>    # import, analyse, stems, export, verify card
    npx @tauri-apps/cli@2 build         # installers (CI: .github/workflows/build.yml, Win + both Macs)

One-time: the Demucs model isn't in git. Make it with the old venv or plain uv (see the script's
docstring): `uv run ... tools/export_demucs_onnx.py models/htdemucs.onnx`.
Data lives in %APPDATA%\PocketPrep or ~/Library/Application Support/PocketPrep (override with
POCKET_PREP_DATA). The database and analysis JSON are the same format as the Python version, so an
existing library carries over. Old MP3 stems are marked "not made" and need remaking (new format).

## Layout (src/)
- `server.rs`: axum routes, same /api/* as before, so `ui/` (vanilla JS) barely changed. Listens on
  127.0.0.1 only and rejects foreign Host headers (DNS rebinding).
- `main.rs`: starts the server, opens the Tauri window. `--browser`, `--headless`.
- `library.rs`, `db.rs`: import (lofty tags, artwork), playlists, cues, undo history.
- `audio.rs`: Symphonia decode; rubato resample (used only to feed Demucs).
- `analysis.rs`: runs at the file's own sample rate (no resampling). Tempo = onset autocorrelation
  then the same 0.01 BPM grid search as before. On the 4 test tracks the grids lock to the beat
  better than the Python ones (Python drifted on 3 of 4). Key: STFT chroma + Krumhansl (was CQT).
- `stems.rs`: htdemucs in ONNX Runtime, CPU (~35 s for a 4 min track). STFT/iSTFT done in Rust,
  verified against PyTorch. Writes one 8-channel 16-bit 44.1 kHz WAV per track.
- `export/`: pdb.rs, anlz.rs (straight ports), usb.rs (card layout, deck-decodability warnings).
- `deck.rs`: OTA over Wi-Fi (reqwest), wired flash with the espflash library (no Python/esptool).
- `jobs.rs`: cpu / gpu / io worker lanes.

## Card format decisions
- Audio files are copied **byte for byte**, never transcoded or resampled. The PDB sample_rate,
  bitrate and bit depth come from the file itself. The deck resamples everything to its 48 kHz output.
- The deck decodes MP3, FLAC and 16-bit PCM WAV only. Anything else (M4A, AIFF, 24-bit WAV) is
  still copied (CDJs play it) but listed as a warning in the export plan, the result and export.json.
- Stems: see docs/STEMS_FORMAT.md (8-ch interleaved, data at byte 4096, timed in seconds).

## Not verified / known gaps (priority order)
1. **UI not clicked through** in the Tauri window (routes all smoke-tested over HTTP; the window opens).
2. **macOS not built or run here.** CI builds both Mac targets; the first Mac run needs a person.
   Unsigned builds need right-click > Open; signing/notarisation isn't set up.
3. **Dragging files from Explorer/Finder no longer imports.** The window keeps HTML drag and drop
   (needed for tracks onto playlists), and web views don't expose dropped file paths. Use Add files/folder.
4. GPU stems removed: DirectML returned wrong results (its whole-tensor ReduceMean, then the first
   time-branch Conv on 86k-wide tensors); CoreML untested. Revisit with a newer ONNX Runtime.
5. Stems are hard-clipped at 16-bit: on a brickwalled master ~0.01 % of drum samples clip.
6. CDJ compatibility of export.pdb untested (data pages only, no index pages). Pajoniiir accepts it.
7. No PQT2 / PCO2 / PWV6/7, constant-tempo grid only, key and energy are heuristics.
8. Export rewrites all ANLZ files and the PDB each time; doesn't remove deleted tracks from the stick.
9. Firmware doesn't read POCKETDECK/ yet (stems.json, STEMS/, analysis/*.json).

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
