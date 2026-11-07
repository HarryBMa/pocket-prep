# DKPRP

Preparation app for the DECK001 (formerly Pocket Prep / Pocket Deck; an ESP32-P4 standalone DJ deck). Import music, analyse it (tempo, beat
grid, key, energy, sections, hot cues), split stems (Demucs), and export a
rekordbox-format USB drive that the DECK001 and CDJs read. Works offline. Windows and macOS.

## Install

Download from [Releases](../../releases).

**macOS (Apple Silicon, macOS 13 Ventura or newer):** open the `.dmg` and drag DKPRP
to Applications. The app isn't notarized by Apple yet, so the first time:

1. Open DKPRP once; macOS says it can't verify the developer. Click **Done**.
2. Open **System Settings > Privacy & Security**, scroll down, click **Open Anyway** next
   to "DKPRP", and confirm.

After that it opens normally.

**Windows 10/11 (x64):** run `DKPRP_x.y.z_x64-setup.exe`. SmartScreen may warn
because the installer isn't code-signed: click **More info > Run anyway**.

## Using it

1. **Add folder** (Library) to import music; analysis runs automatically.
2. Select tracks and **Make stems** (about 10-30 s per track with a GPU).
3. **Export** to an exFAT or FAT32 USB drive. Drives that already hold a rekordbox library
   from another program are never overwritten.

## Building from source

See [HANDOVER.md](HANDOVER.md) (layout, build, tests) and [docs/](docs/) (card layout,
stem format, design notes). In short: Rust, cmake, Node (for the Tauri CLI), the Demucs
model from `tools/export_demucs_onnx.py`, then `npx @tauri-apps/cli@2 build`.

## Third-party parts

Demucs (Meta, MIT), [ClubCraft](https://github.com/beneaze/ClubCraft) crates (MIT/Apache-2.0,
vendored in `vendor/clubcraft`), [rekordcrate](https://github.com/Holzhaus/rekordcrate)
(MPL-2.0, tests only), wavesurfer.js (BSD-3-Clause), Work Sans and Fugaz One fonts
(SIL OFL 1.1), ONNX Runtime (MIT), Tauri (MIT/Apache-2.0).
