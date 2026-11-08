# What Pocket Prep writes to a drive

A Pocket Prep export is a **standard rekordbox USB export** plus an optional Pocket Deck
folder. CDJs, XDJs and rekordbox read the standard part; the Pocket Deck reads both.

```
Contents/<Artist>/<Album>/<file>                  audio, copied byte for byte
PIONEER/rekordbox/export.pdb                      the library: tracks, playlists, keys, artwork
PIONEER/USBANLZ/Pxxx/xxxxxxxx/ANLZ0000.DAT|.EXT   beat grid, waveforms, hot and memory cues
PIONEER/Artwork/00001/a<id>.jpg (+ _m.jpg)        cover art

POCKETDECK/export.json                            what wrote the drive, when, warnings,
                                                  and the hash of the export.pdb it wrote
POCKETDECK/stems.json                             stem file per track (docs/STEMS_FORMAT.md)
POCKETDECK/STEMS/xxxxxxxx.flac                    8-channel stems
POCKETDECK/analysis/xxxxxxxx.json                 sections, phrases, energy, loudness_lufs, gain_db, key, key_shift, mix_in_ms, mix_out_ms,
                                                  audible_start_ms, audible_end_ms, cue_labels
```

## Rule: everything in POCKETDECK/ is optional

**The deck must work fully without `POCKETDECK/`.** Browsing, loading, playing, beat
grid, waveforms, hot cues, memory cues and loops all come from the rekordbox part
(`export.pdb` and the ANLZ files), exactly as on a CDJ. So:

- A drive exported by rekordbox itself (no `POCKETDECK/`) is a first-class library.
- A drive where `POCKETDECK/` is missing, partly copied, from an older or newer Pocket
  Prep, or unreadable must load and play normally. Firmware treats every
  `POCKETDECK/` file as a hint: read it if it parses and matches the track (by the
  `export.pdb` file path), ignore it otherwise, and never refuse a track or a drive
  because of it.
- Features that need it degrade quietly: no stems (the stem controls are greyed out),
  no autopilot phrases or energy (fall back to the beat grid and the waveform), no
  loudness gain (use 0 dB).
- Nothing in `POCKETDECK/` repeats or overrides rekordbox data the deck relies on. Cues,
  the grid, key and BPM are read from rekordbox files only, so there is one truth.

For Pocket Prep, `POCKETDECK/export.json` has one more job: it marks the library as
ours. If a drive has `PIONEER/rekordbox/export.pdb` but no Pocket Prep `export.json`,
or `export.pdb` no longer matches the hash in it (rekordbox rewrote it), Pocket Prep
refuses to export there, so it never overwrites someone else's library. See
`foreign_library` in `src/export/usb.rs`.

## POCKETDECK/sets/<playlist id>.json

One file per exported playlist (`sets.rs::card_json`): `name`, `duration_s`, `tracks` (track id, `file_path` on the
card, title, artist, bpm, camelot after the key shift, energy) in play order, and `transitions[i]` between
`tracks[i]` and `tracks[i+1]`: `beats`, `out_ms`, `in_ms`, `verdict` (good | ok | no), `score` (0-100) and the
`reasons` as text, or null when a side isn't analysed.

## Time signature

`POCKETDECK/analysis/<id>.json` carries `beats_per_bar` (3 or 4 from the analysis, or what the DJ set, 2-12). The
ANLZ beat grid (PQTZ) numbers its beats 1..`beats_per_bar` from the downbeat, as rekordbox numbers 1-4; a CDJ's
phase meter shows those numbers as they are.
