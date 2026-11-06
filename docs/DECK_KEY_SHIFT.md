# Key shift: what DKPRP writes and what the deck does with it

A track can carry a **key shift** of −12…+12 whole semitones, set in DKPRP (Track, "Key shift").
It lets you bring a track into the key of the one before it. The audio file is **not** changed: DKPRP
copies it byte for byte as always, and the deck shifts the pitch while it plays.

Why the deck and not the file: a shifted copy would have to be re-encoded (MP3 and FLAC are copied
as they are, and the stems would need shifting too), it would double the space on the drive, and
the shift could no longer be changed or undone on the deck. CDJs and rekordbox ignore the field and
play the original, so the same drive stays correct everywhere.

## Where it is

`POCKETDECK/analysis/<id>.json`, next to `bpm`, `key` and `camelot`:

```json
{ "file_path": "/Contents/...", "bpm": 126.0, "key": "C", "camelot": "8B", "key_shift": 3 }
```

- `key_shift`: integer, −12…12, **0 when there is none**. A missing file, a missing field, or a value
  that isn't an integer in range means 0.
- `key` and `camelot` stay the key **as analysed**, and so does the rekordbox key in `export.pdb`.
  The key the DJ should think in is the shifted one.

Shifted key (what DKPRP shows): move the tonic by `key_shift` semitones and keep the mode. On the
Camelot wheel that is **7 steps per semitone**: 8B (C) shifted +3 is 5B (Eb). DKPRP names the keys
C C# D Eb E F F# G Ab A Bb B, with an `m` for minor (`analysis::shifted_key`).

## What the deck does

1. When a track loads, read `key_shift` from its analysis file (match by `file_path`, like the other
   hints in `POCKETDECK/`). Anything unreadable means 0. Never refuse a track because of it.
2. Call `ds_player_set_key_shift(player, key_shift)` (deck-stretch, `include/ds_player.h`). The shift is
   applied by the same stretcher as keylock: the tempo is not affected.
   - Keylock on: heard pitch = track pitch + shift.
   - Keylock off: heard pitch = varispeed pitch + shift.
3. Show the shifted key (and the shift, for example `+3`) wherever the deck shows the key.
4. The deck's own key shift control, if it gets one, starts from this value and may change it for
   the session only. Writing it back is a separate decision.

Cost: a track with `key_shift` ≠ 0 uses about a third more CPU in the stretcher than a keylocked
track (measured on PC; not yet on the M85). A track with 0 costs nothing extra.
Quality is best within about ±7 semitones.
