# Pocket Deck stem files (POCKETDECK/STEMS)

Written by Pocket Prep 0.2+. Other players ignore the `POCKETDECK/` folder.
The firmware doesn't read stems yet; this is the contract for when it does.

## Files on the card

```
POCKETDECK/stems.json              index: which stem file belongs to which track
POCKETDECK/STEMS/<ID>.wav          one file per track, ID = rekordbox track id as 8 hex digits
Contents/.../<track>               the original audio, copied byte for byte (untouched)
```

`stems.json`:

```json
{
  "version": 2,
  "tracks": {
    "/Contents/Artist/Album/Track.mp3": {
      "file": "/POCKETDECK/STEMS/0000002A.wav",
      "sample_rate": 44100,
      "channels": 8,
      "frames": 11192276,
      "data_offset": 4096,
      "order": ["drums", "bass", "other", "vocals"]
    }
  }
}
```

The key is the same path as the track row's file path in `export.pdb`
(string 20) and the `PPTH` in its ANLZ files.

## The WAV

| Field            | Value                                                   |
|------------------|---------------------------------------------------------|
| Container        | RIFF/WAVE: `fmt `, `JUNK` (padding), `data`, in that order |
| Format tag       | 1 (plain PCM, not WAVE_FORMAT_EXTENSIBLE)               |
| Channels         | 8                                                       |
| Sample rate      | 44100 Hz (Demucs's own rate, whatever the track's rate) |
| Bits             | 16, signed little-endian, TPDF-dithered                 |
| Block align      | 16 bytes = one frame                                    |
| `data` payload   | starts at byte **4096** of the file                     |

Channel order inside each 16-byte frame:

| Bytes | 0-1     | 2-3     | 4-5    | 6-7    | 8-9     | 10-11   | 12-13    | 14-15    |
|-------|---------|---------|--------|--------|---------|---------|----------|----------|
|       | drums L | drums R | bass L | bass R | other L | other R | vocals L | vocals R |

"other" is everything that isn't drums, bass or vocals (melody, pads, FX).
The four stems add back up to the original mix (within the model's error).

## Why this layout

- **No decoding.** Four compressed streams plus the main track is too much for
  the MCU; PCM is a straight copy into the mixer.
- **One file, one sequential read.** All four stems for a moment in time sit
  in one 16-byte frame, so the deck streams a single file instead of seeking
  between four. ~1.4 MB/s at 44.1 kHz, well inside SD bandwidth.
- **Sector aligned.** Data starting at 4096 means frame `n` is at byte
  `4096 + 16 n`, and reads of whole 512-byte sectors never straddle the header.
  The `JUNK` chunk is skipped by any RIFF parser (Pajoniiir's
  `audio_wav_decoder.c` already walks and skips unknown chunks).

## Timing and sample rate

Stems are always 44.1 kHz. The main track keeps its own rate (44.1, 48, 96 kHz
...). Both describe the same real time: stem frame `n` is at `n / 44100` s
from the start of the track, the same moment as main-track sample
`n × track_rate / 44100`. The deck already resamples each source to its 48 kHz
output, so it resamples the stem stream with its own ratio (44100 → 48000)
and drives both from the same playhead in seconds. `frames / 44100` equals the
track's duration.

Beat grid, cues and phrases on the card are all in milliseconds, so they apply
to the stems unchanged.

## Firmware checklist

1. Look up the track's file path in `stems.json`; no entry = no stems.
2. Open the WAV, check `fmt ` = PCM, 8 ch, 16 bit, 44100; seek to the `data` chunk.
3. Read frames in step with the main track's playhead (in seconds); resample 44.1 → 48 kHz.
4. Per stem: gain (mute/solo/fader) per L/R pair, then sum the four into the deck's channel.
