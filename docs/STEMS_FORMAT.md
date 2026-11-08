# Pocket Deck stem files (POCKETDECK/STEMS), format version 3

Written by Pocket Prep. Other players ignore the `POCKETDECK/` folder.
The firmware doesn't read stems yet; this is the contract for when it does.

| Version | Stem file | Notes |
|---|---|---|
| 2 | 8-channel 16-bit WAV, data at byte 4096 | Pocket Prep 0.2. Converted to version 3 automatically, losslessly. |
| **3** | **8-channel 16-bit FLAC, SEEKTABLE point every 100 ms** | Current. Half the card space of version 2. |

## Files on the card

```
POCKETDECK/stems.json              index: which stem file belongs to which track
POCKETDECK/STEMS/<ID>.flac         one file per track, ID = rekordbox track id as 8 hex digits
Contents/.../<track>               the original audio, copied byte for byte (untouched)
```

`stems.json`:

```json
{
  "version": 3,
  "tracks": {
    "/Contents/Artist/Album/Track.mp3": {
      "file": "/POCKETDECK/STEMS/0000002A.flac",
      "format": "flac",
      "sample_rate": 44100,
      "channels": 8,
      "frames": 11192276,
      "seek_interval_frames": 4410,
      "order": ["drums", "bass", "other", "vocals"]
    }
  }
}
```

The key is unchanged from version 2: the same path as the track row's file path in
`export.pdb` (string 20) and the `PPTH` in its ANLZ files. `frames` is samples per channel.

## The FLAC file

| Field | Value |
|---|---|
| Channels | 8, independent (no stereo decorrelation across stems) |
| Sample rate | 44100 Hz (Demucs's own rate, whatever the track's rate) |
| Bits | 16, TPDF-dithered from the model's float output, hard-clipped at full scale |
| Block size | **fixed 4410 samples = 100 ms** (the last block is shorter). STREAMINFO min = max = 4410 |
| Metadata | STREAMINFO, SEEKTABLE, VORBIS_COMMENT (`POCKETDECK_STEMS=drums,bass,other,vocals`) |
| Encoder | flacenc 0.5, default settings |

Channel order in every decoded frame, as FLAC channels 0-7:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| drums L | drums R | bass L | bass R | other L | other R | vocals L | vocals R |

"other" is everything that isn't drums, bass or vocals (melody, pads, FX). The four
stems add back up to the original mix (within the model's error).

## How seeking works

Every FLAC frame holds exactly 4410 samples, and the SEEKTABLE has one point per frame:

| Bytes | Meaning |
|---|---|
| 0-7 | first sample of the frame: `n × 4410` (u64, big-endian) |
| 8-15 | byte offset of that frame, counted from the first frame's header (u64, big-endian) |
| 16-17 | samples in the frame: 4410 (less for the last one) (u16, big-endian) |

To start playing at time `t` (seconds):

1. `sample = floor(t × 44100)`, `n = sample / 4410`, `skip = sample % 4410`.
2. Read seek point `n` (it sits at `n × 18` bytes into the SEEKTABLE; no search needed,
   because the points are evenly spaced).
3. Seek the file to `first_frame_offset + offset`, where `first_frame_offset` is the byte
   just after the last metadata block (the deck learns it once, when it parses the header).
4. Decode that one frame and throw away the first `skip` samples (at most 99.9 ms of audio).

So a hot cue or a jump costs one file seek and one frame decode, never a scan. dr_flac
(already in the firmware for FLAC tracks) uses the SEEKTABLE the same way through
`drflac_seek_to_pcm_frame()`. The table is about 18 bytes per 100 ms: 43 KB for a
4-minute track, read once when the track loads.

## Timing and sample rate

Stems are always 44.1 kHz. The main track keeps its own rate (44.1, 48, 96 kHz...).
Both describe the same real time: stem sample `n` is at `n / 44100` s from the start,
the same moment as main-track sample `n × track_rate / 44100`. The deck already
resamples each source to its 48 kHz output, so it resamples the stem stream with its
own ratio (44100 → 48000) and drives both from the same playhead in seconds.
`frames / 44100` equals the track's duration.

Beat grid, cues and phrases on the card are all in milliseconds, so they apply to the
stems unchanged.

## Cost on the card and on the deck

Measured on four real stem files (4 to 6.5 minutes of house and DnB):

| | Version 2 (WAV) | Version 3 (FLAC) |
|---|---|---|
| Card space per hour of music | 2.54 GB | 1.17 GB (46 % of WAV) |
| SD read per deck while playing | 1.41 MB/s | about 0.65 MB/s |
| Decode | none | estimate 20-35 MHz of a Cortex-M85 per deck (to be measured on the device) |
| Decoder memory | none | about 128 KB per deck (8 channels × 4410 samples × 32 bits, plus frame buffer) |

## Firmware checklist

1. Look up the track's file path in `stems.json`; no entry = no stems.
2. Open the FLAC, check STREAMINFO: 8 channels, 16 bit, 44100 Hz, block size 4410.
3. Keep the SEEKTABLE and the first-frame offset; seek as above on load, cue and jump.
4. Decode frames in step with the main track's playhead (in seconds); resample 44.1 → 48 kHz.
5. Per stem: gain (mute/solo/fader) per L/R pair, then sum the four into the deck's channel.

## Partial stems (0.6)

A stems file may cover only a section of the track: DKPRP separates the requested range (with 10 s of context
on each side, cut away afterwards) and writes the whole-length file with digital silence outside the range, so a
reader needs nothing new. `POCKETDECK/analysis/<id>.json` carries `stems_range: [start_ms, end_ms]` when that is
the case (absent or null = the whole track).
