# Mood

Every analysed track gets one mood, a label that joins genre and the kind of track:

| Mood | Meant for |
|---|---|
| Deep | deep house, deep tech: warm, rolling, low-key, few vocals |
| Melodic | melodic techno, progressive: big melodies, long breakdowns, minor keys |
| Tech house | groovy, percussive, bass-led, vocal snippets |
| Peak time | driving peak-time techno and house: loud, dense, full on |
| Hard techno | hard, industrial, schranz: fast, loud, distorted kicks |
| Drum & bass | drum & bass, jungle, breaks at 160–180 BPM |
| Mainstream pop | song-shaped pop and dance-pop: vocals most of the way |
| Downtempo | chill-out, downtempo, ambient: slow, soft, sparse drums |

The library can be grouped by mood (sort menu, "Mood (grouped)"), and the empty track panel
shows how many tracks each mood has.

## How it is worked out (`src/mood.rs`)

The same audio always gets the same mood. Nothing is random and no model is downloaded.

1. **Measure.** The analysis measures nine numbers:

   | Number | What it is | Weight |
   |---|---|---|
   | Tempo | BPM | 2 |
   | Loudness | integrated LUFS | 0.75 |
   | Bass | share of the energy below 200 Hz | 0.75 |
   | Treble | share above 2.5 kHz | 0.5 |
   | Drums | share of the track with drums playing | 1 |
   | Melody | share with melody playing | 0.5 |
   | Vocals | share with vocals; **only with stems**, skipped otherwise | 1.5 |
   | Breakdowns | share in breakdowns and build-ups | 0.5 |
   | Key | 1 = minor, 0 = major | 0.5 |

   The drums/melody/vocals numbers are ClubCraft's stem activity per section, averaged over
   the track. Without stems ClubCraft estimates drums and melody from the mix; vocals can't be
   told apart from the mix, so that number is left out instead of read as "no vocals".

2. **Compare.** Each mood has a typical value and a spread for every number. A track's
   distance to a mood is, for every number, (value − typical) / spread, squared, weighted,
   averaged, and square-rooted: how many spreads away it is. The nearest mood wins.
   The confidence is that mood's share of exp(−d²/2τ²) over all moods (τ = 0.35). Under 50 %
   shows a "?". In statistics terms: a weighted, standardised Euclidean distance with a
   Gaussian kernel, i.e. a nearest-centroid classifier.

3. **Learn.** Setting a mood by hand pulls that mood's typical values towards the track:
   typical = (3 × built-in + sum of your tracks) / (3 + number of your tracks). The built-in
   value counts as three tracks, so one odd pick can't swing a mood. Every automatic mood is
   then worked out again; "Automatic" in the picker gives a track back. Undo works too.

The track panel ("Track details" → "Why this mood") shows the three nearest moods and the
measured numbers, so a surprising mood can be traced to the number that caused it.

## Where the typical values come from

They started as rules of thumb (tempo ranges are well known per genre). Then their scale was
set from 32 Creative Commons tracks from archive.org netlabels, four per mood, analysed with
the `mood_calibration` test:

```
POCKET_PREP_MOOD_DIR=<folder with one sub-folder per mood id> \
  cargo test --release mood_calibration -- --ignored --nocapture
```

The netlabel tags turned out to be poor labels: a "deep" track at 160 BPM, "drum & bass" at 90
and 120 BPM, and the same file tagged both deep and tech house. So that set gave the **scale** of
each number (e.g. bass share 0.2–0.5, treble under 0.06, breakdowns 0–0.25), not a fair
accuracy. 10 of 32 matched their tags, and many of the tags are themselves wrong.

**Expect to correct moods at first.** The built-in values are a starting point; the hand-picked
tracks are what make it fit your library. A few per mood is enough to move it.

## Limits and next steps

- Loudness mostly reflects mastering (netlabel releases are 4–6 dB quieter than club
  releases), so it has a low weight.
- Deep vs peak time vs tech house is the hardest split: the nine numbers don't hear kick
  timbre, groove or "darkness". A better next step is an audio embedding: a network turns
  each track into a vector, and the same nearest-centroid idea runs on it. The Essentia
  Discogs-EffNet models know 400 styles but are CC BY-NC-SA (non-commercial) with an AGPL
  library, so they need a licence decision first. Another option is to train a small model on
  your own labelled library.
- Mood isn't written to the USB card (the rekordbox database has no mood field).
