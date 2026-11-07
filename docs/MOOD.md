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

**Deep, Melodic, Tech house and Peak time** are fitted on a DJ library: 88 club tracks, labelled
from their Beatport genre tags (Deep House and Minimal / Deep Tech = deep; Melodic House & Techno
and Progressive House = melodic; Tech House = tech house; Techno and Techno (Peak Time / Driving)
= peak time). Per number, the typical value is the median and the spread is IQR / 1.35, never
below a floor (tempo 2, loudness 1, shares 0.03, key 0.3).

**Hard techno, Drum & bass, Mainstream pop and Downtempo** are still rules of thumb. No clean
labelled set exists for them: 120 Creative Commons tracks from MTG-Jamendo were tried, but their
genre tags are the artists' own. Only 6 of the 29 "drum & bass" tracks are above 160 BPM, and
the fitted tempo spread came out at 38 BPM, so fitting them would have made those moods worse.
They were used only to check the fit, not to set the values.

### How accurate it is

Measured with 5-fold cross-validation: every track is predicted by values fitted *without* it
(the moods are split evenly over five groups; fit on four, test on the fifth, five times).
Never counted on the tracks the values were fitted on.

| Test set | Built-in (before) | Fitted (after) | Folds better |
|---|---|---|---|
| The 4 club moods only, 88 tracks | 28.4 % | 48.9 % | 5 of 5 |
| All 8 moods, 88 club + 72 Jamendo tracks* | 28.9 % | 39.0 % | 4 of 5 |

\* Only the Jamendo tracks where an independent classifier (Essentia Discogs-EffNet) agreed with the
artist's tag are included; the rest are mislabelled too often. Chance for eight moods is 12.5 %.

Club moods, held-out predictions (rows: Beatport label, columns: predicted):

| | deep | melodic | tech house | peak time |
|---|---|---|---|---|
| deep (21) | 5 | 6 | 10 | 0 |
| melodic (7) | 1 | 1 | 3 | 2 |
| tech house (44) | 4 | 6 | 31 | 3 |
| peak time (16) | 0 | 4 | 6 | 6 |

Tech house is found 70 % of the time; deep 24 %, melodic 14 %, peak time 38 %. The most
confused pairs are deep -> tech house, tech house -> melodic and peak time -> tech house: the nine
numbers can't hear kick timbre, groove or darkness. The weights were left unchanged: a
weight search gave +3 points on the pooled set, but not consistently across folds.

A comparison: Essentia's Discogs-EffNet style classifier (400 styles, summed into our moods) agrees
with the Beatport label on 60 % of the 89 club tracks without any fitting. It is the better next
step, but its models are CC BY-NC-SA (non-commercial; a commercial licence is available from the
authors), so it needs a licence decision before it ships.

### Redoing it

```
# 1. a TSV of "mood id<TAB>path" (the music stays where it is), then measure it:
POCKET_PREP_MOOD_LIST=tracks.tsv cargo test --release mood_calibration -- --ignored --nocapture > run.jsonl
# 2. fit and cross-validate (python 3, standard library only):
python tools/mood_fit.py run.jsonl [--keep approved-files.txt] [--fit-only deep,melodic,tech_house,peaktime] [--weights] [--rust]
```

`--rust` prints the values to paste into `MOODS` in `src/mood.rs`. Compare the printed
cross-validated accuracy for the built-in and the fitted values before you keep a change.

**Expect to correct moods at first.** The values are a starting point; the hand-picked tracks
are what make it fit your library. A few per mood is enough to move it.

## Needle Tags: an embedding model instead of nine numbers (`tools/needle/`)

An Essentia-style two-stage model with no non-commercial licence in it. `mood_embed.py` runs htdemucs
(MIT, Meta) on the middle two minutes of a track and takes the output of its cross-transformer
(where the time and the spectrum branch meet), pooled over time, plus the level and busyness of
each stem: 2,056 numbers per track. `mood_head.py` trains a small logistic regression on top
(standardised numbers, L2 penalty chosen by an inner cross-validation, every mood weighted equally).
The trained head is `tools/needle/mood_head_v1.json` (8 moods, 2,048 weights each).

5-fold cross-validation, with all tracks of one artist kept in the same fold (so it can't learn an
artist's sound):

| Test set | DKPRP's nine numbers | Essentia (no training) | Needle Tags |
|---|---|---|---|
| 4 club moods, 88 tracks (Beatport tags) | 49 % | 60 % | **65 %** |
| 4 other moods, 120 Jamendo tracks (the artists' tags, noisy) | 38 % | about 55 % | **60 %** |
| All 8 moods, 88 club + 72 Jamendo | 39 % | - | **69 %** |

The eight-mood result is flattering: the club tracks and the Jamendo tracks differ in how they are
mastered, so part of it is telling the two sources apart. Read it with the first two rows. Weak
spots: melodic (7 tracks, 29 % recall) and deep, which is mistaken for tech house about half the time.
The label sets are small and the tags are not your own. Both go up with your corrections.

It isn't wired into the app yet. The app already runs htdemucs to make stems, so the same pass could
produce the embedding at no extra cost; the head is a 2,048 x 8 matrix multiply. When Needle is
trained, the same recipe can use its encoder instead.

Redo it: `python mood_embed.py list.tsv out.npz` (Needle venv; `list.tsv` is "mood<TAB>path"), then
`python mood_head.py --luke a.npz --jam b.npz --groups artists.tsv --save mood_head.json`.

## Limits and next steps

- Loudness mostly reflects mastering (free-music releases are quieter than club releases, and a
  pooled test can "succeed" by hearing that), so it has a low weight.
- Tempo can come out at half or double the real value (several drum & bass tracks measured at
  87 BPM): the analysis range is 85-175 BPM.
- Deep vs peak time vs tech house is the hardest split: the nine numbers don't hear kick
  timbre, groove or "darkness". A better next step is an audio embedding: a network turns
  each track into a vector, and the same nearest-centroid idea runs on it. The Essentia
  Discogs-EffNet models know 400 styles but are CC BY-NC-SA (non-commercial) with an AGPL
  library, so they need a licence decision first. Another option is to train a small model on
  your own labelled library.
- Mood isn't written to the USB card (the rekordbox database has no mood field).
