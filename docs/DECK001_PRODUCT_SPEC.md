# DKPRP: product specification for the DECK001

Draft, 2026-11-07. DKPRP is the preparation layer for the DECK001: it answers three questions
about every track before the gig, so the deck never has to.

1. **Where can I mix this track?** (grid, key, phrases, mix-in and mix-out points)
2. **What can I do creatively with it?** (stems, sections, vocals to layer, key shift)
3. **Is it ready to perform?** (file plays on the deck, analysed, cues set, on the card)

A feature that doesn't help answer one of those doesn't go in. The app stays a preparation tool: it
never performs, and it doesn't become a DAW.

The three layers:

| Layer | Job | Where it runs |
|---|---|---|
| Preparation | analyse, organise, edit, render | this app, at home |
| Performance intelligence | find the right track, section or transition fast | prepared here, **read** on the deck |
| Hardware readiness | everything processed, cached, validated before stage | this app, the card, the deck's own check |

## 1. What exists today (0.5)

Measured against the brief, so the gaps are honest. "✓" means built and tested, "~" partial.

| Area | Feature | State | Where |
|---|---|---|---|
| Analysis | BPM, beat grid (nudge, downbeat), key (Camelot), energy 1-10 | ✓ | ClubCraft `analysis`, `src/analysis.rs` |
| | Sections: intro, verse, build, drop, breakdown, outro, with bars and stem activity | ✓ | ClubCraft `structure.rs` |
| | Loudness (LUFS) + gain for the deck's target, near-silent start/end | ✓ | 0.5.0 |
| | Mood (12 MB embedding + trained head, or the nine-number fallback) | ✓ | `src/mood.rs`, `src/embed.rs` |
| | Vocal / instrumental detection | ~ | ClubCraft `vocals.rs` has a per-16th "singing" map; DKPRP only uses stem activity per section |
| | Time signature | ✗ | 4/4 assumed everywhere (ClubCraft too) |
| | Genre | ~ | mood is a genre-plus-feel label; tag genre is editable |
| Cues | A = start, B = first drop, C = breakdown, D = outro, phrase lines ≤ 1 per 16 bars | ✓ | `auto_cues` |
| | Hot cues 1-8 and memory cues, beat-snapped, named, keys 1-8 / M | ✓ | |
| | First vocal, last usable mix-out | ✗ | ClubCraft has `default_mix_in/out` (planner.rs) and the vocal map: not wired |
| Library | Duplicates, missing files, unplayable formats (M4A, AIFF, 24-bit WAV) + convert for the deck | ✓ | 0.5.0 |
| | Clipping detection, bitrate validation | ✗ | cheap to add in the loudness pass |
| | Playlists, folders, tags (artist, album, genre, year, comment, colour, rating) | ✓ | |
| | Smart playlists | ✗ | |
| | Mixes-with filter (key ±1 on the wheel or relative, tempo ±6 %) | ✓ | |
| Stems | 4-stem htdemucs, 8-channel FLAC, DirectML / CoreML / CPU, per-track | ✓ | |
| | Section-only separation, vocal-only, stem quality score | ✗ | |
| Export | rekordbox PDB + ANLZ, POCKETDECK/analysis hints (grid, key shift, gain, audible start/end, cue names), dry-run check | ✓ | |
| | Export validation after writing | ~ | tests read the card back with rekordcrate; the app doesn't |
| Deck | Status, OTA (signed), wired recovery (typed FLASH) | ✓ | |
| Sets | setlists, energy curve, transition planning | ✗ | ClubCraft `planner.rs` and `mix.rs` exist, unused |

## 2. The features, by the question they answer

### Q1: Where can I mix this track?

**Mix-in and mix-out points (P0).** Two memory cues the analysis places and the DJ can move: the
first full-groove bar (ClubCraft `default_mix_in`: after the intro, on a phrase) and the last bar
before the energy falls for good (`default_mix_out`). Shown as `IN` / `OUT` flags on the waveform,
written to the card as memory cues and to POCKETDECK/analysis as `mix_in_ms` / `mix_out_ms`. The
deck's autopilot uses them instead of `audible_start_ms` when they exist.

**First vocal (P1).** From ClubCraft's vocal map (needs stems): the first 16th where a sung line
starts. A memory cue `V` and a hint. Without stems, nothing: never guess from the mix.

**Phrase-aligned everything (P0, mostly done).** Every automatic cue already sits on a phrase
start. Add: the phrase strip in the focus view is clickable per 8 bars, and selections snap to
phrases (see sections below).

**Transition compatibility (P1).** Today: key and tempo. The score the brief asks for, as a table
the app can actually compute from what it has:

| Parameter | Source today | Verdict |
|---|---|---|
| BPM | `bpm` | native and ±6 % |
| Key | `camelot`, `key_shift` | same, neighbour, relative; with the shift the deck will apply |
| Energy | `energy` | up / same / down |
| Loudness | `lufs`, `gain_db` | difference after the deck's gain |
| Structure | sections, bars | does the outgoing track's OUT phrase length match the incoming IN |
| Drums | section stem activity | both have drums at the overlap (clash) or one is a break |
| Vocals | vocal map (stems) | sung lines overlap in the overlap window |
| Frequency | ClubCraft `mashup.rs` bar chroma / bass level | bass-on-bass in the overlap |

The output is **a transition window** ("mix out of A at bar 97, into B at bar 17, 16 bars, key
shift 0"), not a number. The verdict lines explain it. ClubCraft `planner::choose_transition_beats`
and `mashup::score` cover the first six rows; drums/vocals/frequency need stems on both tracks.

**Suggestions (P1).** "Mixes with" grows from a filter into a ranked list with the window and the
reasons. Same code, shown in the card column when one track is selected.

### Q2: What can I do creatively with it?

**Sections (P0).** The analysis already has the bars. Add a selection on the waveform (drag,
snapped to phrases: 8, 16, 32, 64 bars) that the app can:
- preview (loop in the player),
- separate stems for *only that range* (the separator takes a sample range; the stem file keeps
  the whole-track layout with silence outside, so the deck's format doesn't change),
- save as a named region (`regions` table: track, start/end bar, name, kind) and export to the
  card as a memory-cue pair plus a `regions` list in POCKETDECK/analysis.

Automatic candidates, each from data that exists: *clean instrumental passage* (vocal activity
≈ 0 over ≥ 8 bars, stems), *strong drums* (drum activity high, melody low), *isolated vocal*
(ClubCraft `vocal_units`: hook, line starts, not a loop), *break*, *drop*. Listed under the
waveform as chips; clicking one makes the selection.

**Stems for the way you'll use the track (P0, "adaptive preparation").** A per-track *intent*:

| Intent | What is made | Cost |
|---|---|---|
| Mix | analysis, waveform, cues | seconds |
| Vocal layer | vocals + instrumental (2 stems: the separator's vocals and the residual) | half a run |
| Remix | all 4 stems | one run |
| Section | stems for one region | proportional |
| Backup | analysis + waveform, low priority | seconds |

`tracks.intent` drives the queue (the "gpu" lane takes Remix last, Section first) and the
readiness score (a Mix track without stems is ready). Set from the row menu, the focus view and
"Edit selected…".

**Stem quality (P1).** After separation: residual vocal energy in the instrumental, bleed of drums
into bass, and the vocal stem's "singing vs leakage" ratio from the vocal map. Three numbers, shown
as one glyph (good / usable / poor) with the numbers on hover. Poor stems get the intent downgraded
to Vocal layer or Mix with a note.

**Key shift (done).** Stays as is: the deck shifts, the file doesn't change.

### Q3: Is it ready to perform?

**Readiness (P0).** Diagnostic, per track, against its intent:

| Check | Pass when | Already have |
|---|---|---|
| Plays on the deck | MP3 / FLAC / 16-bit WAV, or a converted copy | ✓ `deck_problem`, convert |
| File present | path exists | ✓ |
| Analysed | current analysis version | ✓ |
| Grid confirmed | the DJ pressed "grid OK" (or nudged it) | new: `tracks.grid_ok_at` |
| Cues | ≥ 1 hot cue, or auto cues not rejected | ✓ |
| Mix points | IN and OUT exist | new |
| Stems | as the intent asks, quality ≥ usable | ✓ + quality |
| Loudness | measured, gain within ±12 / +6 dB | ✓ |
| On the card | in the last export, file unchanged since | new: `exported` table (track, card id, mtime, hash of the analysis) |

One glyph per row (ring: full = ready, partial = what's missing on hover, red = blocked). No score
out of 100: the ring segments *are* the score, and they say what to do.

**Card validation (P1).** After an export, read the card back the way the tests do (rekordcrate +
our ANLZ reader): every track row resolves to a file, every ANLZ parses, stems match the
format version, sizes match. Shown as the last line in the card column ("Checked ✓" or the
first problem).

**Performance simulation (P1).** Not audio: a walk through a set (or the card) that asserts the
readiness rows per track, the transition windows between consecutive tracks, and the deck's load
time estimate (file size ÷ the measured USB read speed from `docs/SYNC_OPTIONS.md`). The result is
a list of "at the venue this would…" lines. It's the dry-run we already have, extended to the set.

**Processing queue (done, extend).** Lanes cpu / gpu / io exist; add priority by intent and a
"prepare everything" action (analyse → stems by intent → convert) with resume after restart
(jobs are already re-queued at startup for stale analyses; do the same for stems "queued").

### Sets (P1)

A set is a playlist with order, a mix point per pair and an energy curve. ClubCraft's
`planner::order_tracks` (simulated annealing on key, tempo, energy) and `plan_mix` (mix points,
transition beats, key shifts) give a first order and the windows; the DJ reorders by drag. The
card column becomes the set when a set is chosen: order, energy curve under the drive bar,
alternatives per slot (the suggestions list, filtered to the slot's neighbours), and the export
writes the set as a playlist plus `set.json` in POCKETDECK with the windows, so the deck can show
"next: … in 16 bars". The rendered-mix parts of ClubCraft (`mix.rs` automation) stay out: the deck
performs, not the app.

## 3. Data model additions

```
tracks         + intent TEXT ('mix'|'vocal'|'remix'|'section'|'backup', default 'mix')
               + grid_ok_at REAL
               + mix_in_ms REAL, mix_out_ms REAL, first_vocal_ms REAL   (from analysis, user-editable)
               + stem_quality TEXT (JSON: {residual_db, bleed_db, vocal_ratio, grade})
regions        id, track_id, start_ms, end_ms, start_bar, bars, name, kind ('section'|'vocal'|'drums'|'instrumental'), stems INTEGER
sets           id, name, created_at
set_tracks     set_id, position, track_id, mix_out_ms, mix_in_ms, transition_beats, key_shift, note
exported       card_id, track_id, exported_at, file_mtime, analysis_hash, stems_version
```

Analysis JSON gains `mix_in_ms`, `mix_out_ms`, `first_vocal_ms`, `candidates: [{kind, start_bar,
bars, score}]`. POCKETDECK/analysis gains `mix_in_ms`, `mix_out_ms`, `regions`, `intent`;
POCKETDECK/sets/<id>.json carries the set. The rekordbox files don't change.

## 4. What runs where

| | Prep app (PC / Mac) | Card | DECK001 |
|---|---|---|---|
| Analysis, stems, mood, regions, sets, validation | compute | | |
| Waveforms, cues, grid, hints | write | PDB + ANLZ + POCKETDECK | read |
| Transition windows, set order | compute | `set.json` | show "next in N bars" |
| Readiness | compute and show | | the deck's own check on load (file, ANLZ, stems) |
| Live stems | pre-rendered only | stems FLAC | play |

Nothing needs a network at any point; the models ship with the app or are fetched once (below).

## 5. Priorities

| P | Feature | Why now |
|---|---|---|
| P0 | Mix-in / mix-out cues | one function call in ClubCraft, big win on the deck |
| P0 | Intent per track + queue priority | the cheapest way to cut waiting and disk |
| P0 | Readiness ring (with grid OK, on the card) | replaces four columns with one answer |
| P0 | Sections: select, preview, separate a range, save | the creative core for a stems deck |
| P1 | Stem quality grade | stops bad stems reaching the stage |
| P1 | Transition window + reasons, suggestions list | "mixes with" done properly |
| P1 | Card validation after export | find it at home |
| P1 | Sets with order, energy curve, `set.json` | the planning layer |
| P1 | Performance simulation (the set-level dry run) | |
| P2 | First vocal cue, vocal units as chips | needs stems; land after sections |
| P2 | Clipping, bitrate checks; smart playlists | |
| P2 | Time signature | ClubCraft would need it first |

Not planned: automatic set construction as a product feature (the planner's order is a starting
point, never the set), AI recommendations beyond the library, rendered mixes.

## 6. Lean installation

Today's Windows installer is ~301 MB: 215 MB is the **offline WebView2 runtime**, ~60 MB the
binary (ONNX Runtime + DirectML), 174 MB the Demucs model and 12 MB the mood model (compressed in
the installer), the UI under 1 MB. Proposed, in order of effect:

1. **WebView2: ship the bootstrapper, not the offline runtime** (−213 MB). Windows 10/11 have
   WebView2 already; the 2 MB bootstrapper fetches it only on a machine that lacks it. Trade-off:
   that one case needs internet once. The *app* stays offline.
2. **Models fetched on first run** (−~120 MB compressed), from a URL in `tauri.conf.json`, verified
   by SHA-256 pinned in the binary, into the data folder; the installer carries only the 12 MB mood
   model. Stems and mood work offline after that. Trade-off: first run needs internet once, or the
   user copies `htdemucs.onnx` next to the app. Keep a "full" installer target for air-gapped
   machines (the current one, built on request).
3. **Binary**: `[profile.release] lto = "fat", codegen-units = 1, strip = true, panic = "abort"`,
   `opt-level = 3` kept for DSP. Expect −15-25 % of the binary.
4. **ONNX Runtime**: one provider per platform (DirectML on Windows, CoreML on Mac, CPU on Linux);
   don't ship the CPU-only duplicate where the accelerated one exists.
5. **Don't bundle** `tools/`, `docs/`, test audio, the vendored ClubCraft sources (resources are
   explicit in `tauri.conf.json`; check the NSIS target includes nothing else).

Target: a ~75 MB Windows installer, ~70 MB DMG, with the full offline variant at ~300 MB.
Items 3-5 are safe to do now; 1 and 2 change what "offline" means on first run and need a yes.

## 7. Open questions for the DJ

- Does the deck's autopilot prefer `mix_in/out` over `audible_start/end` when both exist? (Firmware.)
- Should a Section region's stems be a separate file or the full-length stems file with silence
  outside the region? (Full-length keeps the deck's reader as it is; separate saves card space.)
- Is "grid OK" worth a key press per track, or should a nudge count as confirmation only?
- Should the card column *be* the set once a set is chosen (one place), or a separate Sets view?
