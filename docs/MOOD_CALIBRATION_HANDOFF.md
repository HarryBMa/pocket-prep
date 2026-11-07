# Handoff: calibrate the moods on a real music library

Paste everything below the line into the local AI (Claude Code) on the computer that has the
music. It works in this repository.

---

You are working in the Pocket Prep / DKPRP repository (Rust + Tauri, vanilla JS UI). Check out
the branch `ui/dkprp-redesign` (pull request #4, "Mood"). Read `CLAUDE.md`, `docs/MOOD.md` and
`src/mood.rs` first.

**Goal:** replace the guessed typical values in `MOODS` (`src/mood.rs`) with values measured on
my own music, and show how accurate the mood is before and after, honestly measured.

**Background:** every analysed track gets one of eight moods (deep, melodic, tech_house,
peaktime, hard_techno, dnb, pop, downtempo). The method is a nearest-centroid classifier: nine
measured numbers per track (bpm, lufs, bass, bright, drums, melody, vocals, breaks, minor), each
mood has a (typical, spread) per number, weighted standardised distance, nearest wins. The
current typical values came from 32 Creative Commons netlabel tracks with unreliable tags, so
only their scale is trustworthy.

**Rules:**
- My music files are read-only. Never move, rename, retag or delete them. Build the calibration
  set as a folder of symlinks (or a list of paths), outside the repository. Never commit audio.
- Stop processes only by exact PID, never by name (`CLAUDE.md`).
- Commit after every logical change and push to `ui/dkprp-redesign`; never force-push.
- Don't change what is measured (`mood_features` in `src/analysis.rs`) without asking me first:
  that forces every library to be re-analysed.

**Steps:**
1. **Labelled set.** Propose 15–40 tracks per mood from my library, using my playlists, crates,
   folders and genre tags as hints. Show me the list per mood and let me remove wrong ones
   before you use it: the labels must be mine, not guessed. Skip a mood rather than fill it
   with doubtful tracks.
2. **Measure.** Make `<calib>/<mood id>/` folders of symlinks and run
   `POCKET_PREP_MOOD_DIR=<calib> cargo test --release mood_calibration -- --ignored --nocapture`.
   Save the JSON lines. This analyses without stems, so `vocals` is empty. If stems can be made
   here (Demucs model in `models/`), also measure a stems run, because vocals are the main
   signal for "pop".
3. **Fit, in Python.** Write `tools/mood_fit.py` (standard library, or numpy/pandas if
   installed). It reads the JSON lines, and per mood and number computes:
   - the typical value: the median;
   - the spread: IQR / 1.35, with a floor so it never collapses (bpm 2, lufs 1, shares 0.03,
     minor 0.3).
   Use the same distance and confidence as `rank()` in `src/mood.rs`, with τ = 0.35.
4. **Evaluate honestly.** Use 5-fold cross-validation, stratified by mood: fit on four folds,
   test on the fifth. Report:
   - accuracy and a confusion matrix, for the current built-in values and for the fitted values;
   - which mood pairs get confused most.
   Never report accuracy on the same tracks the values were fitted on.
5. **Weights (optional).** Try a small grid of feature weights, chosen by cross-validated
   accuracy. Keep the current weights unless the gain is clear: more than 3 percentage points
   and consistent across folds.
6. **Apply.** Write the fitted values (rounded sensibly) into `MOODS` in `src/mood.rs`, keeping
   the mood order and ids. Run `cargo test --release`.
7. **Document.** In `docs/MOOD.md`, replace the calibration section with:
   - the number of tracks per mood;
   - the before/after cross-validated accuracy and the confusion matrix;
   - what still gets confused;
   - how to rerun `tools/mood_fit.py`.
8. **Report back to me in Swedish,** in plain words: how accurate it is now, where it fails,
   and whether more labelled tracks or an audio-embedding model (see "Limits and next steps"
   in `docs/MOOD.md`) would be the better next step. I'm learning Python and statistics, so
   explain the cross-validation and the confusion matrix briefly.
