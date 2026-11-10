# DKPRP: test checklist (0.8)

Tick each box as you go. Where to test:

- **Preview** = the cloud preview of this session (a silent test engine, no sound card, no GPU, fake card folder).
- **Your PC** = needs your own machine: real sound, a real GPU, a real USB card or a DECK001. Build the app from PR #11.
- **Both** = works in either; trust it more on your PC.

Test library in the preview: 8 synthetic tracks. **Bass Test** (128 BPM, 8A) has stems; the rest don't. Rising Sun and
Ocean Drive were used for tests and may look odd. Anything marked *seeded* was put there by me for the test.

---

## 0. First look
- [ ] **Brand** (Both): top-left shows the waveform mark and **DKPRP** in Fugaz One (a chunky italic face). Dark and light theme (sun/moon button). Narrow the window below 1100 px: only the mark stays.
- [ ] **One bar** (Both): search, playlists button, jobs ring, deck pill, card button, add files, `+` import, theme, settings. No sidebar.
- [ ] **Theme** (Both): the theme button switches dark/light and it survives a reload.
- [ ] **Search** (Both): press `/`, type "drive": the list narrows to Midnight Drive and Ocean Drive. Clear it.

## 1. Library list
- [ ] **Columns** (Both): #, art, Track, Artist, BPM, Key, Mood, Energy bars, LUFS, Ready ring. Click a header to sort; click again to reverse.
- [ ] **Filters** (Both): the six glyphs (all, analysed, not analysed, with stems, what follows, needs attention) change what's shown; hover shows the name.
- [ ] **Select** (Both): click a row, Shift-click another, Ctrl/Cmd-click a third. The actions strip appears.
- [ ] **Quick edit** (Both): click the title of a selected row, edit, Enter. Then Ctrl/Cmd+Z or the Undo in the toast brings it back.
- [ ] **Needs attention** (Both): the red-badge filter lists tracks with a note. Bass Test shows clipping and a low bitrate.
- [ ] **Open a track** (Both): double-click a row or press Enter: it opens full screen. Esc or the back arrow returns.

## 2. Playlists
- [ ] **Popover** (Both): the list button opens playlists; Esc closes it.
- [ ] **New playlist / folder** (Both): "Playlist" and "Folder" buttons ask for a name. Select tracks first: they go into the new playlist.
- [ ] **Drag to a playlist** (Both): drag a row onto the playlists button (it opens) and drop it on a playlist.
- [ ] **Reorder** (Both): inside a normal playlist, drag rows to reorder; Undo restores.
- [ ] **Smart playlist** (Both): "Smart" asks for a name and a rule. Make "Peak 124-130": tempo 124 to 130, intent Mix or Vocal layer. It shows 6 tracks, with a funnel glyph. Right-click it, **Edit rule**, change the max to 126: the list shrinks. **Make ordinary** keeps the tracks.
- [ ] **Smart rules** (Both): try words, key codes (e.g. `8A 9A`), energy range, mood, "only with stems". A smart playlist refuses drops and reorder.

## 3. Right-click menu
- [ ] **Row menu** (Both): right-click a row (or the ⋮ button, or Shift+F10). Groups: Open, Mixes with this, Prepare as, Re-analyse, Make stems, Add to playlist, Remove.
- [ ] **Multi-select** (Both): select three, right-click one: the menu acts on all three ("Prepare as" ticks only if they agree).
- [ ] **Playlist menu** (Both): right-click a playlist: Open, Export only this playlist, Rename (F2), Delete.

## 4. Intent and readiness
- [ ] **Prepare as** (Both): Mix, Vocal layer, Remix, Section, Backup. Choose Vocal layer for a track without stems: a stems job appears in the jobs drawer (it fails here: this checkout has no real model; on your PC it runs).
- [ ] **Ready ring** (Both): hover a ring: it lists what is missing (file, analysed, grid, cues, stems, loudness, on the card).
- [ ] **Prepare everything** (Both): jobs drawer, button at the top: queues analysis and the stems each intent asks for.

## 5. The open track
- [ ] **Waveforms** (Both): full mix on top, four stem lanes below. Hover a lane: its name appears top-right. Click a lane label: that stem is soloed.
- [ ] **Cue pads** (Both): A to H. Click an empty pad: sets a cue at the playhead. Click a set pad: jumps. Keys `1`–`8` do the same. **Pencil** edits, names and clears cues.
- [ ] **Mix in / out** (Both): `I` and `O` set the points at the playhead (beat-snapped); the pill shows IN / OUT; the circular arrow goes back to the analysed points.
- [ ] **Beat grid** (Both): the nudge pill moves the grid in 1 ms steps (`[` `]`; Shift for 10 ms); **+1** moves the bar start; ✓ confirms the grid; the circular arrow resets.
- [ ] **Time signature** (Both): the `4/4` button in the grid pill cycles 4/4, 3/4, 6/4, back to the analysis. The downbeat lines and bar numbers follow. Rising Sun and Parallel Lines were detected as 3/4 (synthetic accents).
- [ ] **Key shift** (Both): the key pill shifts the deck's key by semitones; the Camelot code updates.
- [ ] **Phrase strip and sections** (Both): chips under the waveform (Drop 32, Instrumental 32, Chorus 16 *seeded*, Verse 8 *seeded*): click one to select it.
- [ ] **Drag a selection** (Both): drag on the full-mix lane: it snaps to bars (4, 8, 16, 32, 64). Esc clears it.
- [ ] **Loop** (Both): with a selection press `L` or the loop button: playback loops. **+** saves it as a loop (a memory cue with an end). Delete it with the × on its chip.
- [ ] **Stems for the selection** (Your PC): the stems button on the selection pill makes stems for that range only.
- [ ] **Details** (Both): the ⓘ button.

## 6. Checks and notes on a track
- [ ] **Stem quality** (Both): Bass Test shows "Poor stems: drums clips…" (*seeded*) with **Redo**. The ring's stems segment is red.
- [ ] **Clipping / bitrate** (Both): "Clips: 0.2 %…" and "Low bitrate (64 kbps)" notes.
- [ ] **First vocal** (Your PC, needs real stems): after a real separation a "Vocal" memory cue appears at the first sung beat, and sung-phrase chips (Chorus 16, Verse 8). In the preview they were *seeded*.

## 7. What follows (transitions)
- [ ] **Rank** (Both): right-click Midnight Drive, **Mixes with this**: the list shows only tracks that work, best first, each with a score pill (0-100).
- [ ] **Reasons** (Both): click a pill: tempo, key, energy, level, phrase lines, vocal/bass clashes, each with ✓ or !.

## 8. The practice player
Sound itself is silent in the preview. On **Your PC** listen for real.
- [ ] **Button** (Both): the sliders button in the open track's tool row opens **Practice**. If it is missing, the browser player is in use (Settings, System, Player).
- [ ] **Play** (Both): Space plays; the playhead moves; Space pauses; clicking the waveform seeks.
- [ ] **Tempo** (Both): slide to 50 %: the playhead moves at half speed and the readout shows the BPM it makes (64). −5 / +5 and the reset arrow work. **Your PC:** pitch stays the same.
- [ ] **Transpose** (Your PC): +3 st raises the pitch without changing the speed. Drums keep their pitch when "Drums keep their pitch" is ticked (needs stems).
- [ ] **Click and count-in** (Your PC): tick Click and Count-in, press play: a bar of clicks, then the song, with the click on the beat. Try the sounds, ears (left / both / right) and volume.
- [ ] **Stem mixer** (Both): "Load the stems to mix them": four rows appear (six after a RoFormer separation). Volume, pan, **M**ute, **S**olo, offset. Double-click a slider to reset it.
- [ ] **Solo from the lanes** (Both): clicking a lane label solos that stem in the mixer; the full-mix lane clears the solo; your mutes stay.
- [ ] **Loop at any tempo** (Both): loop a chip, set tempo 200 %: the playhead stays in the loop with no gap.
- [ ] **Saved per track** (Both): set tempo 80 % and mute drums, wait a second, reload the page, open the track again: they come back. Open another track: back to defaults.
- [ ] **Output device** (Your PC): choose another device in the panel's Output list; playback continues from the same place. Unplug it while playing: you get a message, not a crash.
- [ ] **Browser player fallback** (Both): Settings, System, Player, "Use the browser player", reload: the Practice button disappears and playback still works. Switch back the same way.
- [ ] **Volume and seek** (Both): the preview bar's volume and seek bar; previous / next track.

## 9. Stems engines
- [ ] **Settings, Stems, Engine** (Both): htdemucs or BS-RoFormer. The RoFormer row shows "Not downloaded (699 MB once, 350 MB kept)" and a Download button.
- [ ] **Download the model** (Your PC): Download: progress in the row and the jobs drawer, then "Ready". It is checked against a hash and converted; the original is deleted.
- [ ] **Separate with RoFormer** (Your PC, needs a GPU): choose BS-RoFormer, Make stems on a real song. Expect 1 to 2 minutes for a song on a laptop GPU. Check: the stems sound right; the practice mixer has six rows (guitar, piano); the vocals are cleaner than htdemucs. **Never tested on a real GPU: tell me what happens.**
- [ ] **No GPU** (Your PC, if you can): a machine without a usable GPU should fall back to htdemucs with a note in the job, not crash.
- [ ] **htdemucs** (Your PC): Make stems with the default engine works as before (the model downloads from Settings if the build has no bundled one).

## 10. Export to a card
- [ ] **Drives** (Your PC): insert a USB stick or SD card: the card column lists it. Non-removable drives are refused.
- [ ] **Plan** (Both): choose a drive: tracks, playlists, stems, size, and "Check before exporting" with notes per track.
- [ ] **Scope** (Both): All tracks or Playlists. With **one** playlist ticked a **Set** strip appears (see 11).
- [ ] **Export** (Your PC): the arrow button. Afterwards: "Checked: N files on the card read back as written" (or the first problem).
- [ ] **Foreign library** (Your PC, careful): a drive that already has someone else's rekordbox library is **refused**, not overwritten.
- [ ] **What lands** (Your PC): `PIONEER/` (database, ANLZ files, artwork), `Contents/` (audio, byte for byte), `POCKETDECK/` (stems.json, analysis hints, sets/).
- [ ] **Rekordbox / CDJ** (Your PC, if you have one): does it read the stick? Not tested on hardware.

## 11. Sets
- [ ] **Set strip** (Both): in the card column, tick one playlist: energy curve, tracks in order with rings, a coloured bar between each pair (green good, grey ok, red no), duration, load time.
- [ ] **Transition bars** (Both): click a bar: reasons and the window length.
- [ ] **Order by…** (Both): Balanced, By key, By tempo, Rising energy: the playlist reorders; Undo restores.
- [ ] **sets/ on the card** (Your PC): `POCKETDECK/sets/<id>.json` per playlist.

## 12. Settings and deck
- [ ] **Analysis** (Both): BPM range, loudness target (LUFS), analyse on import. Save, reload: kept.
- [ ] **System** (Both): application, player (engine or browser, device name), stems model, RoFormer model, data folder.
- [ ] **History** (Both): the list of what you changed, with times.
- [ ] **Deck view** (Your DECK001): status dot, update (OTA) and flashing. Flashing needs you to type FLASH; a wrong word does nothing.

## 13. Import
- [ ] **Add files / folder** (Your PC): the buttons in the top bar import audio (mp3, wav, flac, aiff, m4a); analysis starts by itself.
- [ ] **Convert for the deck** (Your PC): a file the deck can't play (24-bit WAV, M4A) shows a note and "Convert for the deck".

## 14. Safety rules (don't skip)
- [ ] The export **never overwrites** a foreign rekordbox library.
- [ ] Only **removable drives** are offered (unless a developer override is set).
- [ ] **No transcoding**: the card gets your original files.
- [ ] Flashing needs the typed word **FLASH**.

---
When something fails, note: what you did, what you expected, what happened, Windows / Mac / Linux, and any message.
