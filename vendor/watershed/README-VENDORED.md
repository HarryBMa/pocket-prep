# Vendored Watershed audio engine (MIT OR Apache-2.0)

From https://github.com/edarbieto/watershed at commit 6ff334dd53d0b59abea462c2441676782838984a,
four crates: `audio` (the real-time playback engine), `dsp` (trimmed), `roformer` (BS-RoFormer on burn + wgpu) and
`separation` (trimmed: model catalog, resumable download, the separator). Licences: LICENSE-MIT, LICENSE-APACHE (unchanged).
Used by `src/player.rs` (the practice player behind `/api/player/*`) and `src/roformer.rs` (the second stems engine).

What it is: stems -> mixer (gain, pan, mute, solo, pitch lock, offset per stem) -> tempo and pitch
(Signalsmith Stretch, MIT) -> master gain -> limiter -> metronome click with count-in -> one cpal stream.
The audio callback allocates nothing; settings are atomics, songs and beats pass through lock-free queues.

Local changes (keep this list complete; offer them upstream):

1. Cargo.toml: no workspace inheritance (versions and lints written out); `[dev-dependencies]` and
   `examples/` removed (they need Watershed's `dsp` crate).
2. src/engine.rs: `Engine::start_null(sample_rate)` and `Engine::null_peak()`. A thread runs the same
   callback at real time into a buffer nobody hears, so the player can be tested, and used, on a machine
   with no sound card. `_stream` became `Option<cpal::Stream>`.

3. dsp: only `StereoBuffer`, `Stft` and the chunking of the original (decode, FLAC / Vorbis writing, resampling, waveform
   peaks, tempo and key estimation are not vendored: ClubCraft has its own). Cargo.toml written out for a standalone crate.
4. roformer: Cargo.toml written out (versions pinned as in Watershed's workspace); `examples/` removed (they need dsp's
   decoder). Its tests, including the layer-by-layer PyTorch equivalence on the tiny model, are kept and pass.
5. separation: `song.rs` (decode, write stems, sessions) and the `examples/` removed; `lib.rs` re-exports `StereoBuffer`;
   `download.rs` uses reqwest's blocking client instead of ureq (one HTTP stack in the app), same resume / mirror /
   SHA-256 logic. `models.toml` unchanged: the BS-RoFormer SW checkpoint (no licence declared, so never bundled), pinned
   to a Hugging Face commit.

Not taken from Watershed: the egui app, its session files, the stem players' waveform and tempo / key code.
