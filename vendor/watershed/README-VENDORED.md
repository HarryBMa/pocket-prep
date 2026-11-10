# Vendored Watershed audio engine (MIT OR Apache-2.0)

From https://github.com/edarbieto/watershed at commit 6ff334dd53d0b59abea462c2441676782838984a,
only `crates/audio` (the real-time playback engine). Licences: LICENSE-MIT, LICENSE-APACHE (unchanged).
Used by `src/player.rs`, the practice player behind `/api/player/*`.

What it is: stems -> mixer (gain, pan, mute, solo, pitch lock, offset per stem) -> tempo and pitch
(Signalsmith Stretch, MIT) -> master gain -> limiter -> metronome click with count-in -> one cpal stream.
The audio callback allocates nothing; settings are atomics, songs and beats pass through lock-free queues.

Local changes (keep this list complete; offer them upstream):

1. Cargo.toml: no workspace inheritance (versions and lints written out); `[dev-dependencies]` and
   `examples/` removed (they need Watershed's `dsp` crate).
2. src/engine.rs: `Engine::start_null(sample_rate)` and `Engine::null_peak()`. A thread runs the same
   callback at real time into a buffer nobody hears, so the player can be tested, and used, on a machine
   with no sound card. `_stream` became `Option<cpal::Stream>`.

Not taken from Watershed (yet): its separation crates (`roformer`, `separation`), the dsp crate (decode,
resample, tempo and key estimation: we have ClubCraft's), the egui app.
