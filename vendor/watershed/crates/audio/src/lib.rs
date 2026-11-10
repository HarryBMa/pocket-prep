//! Real-time playback engine: mixes stems in a single cpal callback.
//!
//! The UI thread owns an [`Engine`]. Songs are handed to the audio thread through a
//! lock-free queue and handed back for dropping, so the audio callback never
//! allocates or frees memory. Mixer parameters are plain atomics.
//!
//! Signal path: stems → mixer → tempo and pitch (Signalsmith Stretch) → master gain
//! → limiter → plus the metronome click → device.

mod atomic;
mod beats;
mod click;
mod engine;
mod limiter;
mod mixer;
mod samples;
mod song;
mod stretch;

pub use atomic::AtomicF32;
pub use beats::{Beats, Meter};
pub use click::{ClickChannels, ClickSound};
pub use engine::{Engine, EngineError, MIN_LOOP_SECS, Notify, OutputDevice, output_devices};
pub use samples::Samples;
pub use song::{Song, Track, TrackParams};
pub use stretch::{MAX_SEMITONES, MAX_TEMPO, MIN_TEMPO};
