//! Source separation: the model catalog and downloads, and the BS-RoFormer separator. (Pocket Prep decodes songs and
//! writes stems itself, so Watershed's `song` module and its session code are not vendored.)

use std::path::PathBuf;

// (StereoBuffer is re-exported below)

mod burn;
pub mod catalog;
mod download;

pub use burn::BurnSeparator;
pub use watershed_dsp::StereoBuffer;
pub use catalog::ModelEntry;
pub use download::{DownloadError, Expected, download, sha256_file};

/// Splits a stereo mix into stems. [`BurnSeparator`] is the real one; tests use their
/// own.
pub trait Separator: Send {
    /// The sample rate the input must have; stems come out at the same rate.
    fn sample_rate(&self) -> u32;

    /// Stem names in output order, lower case (`vocals`, `drums`, ...). They become
    /// the stem file names.
    fn stem_names(&self) -> &[String];

    /// Where it runs, for the UI, e.g. `Vulkan GPU`.
    fn device(&self) -> String;

    /// Returns one buffer per stem, in [`Self::stem_names`] order, each as long as
    /// `input`.
    ///
    /// `progress` gets the fraction done (0..1), possibly from another thread.
    /// `cancel` is polled while it runs; returning `true` stops with
    /// [`SeparationError::Cancelled`].
    fn separate(
        &mut self,
        input: &StereoBuffer,
        progress: &(dyn Fn(f32) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<StereoBuffer>, SeparationError>;
}

#[derive(Debug, thiserror::Error)]
pub enum SeparationError {
    #[error("the model has {model} stems but {names} stem names were given")]
    StemCount { model: usize, names: usize },
    #[error("input is at {got} Hz but the model needs {expected} Hz")]
    SampleRate { expected: u32, got: u32 },
    /// From the model, including GPU errors, which arrive as panics.
    #[error("{0}")]
    Model(String),
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the output folder already exists: {}", .0.display())]
    OutputExists(PathBuf),
    #[error("separation cancelled")]
    Cancelled,
}
