//! The slice of Watershed's dsp crate that the RoFormer model needs: STFT / iSTFT, overlapping chunks and the
//! stereo buffer. (Decoding, FLAC / Vorbis writing, resampling, tempo and key estimation are not vendored: Pocket Prep
//! has its own, ClubCraft's.)

mod chunks;
mod stft;

pub use chunks::{Chunking, overlap_add};
pub use realfft::num_complex::Complex32;
pub use stft::Stft;

/// Stereo audio with interleaved `[left, right, left, right, ...]` samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StereoBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl StereoBuffer {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        debug_assert!(
            samples.len().is_multiple_of(2),
            "stereo buffer needs an even sample count"
        );
        Self {
            sample_rate,
            samples,
        }
    }

    /// Number of frames (one left + one right sample each).
    pub fn frames(&self) -> usize {
        self.samples.len() / 2
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }
}
