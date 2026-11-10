//! From a song to its stems: STFT, masks, inverse STFT, chunk by chunk.

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use watershed_dsp::{Complex32, StereoBuffer, overlap_add};

use crate::model::BsRoformer;

const CHANNELS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("separation cancelled")]
pub struct Cancelled;

impl BsRoformer {
    /// One buffer per stem, each as long as `input`.
    ///
    /// `progress` gets the fraction done after each chunk. `cancel` is polled before
    /// each one, so stopping takes at most the time of a chunk.
    pub fn separate(
        &self,
        input: &StereoBuffer,
        progress: &dyn Fn(f32),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<StereoBuffer>, Cancelled> {
        let stems = self.mask_estimators.len();
        overlap_add(input, stems, self.chunking, |chunk, done| {
            if cancel() {
                return Err(Cancelled);
            }
            let stems = self.separate_chunk(chunk);
            progress(done);
            Ok(stems)
        })
    }

    /// One interleaved stereo chunk of the model's size to the same for each stem.
    pub fn separate_chunk(&self, chunk: &[f32]) -> Vec<Vec<f32>> {
        let length = chunk.len() / CHANNELS;
        let spectra = std::thread::scope(|scope| {
            [0, 1]
                .map(|channel| {
                    scope.spawn(move || {
                        let signal: Vec<f32> = chunk
                            .iter()
                            .skip(channel)
                            .step_by(CHANNELS)
                            .copied()
                            .collect();
                        self.stft.forward(&signal)
                    })
                })
                .map(|thread| thread.join().expect("STFT thread panicked"))
        });

        let freqs = self.stft.freqs();
        let frames = self.stft.frames(length);
        let input = TensorData::new(
            model_input(&spectra, freqs, frames),
            [1, freqs * CHANNELS, frames, 2],
        );
        let masks = self
            .forward(Tensor::from_data(input, &self.device))
            .cast(DType::F32);

        // One stem at a time, so that the staging buffer each read goes through, which
        // the GPU backend keeps for good, is the size of one stem's masks.
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..self.mask_estimators.len())
                .map(|stem| {
                    let masks: Vec<f32> = masks
                        .clone()
                        .narrow(1, stem, 1)
                        .into_data()
                        .try_to_vec()
                        .expect("cast to f32");
                    let spectra = &spectra;
                    scope.spawn(move || {
                        let channels = [0, 1].map(|channel| {
                            let masked =
                                apply_mask(&spectra[channel], &masks, channel, freqs, frames);
                            self.stft.inverse(&masked, length)
                        });
                        channels[0]
                            .iter()
                            .zip(&channels[1])
                            .flat_map(|(left, right)| [*left, *right])
                            .collect()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().expect("inverse STFT thread panicked"))
                .collect()
        })
    }
}

/// Real and imaginary parts as (frequency * channel, time, 2), which is what the
/// model reads, from spectrograms stored frame after frame.
fn model_input(spectra: &[Vec<Complex32>; CHANNELS], freqs: usize, frames: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(freqs * CHANNELS * frames * 2);
    for freq in 0..freqs {
        for spectrum in spectra {
            for frame in 0..frames {
                let bin = spectrum[frame * freqs + freq];
                input.extend([bin.re, bin.im]);
            }
        }
    }
    input
}

/// `spectrum` times one stem's complex masks, with the DC bin silenced as the original
/// model does. `masks` is (frequency * channel, time, 2).
fn apply_mask(
    spectrum: &[Complex32],
    masks: &[f32],
    channel: usize,
    freqs: usize,
    frames: usize,
) -> Vec<Complex32> {
    let mut masked = vec![Complex32::ZERO; spectrum.len()];
    for freq in 1..freqs {
        let row = &masks[(freq * CHANNELS + channel) * frames * 2..][..frames * 2];
        for (frame, mask) in row.as_chunks::<2>().0.iter().enumerate() {
            let index = frame * freqs + freq;
            masked[index] = spectrum[index] * Complex32::new(mask[0], mask[1]);
        }
    }
    masked
}
