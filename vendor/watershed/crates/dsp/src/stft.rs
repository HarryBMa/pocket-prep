//! STFT and inverse STFT, matching `torch.stft` and `torch.istft` with `center=True`,
//! reflect padding, a periodic Hann window and no normalization.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// A transform of a fixed size. Spectrograms are flat, frame after frame, with
/// [`Self::freqs`] bins per frame.
pub struct Stft {
    hop: usize,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let window = (0..n_fft)
            .map(|i| {
                let phase = std::f64::consts::TAU * i as f64 / n_fft as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        Self {
            hop,
            window,
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    fn n_fft(&self) -> usize {
        self.window.len()
    }

    /// Bins per frame.
    pub fn freqs(&self) -> usize {
        self.n_fft() / 2 + 1
    }

    /// Frames of the spectrogram of a signal of `samples` samples.
    pub fn frames(&self, samples: usize) -> usize {
        1 + samples / self.hop
    }

    /// The spectrogram of `signal`, which must be longer than half a window.
    pub fn forward(&self, signal: &[f32]) -> Vec<Complex32> {
        let half = self.n_fft() / 2;
        let padded = pad_reflect(signal, 1, half, half);
        let mut input = self.forward.make_input_vec();
        let mut scratch = self.forward.make_scratch_vec();
        let mut spectrum = vec![Complex32::ZERO; self.frames(signal.len()) * self.freqs()];

        for (frame, output) in spectrum.chunks_exact_mut(self.freqs()).enumerate() {
            let samples = &padded[frame * self.hop..][..self.n_fft()];
            for ((x, sample), window) in input.iter_mut().zip(samples).zip(&self.window) {
                *x = sample * window;
            }
            self.forward
                .process_with_scratch(&mut input, output, &mut scratch)
                .expect("buffers come from the plan");
        }
        spectrum
    }

    /// The signal of `length` samples behind a spectrogram.
    ///
    /// The imaginary parts of the first and last bin are ignored, as they are in
    /// `torch.istft`. A real signal has none, but a masked spectrogram can.
    pub fn inverse(&self, spectrum: &[Complex32], length: usize) -> Vec<f32> {
        let n_fft = self.n_fft();
        let frames = spectrum.len() / self.freqs();
        let total = n_fft + self.hop * (frames - 1);
        let mut signal = vec![0.0; total];
        let mut envelope = vec![0.0f32; total];
        let mut input = self.inverse.make_input_vec();
        let mut output = self.inverse.make_output_vec();
        let mut scratch = self.inverse.make_scratch_vec();
        let scale = 1.0 / n_fft as f32;

        for (frame, bins) in spectrum.chunks_exact(self.freqs()).enumerate() {
            input.copy_from_slice(bins);
            input[0].im = 0.0;
            input[self.freqs() - 1].im = 0.0;
            self.inverse
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .expect("buffers come from the plan");

            let start = frame * self.hop;
            let target = signal[start..][..n_fft].iter_mut();
            let weight = envelope[start..][..n_fft].iter_mut();
            for (((y, e), x), w) in target.zip(weight).zip(&output).zip(&self.window) {
                *y += x * scale * w;
                *e += w * w;
            }
        }

        let start = n_fft / 2;
        signal[start..][..length]
            .iter()
            .zip(&envelope[start..])
            .map(|(y, e)| y / e)
            .collect()
    }
}

/// Pads frames of `channels` samples on each side with their mirror image around the
/// first and last frame, as `torch.nn.functional.pad` does in reflect mode.
///
/// `left` and `right` must be below the number of frames.
pub(crate) fn pad_reflect(samples: &[f32], channels: usize, left: usize, right: usize) -> Vec<f32> {
    let frames = samples.len() / channels;
    let frame = |i: usize| &samples[i * channels..][..channels];
    let mut padded = Vec::with_capacity((frames + left + right) * channels);
    for i in (1..=left).rev() {
        padded.extend_from_slice(frame(i));
    }
    padded.extend_from_slice(samples);
    for i in 1..=right {
        padded.extend_from_slice(frame(frames - 1 - i));
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| {
                let t = i as f32;
                0.5 * (t * 0.03).sin() + 0.3 * (t * 0.41).cos() + 0.1 * (t * 1.7 + 1.0).sin()
            })
            .collect()
    }

    #[test]
    fn reflect_padding_mirrors_around_the_edges() {
        let padded = pad_reflect(&[1.0, 2.0, 3.0, 4.0], 1, 2, 3);
        assert_eq!(padded, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn reflect_padding_keeps_frames_together() {
        let padded = pad_reflect(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 2, 1, 1);
        assert_eq!(
            padded,
            [2.0, 20.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 2.0, 20.0]
        );
    }

    #[test]
    fn sizes_match_torch() {
        let stft = Stft::new(2048, 512);
        assert_eq!(stft.freqs(), 1025);
        assert_eq!(stft.frames(588800), 1151);
    }

    #[test]
    fn a_tone_lands_in_its_bin() {
        let stft = Stft::new(64, 16);
        let tone: Vec<f32> = (0..640)
            .map(|i| (std::f32::consts::TAU * 8.0 * i as f32 / 64.0).sin())
            .collect();
        let spectrum = stft.forward(&tone);
        let frame = &spectrum[10 * stft.freqs()..][..stft.freqs()];
        let loudest = (0..stft.freqs())
            .max_by(|a, b| frame[*a].norm().total_cmp(&frame[*b].norm()))
            .unwrap();
        assert_eq!(loudest, 8);
        // Half of the sine's energy is in this bin; the window's mean is 1/2.
        assert!((frame[8].norm() - 16.0).abs() < 1e-3);
    }

    #[test]
    fn inverse_restores_the_signal() {
        let stft = Stft::new(64, 16);
        for length in [496, 500, 511] {
            let signal = signal(length);
            let restored = stft.inverse(&stft.forward(&signal), length);
            assert_eq!(restored.len(), length);
            for (a, b) in restored.iter().zip(&signal) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn inverse_ignores_the_imaginary_part_of_the_outer_bins() {
        let stft = Stft::new(64, 16);
        let mut spectrum = stft.forward(&signal(496));
        let clean = stft.inverse(&spectrum, 496);
        for frame in spectrum.chunks_exact_mut(stft.freqs()) {
            frame[0].im = 1.0;
            frame[stft.freqs() - 1].im = -1.0;
        }
        assert_eq!(stft.inverse(&spectrum, 496), clean);
    }
}
