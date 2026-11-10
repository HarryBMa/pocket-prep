/// Interleaved stereo audio stored as packed 24-bit integers: 3 bytes per sample
/// instead of 4 for `f32`, a quarter less memory for every stem of a song.
///
/// Each track is scaled so that its own peak uses the full 24-bit range, so the
/// rounding error stays 144 dB under the track's peak: the resolution of the
/// 24-bit FLAC stems it is usually decoded from. `f32` input of any level
/// (stems often exceed full scale) round-trips within one step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Samples {
    data: Vec<[u8; 3]>,
    /// Value of one integer step.
    scale: f32,
}

const MAX_INT: f32 = ((1 << 23) - 1) as f32;

impl Samples {
    pub fn from_f32(samples: &[f32]) -> Self {
        let peak = samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        if peak == 0.0 || !peak.is_finite() {
            return Self {
                data: vec![[0; 3]; samples.len()],
                scale: 0.0,
            };
        }
        let to_int = MAX_INT / peak;
        let data = samples
            .iter()
            .map(|&s| {
                let int = (s * to_int).round() as i32;
                let [b0, b1, b2, _] = int.to_le_bytes();
                [b0, b1, b2]
            })
            .collect();
        Self {
            data,
            scale: peak / MAX_INT,
        }
    }

    /// Number of samples (twice the number of frames).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The sample at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        decode(self.data[index], self.scale)
    }

    /// The left and right samples of each frame in `frames`.
    #[inline]
    pub fn frames(
        &self,
        frames: std::ops::Range<usize>,
    ) -> impl Iterator<Item = [f32; 2]> + use<'_> {
        let scale = self.scale;
        self.data[frames.start * 2..frames.end * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(move |[l, r]| [decode(*l, scale), decode(*r, scale)])
    }

    /// All samples, unpacked.
    pub fn to_f32(&self) -> Vec<f32> {
        let scale = self.scale;
        self.data.iter().map(|&s| decode(s, scale)).collect()
    }

    /// Pads with silence (or truncates) to `len` samples.
    pub(crate) fn resize(&mut self, len: usize) {
        self.data.resize(len, [0; 3]);
    }
}

impl From<Vec<f32>> for Samples {
    fn from(samples: Vec<f32>) -> Self {
        Self::from_f32(&samples)
    }
}

#[inline]
fn decode([b0, b1, b2]: [u8; 3], scale: f32) -> f32 {
    // The top byte takes the sign bit, and the arithmetic shift extends it.
    (i32::from_le_bytes([0, b0, b1, b2]) >> 8) as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_24_bits_of_the_peak() {
        // Over full scale, as separated stems often are, with quiet details.
        let input: Vec<f32> = (0..10_000)
            .map(|i| (i as f32 * 0.01).sin() * 1.6 + (i as f32 * 0.37).sin() * 1e-4)
            .collect();
        let samples = Samples::from_f32(&input);
        let peak = input.iter().fold(0.0f32, |p, s| p.max(s.abs()));
        let max_error = input
            .iter()
            .enumerate()
            .map(|(i, &s)| (samples.get(i) - s).abs())
            .fold(0.0f32, f32::max);
        // Half a step of 24 bits plus f32's own rounding, which near the peak is
        // about as coarse: at most one step.
        assert!(max_error <= peak / MAX_INT, "{max_error}");
        assert_eq!(samples.get(0), 0.0);
    }

    #[test]
    fn keeps_the_sign_and_the_extremes() {
        let samples = Samples::from_f32(&[-2.0, 2.0, -0.5, 0.0]);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(samples.get(0), -2.0));
        assert!(close(samples.get(1), 2.0));
        assert!(close(samples.get(2), -0.5));
        assert_eq!(samples.get(3), 0.0);
        let frames: Vec<_> = samples.frames(0..2).collect();
        assert!(close(frames[0][0], -2.0) && close(frames[0][1], 2.0));
        let unpacked = samples.to_f32();
        assert_eq!(unpacked.len(), 4);
        assert!(
            unpacked
                .iter()
                .enumerate()
                .all(|(i, &s)| s == samples.get(i))
        );
    }

    #[test]
    fn silence_and_padding() {
        let mut samples = Samples::from_f32(&[0.0; 4]);
        assert!((0..4).all(|i| samples.get(i) == 0.0));
        samples.resize(6);
        assert_eq!(samples.len(), 6);
        assert_eq!(samples.get(5), 0.0);
    }
}
