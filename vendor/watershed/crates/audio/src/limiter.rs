/// Peak limiter for the master output. Keeps a hot stem soloed at full volume from
/// clipping at the device.
///
/// It looks ahead a few milliseconds: the input is delayed so the gain can come down
/// smoothly before a peak reaches the output, instead of squaring it off.
pub(crate) struct Limiter {
    /// Peaks above this level are turned down.
    threshold: f32,
    /// Lookahead delay line, one stereo frame per slot.
    delay: Vec<[f32; 2]>,
    index: usize,
    /// Largest peak seen in the last `delay.len()` frames, and how long to keep it.
    held: f32,
    hold: usize,
    gain: f32,
    attack: f32,
    release: f32,
}

/// -1 dBFS, a little headroom below full scale.
const THRESHOLD: f32 = 0.891;
const LOOKAHEAD_SECS: f32 = 0.003;
const RELEASE_SECS: f32 = 0.1;

impl Limiter {
    pub fn new(sample_rate: u32) -> Self {
        let rate = sample_rate as f32;
        let lookahead = ((LOOKAHEAD_SECS * rate) as usize).max(1);
        Self {
            threshold: THRESHOLD,
            delay: vec![[0.0; 2]; lookahead],
            index: 0,
            held: 0.0,
            hold: 0,
            gain: 1.0,
            // Settles to within 1% of the target gain during the lookahead.
            attack: (-5.0 / lookahead as f32).exp(),
            release: (-1.0 / (RELEASE_SECS * rate)).exp(),
        }
    }

    /// Delay added to the signal, in frames.
    pub fn latency(&self) -> usize {
        self.delay.len()
    }

    /// Limits interleaved stereo samples in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        let lookahead = self.delay.len();
        for frame in samples.as_chunks_mut::<2>().0 {
            let peak = frame[0].abs().max(frame[1].abs());
            if peak >= self.held {
                self.held = peak;
                self.hold = lookahead;
            } else if self.hold > 0 {
                self.hold -= 1;
            } else {
                self.held = peak;
            }

            let target = if self.held > self.threshold {
                self.threshold / self.held
            } else {
                1.0
            };
            let coef = if target < self.gain {
                self.attack
            } else {
                self.release
            };
            self.gain = target + (self.gain - target) * coef;

            let delayed = std::mem::replace(&mut self.delay[self.index], *frame);
            self.index = (self.index + 1) % lookahead;
            frame[0] = delayed[0] * self.gain;
            frame[1] = delayed[1] * self.gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(amplitude: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let s = amplitude * (i as f32 * 0.05).sin();
                [s, s]
            })
            .collect()
    }

    #[test]
    fn quiet_signal_is_only_delayed() {
        let mut limiter = Limiter::new(48_000);
        let input = sine(0.5, 1000);
        let mut out = input.clone();
        limiter.process(&mut out);
        let delay = limiter.latency() * 2;
        assert!(out[..delay].iter().all(|&s| s == 0.0));
        for (o, i) in out[delay..].iter().zip(&input) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn hot_signal_stays_below_full_scale() {
        let mut limiter = Limiter::new(48_000);
        // +6 dB over full scale, starting abruptly.
        let mut out = sine(2.0, 48_000);
        limiter.process(&mut out);
        let max = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(max <= 1.0, "peak {max}");
        // Once settled, the level sits at the threshold.
        let tail = out[out.len() / 2..]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((tail - THRESHOLD).abs() < 0.01, "settled peak {tail}");
    }

    #[test]
    fn single_spike_is_caught() {
        let mut limiter = Limiter::new(48_000);
        let mut out = vec![0.1; 2000];
        out[1000] = 3.0;
        out[1001] = -3.0;
        limiter.process(&mut out);
        assert!(out.iter().all(|s| s.abs() <= 1.0), "{out:?}");
    }
}
