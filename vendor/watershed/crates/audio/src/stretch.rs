use std::f32::consts::FRAC_PI_2;
use std::ops::Range;

use signalsmith_stretch::Stretch;

use crate::mixer::{Bus, MAX_BLOCK, Voice};

/// Tempo range the engine accepts, as a playback speed factor.
pub const MIN_TEMPO: f32 = 0.25;
pub const MAX_TEMPO: f32 = 2.0;
/// Transposition range the engine accepts, in semitones.
pub const MAX_SEMITONES: f32 = 24.0;

/// Length of the crossfade when a loop jumps from its end back to its start.
const LOOP_CROSSFADE_SECS: f32 = 0.01;

/// Above this frequency the vocal bus's tonality limit shifts the spectrum by a fixed
/// amount instead of scaling it, which keeps more of the timbre (the value in
/// Signalsmith's README and web demo).
const TONALITY_LIMIT_HZ: f32 = 8000.0;

/// How the vocal bus is transposed. Both off, it is transposed like the pitched bus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VocalShift {
    /// Keep the formants (the resonances of the throat and mouth) where they were,
    /// as when a singer sings lower, instead of moving them with the pitch.
    pub formants: bool,
    /// Apply the tonality limit above [`TONALITY_LIMIT_HZ`].
    pub tonality_limit: bool,
}

/// Tempo and pitch change on the stereo mix, with Signalsmith Stretch.
///
/// Stretchers with the same configuration, one per [`Bus`]: the pitched bus gets the
/// transposition, the pitch-locked one only the tempo, the vocal one the
/// transposition with its [`VocalShift`]. Same configuration means same latency, so
/// the buses stay aligned and are summed at the output.
///
/// It reads the song through a [`Voice`] at `tempo` song frames per output frame,
/// jumping back at the end of a loop, and tracks which song frame is being heard,
/// since the stretcher delays the audio by about 120 ms. Nothing here allocates
/// after [`Stretcher::new`].
pub(crate) struct Stretcher {
    /// Indexed by [`Bus`].
    stretch: [Stretch; 3],
    /// Input fed before the seek point after a seek, so the output starts right away:
    /// one block plus one interval, as the library recommends.
    preroll: usize,
    input_latency: usize,
    output_latency: usize,
    /// Input frames fed so far, counting on through loop jumps. Fractional so that
    /// slow tempos do not drift.
    fed: f64,
    /// Song frame of the next input frame.
    next: usize,
    /// Maps fed frames to song frames: `(fed, song frame)` where the current pass
    /// started, and the same for the previous pass, which is still being heard for a
    /// moment after a loop jump.
    pass: (usize, usize),
    previous_pass: (usize, usize),
    /// After a loop jump: the song frame after the loop end, faded out over the
    /// start of the next pass, and how many crossfade frames are done.
    tail: Option<(usize, usize)>,
    crossfade: usize,
    /// Lowest position to report after a seek, while the latency is filling up.
    floor: usize,
    /// Transposition currently applied to the pitched and vocal buses, and how the
    /// vocal one is transposed.
    semitones: f32,
    vocal: VocalShift,
    sample_rate: f32,
    /// Stereo scratch buffers: the mixed input of each bus, the crossfade tail and
    /// the output of the buses after the first.
    input: [Vec<f32>; 3],
    tail_input: Vec<f32>,
    bus_output: Vec<f32>,
}

impl Stretcher {
    pub fn new(sample_rate: u32) -> Self {
        // Same values as Signalsmith's default preset, which does not expose them.
        let rate = sample_rate as f32;
        let block = (rate * 0.12) as usize;
        let interval = (rate * 0.03) as usize;
        let stretch = Bus::ALL.map(|_| Stretch::new(2, block, interval));
        let preroll = block + interval;
        let max_input = (MAX_BLOCK as f32 * MAX_TEMPO).ceil() as usize + 1;
        Self {
            input_latency: stretch[0].input_latency(),
            output_latency: stretch[0].output_latency(),
            stretch,
            preroll,
            fed: 0.0,
            next: 0,
            pass: (0, 0),
            previous_pass: (0, 0),
            tail: None,
            crossfade: ((rate * LOOP_CROSSFADE_SECS) as usize).max(1),
            floor: 0,
            semitones: 0.0,
            vocal: VocalShift::default(),
            sample_rate: rate,
            input: Bus::ALL.map(|_| vec![0.0; preroll.max(max_input) * 2]),
            tail_input: vec![0.0; max_input * 2],
            bus_output: vec![0.0; MAX_BLOCK * 2],
        }
    }

    /// Clears the stretcher and positions it so that `frame` is heard next.
    pub fn seek(&mut self, voice: &mut Voice, frame: usize, tempo: f32) {
        // The stretcher works on input from `input_latency` frames ago, so feeding up
        // to `frame + input_latency` centers its first output on `frame`.
        let end = frame + self.input_latency;
        let start = end.saturating_sub(self.preroll);
        let pad = self.preroll - (end - start);
        // Otherwise a mute or solo changed while paused would ramp in across the pre-roll.
        voice.snap_gains();
        for bus in Bus::ALL {
            let stretch = &mut self.stretch[bus as usize];
            stretch.reset();
            let input = &mut self.input[bus as usize][..self.preroll * 2];
            input.fill(0.0);
            voice.mix_into(start, bus, &mut input[pad * 2..]);
            stretch.seek(&*input, f64::from(tempo));
        }
        self.fed = end as f64;
        self.next = end;
        self.pass = (0, 0);
        self.previous_pass = (0, 0);
        self.tail = None;
        self.floor = frame;
    }

    /// Fills `out` (interleaved stereo) and advances the song by `tempo` frames per
    /// output frame. When the input crosses the end of `looped` it jumps back to its
    /// start. Past the end of the song the input is silence.
    pub fn render(
        &mut self,
        voice: &mut Voice,
        tempo: f32,
        semitones: f32,
        vocal: VocalShift,
        looped: Option<Range<usize>>,
        out: &mut [f32],
    ) {
        if semitones != self.semitones {
            self.stretch[Bus::Pitched as usize].set_transpose_factor_semitones(semitones, None);
        }
        if semitones != self.semitones || vocal != self.vocal {
            let stretch = &mut self.stretch[Bus::Vocal as usize];
            let limit = vocal
                .tonality_limit
                .then_some(TONALITY_LIMIT_HZ / self.sample_rate);
            stretch.set_transpose_factor_semitones(semitones, limit);
            // A factor of 1 with pitch compensation: the formants stay where they were.
            stretch.set_formant_factor(1.0, vocal.formants);
            self.semitones = semitones;
            self.vocal = vocal;
        }
        let before = self.fed as usize;
        self.fed += (out.len() / 2) as f64 * f64::from(tempo);
        let frames = self.fed as usize - before;
        for input in &mut self.input {
            input[..frames * 2].fill(0.0);
        }

        let mut done = 0;
        while done < frames {
            // Only a pass that reaches the loop end from before it jumps back, so
            // playing after the loop simply goes on.
            let loop_end = looped.as_ref().filter(|r| self.next < r.end).map(|r| r.end);
            let len = loop_end.map_or(frames - done, |end| (frames - done).min(end - self.next));
            let mut tail = self.tail;
            for bus in Bus::ALL {
                let chunk = &mut self.input[bus as usize][done * 2..(done + len) * 2];
                voice.mix_into(self.next, bus, chunk);
                tail = fade_tail(
                    self.tail,
                    &mut self.tail_input,
                    self.crossfade,
                    voice,
                    bus,
                    chunk,
                );
            }
            self.tail = tail;
            self.next += len;
            done += len;

            if let (Some(end), Some(range)) = (loop_end, &looped)
                && self.next == end
            {
                self.previous_pass = self.pass;
                self.pass = (before + done, range.start);
                self.next = range.start;
                self.tail = Some((end, 0));
                self.floor = 0;
            }
        }
        let [first, rest @ ..] = &mut self.stretch;
        first.process(&self.input[0][..frames * 2], &mut *out);
        let bus_output = &mut self.bus_output[..out.len()];
        for (stretch, input) in rest.iter_mut().zip(&self.input[1..]) {
            stretch.process(&input[..frames * 2], &mut *bus_output);
            for (o, b) in out.iter_mut().zip(bus_output.iter()) {
                *o += b;
            }
        }
    }

    /// The song frame being heard, given `extra` frames of output latency added after
    /// the stretcher (e.g. a limiter). Output latency counts at the current tempo.
    /// Assumes a loop is longer than that latency (about 0.2 s at most).
    pub fn heard(&self, tempo: f32, extra: usize) -> usize {
        (self.heard_exact(tempo, extra).max(0.0) as usize).max(self.floor)
    }

    /// Like [`Stretcher::heard`], but fractional and without holding the seek target
    /// while the latency fills: after a seek it starts the output latency before the
    /// target, and reaches it when the target is really heard.
    pub fn heard_exact(&self, tempo: f32, extra: usize) -> f64 {
        let behind =
            self.input_latency as f64 + (self.output_latency + extra) as f64 * f64::from(tempo);
        let fed = self.fed - behind;
        let (start, song) = if fed >= self.pass.0 as f64 {
            self.pass
        } else {
            self.previous_pass
        };
        song as f64 + (fed - start as f64)
    }
}

/// Crossfades the start of a loop pass with what followed the loop end, so the jump
/// does not click. `tail` is `(song frame, crossfade frames done)`; returns it
/// advanced past `chunk`.
fn fade_tail(
    tail: Option<(usize, usize)>,
    scratch: &mut [f32],
    crossfade: usize,
    voice: &mut Voice,
    bus: Bus,
    chunk: &mut [f32],
) -> Option<(usize, usize)> {
    let (from, done) = tail?;
    let frames = (chunk.len() / 2).min(crossfade - done);
    let old_audio = &mut scratch[..frames * 2];
    old_audio.fill(0.0);
    voice.mix_into(from, bus, old_audio);
    for (i, (new, old)) in chunk
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(old_audio.as_chunks::<2>().0)
        .enumerate()
    {
        // Equal power: the two sides are unrelated audio.
        let t = (done + i) as f32 + 0.5;
        let angle = t / crossfade as f32 * FRAC_PI_2;
        let (fade_in, fade_out) = (angle.sin(), angle.cos());
        new[0] = new[0] * fade_in + old[0] * fade_out;
        new[1] = new[1] * fade_in + old[1] * fade_out;
    }
    (done + frames < crossfade).then_some((from + frames, done + frames))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::Song;

    const RATE: u32 = 48_000;

    /// A song with one track: silence until `onset`, then a 440 Hz sine.
    fn voice(frames: usize, onset: usize) -> Voice {
        let samples: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let phase = i as f32 * 440.0 * std::f32::consts::TAU / RATE as f32;
                let s = if i < onset { 0.0 } else { 0.5 * phase.sin() };
                [s, s]
            })
            .collect();
        Voice::new(Arc::new(Song::new(RATE, vec![("sine".into(), samples)])))
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn seek_reports_the_target_and_sound_starts_soon() {
        let mut voice = voice(RATE as usize * 4, 0);
        let mut stretcher = Stretcher::new(RATE);
        stretcher.seek(&mut voice, 48_000, 1.0);
        assert_eq!(stretcher.heard(1.0, 0), 48_000);

        // Sound arrives within the output latency (about 60 ms).
        let mut out = vec![0.0; 512 * 2];
        let mut first_loud = None;
        for block in 0..20 {
            stretcher.render(&mut voice, 1.0, 0.0, VocalShift::default(), None, &mut out);
            if first_loud.is_none() && rms(&out) > 0.1 {
                first_loud = Some(block * 512);
            }
        }
        let first_loud = first_loud.expect("no output");
        assert!(first_loud <= stretcher.output_latency, "{first_loud}");
    }

    #[test]
    fn position_follows_tempo() {
        let mut voice = voice(RATE as usize * 4, 0);
        let mut stretcher = Stretcher::new(RATE);
        stretcher.seek(&mut voice, 0, 0.5);
        let mut out = vec![0.0; 500 * 2];
        // Past the initial latency, then measure one second of output.
        for _ in 0..96 {
            stretcher.render(&mut voice, 0.5, 0.0, VocalShift::default(), None, &mut out);
        }
        let before = stretcher.heard(0.5, 0);
        for _ in 0..96 {
            stretcher.render(&mut voice, 0.5, 0.0, VocalShift::default(), None, &mut out);
        }
        assert_eq!(stretcher.heard(0.5, 0) - before, 24_000);
    }

    #[test]
    fn reported_position_matches_what_is_heard() {
        // Silence, then a sine starting at one second.
        let onset = RATE as usize;
        let mut voice = voice(RATE as usize * 3, onset);

        for tempo in [0.5, 1.0, 1.25] {
            let mut stretcher = Stretcher::new(RATE);
            stretcher.seek(&mut voice, onset - RATE as usize / 2, tempo);
            let mut out = vec![0.0; 64 * 2];
            let heard_at_onset = loop {
                stretcher.render(
                    &mut voice,
                    tempo,
                    0.0,
                    VocalShift::default(),
                    None,
                    &mut out,
                );
                if rms(&out) > 0.1 {
                    break stretcher.heard(tempo, 0);
                }
            };
            // Within 10 ms of the true onset (measured: about one 64-frame block).
            let error = heard_at_onset.abs_diff(onset);
            assert!(error < RATE as usize / 100, "tempo {tempo}: off by {error}");
        }
    }

    #[test]
    fn loops_between_a_and_b() {
        let mut voice = voice(RATE as usize * 10, 0);
        let mut stretcher = Stretcher::new(RATE);
        let looped = 48_000..96_000;
        stretcher.seek(&mut voice, 0, 1.25);
        let mut out = vec![0.0; 256 * 2];
        let mut previous = 0;
        let mut jumps = 0;
        // 8 s of output is 10 s of song at 1.25: one to reach A, then one per pass.
        for _ in 0..1500 {
            stretcher.render(
                &mut voice,
                1.25,
                0.0,
                VocalShift::default(),
                Some(looped.clone()),
                &mut out,
            );
            let heard = stretcher.heard(1.25, 0);
            assert!(heard < looped.end, "heard {heard} past the loop end");
            if heard < previous {
                // Jumps land on A, within one block.
                assert!(heard - looped.start <= 320, "jumped to {heard}");
                jumps += 1;
            }
            previous = heard;
        }
        assert_eq!(jumps, 8);
    }

    #[test]
    fn plays_on_after_the_loop() {
        let mut voice = voice(RATE as usize * 4, 0);
        let mut stretcher = Stretcher::new(RATE);
        stretcher.seek(&mut voice, 100_000, 1.0);
        let mut out = vec![0.0; 512 * 2];
        for _ in 0..100 {
            stretcher.render(
                &mut voice,
                1.0,
                0.0,
                VocalShift::default(),
                Some(48_000..96_000),
                &mut out,
            );
        }
        assert!(stretcher.heard(1.0, 0) > 140_000);
    }

    #[test]
    fn loop_jump_crossfades() {
        // Silence up to the loop end, then a constant: after the jump to A the
        // constant fades out smoothly instead of stopping dead.
        let samples = (0..2000).flat_map(|i| [f32::from(u8::from(i >= 1000)); 2]);
        let mut voice = Voice::new(Arc::new(Song::new(
            RATE,
            vec![("step".into(), samples.collect::<Vec<f32>>())],
        )));
        let mut scratch = vec![0.0; 100 * 2];
        let mut chunk = vec![0.0; 60 * 2];
        let tail = fade_tail(
            Some((1000, 0)),
            &mut scratch,
            50,
            &mut voice,
            Bus::Pitched,
            &mut chunk,
        );
        let left: Vec<f32> = chunk.iter().step_by(2).copied().collect();
        assert!(left[0] > 0.99 && left[49] < 0.05, "{left:?}");
        assert!(left[..50].windows(2).all(|w| w[1] < w[0]), "{left:?}");
        assert!(left[50..].iter().all(|&s| s == 0.0));
        assert_eq!(tail, None);
    }

    /// Frequency of a steady stereo signal, from its zero crossings.
    fn frequency(samples: &[f32]) -> f32 {
        let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
        let crossings = left
            .windows(2)
            .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
            .count();
        crossings as f32 * RATE as f32 / left.len() as f32
    }

    #[test]
    fn pitch_locked_tracks_keep_their_pitch() {
        for (locked, expected) in [(false, 880.0), (true, 440.0)] {
            let mut voice = voice(RATE as usize * 4, 0);
            voice.song.tracks[0].params.set_pitch_lock(locked);
            let mut stretcher = Stretcher::new(RATE);
            stretcher.seek(&mut voice, 0, 1.0);
            let mut out = vec![0.0; 512 * 2];
            // Past the latency, then one second.
            for _ in 0..20 {
                stretcher.render(&mut voice, 0.8, 12.0, VocalShift::default(), None, &mut out);
            }
            let mut heard = Vec::new();
            for _ in 0..94 {
                stretcher.render(&mut voice, 0.8, 12.0, VocalShift::default(), None, &mut out);
                heard.extend_from_slice(&out);
            }
            let f = frequency(&heard);
            assert!((f - expected).abs() < 10.0, "locked {locked}: {f} Hz");
        }
    }

    #[test]
    fn vocals_are_transposed_with_their_options() {
        let options = [
            VocalShift::default(),
            VocalShift {
                formants: true,
                tonality_limit: true,
            },
        ];
        for vocal in options {
            let mut voice = voice(RATE as usize * 4, 0);
            voice.song.tracks[0].params.set_vocal(true);
            let mut stretcher = Stretcher::new(RATE);
            stretcher.seek(&mut voice, 0, 1.0);
            let mut out = vec![0.0; 512 * 2];
            for _ in 0..20 {
                stretcher.render(&mut voice, 0.8, 12.0, vocal, None, &mut out);
            }
            let mut heard = Vec::new();
            for _ in 0..94 {
                stretcher.render(&mut voice, 0.8, 12.0, vocal, None, &mut out);
                heard.extend_from_slice(&out);
            }
            let f = frequency(&heard);
            assert!((f - 880.0).abs() < 10.0, "{vocal:?}: {f} Hz");
            // A lone sine is the worst case for the formant correction, which reshapes
            // the spectrum's envelope: it comes out at ~0.13 instead of ~0.35. A voice
            // has harmonics all along the envelope, so it should keep its level better.
            let level = rms(&heard);
            assert!(level > 0.1, "{vocal:?}: rms {level}");
        }
    }

    #[test]
    fn keeps_the_level_at_unity() {
        let mut voice = voice(RATE as usize * 4, 0);
        let mut stretcher = Stretcher::new(RATE);
        stretcher.seek(&mut voice, 0, 1.0);
        let mut out = vec![0.0; 512 * 2];
        for _ in 0..40 {
            stretcher.render(
                &mut voice,
                0.75,
                -2.0,
                VocalShift::default(),
                None,
                &mut out,
            );
        }
        // A 0.5 amplitude sine has an RMS of about 0.354.
        let level = rms(&out);
        assert!((level - 0.354).abs() < 0.05, "rms {level}");
    }
}
