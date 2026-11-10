//! Metronome click and count-in, mixed after the stretcher so they are never
//! time-stretched or transposed.
//!
//! The beats come from the song's grid, in song frames: steady, or following the
//! song's own beats when its tempo moves. The click follows the song frame being heard, so it lands on the beat the user
//! hears at any tempo. The sounds are synthesized once, when the engine starts.

use std::f32::consts::TAU;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::{AtomicF32, Beats};

/// Beats closer than this to a seek target still click: a playhead snapped to a
/// beat can land a frame or two before or after it.
const SEEK_TOLERANCE_SECS: f64 = 0.002;

/// The sound of the click.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClickSound {
    /// A short, dry tick.
    #[default]
    Click,
    /// The classic beep of a DAW metronome.
    Metronome,
    /// A woodblock.
    Wood,
}

impl ClickSound {
    pub const ALL: [ClickSound; 3] = [Self::Click, Self::Metronome, Self::Wood];
}

/// Which output channels the click plays on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClickChannels {
    Left,
    #[default]
    Both,
    Right,
}

/// Click settings shared with the UI. The beats come through the command queue.
#[derive(Debug, Default)]
pub(crate) struct ClickParams {
    /// Metronome while playing.
    pub on: AtomicBool,
    /// One bar of clicks before playback starts.
    pub count_in: AtomicBool,
    pub sound: AtomicU8,
    pub channels: AtomicU8,
    pub gain: AtomicF32,
}

impl ClickParams {
    pub fn sound(&self) -> ClickSound {
        ClickSound::ALL[usize::from(self.sound.load(Ordering::Relaxed)).min(2)]
    }

    pub fn channels(&self) -> ClickChannels {
        match self.channels.load(Ordering::Relaxed) {
            0 => ClickChannels::Left,
            2 => ClickChannels::Right,
            _ => ClickChannels::Both,
        }
    }
}

pub(crate) fn sound_index(sound: ClickSound) -> u8 {
    match sound {
        ClickSound::Click => 0,
        ClickSound::Metronome => 1,
        ClickSound::Wood => 2,
    }
}

pub(crate) fn channels_index(channels: ClickChannels) -> u8 {
    match channels {
        ClickChannels::Left => 0,
        ClickChannels::Both => 1,
        ClickChannels::Right => 2,
    }
}

/// A count-in in progress: a silent bar before `target`, timed like the song.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CountIn {
    /// Song frame "heard" so far, counting up from one bar before where playback
    /// starts.
    pub heard: f64,
}

/// Plays the click on the audio thread. Nothing here allocates after
/// [`Clicker::new`].
pub(crate) struct Clicker {
    /// The song's beats, in song frames; `None` without a grid.
    beats: Option<Box<Beats>>,
    /// Indexed by sound, then accent (1 for the first beat of a bar). Mono.
    sounds: [[Vec<f32>; 2]; 3],
    /// The click sounding now: sound, accent and next sample.
    voice: Option<(usize, usize, usize)>,
    /// Song frame heard at the end of the previous block, while playing.
    last: Option<f64>,
    /// After a seek, beats from here on click once the heard position reaches it.
    waiting: bool,
    /// Beats before bar 1 click up to here: the end of a count-in.
    count_in_end: f64,
    tolerance: f64,
}

impl Clicker {
    pub fn new(sample_rate: u32) -> Self {
        let sounds = ClickSound::ALL.map(|s| [false, true].map(|a| synthesize(s, a, sample_rate)));
        Self {
            beats: None,
            sounds,
            voice: None,
            last: None,
            waiting: false,
            count_in_end: f64::NEG_INFINITY,
            tolerance: SEEK_TOLERANCE_SECS * f64::from(sample_rate),
        }
    }

    /// Replaces the beats, returning the old ones so they are freed elsewhere.
    pub fn set_beats(&mut self, beats: Option<Box<Beats>>) -> Option<Box<Beats>> {
        std::mem::replace(&mut self.beats, beats)
    }

    /// Playback will start at `frame`: beats from there on click.
    pub fn seek(&mut self, frame: usize) {
        self.last = Some(frame as f64 - self.tolerance);
        self.waiting = true;
        self.count_in_end = f64::NEG_INFINITY;
    }

    /// Playback stopped; a click already sounding rings out.
    pub fn stop(&mut self) {
        self.last = None;
    }

    /// Starts a count-in before `target`, or `None` without a grid. It counts the
    /// bar before the first beat from `target`, at its pace.
    pub fn count_in(&mut self, target: usize) -> Option<CountIn> {
        let beats = self.beats.as_ref()?;
        let target = target as f64;
        let heard = target - beats.bar_before(target - self.tolerance);
        self.count_in_end = target;
        Some(CountIn { heard })
    }

    /// The count-in is over and the song is heard from `heard` on.
    pub fn count_in_done(&mut self, heard: f64) {
        self.last = Some(heard);
        self.waiting = false;
    }

    /// Clicks for a block of `out` (interleaved stereo) that played `advanced` song
    /// frames and ended with `heard` being heard.
    pub fn play(
        &mut self,
        params: &ClickParams,
        heard: f64,
        advanced: f64,
        looped: Option<&Range<usize>>,
        out: &mut [f32],
    ) {
        let block = heard - advanced..heard;
        // Beats from `floor` on click.
        let floor = match self.last {
            Some(last) if heard >= last => {
                self.waiting = false;
                last
            }
            // Still filling the stretcher's latency after a seek.
            Some(_) if self.waiting => {
                self.ring(params, out);
                return;
            }
            // Jumped back to the start of a loop.
            _ => block.start.max(looped.map_or(0, |r| r.start) as f64),
        };
        self.last = Some(heard);
        // The metronome starts at bar 1; the end of a count-in may come before it.
        let on = params.on.load(Ordering::Relaxed);
        let count_in_end = self.count_in_end;
        self.render(
            params,
            true,
            block,
            floor,
            |index, frame| (on && index >= 0) || frame < count_in_end,
            out,
        );
    }

    /// Lets a click already sounding ring out over `out`, starting no new one.
    pub fn ring(&mut self, params: &ClickParams, out: &mut [f32]) {
        self.render(params, false, 0.0..0.0, 0.0, |_, _| false, out);
    }

    /// Clicks for a count-in block over `out`, advancing it by `tempo` song frames
    /// per output frame.
    pub fn play_count_in(
        &mut self,
        params: &ClickParams,
        count_in: &mut CountIn,
        tempo: f32,
        out: &mut [f32],
    ) {
        let from = count_in.heard;
        count_in.heard += (out.len() / 2) as f64 * f64::from(tempo);
        let block = from..count_in.heard;
        self.render(params, true, block, from, |_, _| true, out);
    }

    /// Adds the click to `out`, which plays the song frames in `block`. With `beats`,
    /// a new click starts at each beat from `floor` to the end of the block that
    /// `keep` accepts.
    fn render(
        &mut self,
        params: &ClickParams,
        beats: bool,
        block: Range<f64>,
        floor: f64,
        keep: impl Fn(i64, f64) -> bool,
        out: &mut [f32],
    ) {
        let frames = out.len() / 2;
        let gain = params.gain.load();
        let channels = match params.channels() {
            ClickChannels::Left => [1.0, 0.0],
            ClickChannels::Both => [1.0, 1.0],
            ClickChannels::Right => [0.0, 1.0],
        };
        let sound = sound_index(params.sound()) as usize;

        let gains = (gain, channels);
        let mut done = 0;
        if beats && let Some(beats) = self.beats.take() {
            if block.end > block.start {
                let span = block.end - block.start;
                for (index, frame) in beats.within(floor..block.end) {
                    if !keep(index, frame) {
                        continue;
                    }
                    let offset = (frame - block.start) / span * frames as f64;
                    let at = (offset.max(0.0).round() as usize).clamp(done, frames);
                    self.add_voice(&mut out[done * 2..at * 2], gains);
                    done = at;
                    self.voice = Some((sound, usize::from(beats.accented(index)), 0));
                }
            }
            self.beats = Some(beats);
        }
        self.add_voice(&mut out[done * 2..], gains);
    }

    /// Adds the click sounding now to `out`, as far as it lasts, at `gain` and on
    /// the channels set to 1.0.
    fn add_voice(&mut self, out: &mut [f32], (gain, [left, right]): (f32, [f32; 2])) {
        let Some((sound, accent, next)) = &mut self.voice else {
            return;
        };
        let samples = &self.sounds[*sound][*accent][*next..];
        let n = (out.len() / 2).min(samples.len());
        for (frame, sample) in out.as_chunks_mut::<2>().0.iter_mut().zip(&samples[..n]) {
            let sample = soft_clip(sample * gain);
            frame[0] += sample * left;
            frame[1] += sample * right;
        }
        *next += n;
        if n == samples.len() {
            self.voice = None;
        }
    }
}

/// Level up to which the click passes unchanged; above it, peaks bend smoothly
/// towards 1.0. A wide knee, so a click turned far up thickens instead of being
/// squared off. At 0 dB only the Wood click's first peak (0.9) is touched, by
/// less than 1 dB.
const SOFT_CLIP_KNEE: f32 = 0.5;

/// Keeps a click turned up past 0 dB below full scale: it is mixed after the
/// limiter, so it would otherwise be clipped hard at the output. Louder, not harsh.
fn soft_clip(x: f32) -> f32 {
    let magnitude = x.abs();
    if magnitude <= SOFT_CLIP_KNEE {
        return x;
    }
    let room = 1.0 - SOFT_CLIP_KNEE;
    let bent = SOFT_CLIP_KNEE + room * ((magnitude - SOFT_CLIP_KNEE) / room).tanh();
    bent.copysign(x)
}

/// Peak level of each click, after Ableton Live's metronome samples so that the
/// three sounds are about as loud: a sustained tone needs a lower peak than a dry
/// pulse. The accent is told apart by its pitch, and in the Metronome sound also by
/// its level.
fn level(sound: ClickSound, accent: bool) -> f32 {
    match (sound, accent) {
        (ClickSound::Click, _) => 0.5,
        (ClickSound::Metronome, false) => 0.23,
        (ClickSound::Metronome, true) => 0.34,
        (ClickSound::Wood, _) => 0.9,
    }
}

/// One click, mono. Modelled on what Ableton Live's Click, Classic and Wood samples
/// measure (spectrum and envelope), not copied from them.
fn synthesize(sound: ClickSound, accent: bool, sample_rate: u32) -> Vec<f32> {
    let rate = sample_rate as f32;
    let mut samples = match sound {
        ClickSound::Click => click(accent, rate),
        ClickSound::Metronome => beep(accent, rate),
        ClickSound::Wood => woodblock(accent, rate),
    };
    let peak = samples.iter().fold(0.0f32, |p, s| p.max(s.abs()));
    if peak > 0.0 {
        let gain = level(sound, accent) / peak;
        samples.iter_mut().for_each(|s| *s *= gain);
    }
    samples
}

/// A damped sine starting at zero.
fn mode(t: f32, frequency: f32, decay: f32) -> f32 {
    (TAU * frequency * t).sin() * (-t / decay).exp()
}

/// Rises from 0 to 1 over `secs` along half a cosine.
fn rise(t: f32, secs: f32) -> f32 {
    if t >= secs {
        1.0
    } else {
        0.5 - 0.5 * (std::f32::consts::PI * t / secs).cos()
    }
}

/// A dry tick: one short pulse, a low thump around 240 Hz and a 1 kHz ring, all
/// gone within 5 ms. The accent has a wider pulse and a lower thump, so it sounds
/// fuller, and a higher ring.
fn click(accent: bool, rate: f32) -> Vec<f32> {
    // (pulse width, thump frequency, level and decay, ring frequency and level)
    let (width, thump, ring) = if accent {
        (0.0006, (186.0, 0.5, 0.0015), (1600.0, 0.15))
    } else {
        (0.0003, (240.0, 0.35, 0.0012), (1050.0, 0.3))
    };
    let ramp = 0.00015;
    // One-pole high-pass at 200 Hz: what follows the pulse swings below zero.
    let coef = (-TAU * 200.0 / rate).exp();
    let (mut previous_in, mut previous_out) = (0.0, 0.0);
    (0..(0.015 * rate) as usize)
        .map(|i| {
            let t = i as f32 / rate;
            let pulse = rise(t, ramp) * rise(width - t, ramp) * f32::from(u8::from(t < width));
            let high_passed = coef * (previous_out + pulse - previous_in);
            (previous_in, previous_out) = (pulse, high_passed);
            let tail = rise((0.015 - t).max(0.0), 0.002);
            let low = thump.1 * mode(t, thump.0, thump.2);
            (high_passed + low + ring.1 * mode(t, ring.0, 0.0015)) * tail
        })
        .collect()
}

/// The classic DAW beep: a 1.1 kHz tone (1.575 kHz on the accent) with a little
/// third and fifth harmonic, decaying for 17 ms, then cut off in 3 ms.
fn beep(accent: bool, rate: f32) -> Vec<f32> {
    let f = if accent { 1575.0 } else { 1100.0 };
    let length = 0.02;
    (0..(length * rate) as usize)
        .map(|i| {
            let t = i as f32 / rate;
            let w = TAU * f * t;
            let tone = w.sin() + 0.22 * (3.0 * w).sin() + 0.025 * (5.0 * w).sin();
            let envelope = rise(t, 0.0005) * (-t / 0.0082).exp() * rise(length - t, 0.003);
            tone * envelope
        })
        .collect()
}

/// A woodblock at 1.08 kHz (1.555 kHz on the accent): a soft 2 ms strike, a body
/// that holds for a few milliseconds and drops fast, weaker inharmonic modes, and
/// the hollow block ringing on quietly for about 0.3 s.
fn woodblock(accent: bool, rate: f32) -> Vec<f32> {
    let f = if accent { 1555.0 } else { 1080.0 };
    // (frequency ratio, level, decay): the accent has a strong mode close to the
    // main one, which beats against it.
    let modes: &[(f32, f32, f32)] = if accent {
        &[
            (0.94, 0.6, 0.006),
            (0.87, 0.05, 0.005),
            (2.23, 0.25, 0.0025),
            (2.48, 0.25, 0.002),
        ]
    } else {
        &[
            (0.86, 0.06, 0.005),
            (1.6, 0.2, 0.0025),
            (2.21, 0.6, 0.0025),
            (2.66, 0.15, 0.002),
            (3.36, 0.2, 0.002),
        ]
    };
    // The hollow ringing: the main mode and two close ones, so it is not a pure tone.
    // Quieter on the accent, which rings for less time in the reference.
    let ring_level = if accent { 0.6 } else { 1.0 };
    let rings = [(1.0, 0.16), (0.965, 0.08), (1.035, 0.06)].map(|(r, l)| (r, l * ring_level));
    let length = 0.3;
    // Deterministic noise for the strike, so every click is the same.
    let mut seed = 0x9e37_79b9_u32;
    (0..(length * rate) as usize)
        .map(|i| {
            let t = i as f32 / rate;
            let strike = rise(t, 0.002);
            let body = strike
                * (-(t - 0.002).max(0.0) / 0.02).exp()
                * (-(t - 0.007).max(0.0) / 0.0045).exp();
            let main = (TAU * f * t).sin() * body;
            let others: f32 = modes
                .iter()
                .map(|&(ratio, level, decay)| level * rise(t, 0.0005) * mode(t, f * ratio, decay))
                .sum();
            let ring: f32 = rings
                .iter()
                .map(|&(ratio, level)| level * (TAU * f * ratio * t).sin())
                .sum::<f32>()
                * (1.0 - (-t / 0.005).exp())
                * (-t / 0.055).exp();
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let white = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
            let noise = 0.12 * white * rise(t, 0.0001) * (-t / 0.0007).exp();
            (main + others + ring + noise) * rise(length - t, 0.005)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const BLOCK: usize = 256;

    /// Where the steady grids of the tests stop being listed (they go on after).
    const SONG_END: f64 = 1e6;

    fn clicker(beats: &Beats) -> Clicker {
        let mut clicker = Clicker::new(RATE);
        clicker.set_beats(Some(Box::new(beats.clone())));
        clicker
    }

    fn params() -> ClickParams {
        let params = ClickParams {
            gain: AtomicF32::new(1.0),
            ..Default::default()
        };
        params.on.store(true, Ordering::Relaxed);
        // Its accent is louder, which `clicks` tells apart.
        params
            .sound
            .store(sound_index(ClickSound::Metronome), Ordering::Relaxed);
        params
    }

    /// Where clicks start in `out` (interleaved stereo), and whether each is accented.
    /// A click starts with an exact zero, after a stretch of silence.
    fn clicks(out: &[f32]) -> Vec<(usize, bool)> {
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        let mut found = Vec::new();
        let mut zeros = usize::MAX / 2;
        for (i, &s) in left.iter().enumerate() {
            if s == 0.0 {
                zeros += 1;
                continue;
            }
            if zeros >= 200 {
                let peak = left[i..(i + 200).min(left.len())]
                    .iter()
                    .fold(0.0f32, |p, s| p.max(s.abs()));
                let threshold = (level(ClickSound::Metronome, false)
                    + level(ClickSound::Metronome, true))
                    / 2.0;
                found.push((i - 1, peak > threshold));
            }
            zeros = 0;
        }
        found
    }

    /// Plays `blocks` blocks heard from `from`, `tempo` song frames per output frame.
    fn play(clicker: &mut Clicker, params: &ClickParams, from: f64, tempo: f64) -> Vec<f32> {
        let mut out = vec![0.0; 400 * BLOCK * 2];
        let mut heard = from;
        for block in out.chunks_mut(BLOCK * 2) {
            heard += BLOCK as f64 * tempo;
            clicker.play(params, heard, BLOCK as f64 * tempo, None, block);
        }
        out
    }

    #[test]
    fn clicks_land_on_the_beats_at_any_tempo() {
        // 120 BPM from bar 1 at 0.5 s, in 3/4.
        let beats = Beats::steady(24_000.0, 24_000.0, 3, SONG_END).unwrap();
        for tempo in [0.5, 1.0, 1.25] {
            let params = params();
            let mut clicker = clicker(&beats);
            clicker.seek(0);
            let found = clicks(&play(&mut clicker, &params, 0.0, tempo));
            let expected: Vec<(usize, bool)> = (0..)
                .map(|k| (((24_000 + k * 24_000) as f64 / tempo) as usize, k % 3 == 0))
                .take_while(|&(f, _)| f < 400 * BLOCK - 200)
                .collect();
            assert_eq!(found.len(), expected.len(), "tempo {tempo}: {found:?}");
            for (f, e) in found.iter().zip(&expected) {
                assert!(
                    f.0.abs_diff(e.0) <= 1 && f.1 == e.1,
                    "tempo {tempo}: {f:?} vs {e:?}"
                );
            }
        }
    }

    #[test]
    fn nothing_before_bar_one() {
        let beats = Beats::steady(48_000.0, 12_000.0, 4, SONG_END).unwrap();
        let mut clicker = clicker(&beats);
        clicker.seek(0);
        let found = clicks(&play(&mut clicker, &params(), 0.0, 1.0));
        assert_eq!(found[0], (48_000, true));
    }

    #[test]
    fn a_seek_onto_a_beat_clicks_it() {
        let beats = Beats::steady(0.0, 24_000.0, 4, SONG_END).unwrap();
        let mut clicker = clicker(&beats);
        // Snapped a frame past beat 4; the stretcher's latency shows first.
        clicker.seek(72_001);
        let found = clicks(&play(&mut clicker, &params(), 72_001.0 - 3000.0, 1.0));
        assert_eq!(found[0], (3000 - 1, false));
    }

    #[test]
    fn loop_jumps_click_from_the_loop_start() {
        let beats = Beats::steady(0.0, 24_000.0, 4, SONG_END).unwrap();
        let params = params();
        let mut clicker = clicker(&beats);
        clicker.seek(30_000);
        let mut out = vec![0.0; 300 * BLOCK * 2];
        let looped = 24_000..60_000;
        let mut heard = 30_000.0;
        for block in out.chunks_mut(BLOCK * 2) {
            heard += BLOCK as f64;
            if heard >= looped.end as f64 {
                heard -= (looped.end - looped.start) as f64;
            }
            clicker.play(&params, heard, BLOCK as f64, Some(&looped), block);
        }
        // Beat 3 at 48 000, then the loop start (beat 2) and beat 3 at every pass,
        // never beat 4 after the loop end.
        let found: Vec<usize> = clicks(&out).iter().map(|c| c.0).collect();
        let expected = [18_000, 30_000, 54_000, 66_000];
        assert_eq!(found.len(), expected.len(), "{found:?}");
        for (f, e) in found.iter().zip(expected) {
            assert!(f.abs_diff(e) <= 1, "{found:?}");
        }
    }

    #[test]
    fn count_in_plays_one_bar() {
        // A bar is 200 blocks, so the count-in ends between two blocks.
        let beats = Beats::steady(0.0, 12_800.0, 4, SONG_END).unwrap();
        let params = params();
        params.on.store(false, Ordering::Relaxed);
        let mut clicker = clicker(&beats);
        let mut count_in = clicker.count_in(102_400).unwrap();
        let mut out = vec![0.0; 300 * BLOCK * 2];
        for block in out.chunks_mut(BLOCK * 2) {
            if count_in.heard < 102_400.0 {
                clicker.play_count_in(&params, &mut count_in, 1.0, block);
            } else {
                clicker.play(&params, count_in.heard, BLOCK as f64, None, block);
                count_in.heard += BLOCK as f64;
            }
        }
        // Bar 2 of the grid, then nothing: the click is off.
        let found = clicks(&out);
        let expected: Vec<(usize, bool)> = (0..4).map(|k| (k * 12_800, k == 0)).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn clicks_and_count_in_follow_a_tempo_map() {
        // Bar 1 at 48 000, then beats 24 000, 22 000 and 20 000 frames apart.
        let beats = Beats::new(vec![48_000.0, 72_000.0, 94_000.0, 114_000.0], 4).unwrap();
        let mut clicker = clicker(&beats);
        clicker.seek(0);
        let found = clicks(&play(&mut clicker, &params(), 0.0, 1.0));
        assert_eq!(found, [(48_000, true), (72_000, false), (94_000, false)]);

        // A count-in to the beat at 94 000: the bar before it, at its own pace,
        // counted on from bar 1 backwards at the first beat's pace.
        let params = params();
        params.on.store(false, Ordering::Relaxed);
        let mut count_in = clicker.count_in(94_000).unwrap();
        assert_eq!(count_in.heard, 0.0);
        let mut out = vec![0.0; 400 * BLOCK * 2];
        for block in out.chunks_mut(BLOCK * 2) {
            // Up to where the song starts, as the engine does.
            let left = (94_000.0 - count_in.heard).max(0.0) as usize;
            let block = &mut block[..(left * 2).min(BLOCK * 2)];
            clicker.play_count_in(&params, &mut count_in, 1.0, block);
        }
        let found: Vec<usize> = clicks(&out).iter().map(|c| c.0).collect();
        assert_eq!(found, vec![0, 24_000, 48_000, 72_000]);
    }

    #[test]
    fn soft_clip_bends_only_loud_peaks() {
        assert_eq!(soft_clip(0.4), 0.4);
        assert_eq!(soft_clip(-0.5), -0.5);
        // The loudest click at 0 dB loses less than 1 dB.
        assert!(soft_clip(0.9) > 0.9 * 0.891, "{}", soft_clip(0.9));
        for x in [1.0, 2.0, 3.6, -3.6, 100.0] {
            let y = soft_clip(x);
            assert!(
                y.abs() <= 1.0 && y.abs() > 0.8 && y.signum() == x.signum(),
                "{x} → {y}"
            );
        }
        // Still growing with the input, so louder means louder.
        assert!(soft_clip(2.0) > soft_clip(1.5));
    }

    #[test]
    fn sounds_are_short_and_normalized() {
        for sound in ClickSound::ALL {
            for accent in [false, true] {
                let samples = synthesize(sound, accent, RATE);
                assert!(samples.len() <= RATE as usize / 2);
                let peak = samples.iter().fold(0.0f32, |p, s| p.max(s.abs()));
                let level = level(sound, accent);
                assert!((peak - level).abs() < 1e-3, "{sound:?} {accent}: {peak}");
                assert_eq!(samples[0], 0.0);
                assert!(samples.last().unwrap().abs() < 1e-3);
            }
        }
    }
}
