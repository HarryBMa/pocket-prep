use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use crate::{AtomicF32, Samples};

/// Live mixer settings for one track. The UI writes them, the audio thread reads them.
#[derive(Debug)]
pub struct TrackParams {
    /// Linear gain, 1.0 = unity.
    gain: AtomicF32,
    /// Balance from -1.0 (left) to 1.0 (right).
    pan: AtomicF32,
    mute: AtomicBool,
    solo: AtomicBool,
    /// Follows the tempo but not the transposition, e.g. drums.
    pitch_lock: AtomicBool,
    /// A sung voice: transposed on its own bus, which can keep its formants.
    vocal: AtomicBool,
    /// Song frame where the track starts: positive starts it later, negative cuts
    /// its beginning.
    offset: AtomicI64,
}

impl Default for TrackParams {
    fn default() -> Self {
        Self {
            gain: AtomicF32::new(1.0),
            pan: AtomicF32::new(0.0),
            mute: AtomicBool::new(false),
            solo: AtomicBool::new(false),
            pitch_lock: AtomicBool::new(false),
            vocal: AtomicBool::new(false),
            offset: AtomicI64::new(0),
        }
    }
}

impl TrackParams {
    pub fn gain(&self) -> f32 {
        self.gain.load()
    }
    pub fn set_gain(&self, gain: f32) {
        self.gain.store(gain.max(0.0));
    }
    pub fn pan(&self) -> f32 {
        self.pan.load()
    }
    pub fn set_pan(&self, pan: f32) {
        self.pan.store(pan.clamp(-1.0, 1.0));
    }
    pub fn mute(&self) -> bool {
        self.mute.load(Ordering::Relaxed)
    }
    pub fn set_mute(&self, mute: bool) {
        self.mute.store(mute, Ordering::Relaxed);
    }
    pub fn solo(&self) -> bool {
        self.solo.load(Ordering::Relaxed)
    }
    pub fn set_solo(&self, solo: bool) {
        self.solo.store(solo, Ordering::Relaxed);
    }
    pub fn pitch_lock(&self) -> bool {
        self.pitch_lock.load(Ordering::Relaxed)
    }
    pub fn set_pitch_lock(&self, locked: bool) {
        self.pitch_lock.store(locked, Ordering::Relaxed);
    }
    pub fn vocal(&self) -> bool {
        self.vocal.load(Ordering::Relaxed)
    }
    pub fn set_vocal(&self, vocal: bool) {
        self.vocal.store(vocal, Ordering::Relaxed);
    }
    pub fn offset(&self) -> i64 {
        self.offset.load(Ordering::Relaxed)
    }
    pub fn set_offset(&self, frames: i64) {
        self.offset.store(frames, Ordering::Relaxed);
    }

    /// Whether the track is heard, given whether any track in the song is soloed.
    /// Solo overrides mute: a muted track that is soloed plays.
    pub fn audible(&self, any_solo: bool) -> bool {
        if any_solo { self.solo() } else { !self.mute() }
    }

    /// Left/right gains, given whether any track in the song is soloed.
    pub(crate) fn channel_gains(&self, any_solo: bool) -> [f32; 2] {
        if !self.audible(any_solo) {
            return [0.0, 0.0];
        }
        // Balance law for stereo sources: the center leaves both sides at unity.
        let gain = self.gain();
        let pan = self.pan();
        [gain * (1.0 - pan.max(0.0)), gain * (1.0 + pan.min(0.0))]
    }
}

/// One stem: interleaved stereo samples at the song's sample rate.
#[derive(Debug)]
pub struct Track {
    pub name: String,
    pub samples: Samples,
    pub params: TrackParams,
}

/// A set of stems that play in sync. All tracks hold the same number of frames;
/// each can be moved in time by its offset.
#[derive(Debug)]
pub struct Song {
    pub sample_rate: u32,
    pub tracks: Vec<Track>,
    /// Frames in every track.
    track_frames: usize,
}

impl Song {
    /// Builds a song from `(name, interleaved stereo samples)` pairs. Shorter tracks
    /// are padded with silence to the longest one.
    pub fn new<S: Into<Samples>>(sample_rate: u32, tracks: Vec<(String, S)>) -> Self {
        let tracks: Vec<(String, Samples)> = tracks
            .into_iter()
            .map(|(name, samples)| (name, samples.into()))
            .collect();
        let frames = tracks.iter().map(|(_, s)| s.len() / 2).max().unwrap_or(0);
        let tracks = tracks
            .into_iter()
            .map(|(name, mut samples)| {
                samples.resize(frames * 2);
                Track {
                    name,
                    samples,
                    params: TrackParams::default(),
                }
            })
            .collect();
        Self {
            sample_rate,
            tracks,
            track_frames: frames,
        }
    }

    /// Frames in every track, before offsets.
    pub fn track_frames(&self) -> usize {
        self.track_frames
    }

    /// Length of the song: up to the end of the track that ends last, offsets
    /// included.
    pub fn frames(&self) -> usize {
        self.tracks
            .iter()
            .map(|t| (self.track_frames as i64 + t.params.offset()).max(0) as usize)
            .max()
            .unwrap_or(0)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }
}
