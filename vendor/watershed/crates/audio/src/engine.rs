use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample, StreamConfig};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::click::{
    ClickChannels, ClickParams, ClickSound, Clicker, CountIn, channels_index, sound_index,
};
use crate::limiter::Limiter;
use crate::mixer::{MAX_BLOCK, Voice};
use crate::stretch::{MAX_SEMITONES, MAX_TEMPO, MIN_TEMPO, Stretcher, VocalShift};
use crate::{AtomicF32, Beats, Song};

/// How long pause and resume fade, in seconds. Avoids clicks.
const FADE_SECS: f32 = 0.01;

/// Shortest A-B loop, in seconds. Must stay above the tempo and pitch latency
/// (about 0.2 s at most) so the displayed position can follow the jumps.
pub const MIN_LOOP_SECS: f64 = 0.5;

/// Click level before the user changes it: -6 dB.
pub const DEFAULT_CLICK_GAIN: f32 = 0.5;
/// Loudest click: +12 dB.
pub const MAX_CLICK_GAIN: f32 = 4.0;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("no audio output device found")]
    NoDevice,
    #[error("audio device error: {0}")]
    Device(#[from] cpal::Error),
    #[error("unsupported sample format {0}")]
    UnsupportedFormat(SampleFormat),
}

/// Called when the engine needs attention: it must be restarted or it failed. Runs on
/// an audio system thread, so it should only wake the UI (e.g. request a repaint).
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// An output device as shown to the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputDevice {
    /// Stable identifier, suitable for saving in the config.
    pub id: String,
    pub name: String,
}

/// Lists the output devices of the default host.
pub fn output_devices() -> Vec<OutputDevice> {
    let host = cpal::default_host();
    let Ok(devices) = host.output_devices() else {
        return Vec::new();
    };
    devices.filter_map(|d| describe(&d)).collect()
}

fn describe(device: &cpal::Device) -> Option<OutputDevice> {
    let id = device.id().ok()?.to_string();
    let name = device
        .description()
        .map(|d| d.name().to_owned())
        .unwrap_or_else(|_| id.clone());
    Some(OutputDevice { id, name })
}

enum Command {
    Load(Box<Voice>),
    Seek(usize),
    Beats(Option<Box<Beats>>),
}

/// What the audio thread hands back to be freed on the UI thread. Only held until
/// it is dropped, never read.
#[allow(dead_code)]
enum Garbage {
    Voice(Box<Voice>),
    Beats(Box<Beats>),
}

/// State shared between the UI and the audio thread.
#[derive(Default)]
struct Shared {
    playing: AtomicBool,
    /// Song frame being heard, published by the audio thread. Latency compensated.
    position: AtomicU64,
    master_gain: AtomicF32,
    /// Playback speed factor, 1.0 = original tempo.
    tempo: AtomicF32,
    /// Transposition in semitones.
    semitones: AtomicF32,
    /// How vocal tracks are transposed: see [`VocalShift`].
    vocal_formants: AtomicBool,
    vocal_tonality_limit: AtomicBool,
    /// A-B loop as `start << 32 | end` in frames, 0 when off. One atomic so the audio
    /// thread never sees the start of one loop with the end of another.
    looped: AtomicU64,
    /// Set when the device is gone or the stream hit an unrecoverable error.
    failed: AtomicBool,
    /// Set when the stream must be rebuilt, e.g. the device sample rate changed because
    /// Bluetooth headphones switched to their headset profile when a mic was opened.
    invalidated: AtomicBool,
    /// Buffer underruns reported by the device. Counted here because the report can
    /// arrive on the audio thread, where logging is not allowed.
    xruns: AtomicU64,
    click: ClickParams,
    /// Set while a count-in plays, before the song starts.
    counting_in: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        let shared = Self {
            master_gain: AtomicF32::new(1.0),
            tempo: AtomicF32::new(1.0),
            ..Default::default()
        };
        shared.click.gain.store(DEFAULT_CLICK_GAIN);
        shared
            .click
            .channels
            .store(channels_index(ClickChannels::default()), Ordering::Relaxed);
        shared
    }
}

/// Audio output. Owns the cpal stream; dropping the engine stops playback.
pub struct Engine {
    _stream: Option<cpal::Stream>,
    /// Pocket Prep: a thread that runs the callback without a device (tests, machines with no sound card).
    _null: Option<NullDriver>,
    commands: Producer<Command>,
    garbage: Consumer<Garbage>,
    shared: Arc<Shared>,
    device: OutputDevice,
    sample_rate: u32,
    song: Option<Arc<Song>>,
    /// The beats last sent to the audio thread.
    beats: Option<Beats>,
}

impl Engine {
    /// Opens the device with the given id, or the system default device when `None` or
    /// not found. A stream on the default device follows it: when the system default
    /// changes, [`Engine::needs_restart`] becomes true. `notify` is called when [`Engine::needs_restart`] or [`Engine::failed`] becomes true.
    pub fn start(device_id: Option<&str>, notify: Notify) -> Result<Self, EngineError> {
        let host = cpal::default_host();
        let device = device_id
            .and_then(|id| id.parse().ok())
            .and_then(|id| host.device_by_id(&id))
            .or_else(|| host.default_output_device())
            .ok_or(EngineError::NoDevice)?;
        let info = describe(&device).ok_or(EngineError::NoDevice)?;

        let supported = device.default_output_config()?;
        let format = supported.sample_format();
        let config: StreamConfig = supported.config();
        log::info!(
            "opening {} at {} Hz, {} channels, {format}",
            info.name,
            config.sample_rate,
            config.channels
        );

        let shared = Arc::new(Shared::new());
        let (commands, command_rx) = RingBuffer::new(64);
        let (garbage_tx, garbage) = RingBuffer::new(64);
        let state = CallbackState {
            commands: command_rx,
            garbage: garbage_tx,
            shared: shared.clone(),
            voice: None,
            // Allocates: done here, on the UI thread, never in the callback.
            stretcher: Stretcher::new(config.sample_rate),
            limiter: Limiter::new(config.sample_rate),
            clicker: Clicker::new(config.sample_rate),
            count_in: None,
            was_playing: false,
            fade: 0.0,
            fade_step: 1.0 / (FADE_SECS * config.sample_rate as f32),
            mix: vec![0.0; MAX_BLOCK * 2],
            clicks: vec![0.0; MAX_BLOCK * 2],
            channels: usize::from(config.channels),
        };

        let stream = match format {
            SampleFormat::F32 => build::<f32>(&device, &config, state, &shared, notify)?,
            SampleFormat::I16 => build::<i16>(&device, &config, state, &shared, notify)?,
            SampleFormat::I32 => build::<i32>(&device, &config, state, &shared, notify)?,
            SampleFormat::U16 => build::<u16>(&device, &config, state, &shared, notify)?,
            other => return Err(EngineError::UnsupportedFormat(other)),
        };
        stream.play()?;

        Ok(Self {
            _stream: Some(stream),
            _null: None,
            commands,
            garbage,
            shared,
            device: info,
            sample_rate: config.sample_rate,
            song: None,
            beats: None,
        })
    }

    /// Pocket Prep addition: an engine with no sound card. A thread runs the same callback at real time,
    /// 512 frames at a time, into a buffer that nobody hears; [`Engine::null_peak`] reports its loudest sample
    /// so tests can check what would have been played. Nothing in the audio path differs from a real device.
    pub fn start_null(sample_rate: u32) -> Self {
        let shared = Arc::new(Shared::new());
        let (commands, command_rx) = RingBuffer::new(64);
        let (garbage_tx, garbage) = RingBuffer::new(64);
        let mut state = CallbackState {
            commands: command_rx,
            garbage: garbage_tx,
            shared: shared.clone(),
            voice: None,
            stretcher: Stretcher::new(sample_rate),
            limiter: Limiter::new(sample_rate),
            clicker: Clicker::new(sample_rate),
            count_in: None,
            was_playing: false,
            fade: 0.0,
            fade_step: 1.0 / (FADE_SECS * sample_rate as f32),
            mix: vec![0.0; MAX_BLOCK * 2],
            clicks: vec![0.0; MAX_BLOCK * 2],
            channels: 2,
        };
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicF32::new(0.0));
        let (stop_t, peak_t) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            const FRAMES: usize = 512;
            let mut buf = vec![0.0f32; FRAMES * 2];
            let started = std::time::Instant::now();
            let mut done = 0u64;
            while !stop_t.load(Ordering::Relaxed) {
                state.process(&mut buf);
                let loudest = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                peak_t.store(loudest.max(peak_t.load() * 0.9));
                done += FRAMES as u64;
                let due = std::time::Duration::from_secs_f64(done as f64 / f64::from(sample_rate));
                if let Some(wait) = due.checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
        });
        Self {
            _stream: None,
            _null: Some(NullDriver { stop, peak, handle: Some(handle) }),
            commands,
            garbage,
            shared,
            device: OutputDevice { id: "null".into(), name: "No sound card (silent)".into() },
            sample_rate,
            song: None,
            beats: None,
        }
    }

    /// The loudest sample the null engine rendered lately (decaying), 0 for a real device.
    pub fn null_peak(&self) -> f32 {
        self._null.as_ref().map_or(0.0, |n| n.peak.load())
    }

    pub fn device(&self) -> &OutputDevice {
        &self.device
    }

    /// The device sample rate. Songs must be resampled to it before loading.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// True if the stream failed, e.g. because the device was disconnected.
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Relaxed)
    }

    /// True if the stream must be rebuilt: start a new engine on the same choice of
    /// device. The device may now have a different sample rate, or be another one when
    /// following the system default.
    pub fn needs_restart(&self) -> bool {
        self.shared.invalidated.load(Ordering::Relaxed)
    }

    /// Number of buffer underruns (audible dropouts) reported so far.
    pub fn xruns(&self) -> u64 {
        self.shared.xruns.load(Ordering::Relaxed)
    }

    pub fn song(&self) -> Option<&Arc<Song>> {
        self.song.as_ref()
    }

    /// Replaces the current song and rewinds to the start, paused.
    pub fn load(&mut self, song: Arc<Song>) {
        debug_assert_eq!(song.sample_rate, self.sample_rate);
        self.pause();
        self.send(Command::Load(Box::new(Voice::new(song.clone()))));
        self.shared.position.store(0, Ordering::Relaxed);
        self.song = Some(song);
    }

    pub fn play(&mut self) {
        if let Some(song) = &self.song {
            if self.position() >= song.frames() {
                self.seek(0);
            }
            self.shared.playing.store(true, Ordering::Relaxed);
        }
    }

    pub fn pause(&mut self) {
        self.shared.playing.store(false, Ordering::Relaxed);
    }

    pub fn toggle(&mut self) {
        if self.is_playing() {
            self.pause();
        } else {
            self.play();
        }
    }

    pub fn stop(&mut self) {
        self.pause();
        self.seek(0);
    }

    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    /// True while the count-in plays: playing, but the song has not started yet.
    pub fn counting_in(&self) -> bool {
        self.is_playing() && self.shared.counting_in.load(Ordering::Relaxed)
    }

    pub fn seek(&mut self, frame: usize) {
        let frame = frame.min(self.song.as_ref().map_or(0, |s| s.frames()));
        self.send(Command::Seek(frame));
        // Show the new position right away instead of waiting for the next callback.
        self.shared.position.store(frame as u64, Ordering::Relaxed);
    }

    /// Song frame being heard now, in frames. Accounts for the tempo and pitch
    /// processing latency.
    pub fn position(&self) -> usize {
        self.shared.position.load(Ordering::Relaxed) as usize
    }

    pub fn master_gain(&self) -> f32 {
        self.shared.master_gain.load()
    }

    pub fn set_master_gain(&self, gain: f32) {
        self.shared.master_gain.store(gain.max(0.0));
    }

    /// Playback speed factor: 0.5 is half speed, 1.0 the original tempo.
    pub fn tempo(&self) -> f32 {
        self.shared.tempo.load()
    }

    pub fn set_tempo(&self, tempo: f32) {
        self.shared.tempo.store(tempo.clamp(MIN_TEMPO, MAX_TEMPO));
    }

    /// Transposition in semitones, without changing the tempo.
    pub fn semitones(&self) -> f32 {
        self.shared.semitones.load()
    }

    pub fn set_semitones(&self, semitones: f32) {
        self.shared
            .semitones
            .store(semitones.clamp(-MAX_SEMITONES, MAX_SEMITONES));
    }

    /// How tracks marked as vocals are transposed: keeping their formants where
    /// they were, and with a tonality limit that leaves the highs closer to the
    /// original. With both off they are transposed like the other tracks.
    pub fn set_vocal_shift(&self, formants: bool, tonality_limit: bool) {
        self.shared
            .vocal_formants
            .store(formants, Ordering::Relaxed);
        self.shared
            .vocal_tonality_limit
            .store(tonality_limit, Ordering::Relaxed);
    }

    /// Loops playback between two frames, or plays straight through with `None`.
    /// Loops shorter than [`MIN_LOOP_SECS`] are ignored.
    pub fn set_loop(&self, range: Option<Range<usize>>) {
        let min = (MIN_LOOP_SECS * f64::from(self.sample_rate)) as usize;
        let packed = range
            .filter(|r| r.end >= r.start + min)
            .and_then(|r| {
                let start = u64::from(u32::try_from(r.start).ok()?);
                let end = u64::from(u32::try_from(r.end).ok()?);
                Some(start << 32 | end)
            })
            .unwrap_or(0);
        self.shared.looped.store(packed, Ordering::Relaxed);
    }

    /// Turns the metronome on or off. It sounds only where the song has beats.
    pub fn set_click(&self, on: bool) {
        self.shared.click.on.store(on, Ordering::Relaxed);
    }

    /// Plays one bar of clicks each time playback starts.
    pub fn set_count_in(&self, on: bool) {
        self.shared.click.count_in.store(on, Ordering::Relaxed);
    }

    /// The song's beats, in frames at the engine's rate; `None` silences the click
    /// and the count-in. Cheap when they did not change.
    pub fn set_beats(&mut self, beats: Option<Beats>) {
        if beats != self.beats {
            self.beats = beats.clone();
            self.send(Command::Beats(beats.map(Box::new)));
        }
    }

    pub fn set_click_sound(&self, sound: ClickSound) {
        self.shared
            .click
            .sound
            .store(sound_index(sound), Ordering::Relaxed);
    }

    pub fn set_click_channels(&self, channels: ClickChannels) {
        self.shared
            .click
            .channels
            .store(channels_index(channels), Ordering::Relaxed);
    }

    /// Linear gain of the click, independent of the master volume. Up to
    /// [`MAX_CLICK_GAIN`]; above 1.0 its peaks are softly limited.
    pub fn set_click_gain(&self, gain: f32) {
        self.shared
            .click
            .gain
            .store(gain.clamp(0.0, MAX_CLICK_GAIN));
    }

    /// Drops songs the audio thread is done with. Call regularly from the UI thread.
    pub fn collect_garbage(&mut self) {
        while self.garbage.pop().is_ok() {}
    }

    fn send(&mut self, command: Command) {
        self.collect_garbage();
        if self.commands.push(command).is_err() {
            log::warn!("audio command queue full, command dropped");
        }
    }
}

/// The thread behind [`Engine::start_null`].
struct NullDriver {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicF32>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for NullDriver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut state: CallbackState,
    shared: &Arc<Shared>,
    notify: Notify,
) -> Result<cpal::Stream, EngineError>
where
    T: SizedSample + FromSample<f32>,
{
    let shared = shared.clone();
    let stream = device.build_output_stream(
        *config,
        move |data: &mut [T], _| state.process(data),
        move |err| match err.kind() {
            // May be called on the audio thread: count it, the UI thread logs it.
            ErrorKind::Xrun => {
                shared.xruns.fetch_add(1, Ordering::Relaxed);
            }
            // The stream keeps working.
            ErrorKind::RealtimeDenied => {
                log::warn!("audio stream: {err}");
            }
            // A stream on the system default device moved to the new default (macOS,
            // PipeWire). Rebuilt anyway, to open it at its own rate and show its name.
            // WASAPI reports a default change as `StreamInvalidated`.
            ErrorKind::DeviceChanged | ErrorKind::StreamInvalidated => {
                log::warn!("audio stream must be rebuilt: {err}");
                shared.invalidated.store(true, Ordering::Relaxed);
                notify();
            }
            _ => {
                log::error!("audio stream error: {err}");
                shared.failed.store(true, Ordering::Relaxed);
                notify();
            }
        },
        None,
    )?;
    Ok(stream)
}

/// Everything the audio callback owns. Nothing in here allocates after construction.
struct CallbackState {
    commands: Consumer<Command>,
    garbage: Producer<Garbage>,
    shared: Arc<Shared>,
    voice: Option<Box<Voice>>,
    stretcher: Stretcher,
    limiter: Limiter,
    clicker: Clicker,
    count_in: Option<CountIn>,
    /// Whether the previous callback was playing, to tell when playback starts.
    was_playing: bool,
    /// Pause/resume fade level, 0.0 (silent) to 1.0 (playing).
    fade: f32,
    fade_step: f32,
    /// Stereo scratch buffers of `MAX_BLOCK` frames: the song and the click.
    mix: Vec<f32>,
    clicks: Vec<f32>,
    channels: usize,
}

impl CallbackState {
    fn process<T: SizedSample + FromSample<f32>>(&mut self, data: &mut [T]) {
        self.handle_commands();

        let mut playing = self.shared.playing.load(Ordering::Relaxed);
        let master = self.shared.master_gain.load();
        let tempo = self.shared.tempo.load();
        let semitones = self.shared.semitones.load();
        let vocal = VocalShift {
            formants: self.shared.vocal_formants.load(Ordering::Relaxed),
            tonality_limit: self.shared.vocal_tonality_limit.load(Ordering::Relaxed),
        };
        let looped = match self.shared.looped.load(Ordering::Relaxed) {
            0 => None,
            packed => Some((packed >> 32) as usize..(packed & 0xffff_ffff) as usize),
        };
        let channels = self.channels;

        // Resuming from silence: the stretcher still holds about 120 ms of the mix from
        // before the pause, with the mute, solo and faders of then. Start over from the
        // position heard at the pause, so the first thing heard is the current mix.
        if playing && !self.was_playing && self.fade == 0.0 && self.voice.is_some() {
            let target = self.shared.position.load(Ordering::Relaxed) as usize;
            self.seek(target);
            if self.shared.click.count_in.load(Ordering::Relaxed) {
                self.count_in = self.clicker.count_in(target);
            }
        }
        if !playing {
            self.count_in = None;
            self.clicker.stop();
        }
        self.was_playing = playing;

        for out in data.chunks_mut(MAX_BLOCK * channels) {
            let frames = out.len() / channels;
            let mix = &mut self.mix[..frames * 2];
            mix.fill(0.0);
            let clicks = &mut self.clicks[..frames * 2];
            clicks.fill(0.0);

            // The count-in runs until the song frame it counts towards is the one the
            // stretcher will play first, so its beats and the song's line up.
            let mut song_start = 0;
            if let Some(count_in) = &mut self.count_in {
                let begin = self.stretcher.heard_exact(tempo, self.limiter.latency());
                let left = ((begin - count_in.heard) / f64::from(tempo)).ceil();
                song_start = (left.max(0.0) as usize).min(frames);
                self.clicker.play_count_in(
                    &self.shared.click,
                    count_in,
                    tempo,
                    &mut clicks[..song_start * 2],
                );
                if count_in.heard >= begin {
                    self.clicker.count_in_done(begin);
                    self.count_in = None;
                }
            }

            if let Some(voice) = &mut self.voice
                && (playing || self.fade > 0.0)
                && self.count_in.is_none()
            {
                let mix = &mut mix[song_start * 2..];
                self.stretcher
                    .render(voice, tempo, semitones, vocal, looped.clone(), mix);
                for frame in mix.as_chunks_mut::<2>().0 {
                    self.fade = if playing {
                        (self.fade + self.fade_step).min(1.0)
                    } else {
                        (self.fade - self.fade_step).max(0.0)
                    };
                    let gain = master * self.fade;
                    frame[0] *= gain;
                    frame[1] *= gain;
                }

                let end = voice.song.frames();
                let heard = self.stretcher.heard(tempo, self.limiter.latency());
                if heard >= end {
                    self.fade = 0.0;
                    playing = false;
                    self.shared.playing.store(false, Ordering::Relaxed);
                }
                self.shared
                    .position
                    .store(heard.min(end) as u64, Ordering::Relaxed);

                if playing {
                    let song_frames = (frames - song_start) as f64 * f64::from(tempo);
                    self.clicker.play(
                        &self.shared.click,
                        self.stretcher.heard_exact(tempo, self.limiter.latency()),
                        song_frames,
                        looped.as_ref(),
                        &mut clicks[song_start * 2..],
                    );
                }
            }
            if !playing {
                self.clicker.ring(&self.shared.click, clicks);
            }
            // Always run, so that the lookahead delay drains when playback stops.
            self.limiter.process(mix);
            // After the limiter, whose delay is already in the heard position.
            for (m, c) in mix.iter_mut().zip(clicks.iter()) {
                *m += c;
            }

            for (dst, src) in out.chunks_exact_mut(channels).zip(mix.as_chunks::<2>().0) {
                // Hard clip as a last line of defense; stems can sum above full scale.
                let left = src[0].clamp(-1.0, 1.0);
                let right = src[1].clamp(-1.0, 1.0);
                match dst {
                    [mono] => *mono = T::from_sample((left + right) * 0.5),
                    [l, r, rest @ ..] => {
                        *l = T::from_sample(left);
                        *r = T::from_sample(right);
                        rest.fill(T::EQUILIBRIUM);
                    }
                    [] => {}
                }
            }
        }
        self.shared
            .counting_in
            .store(self.count_in.is_some(), Ordering::Relaxed);
    }

    fn handle_commands(&mut self) {
        while let Ok(command) = self.commands.pop() {
            match command {
                Command::Load(voice) => {
                    if let Some(old) = self.voice.replace(voice) {
                        // Hand the old song back so it is freed on the UI thread.
                        // If the queue is full we free it here, which is rare and only a glitch.
                        let _ = self.garbage.push(Garbage::Voice(old));
                    }
                    self.fade = 0.0;
                    self.count_in = None;
                    self.seek(0);
                }
                Command::Seek(frame) => {
                    // A seek during the count-in starts the song there at once.
                    self.count_in = None;
                    self.seek(frame);
                }
                Command::Beats(beats) => {
                    if let Some(old) = self.clicker.set_beats(beats) {
                        let _ = self.garbage.push(Garbage::Beats(old));
                    }
                }
            }
        }
    }

    fn seek(&mut self, frame: usize) {
        if let Some(voice) = &mut self.voice {
            let tempo = self.shared.tempo.load();
            self.stretcher.seek(voice, frame, tempo);
            self.clicker.seek(frame);
            self.shared.position.store(frame as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// A callback with a two-track song loaded: a loud sine and silence.
    fn callback() -> (CallbackState, Arc<Song>) {
        let sine = (0..RATE as usize * 4)
            .flat_map(|i| [0.5 * (i as f32 * 440.0 * std::f32::consts::TAU / RATE as f32).sin(); 2])
            .collect();
        let silence = vec![0.0; RATE as usize * 4 * 2];
        let song = Arc::new(Song::new(
            RATE,
            vec![("sine".into(), sine), ("silence".into(), silence)],
        ));
        let shared = Arc::new(Shared::new());
        let (_, commands) = RingBuffer::new(1);
        let (garbage, _) = RingBuffer::new(1);
        let mut state = CallbackState {
            commands,
            garbage,
            shared,
            voice: Some(Box::new(Voice::new(song.clone()))),
            stretcher: Stretcher::new(RATE),
            limiter: Limiter::new(RATE),
            clicker: Clicker::new(RATE),
            count_in: None,
            was_playing: false,
            fade: 0.0,
            fade_step: 1.0 / (FADE_SECS * RATE as f32),
            mix: vec![0.0; MAX_BLOCK * 2],
            clicks: vec![0.0; MAX_BLOCK * 2],
            channels: 2,
        };
        state.seek(0);
        (state, song)
    }

    /// A callback whose song is a sine starting at `onset`, with a grid of a beat
    /// every `beat` frames from 0, positioned at `from`.
    fn click_callback(onset: usize, beat: f64, from: usize) -> CallbackState {
        let sine = (0..RATE as usize * 4)
            .flat_map(|i| {
                let phase = i as f32 * 440.0 * std::f32::consts::TAU / RATE as f32;
                [if i < onset { 0.0 } else { 0.5 * phase.sin() }; 2]
            })
            .collect::<Vec<f32>>();
        let (mut state, _) = callback();
        let song = Arc::new(Song::new(RATE, vec![("sine".into(), sine)]));
        state.voice = Some(Box::new(Voice::new(song)));
        let beats = Beats::steady(0.0, beat, 4, f64::from(RATE) * 4.0).unwrap();
        state.clicker.set_beats(Some(Box::new(beats)));
        state.shared.click.gain.store(1.0);
        state.seek(from);
        state
    }

    /// Plays `blocks` callbacks of 256 frames and returns the left channel.
    fn record(state: &mut CallbackState, blocks: usize) -> Vec<f32> {
        state.shared.playing.store(true, Ordering::Relaxed);
        let mut out = vec![0.0f32; 256 * 2];
        let mut left = Vec::new();
        for _ in 0..blocks {
            state.process(&mut out);
            left.extend(out.iter().step_by(2));
        }
        left
    }

    /// Output frames where the level crosses `threshold` after at least 100 ms
    /// below it.
    fn onsets(left: &[f32], threshold: f32) -> Vec<usize> {
        let mut found = Vec::new();
        let mut quiet = usize::MAX / 2;
        for (i, s) in left.iter().enumerate() {
            if s.abs() < threshold {
                quiet += 1;
            } else {
                if quiet >= RATE as usize / 10 {
                    found.push(i);
                }
                quiet = 0;
            }
        }
        found
    }

    #[test]
    fn click_lands_with_the_beat_that_is_heard() {
        // Song and grid both start a beat at 1 s; play from 0.5 s at several tempos.
        for tempo in [0.5, 1.0, 1.25] {
            let song_onset = {
                let mut state = click_callback(48_000, 48_000.0, 24_000);
                state.shared.tempo.store(tempo);
                // At 60 % of the sine's level: at slow tempos the stretcher spreads an
                // attack over about 10 ms, and the click lands in its middle.
                onsets(&record(&mut state, 300), 0.3)[0]
            };
            let click_onset = {
                let mut state = click_callback(usize::MAX, 48_000.0, 24_000);
                state.shared.tempo.store(tempo);
                state.shared.click.on.store(true, Ordering::Relaxed);
                onsets(&record(&mut state, 300), 0.1)[0]
            };
            // Within 2 ms of each other.
            let error = song_onset.abs_diff(click_onset);
            assert!(
                error < 96,
                "tempo {tempo}: song {song_onset}, click {click_onset}"
            );
        }
    }

    #[test]
    fn count_in_plays_a_bar_then_the_song() {
        // A beat every 0.25 s; start on a beat at 1 s.
        let beat = 12_000;
        let clicks = {
            let mut state = click_callback(usize::MAX, beat as f64, 48_000);
            state.shared.click.on.store(true, Ordering::Relaxed);
            state.shared.click.count_in.store(true, Ordering::Relaxed);
            onsets(&record(&mut state, 300), 1e-3)
        };
        // Four beats of count-in, then the click goes on at the same pace.
        assert!(clicks.len() >= 7, "{clicks:?}");
        for (k, at) in clicks.iter().enumerate() {
            assert!(at.abs_diff(k * beat) < 96, "{clicks:?}");
        }
        // With the click silent, the song starts after one bar.
        let mut state = click_callback(48_000, beat as f64, 48_000);
        state.shared.click.count_in.store(true, Ordering::Relaxed);
        state.shared.click.gain.store(0.0);
        state.shared.playing.store(true, Ordering::Relaxed);
        let mut out = vec![0.0f32; 256 * 2];
        state.process(&mut out);
        assert!(state.shared.counting_in.load(Ordering::Relaxed));
        let left = record(&mut state, 300);
        let start = onsets(&left, 0.1)[0] + 256;
        assert!(start.abs_diff(4 * beat) < 96, "song started at {start}");
        assert!(!state.shared.counting_in.load(Ordering::Relaxed));
    }

    /// Runs `blocks` callbacks of 512 frames and returns the loudest sample.
    fn peak(state: &mut CallbackState, blocks: usize) -> f32 {
        let mut out = vec![0.0f32; 512 * 2];
        let mut peak = 0.0f32;
        for _ in 0..blocks {
            state.process(&mut out);
            peak = out.iter().fold(peak, |p, s| p.max(s.abs()));
        }
        peak
    }

    #[test]
    fn mix_changes_while_paused_are_heard_at_once() {
        let (mut state, song) = callback();
        state.shared.playing.store(true, Ordering::Relaxed);
        assert!(peak(&mut state, 50) > 0.3);

        state.shared.playing.store(false, Ordering::Relaxed);
        peak(&mut state, 10);
        // Solo the silent track while paused: resuming must not let the sine through.
        song.tracks[1].params.set_solo(true);
        state.shared.playing.store(true, Ordering::Relaxed);
        let leak = peak(&mut state, 30);
        assert!(leak < 1e-3, "the old mix leaked through: peak {leak}");
    }
}
