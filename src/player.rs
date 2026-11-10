//! The practice player: Watershed's playback engine (vendor/watershed, MIT OR Apache-2.0) behind the app's API.
//!
//! One thread owns the audio engine (a cpal stream isn't `Send` everywhere) and runs closures sent to it, so the
//! HTTP handlers never touch the stream. The thread also publishes a snapshot of the state about every 15 ms, which
//! `GET /api/player` reads without waiting for it.
//!
//! What it adds to the browser's `<audio>` element: tempo from 25 % to 200 % without changing the pitch, transposition
//! in semitones without changing the tempo (drums stay put when their pitch lock is on, vocals can keep their formants),
//! the stems as a mixer (volume, pan, mute, solo, offset per stem), an A-B loop done by the engine itself, a metronome
//! on the track's beat grid with a count-in, a limiter on the output, and the choice of output device.
//!
//! Signal path (all in the engine): stems -> mixer -> tempo and pitch (Signalsmith Stretch) -> master -> limiter -> click.
//!
//! `POCKET_PREP_NULL_AUDIO=1` runs the same engine with no sound card (tests, machines without one), see
//! `Engine::start_null`. When no device can be opened the snapshot says `available: false` and the UI keeps using the
//! browser's own player.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::{json, Value};
use watershed_audio::{
    output_devices, Beats, ClickChannels, ClickSound, Engine, Song, MAX_SEMITONES, MAX_TEMPO, MIN_TEMPO,
};

use crate::{db, stems, vocals};

/// How the null engine is clocked when there is no device.
const NULL_RATE: u32 = 48_000;

#[derive(Clone, Debug, Default, Serialize)]
pub struct StemState {
    pub name: String,
    pub gain: f32,
    pub pan: f32,
    pub mute: bool,
    pub solo: bool,
    pub pitch_lock: bool,
    pub vocal: bool,
    pub offset_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    /// An output device is open. False: the browser's player is used instead.
    pub available: bool,
    pub error: Option<String>,
    pub device: Option<Value>,
    pub track_id: Option<i64>,
    /// The stems are loaded as the song (else the original mix is).
    pub stems_loaded: bool,
    pub playing: bool,
    pub counting_in: bool,
    pub position_ms: f64,
    pub duration_ms: f64,
    pub rate: u32,
    pub tempo: f32,
    pub semitones: f32,
    pub gain: f32,
    pub loop_ms: Option<[f64; 2]>,
    pub click: Value,
    pub vocal_formants: bool,
    pub vocal_tonality_limit: bool,
    pub stems: Vec<StemState>,
    pub xruns: u64,
    /// The device changed under us: the song must be loaded again (its sample rate may differ).
    pub needs_reload: bool,
    /// Loudest sample of the null engine lately; 0 on a real device.
    pub null_peak: f32,
    /// Bumped on every change a client didn't cause itself (a track loaded, the device moved).
    pub revision: u64,
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot {
            available: false, error: None, device: None, track_id: None, stems_loaded: false, playing: false, counting_in: false,
            position_ms: 0.0, duration_ms: 0.0, rate: 0, tempo: 1.0, semitones: 0.0, gain: 1.0, loop_ms: None,
            click: json!({"on": false, "count_in": false, "gain": 0.5, "sound": "click", "channels": "both", "beats": false}),
            vocal_formants: false, vocal_tonality_limit: false, stems: vec![], xruns: 0, needs_reload: false, null_peak: 0.0, revision: 0,
        }
    }
}

type Job = Box<dyn FnOnce(&mut Core) + Send>;

/// Everything the audio thread owns.
struct Core {
    engine: Option<Engine>,
    device_id: Option<String>,
    error: Option<String>,
    track_id: Option<i64>,
    stems_loaded: bool,
    loop_ms: Option<[f64; 2]>,
    click_on: bool,
    count_in: bool,
    click_gain: f32,
    click_sound: ClickSound,
    click_channels: ClickChannels,
    has_beats: bool,
    vocal_formants: bool,
    vocal_tonality_limit: bool,
    needs_reload: bool,
    revision: u64,
    notified: Arc<AtomicBool>,
    snap: Arc<RwLock<Snapshot>>,
}

impl Core {
    fn engine_open(&mut self) -> Result<&mut Engine> {
        if self.engine.is_none() {
            let engine = if std::env::var_os("POCKET_PREP_NULL_AUDIO").is_some() {
                Engine::start_null(NULL_RATE)
            } else {
                let flag = self.notified.clone();
                match Engine::start(self.device_id.as_deref(), Arc::new(move || flag.store(true, Ordering::Relaxed))) {
                    Ok(e) => e,
                    Err(e) => { self.error = Some(e.to_string()); return Err(anyhow!("{e}")); }
                }
            };
            self.error = None;
            self.apply_settings(&engine);
            self.engine = Some(engine);
        }
        Ok(self.engine.as_mut().unwrap())
    }

    /// Settings the engine forgets when it is restarted.
    fn apply_settings(&self, e: &Engine) {
        e.set_click(self.click_on);
        e.set_count_in(self.count_in);
        e.set_click_gain(self.click_gain);
        e.set_click_sound(self.click_sound);
        e.set_click_channels(self.click_channels);
        e.set_vocal_shift(self.vocal_formants, self.vocal_tonality_limit);
    }

    fn publish(&mut self) {
        let mut s = Snapshot { revision: self.revision, needs_reload: self.needs_reload, error: self.error.clone(), ..Snapshot::default() };
        if let Some(e) = &self.engine {
            let rate = e.sample_rate();
            let ms = |frames: usize| frames as f64 / f64::from(rate) * 1000.0;
            s.available = !e.failed();
            s.device = Some(json!({"id": e.device().id, "name": e.device().name}));
            s.rate = rate;
            s.playing = e.is_playing();
            s.counting_in = e.counting_in();
            s.position_ms = ms(e.position());
            s.tempo = e.tempo();
            s.semitones = e.semitones();
            s.gain = e.master_gain();
            s.xruns = e.xruns();
            s.null_peak = e.null_peak();
            if let Some(song) = e.song() {
                s.track_id = self.track_id;
                s.stems_loaded = self.stems_loaded;
                s.duration_ms = ms(song.frames());
                s.stems = song.tracks.iter().map(|t| StemState {
                    name: t.name.clone(), gain: t.params.gain(), pan: t.params.pan(), mute: t.params.mute(), solo: t.params.solo(),
                    pitch_lock: t.params.pitch_lock(), vocal: t.params.vocal(), offset_ms: ms(1) * t.params.offset() as f64,
                }).collect();
            }
        }
        s.loop_ms = self.loop_ms;
        s.vocal_formants = self.vocal_formants;
        s.vocal_tonality_limit = self.vocal_tonality_limit;
        s.click = json!({
            "on": self.click_on, "count_in": self.count_in, "gain": self.click_gain, "beats": self.has_beats,
            "sound": match self.click_sound { ClickSound::Click => "click", ClickSound::Metronome => "metronome", ClickSound::Wood => "wood" },
            "channels": match self.click_channels { ClickChannels::Left => "left", ClickChannels::Both => "both", ClickChannels::Right => "right" },
        });
        if self.engine.is_none() && s.error.is_none() { s.error = None; }
        *self.snap.write().unwrap() = s;
    }

    fn tick(&mut self) {
        if let Some(e) = &mut self.engine {
            e.collect_garbage();
            if e.needs_restart() || e.failed() {
                // The device went away or changed its rate: drop the engine; the next load opens it again.
                let was = self.track_id.take();
                self.engine = None;
                self.needs_reload = was.is_some();
                self.revision += 1;
                if was.is_none() { self.needs_reload = false; }
                self.error = Some("The audio device changed. Press play to open it again.".into());
                self.notified.store(false, Ordering::Relaxed);
            }
        }
        self.publish();
    }
}

pub struct Player {
    tx: Sender<Job>,
    snap: Arc<RwLock<Snapshot>>,
}

static PLAYER: OnceLock<Player> = OnceLock::new();

pub fn player() -> &'static Player {
    PLAYER.get_or_init(|| {
        let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();
        let snap = Arc::new(RwLock::new(Snapshot::default()));
        let mut core = Core {
            engine: None, device_id: None, error: None, track_id: None, stems_loaded: false, loop_ms: None, click_on: false, count_in: false,
            click_gain: 0.5, click_sound: ClickSound::Click, click_channels: ClickChannels::Both, has_beats: false,
            vocal_formants: false, vocal_tonality_limit: false, needs_reload: false, revision: 0,
            notified: Arc::new(AtomicBool::new(false)), snap: snap.clone(),
        };
        std::thread::Builder::new().name("player".into()).spawn(move || loop {
            match rx.recv_timeout(Duration::from_millis(15)) {
                Ok(job) => { job(&mut core); core.publish(); }
                Err(RecvTimeoutError::Timeout) => core.tick(),
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }).expect("the player thread");
        Player { tx, snap }
    })
}

impl Player {
    pub fn snapshot(&self) -> Snapshot {
        self.snap.read().unwrap().clone()
    }

    /// Run `f` on the audio thread and wait for its result.
    fn run<R: Send + 'static>(&self, f: impl FnOnce(&mut Core) -> R + Send + 'static) -> Result<R> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Box::new(move |c| { let _ = tx.send(f(c)); })).map_err(|_| anyhow!("the player thread has stopped"))?;
        rx.recv().map_err(|_| anyhow!("the player thread stopped while working"))
    }

    /// The device's sample rate (opening the engine when it isn't), for decoding songs to it.
    pub fn rate(&self) -> Result<u32> {
        self.run(|c| c.engine_open().map(|e| e.sample_rate()))?
    }

    pub fn load(&self, track_id: i64, song: Song, beats: Option<Beats>, position_ms: f64, stems_loaded: bool) -> Result<()> {
        self.run(move |c| -> Result<()> {
            let loop_ms = c.loop_ms;
            let e = c.engine_open()?;
            let rate = f64::from(e.sample_rate());
            e.load(Arc::new(song));
            e.set_beats(beats.clone());
            e.seek((position_ms / 1000.0 * rate) as usize);
            c.has_beats = beats.is_some();
            c.track_id = Some(track_id);
            c.stems_loaded = stems_loaded;
            c.needs_reload = false;
            c.revision += 1;
            // A loop belongs to the track: keep it for the same track (stems loaded over the mix), else clear it.
            if c.engine.as_ref().is_some_and(|_| loop_ms.is_some()) { c.set_loop(loop_ms); }
            Ok(())
        })?
    }

    pub fn unload(&self) -> Result<()> {
        self.run(|c| { if let Some(e) = &mut c.engine { e.pause(); } c.track_id = None; c.loop_ms = None; c.has_beats = false; c.revision += 1; })
    }

    pub fn command(&self, cmd: Command) -> Result<()> {
        self.run(move |c| c.apply(cmd))?
    }

    pub fn devices(&self) -> Vec<Value> {
        output_devices().into_iter().map(|d| json!({"id": d.id, "name": d.name})).collect()
    }

    /// Switch the output device. The song must be loaded again at the new device's rate (the caller does it).
    pub fn set_device(&self, id: Option<String>) -> Result<Option<i64>> {
        self.run(move |c| {
            let was = c.track_id.take();
            c.engine = None;                       // closes the stream
            c.device_id = id;
            c.loop_ms = if was.is_some() { c.loop_ms } else { None };
            c.revision += 1;
            c.engine_open().map(|_| was)
        })?
    }
}

#[derive(Debug)]
pub enum Command {
    Play,
    Pause,
    Toggle,
    Stop,
    SeekMs(f64),
    Tempo(f32),
    Semitones(f32),
    Gain(f32),
    Loop(Option<[f64; 2]>),
    Click { on: Option<bool>, count_in: Option<bool>, gain: Option<f32>, sound: Option<String>, channels: Option<String> },
    VocalShift { formants: bool, tonality_limit: bool },
    Stem { name: String, gain: Option<f32>, pan: Option<f32>, mute: Option<bool>, solo: Option<bool>, pitch_lock: Option<bool>, offset_ms: Option<f64> },
}

impl Core {
    fn set_loop(&mut self, range: Option<[f64; 2]>) {
        let Some(e) = &self.engine else { return };
        let rate = f64::from(e.sample_rate());
        e.set_loop(range.map(|[a, b]| (a / 1000.0 * rate).round() as usize..(b / 1000.0 * rate).round() as usize));
        // Loops the engine refuses (too short) are not kept.
        self.loop_ms = range.filter(|[a, b]| b - a >= watershed_audio::MIN_LOOP_SECS * 1000.0);
    }

    fn apply(&mut self, cmd: Command) -> Result<()> {
        if self.engine.is_none() { return Err(anyhow!("no audio device is open")); }
        match cmd {
            Command::Play => self.engine.as_mut().unwrap().play(),
            Command::Pause => self.engine.as_mut().unwrap().pause(),
            Command::Toggle => self.engine.as_mut().unwrap().toggle(),
            Command::Stop => self.engine.as_mut().unwrap().stop(),
            Command::SeekMs(ms) => {
                let e = self.engine.as_mut().unwrap();
                e.seek((ms.max(0.0) / 1000.0 * f64::from(e.sample_rate())) as usize);
            }
            Command::Tempo(t) => self.engine.as_ref().unwrap().set_tempo(t.clamp(MIN_TEMPO, MAX_TEMPO)),
            Command::Semitones(s) => self.engine.as_ref().unwrap().set_semitones(s.clamp(-MAX_SEMITONES, MAX_SEMITONES)),
            Command::Gain(g) => self.engine.as_ref().unwrap().set_master_gain(g),
            Command::Loop(r) => self.set_loop(r),
            Command::Click { on, count_in, gain, sound, channels } => {
                if let Some(v) = on { self.click_on = v; }
                if let Some(v) = count_in { self.count_in = v; }
                if let Some(v) = gain { self.click_gain = v.clamp(0.0, 4.0); }
                if let Some(v) = sound {
                    self.click_sound = match v.as_str() { "metronome" => ClickSound::Metronome, "wood" => ClickSound::Wood, "click" => ClickSound::Click, _ => return Err(anyhow!("sound is click, metronome or wood")) };
                }
                if let Some(v) = channels {
                    self.click_channels = match v.as_str() { "left" => ClickChannels::Left, "right" => ClickChannels::Right, "both" => ClickChannels::Both, _ => return Err(anyhow!("channels is left, right or both")) };
                }
                self.apply_settings(self.engine.as_ref().unwrap());
            }
            Command::VocalShift { formants, tonality_limit } => {
                self.vocal_formants = formants;
                self.vocal_tonality_limit = tonality_limit;
                self.apply_settings(self.engine.as_ref().unwrap());
            }
            Command::Stem { name, gain, pan, mute, solo, pitch_lock, offset_ms } => {
                let e = self.engine.as_ref().unwrap();
                let rate = f64::from(e.sample_rate());
                let song = e.song().ok_or_else(|| anyhow!("no song is loaded"))?;
                let t = song.tracks.iter().find(|t| t.name == name).ok_or_else(|| anyhow!("no stem called {name}"))?;
                if let Some(v) = gain { t.params.set_gain(v.clamp(0.0, 4.0)); }
                if let Some(v) = pan { t.params.set_pan(v.clamp(-1.0, 1.0)); }
                if let Some(v) = mute { t.params.set_mute(v); }
                if let Some(v) = solo { t.params.set_solo(v); }
                if let Some(v) = pitch_lock { t.params.set_pitch_lock(v); }
                if let Some(v) = offset_ms { t.params.set_offset((v / 1000.0 * rate).round() as i64); }
            }
        }
        Ok(())
    }
}

// ── Loading a track ──────────────────────────────────────────────────────────

/// Interleave two channels.
fn interleave(l: &[f32], r: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(l.len() * 2);
    for (a, b) in l.iter().zip(r) { out.push(*a); out.push(*b); }
    out
}

/// The beats of a track for the click: its grid as the deck gets it (nudge, time signature), in frames at `rate`.
pub fn beats_for(track_id: i64, duration_s: f64, rate: u32) -> Option<Beats> {
    let (grid, _) = vocals::grid_and_sections(track_id)?;
    let rate = f64::from(rate);
    Beats::steady(grid.downbeat * rate, grid.beat_len() * rate, grid.beats_per_bar, duration_s * rate)
}

/// Decode a track (its stems when `with_stems` and they exist, else the original mix) into a song at `rate`.
pub fn build_song(track_id: i64, rate: u32, with_stems: bool) -> Result<(Song, bool)> {
    let t = db::track(track_id)?.ok_or_else(|| anyhow!("unknown track"))?;
    if with_stems && stems::has_stems(track_id) {
        let audio = stems::read_stems(track_id)?;
        let tracks: Vec<(String, Vec<f32>)> = stems::STEM_NAMES.iter().enumerate().map(|(k, name)| {
            let (l, r) = &audio[k];
            let (l, r) = if stems::RATE == rate { (l.clone(), r.clone()) } else {
                (clubcraft_audio::resample::resample_channel(l, stems::RATE, rate), clubcraft_audio::resample::resample_channel(r, stems::RATE, rate))
            };
            (name.to_string(), interleave(&l, &r))
        }).collect();
        let song = Song::new(rate, tracks);
        // Drums keep their pitch when the song is transposed; the vocal stem goes on its own bus.
        for tr in &song.tracks {
            tr.params.set_pitch_lock(tr.name == "drums");
            tr.params.set_vocal(tr.name == "vocals");
        }
        return Ok((song, true));
    }
    let path = crate::export::usb::track_path(&t);
    let buf = clubcraft_audio::decode_file(&path, rate)?;
    let song = Song::new(rate, vec![("mix".to_string(), interleave(&buf.left, &buf.right))]);
    Ok((song, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, seconds: f64, hz: f64, amp: f32) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize).flat_map(|i| {
            let v = amp * (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(rate)).sin() as f32;
            [v, v]
        }).collect()
    }

    fn wait_for(what: &str, cond: impl Fn(&Snapshot) -> bool) -> Snapshot {
        for _ in 0..200 {
            let s = player().snapshot();
            if cond(&s) { return s; }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out waiting for {what}: {:?}", player().snapshot());
    }

    /// The engine runs with no sound card: a loaded song plays, the position advances at the tempo, the stem mixer
    /// changes what comes out, the loop holds the position inside it and a count-in precedes the song.
    #[test]
    fn null_engine_plays_mixes_and_loops() {
        unsafe { std::env::set_var("POCKET_PREP_NULL_AUDIO", "1"); }
        let p = player();
        let rate = p.rate().unwrap();
        assert_eq!(rate, NULL_RATE);
        let song = Song::new(rate, vec![("drums".into(), tone(rate, 20.0, 100.0, 0.5)), ("vocals".into(), tone(rate, 20.0, 800.0, 0.5))]);
        let beats = Beats::steady(0.0, f64::from(rate) * 0.5, 4, f64::from(rate) * 20.0);
        p.load(7, song, beats, 0.0, true).unwrap();
        let s = wait_for("the song", |s| s.track_id == Some(7));
        assert_eq!((s.stems.len(), s.duration_ms.round() as i64, s.playing), (2, 20_000, false));

        // Plays, and the position advances.
        p.command(Command::Play).unwrap();
        let s = wait_for("some playback", |s| s.playing && s.position_ms > 500.0);
        assert!(s.null_peak > 0.1, "peak {}", s.null_peak);

        // Muting every stem silences the output (the limiter and fades settle within a few blocks).
        for name in ["drums", "vocals"] {
            p.command(Command::Stem { name: name.into(), gain: None, pan: None, mute: Some(true), solo: None, pitch_lock: None, offset_ms: None }).unwrap();
        }
        let s = wait_for("silence", |s| s.null_peak < 0.01);
        assert!(s.playing);
        p.command(Command::Stem { name: "vocals".into(), gain: None, pan: None, mute: Some(false), solo: None, pitch_lock: None, offset_ms: None }).unwrap();
        wait_for("sound again", |s| s.null_peak > 0.1);

        // A loop keeps the position between its ends.
        p.command(Command::SeekMs(2000.0)).unwrap();
        p.command(Command::Loop(Some([1000.0, 3000.0]))).unwrap();
        p.command(Command::Tempo(2.0)).unwrap();
        std::thread::sleep(Duration::from_millis(2500));
        let s = p.snapshot();
        assert!(s.position_ms >= 800.0 && s.position_ms <= 3300.0, "left the loop: {}", s.position_ms);
        assert_eq!(s.loop_ms, Some([1000.0, 3000.0]));
        assert_eq!(s.tempo, 2.0);

        // Pause, clear the loop, count-in on: playing starts with a bar of clicks and no song yet.
        p.command(Command::Pause).unwrap();
        std::thread::sleep(Duration::from_millis(100));            // the pause fade (10 ms) ends: a count-in starts from silence
        p.command(Command::Loop(None)).unwrap();
        p.command(Command::Tempo(1.0)).unwrap();
        p.command(Command::Click { on: None, count_in: Some(true), gain: None, sound: None, channels: None }).unwrap();
        p.command(Command::SeekMs(4000.0)).unwrap();
        p.command(Command::Play).unwrap();
        let s = wait_for("the count-in", |s| s.counting_in);
        assert!(s.click["count_in"].as_bool().unwrap() && s.click["beats"].as_bool().unwrap());
        wait_for("the song after the count-in", |s| s.playing && !s.counting_in && s.position_ms > 4000.0);

        // Bad input is refused.
        assert!(p.command(Command::Stem { name: "nope".into(), gain: Some(1.0), pan: None, mute: None, solo: None, pitch_lock: None, offset_ms: None }).is_err());
        assert!(p.command(Command::Click { on: None, count_in: None, gain: None, sound: Some("gong".into()), channels: None }).is_err());
        p.command(Command::Pause).unwrap();
    }
}
