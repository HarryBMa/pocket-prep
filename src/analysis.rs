//! Track analysis: tempo and beat grid, key, energy, sections, auto cues and waveforms.
//!
//! Everything a player needs so it doesn't have to analyse the track itself.
//! Results are saved as JSON per track in the analysis folder (same format as
//! the old Python app, plus sections and loudness); the export step turns them
//! into rekordbox ANLZ files.
//!
//! The musical analysis is ClubCraft's `clubcraft-analysis` crate (MIT OR Apache-2.0):
//! beat grid within 0.05 BPM / 5 ms on its 135-track reference set, EDM-tuned key
//! profiles, EBU R128 loudness, energy, sections and beat-snapped hot cues. When the
//! track has stems they are passed in, which makes beats, key and sections more robust.
//! This file keeps only what is specific to the deck: the rekordbox waveform formats,
//! and mapping ClubCraft's results to our analysis JSON and cue slots.
//!
//! The audio is analysed at its own sample rate, never resampled. Every result is
//! in real time (ms, or 150 waveform columns per second).

use anyhow::{anyhow, Result};
use base64::Engine;
use clubcraft_analysis::{AnalyzeOptions, StemsRef};
use clubcraft_audio::AudioBuffer;
use clubcraft_core::SectionKind;
use rusqlite::params;
use serde_json::{json, Value};

use crate::{config, db, stems};

const COLS_PER_S: f64 = 150.0;
/// Bump when results change; older analyses are redone at startup (3 = ClubCraft + phrase cues).
pub const VERSION: i64 = 3;

pub fn load_analysis(track_id: i64) -> Option<Value> {
    let text = std::fs::read_to_string(config::analysis_dir().join(format!("{track_id}.json"))).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn waveform_bytes(a: &Value, name: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(a["waveforms"][name].as_str().unwrap_or("")).unwrap_or_default()
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// numpy-style rounding (half to even), so results match the Python version.
fn rnd(x: f64) -> f64 {
    x.round_ties_even()
}

fn percentile(x: &[f64], q: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos.fract());
    if i + 1 < s.len() { s[i] * (1.0 - f) + s[i + 1] * f } else { s[i] }
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { 0.0 } else { x.iter().sum::<f64>() / x.len() as f64 }
}

// ── Band levels for the waveforms ────────────────────────────────────────────

/// One biquad section, direct form II transposed (as scipy's sosfilt).
struct Biquad { b: [f64; 3], a: [f64; 2], z: [f64; 2] }

impl Biquad {
    /// RBJ cookbook low/high-pass; two of these with the right Qs make a
    /// 4th-order Butterworth identical to scipy's butter(4, ..., output="sos").
    fn new(high: bool, f0: f64, q: f64, sr: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * f0 / sr;
        let (cw, alpha) = (w.cos(), w.sin() / (2.0 * q));
        let a0 = 1.0 + alpha;
        let b = if high { [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0] } else { [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0] };
        Biquad { b: b.map(|v| v / a0), a: [-2.0 * cw / a0, (1.0 - alpha) / a0], z: [0.0; 2] }
    }

    fn run(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

fn butter4(x: &[f32], high: bool, f0: f64, sr: f64) -> Vec<f64> {
    let mut s = [Biquad::new(high, f0, 0.541_196_100_146_197, sr), Biquad::new(high, f0, 1.306_562_964_876_376_7, sr)];
    x.iter().map(|&v| { let a = s[0].run(v as f64); s[1].run(a) }).collect()
}

/// Sample index where waveform column i starts (columns are exactly 1/150 s).
fn col_start(i: usize, sr: u32) -> usize {
    (i as u64 * sr as u64 / COLS_PER_S as u64) as usize
}

/// Peak level and low/mid/high RMS for every 1/150 s column.
fn band_envelopes(y: &[f32], sr: u32) -> [Vec<f64>; 4] {
    let lo = butter4(y, false, 200.0, sr as f64);
    let hi = butter4(y, true, 2500.0, sr as f64);
    let n = (y.len() as u64 * 150 / sr as u64) as usize;
    let mut out: [Vec<f64>; 4] = Default::default();
    for i in 0..n {
        let (a, b) = (col_start(i, sr), col_start(i + 1, sr).min(y.len()));
        let len = (b - a).max(1) as f64;
        let (mut pk, mut sl, mut sm, mut sh) = (0.0f64, 0.0, 0.0, 0.0);
        for j in a..b {
            let v = y[j] as f64;
            let m = v - lo[j] - hi[j];
            pk = pk.max(v.abs());
            sl += lo[j] * lo[j];
            sm += m * m;
            sh += hi[j] * hi[j];
        }
        out[0].push(pk);
        out[1].push((sl / len).sqrt());
        out[2].push((sm / len).sqrt());
        out[3].push((sh / len).sqrt());
    }
    out
}

// ── Waveforms in rekordbox's formats ─────────────────────────────────────────

fn waveforms(peak: &[f64], lo: &[f64], mid: &[f64], hi: &[f64]) -> Value {
    if peak.is_empty() {
        return json!({"pwav": "", "pwv2": "", "pwv3": "", "pwv4": "", "pwv5": ""});
    }
    let r = percentile(peak, 99.5);
    let r = if r > 0.0 { r } else { 1.0 };
    let h: Vec<f64> = peak.iter().map(|p| (p / r).clamp(0.0, 1.0)).collect();
    let white: Vec<f64> = (0..h.len()).map(|i| (hi[i] / (lo[i] + mid[i] + hi[i] + 1e-9) * 2.0).clamp(0.0, 1.0)).collect();

    // PWV3: 1 byte per column, height 0-31, whiteness 0-7.
    let pwv3: Vec<u8> = (0..h.len()).map(|i| ((rnd(white[i] * 7.0) as u8) << 5) | rnd(h[i] * 31.0) as u8).collect();

    // PWV5: 2 bytes per column, RGB 3 bits each (bass=red, mids=green, highs=blue), height 5 bits.
    let mut pwv5 = Vec::with_capacity(h.len() * 2);
    for i in 0..h.len() {
        let m = lo[i].max(mid[i]).max(hi[i]) + 1e-9;
        let c = |x: f64| rnd(x / m * 7.0) as u16;
        let v = (c(lo[i]) << 13) | (c(mid[i]) << 10) | (c(hi[i]) << 7) | ((rnd(h[i] * 31.0) as u16) << 2);
        pwv5.extend_from_slice(&v.to_be_bytes());
    }

    // Column i of n covers [i·len/n, (i+1)·len/n), at least one value.
    let columns = |n: usize| -> Vec<std::ops::Range<usize>> {
        (0..n).map(|i| {
            let a = (i * h.len() / n).min(h.len() - 1);
            a..((i + 1) * h.len() / n).clamp(a + 1, h.len())
        }).collect()
    };
    let max_of = |x: &[f64], s: &std::ops::Range<usize>| x[s.clone()].iter().cloned().fold(0.0, f64::max);
    let pwav: Vec<u8> = columns(400).iter().map(|s| ((rnd(mean(&white[s.clone()]) * 7.0) as u8) << 5) | rnd(max_of(&h, s) * 31.0) as u8).collect();
    let pwv2: Vec<u8> = columns(100).iter().map(|s| rnd(max_of(&h, s) * 15.0) as u8).collect();

    // PWV4: 1200 columns × 6 bytes: height, height, 0, bass, mids, highs (0-127).
    let refs: Vec<f64> = [lo, mid, hi].iter().map(|x| { let p = percentile(x, 99.5); if p > 0.0 { p } else { 1.0 } }).collect();
    let mut pwv4 = vec![];
    for s in columns(1200) {
        let hh = rnd(max_of(&h, &s) * 127.0) as u8;
        pwv4.extend_from_slice(&[hh, hh, 0]);
        for (j, x) in [lo, mid, hi].iter().enumerate() {
            pwv4.push(rnd(max_of(x, &s) / refs[j] * 127.0).clamp(0.0, 127.0) as u8);
        }
    }
    json!({"pwav": b64(&pwav), "pwv2": b64(&pwv2), "pwv3": b64(&pwv3), "pwv4": b64(&pwv4), "pwv5": b64(&pwv5)})
}

/// Every beat as (beat number 1-4, time in ms, bpm) for the ANLZ PQTZ section.
pub fn beats_from_grid(a: &Value) -> Vec<(u16, u32, f64)> {
    let (mut t, period, dur) = (a["first_beat_ms"].as_f64().unwrap_or(0.0), a["period_ms"].as_f64().unwrap_or(500.0), a["duration_ms"].as_f64().unwrap_or(0.0));
    let (down, bpm) = (a["downbeat_index"].as_i64().unwrap_or(0), a["bpm"].as_f64().unwrap_or(0.0));
    let mut out = vec![];
    let mut i = 0i64;
    while t < dur && period > 0.0 {
        out.push(((i - down).rem_euclid(4) as u16 + 1, rnd(t) as u32, bpm));
        t += period;
        i += 1;
    }
    out
}

// ── ClubCraft analysis, mapped to our JSON ───────────────────────────────────

/// Automatic hot cues: A start, B first drop, C first breakdown after it, D outro.
/// Each sits on a phrase line (every 8 bars from the grid anchor, where ClubCraft's
/// sections start), and there is at most one per 16-bar phrase. When two land in
/// the same phrase the musical event beats "Start" (e.g. a drop at bar 8), otherwise
/// the earlier one in the list wins. Returns (slot 1-4, time in s, label).
pub fn auto_cues(sections: &[clubcraft_core::Section], g: &clubcraft_core::BeatGrid) -> Vec<(i64, f64, &'static str)> {
    let first_drop = sections.iter().find(|s| s.kind == SectionKind::Drop);
    let breakdown = first_drop.and_then(|d| sections.iter().find(|s| s.kind == SectionKind::Breakdown && s.start > d.start));
    let outro = sections.iter().find(|s| s.kind == SectionKind::Outro);
    let wanted = [(1, Some(g.downbeat), "Start"), (2, first_drop.map(|s| s.start), "Drop"),
                  (3, breakdown.map(|s| s.start), "Breakdown"), (4, outro.map(|s| s.start), "Outro")];
    let bar_of = |t: f64| (g.bar_at(t) / 8.0).round() as i64 * 8;          // nearest phrase line
    let mut taken: Vec<(i64, i64)> = vec![];                                   // (16-bar phrase, slot)
    // Musical events first, then Start, so Start gives way when it shares a phrase.
    for &(slot, t, _) in wanted[1..].iter().chain(&wanted[..1]) {
        let Some(t) = t else { continue };
        let phrase = bar_of(t).div_euclid(16);
        if !taken.iter().any(|&(p, _)| p == phrase) {
            taken.push((phrase, slot));
        }
    }
    taken.sort_by_key(|&(_, slot)| slot);
    taken.iter().map(|&(_, slot)| {
        let (_, t, label) = wanted[slot as usize - 1];
        (slot, g.time_of_bar(bar_of(t.unwrap()) as f64).max(0.0), label)
    }).collect()
}

/// Analyse decoded audio (and stems, if any) into our analysis JSON. No database access.
pub fn analyse_audio(audio: &AudioBuffer, stems: Option<&StemsRef>, bpm_range: (f64, f64)) -> Value {
    let mut opts = AnalyzeOptions::default();
    (opts.beat.fold_min, opts.beat.fold_max) = bpm_range;
    let r = clubcraft_analysis::analyze(audio, stems, &opts);

    let sr = audio.sample_rate;
    let mono: Vec<f32> = audio.left.iter().zip(&audio.right).map(|(l, r)| (l + r) / 2.0).collect();
    let duration_ms = mono.len() as f64 / sr as f64 * 1000.0;
    let [peak, lo, mid, hi] = band_envelopes(&mono, sr);

    // Grid: ClubCraft gives a tempo and its anchor downbeat. Ours is the first beat at or
    // after 0 plus which beat (0-3) of the bar that is, as rekordbox's PQTZ counts.
    let g = r.grid.unwrap_or(clubcraft_core::BeatGrid::new(120.0, 0.0));
    let beat = 60.0 / g.bpm;
    let n = (g.downbeat / beat).floor();
    let first = g.downbeat - n * beat;
    let ms = |s: f64| (s * 1000.0).max(0.0);
    let start_of = |k: SectionKind| r.sections.iter().find(|s| s.kind == k).map(|s| ms(s.start));
    // Intro end = where the first non-intro section starts (only when the track has an intro).
    let intro_end = r.sections.first().filter(|s| s.kind == SectionKind::Intro).map(|s| ms(s.end));
    let drops: Vec<f64> = r.sections.iter().filter(|s| s.kind == SectionKind::Drop).map(|s| ms(s.start)).collect();
    json!({
        "version": VERSION,
        "bpm": g.bpm, "period_ms": beat * 1000.0, "first_beat_ms": first * 1000.0,
        "downbeat_index": (n as i64).rem_euclid(4), "duration_ms": duration_ms,
        "key": r.key.map(|k| k.name()), "camelot": r.key.map(|k| k.camelot()),
        "energy": r.energy, "loudness_lufs": r.loudness_lufs, "gain_db": r.gain_db,
        "phrases": {"first_downbeat_ms": ms(g.downbeat), "intro_end_ms": intro_end,
                    "outro_start_ms": start_of(SectionKind::Outro), "drops": drops},
        "sections": r.sections.iter().map(|s| json!({
            "kind": s.kind.label(), "start_ms": ms(s.start), "end_ms": ms(s.end), "bars": s.bars, "energy": s.energy})).collect::<Vec<_>>(),
        // Auto cues: slot 1-4 = hot cue A-D.
        "auto_cues": auto_cues(&r.sections, &g).iter().map(|&(slot, t, label)| json!({
            "slot": slot, "start_ms": ms(t), "label": label})).collect::<Vec<_>>(),
        "waveforms": waveforms(&peak, &lo, &mid, &hi),
    })
}

pub fn analyse(track_id: i64, progress: &dyn Fn(f64, &str)) -> Result<Value> {
    let t = db::track(track_id)?.ok_or_else(|| anyhow!("unknown track"))?;
    progress(0.05, "Loading audio");
    let audio = clubcraft_audio::decode::decode_native(std::path::Path::new(t["path"].as_str().unwrap_or("")))?;
    // Stems help ClubCraft's beat, key and section detection. They are 44.1 kHz, so they
    // can only be used with 44.1 kHz tracks (the mix is never resampled for analysis).
    let stem_audio = if audio.sample_rate == stems::RATE { stems::read_stems(track_id).ok() } else { None };
    let stems_ref = stem_audio.as_ref().filter(|s| s[0].0.len() == audio.left.len()).map(|s| {
        StemsRef::new(std::array::from_fn(|i| (s[i].0.as_slice(), s[i].1.as_slice())))
    });
    progress(0.3, if stems_ref.is_some() { "Analysing (with stems)" } else { "Analysing" });
    let range = (config::setting_f64("bpm_min").max(40.0), config::setting_f64("bpm_max").max(60.0));
    let mut a = analyse_audio(&audio, stems_ref.as_ref(), range);
    a["track_id"] = json!(track_id);
    a["analysed_at"] = json!(db::now());
    std::fs::write(config::analysis_dir().join(format!("{track_id}.json")), a.to_string())?;

    db::tx(|c| {
        c.execute("UPDATE tracks SET bpm=?, musical_key=?, camelot=?, energy=?, duration=?, analysed_at=? WHERE id=?",
                  params![a["bpm"].as_f64(), a["key"].as_str(), a["camelot"].as_str(), a["energy"].as_i64(),
                          a["duration_ms"].as_f64().map(|d| d / 1000.0), a["analysed_at"].as_f64(), track_id])?;
        // Place automatic cues only if the user hasn't set any.
        let user: Option<i64> = c.query_row("SELECT 1 FROM cues WHERE track_id=? AND auto=0", [track_id], |r| r.get(0)).ok();
        if user.is_none() {
            c.execute("DELETE FROM cues WHERE track_id=? AND auto=1", [track_id])?;
            for q in a["auto_cues"].as_array().into_iter().flatten() {
                c.execute("INSERT INTO cues(track_id, slot, start_ms, label, auto) VALUES (?,?,?,?,1)",
                          params![track_id, q["slot"].as_i64(), q["start_ms"].as_f64().map(|v| v as i64), q["label"].as_str()])?;
            }
        }
        Ok(())
    })?;
    progress(1.0, "Done");
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A click track at a known tempo: the grid must come back at that tempo, on the clicks,
    /// at 44.1 and 48 kHz (no resampling), and our first-beat/downbeat mapping must hold.
    #[test]
    fn grid_from_click_track() {
        for sr in [44100u32, 48000] {
            let bpm = 128.0;
            let n = sr as usize * 40;
            let period = 60.0 / bpm * sr as f64;
            let mut y = vec![0.0f32; n];
            let mut k = 0.25 * sr as f64;              // first click at 250 ms
            while (k as usize) < n {
                for j in 0..400.min(n - k as usize) {
                    y[k as usize + j] = (j as f32 * 0.3).sin() * (1.0 - j as f32 / 400.0);
                }
                k += period;
            }
            let audio = AudioBuffer::new(sr, y.clone(), y);
            let a = analyse_audio(&audio, None, (85.0, 175.0));
            assert!((a["bpm"].as_f64().unwrap() - bpm).abs() < 0.05, "{sr}: {}", a["bpm"]);
            let (first, p) = (a["first_beat_ms"].as_f64().unwrap(), a["period_ms"].as_f64().unwrap());
            assert!(first >= 0.0 && first < p, "{sr}: first beat {first}");
            let off = (first - 250.0).rem_euclid(p).min(p - (first - 250.0).rem_euclid(p));
            assert!(off < 15.0, "{sr}: grid {off} ms off the clicks");
            assert!((0..4).contains(&a["downbeat_index"].as_i64().unwrap()));
        }
    }

    /// Music with a known tempo and key: chords in A minor over a kick drum.
    #[test]
    fn synthetic_music_tempo_and_key() {
        for (bpm, sr) in [(124.0, 44100u32), (128.0, 48000), (174.0, 44100)] {
            let y = crate::testaudio::music(sr, 60.0, bpm, 0.1, crate::testaudio::A_MINOR);
            let a = analyse_audio(&AudioBuffer::new(sr, y.clone(), y), None, (85.0, 175.0));
            assert!((a["bpm"].as_f64().unwrap() - bpm).abs() < 0.05, "{bpm}@{sr}: bpm {}", a["bpm"]);
            assert_eq!((a["key"].as_str(), a["camelot"].as_str()), (Some("Am"), Some("8A")), "{bpm}@{sr}");
        }
    }

    /// Real tracks with known values (not in git: copyright). Run with
    ///   POCKET_PREP_KNOWN_TRACKS=<folder> cargo test --release -- --ignored known_tracks
    /// Expected values are in tests/known_tracks.json.
    #[test]
    #[ignore]
    fn known_tracks() {
        let dir = std::path::PathBuf::from(std::env::var("POCKET_PREP_KNOWN_TRACKS").expect("set POCKET_PREP_KNOWN_TRACKS"));
        let list: Value = serde_json::from_str(include_str!("../tests/known_tracks.json")).unwrap();
        let mut failures = vec![];
        for t in list["tracks"].as_array().unwrap() {
            let file = t["file"].as_str().unwrap();
            let audio = clubcraft_audio::decode::decode_native(&dir.join(file)).unwrap();
            let a = analyse_audio(&audio, None, (85.0, 175.0));
            let (bpm, key) = (a["bpm"].as_f64().unwrap(), a["key"].as_str().unwrap_or("").to_string());
            println!("{file}: {bpm} BPM, {key}");
            if (bpm - t["bpm"].as_f64().unwrap()).abs() > 0.05 {
                failures.push(format!("{file}: bpm {bpm}, expected {}", t["bpm"]));
            }
            if key != t["key"].as_str().unwrap() {
                failures.push(format!("{file}: key {key}, expected {}", t["key"]));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    fn section(kind: SectionKind, start_bar: i32, bars: u32, g: &clubcraft_core::BeatGrid) -> clubcraft_core::Section {
        clubcraft_core::Section { kind, start: g.time_of_bar(start_bar as f64), end: g.time_of_bar((start_bar + bars as i32) as f64),
                                  start_bar, bars, energy: 0.5, stems: [0.0; 4] }
    }

    #[test]
    fn auto_cues_one_per_16_bar_phrase() {
        use SectionKind::*;
        let g = clubcraft_core::BeatGrid::new(126.0, 0.02);
        // "Still Here": 8-bar intro, so the drop shares the first 16-bar phrase with Start.
        let s = [section(Intro, 0, 8, &g), section(Drop, 8, 24, &g), section(Breakdown, 32, 16, &g),
                 section(Drop, 48, 16, &g), section(Outro, 136, 10, &g)];
        let c = auto_cues(&s, &g);
        assert_eq!(c.iter().map(|x| (x.0, x.2)).collect::<Vec<_>>(), [(2, "Drop"), (3, "Breakdown"), (4, "Outro")]);
        assert!((c[0].1 - g.time_of_bar(8.0)).abs() < 1e-9);
        // 32-bar intro: all four, each on its own phrase.
        let s = [section(Intro, 0, 32, &g), section(Drop, 32, 64, &g), section(Breakdown, 96, 16, &g), section(Outro, 176, 9, &g)];
        let c = auto_cues(&s, &g);
        assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), [1, 2, 3, 4]);
        // A section starting 1 bar off a phrase line is moved onto it.
        let s = [section(Intro, 0, 33, &g), section(Drop, 33, 64, &g)];
        assert!((auto_cues(&s, &g)[1].1 - g.time_of_bar(32.0)).abs() < 1e-9);
    }

    #[test]
    fn butterworth_matches_scipy_gain() {
        // 4th-order low-pass at 200 Hz: -3 dB at the corner, unity at DC.
        let sr = 44100.0;
        let tone = |f: f64| -> f64 {
            let x: Vec<f32> = (0..44100).map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin() as f32).collect();
            let y = butter4(&x, false, 200.0, sr);
            (y[22050..].iter().map(|v| v * v).sum::<f64>() / 22050.0 * 2.0).sqrt()
        };
        assert!((tone(200.0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);
        assert!((tone(20.0) - 1.0).abs() < 0.01);
    }
}
