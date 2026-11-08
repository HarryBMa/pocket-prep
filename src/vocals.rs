//! Vocals from the stems: where the singing starts, and the sung phrases worth lifting.
//!
//! Runs after every separation (the only time the vocal stem exists). ClubCraft's vocal map says, per
//! 16th note, how surely the vocal stem sings; the first step that is surely sung for a beat is the
//! **first vocal**. Its mashup features find **vocal units**: whole phrases (16 or 8 bars) that start
//! and end with a sung line, each with a quality 0..1. Without stems nothing is guessed from the mix.
//!
//! Saved as `<data>/analysis/<id>.vocals.json` and `tracks.first_vocal_ms`; the open track shows the
//! units as chips (select, loop, save), the waveform gets a memory cue "Vocal" at the first vocal
//! (only when no cue of that name exists), and the card's hints carry both.

use anyhow::Result;
use clubcraft_analysis::mashup::{self, MatchOptions};
use clubcraft_analysis::{vocals, StemsRef};
use clubcraft_core::BeatGrid;
use serde_json::{json, Value};

use crate::{analysis, config, db, export::usb, library, stems::StemAudio};

const SUNG: f32 = 0.5;      // "surely sung" per 16th
const SUNG_STEPS: usize = 4; // for a beat

fn file(track_id: i64) -> std::path::PathBuf { config::analysis_dir().join(format!("{track_id}.vocals.json")) }

pub fn load(track_id: i64) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(file(track_id)).ok()?).ok()
}

/// The grid ClubCraft counts bars from (the anchor, with the DJ's nudge) and the section starts (s).
pub fn grid_and_sections(track_id: i64) -> Option<(BeatGrid, Vec<f64>)> {
    let t = db::track(track_id).ok().flatten()?;
    let a = analysis::user_grid(&analysis::load_analysis(track_id)?, &t);
    let (bpm, p, f) = (a["bpm"].as_f64()?, a["period_ms"].as_f64()?, a["first_beat_ms"].as_f64()?);
    if bpm <= 0.0 || p <= 0.0 { return None; }
    let nudge = t.get("grid_offset_ms").and_then(Value::as_f64).unwrap_or(0.0) + t.get("downbeat_shift").and_then(Value::as_f64).unwrap_or(0.0) * p;
    let anchor = a["phrases"]["first_downbeat_ms"].as_f64().map(|d| d + nudge).unwrap_or(f + a["downbeat_index"].as_f64().unwrap_or(0.0) * p);
    let starts = a["sections"].as_array().map(|v| v.iter().filter_map(|s| s["start_ms"].as_f64().map(|x| x / 1000.0)).collect()).unwrap_or_default();
    Some((BeatGrid { bpm, downbeat: anchor / 1000.0, beats_per_bar: a["beats_per_bar"].as_u64().unwrap_or(4) as u32 }, starts))
}

/// First vocal (ms) and the vocal units of `stems` on `grid`. Pure: no files, no database.
pub fn find(stems: &StemAudio, grid: BeatGrid, section_starts: &[f64], gain_db: f32) -> Value {
    let n = stems[0].0.len();
    let duration = n as f64 / 44100.0;
    let sr = StemsRef::new(std::array::from_fn(|k| (&stems[k].0[..n], &stems[k].1[..n])));
    let map = vocals::compute(&sr, grid, duration);
    let first = map.sung.windows(SUNG_STEPS).position(|w| w.iter().all(|s| *s >= SUNG))
        .map(|j| grid.time_of_beat((map.first_step + j as i32) as f64 / 4.0).max(0.0) * 1000.0);
    let mut f = mashup::compute(&sr, grid, duration, gain_db, section_starts);
    f.set_vocal_map(map);
    let units: Vec<Value> = mashup::vocal_units(&f, &MatchOptions::default(), 12).iter().map(|u| json!({
        "bar": u.bar, "bars": u.bars, "role": u.role.label(), "quality": (u.quality * 100.0).round() / 100.0,
        "start_ms": (grid.time_of_bar(u.bar as f64) * 1000.0).round(), "end_ms": (grid.time_of_bar((u.bar + u.bars as i32) as f64) * 1000.0).round(),
    })).collect();
    json!({"first_vocal_ms": first.map(f64::round), "units": units, "grid": {"bpm": grid.bpm, "downbeat_ms": grid.downbeat * 1000.0}})
}

/// Run after a separation: find, save, place the "Vocal" memory cue.
pub fn after_separation(track_id: i64, stems: &StemAudio) -> Result<()> {
    let Some((grid, starts)) = grid_and_sections(track_id) else { return Ok(()) };
    let t = db::track(track_id)?.unwrap_or_default();
    let v = find(stems, grid, &starts, usb::gain_db(t.get("lufs").and_then(Value::as_f64)) as f32);
    std::fs::write(file(track_id), v.to_string())?;
    let first = v["first_vocal_ms"].as_f64();
    db::tx(|c| Ok(c.execute("UPDATE tracks SET first_vocal_ms=? WHERE id=?", rusqlite::params![first, track_id])?))?;
    if let Some(ms) = first {
        let mut cues = library::cues(track_id)?;
        if !cues.iter().any(|q| q["label"].as_str() == Some("Vocal")) {
            cues.push(json!({"slot": 0, "start_ms": ms, "label": "Vocal", "auto": 1}).as_object().unwrap().clone());
            library::set_cues(track_id, &cues.into_iter().map(Value::Object).collect::<Vec<_>>())?;
        }
    }
    Ok(())
}

/// When the stems go, so does what was read from them (the cue stays: the DJ may have moved it).
pub fn remove(track_id: i64) {
    let _ = std::fs::remove_file(file(track_id));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A track whose vocal stem is silent for 8 bars and then sings: the first vocal lands at bar 8 (±1 beat),
    /// and there are no units before it.
    #[test]
    fn first_vocal_where_the_singing_starts() {
        let g = BeatGrid::new(120.0, 0.0);
        let n = (32.0 * g.bar_len() * 44100.0) as usize;
        let from = (8.0 * g.bar_len() * 44100.0) as usize;
        let tone = |f: f64, amp: f32, start: usize| -> Vec<f32> { (0..n).map(|i| if i < start { 0.0 } else { amp * (2.0 * std::f64::consts::PI * f * i as f64 / 44100.0).sin() as f32 }).collect() };
        // A "voice": a 300 Hz tone with vibrato and syllable-like gaps, loud; the rest quiet tones.
        let voice: Vec<f32> = (0..n).map(|i| {
            if i < from { return 0.0; }
            let t = i as f64 / 44100.0;
            let on = ((t * 3.0).fract() < 0.7) as i32 as f32;
            on * 0.5 * (2.0 * std::f64::consts::PI * (300.0 + 6.0 * (2.0 * std::f64::consts::PI * 5.5 * t).sin()) * t).sin() as f32
        }).collect();
        let stems: StemAudio = [(tone(80.0, 0.2, 0), tone(80.0, 0.2, 0)), (tone(55.0, 0.2, 0), tone(55.0, 0.2, 0)), (tone(440.0, 0.1, 0), tone(440.0, 0.1, 0)), (voice.clone(), voice)];
        let v = find(&stems, g, &[0.0, 8.0 * g.bar_len()], 0.0);
        let first = v["first_vocal_ms"].as_f64().expect("a first vocal");
        let want = 8.0 * g.bar_len() * 1000.0;
        assert!((first - want).abs() <= g.beat_len() * 1000.0 + 1.0, "first vocal at {first} ms, wanted {want}: {v}");
        assert!(v["units"].as_array().unwrap().iter().all(|u| u["start_ms"].as_f64().unwrap() >= want - 1.0), "{v}");
    }
}
