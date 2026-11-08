//! Transition intelligence: how well track B follows track A at their mix points, with reasons.
//!
//! The library's "mixes with" used to be a key-and-tempo rule in the browser. This scores the whole
//! transition the way a DJ judges it at the booth, each factor a reason the DJ can read:
//! tempo (ratio, half/double time allowed), key (Camelot rules, the deck's key shift included),
//! energy step, level after the export's gain, whether both mix points sit on phrase lines, and what
//! clashes inside the transition window (vocals over vocals, bass over bass) from the sections' stem
//! activity. The window length comes from ClubCraft's planner (`choose_transition_beats`), which fits
//! it to the room after A's mix-out and before B's first drop.

use clubcraft_core::analysis::{Analysis, Section, SectionKind};
use clubcraft_core::planner::{bpm_distance, choose_transition_beats};
use clubcraft_core::{BeatGrid, Key};
use serde_json::{json, Value};

use crate::{analysis, db, export::usb};

/// One track as the scorer sees it (times in seconds).
#[derive(Debug, Clone)]
pub struct Side {
    pub id: i64,
    pub bpm: Option<f64>,
    pub key: Option<Key>,
    pub energy: Option<f64>,
    /// Loudness after the export's gain (LUFS), i.e. what the deck plays.
    pub level: Option<f64>,
    pub mix_in: f64,
    pub mix_out: f64,
    pub duration: f64,
    pub grid: Option<BeatGrid>,
    pub sections: Vec<Section>,
    /// Sections carry stem activity (analysis version 6+); without it the clash check is skipped.
    pub has_activity: bool,
}

fn kind_of(label: &str) -> SectionKind {
    match label {
        "Intro" => SectionKind::Intro, "Verse" => SectionKind::Verse, "Build" => SectionKind::Buildup,
        "Drop" => SectionKind::Drop, "Break" => SectionKind::Breakdown, _ => SectionKind::Outro,
    }
}

/// A track from its row and analysis; `None` when it isn't analysed.
pub fn side(t: &db::Row) -> Option<Side> {
    let id = t["id"].as_i64()?;
    let a = analysis::load_analysis(id)?;
    let a = analysis::user_grid(&a, t);
    let ms = |v: &Value| v.as_f64().map(|x| x / 1000.0);
    // The grid anchored where ClubCraft counts phrases from (bar 0: the first downbeat with music, not the first
    // downbeat of the file), with the DJ's nudge applied, so "on a phrase line" means what the sections mean.
    let grid = match (a["bpm"].as_f64(), a["period_ms"].as_f64(), a["first_beat_ms"].as_f64()) {
        (Some(bpm), Some(p), Some(f)) if bpm > 0.0 => {
            let nudge = t.get("grid_offset_ms").and_then(Value::as_f64).unwrap_or(0.0) + t.get("downbeat_shift").and_then(Value::as_f64).unwrap_or(0.0) * p;
            let anchor = a["phrases"]["first_downbeat_ms"].as_f64().map(|d| d + nudge).unwrap_or(f + a["downbeat_index"].as_f64().unwrap_or(0.0) * p);
            Some(BeatGrid { bpm, downbeat: anchor / 1000.0, beats_per_bar: a["beats_per_bar"].as_u64().unwrap_or(4) as u32 })
        }
        _ => None,
    };
    let bpm = t.get("bpm").and_then(Value::as_f64).filter(|b| *b > 0.0).or(grid.map(|g| g.bpm));
    let shift = t.get("key_shift").and_then(Value::as_i64).unwrap_or(0) as i32;
    let key = t.get("camelot").and_then(Value::as_str).and_then(Key::from_camelot).map(|k| k.shifted(shift));
    let lufs = t.get("lufs").and_then(Value::as_f64);
    let mut has_activity = false;
    let sections: Vec<Section> = a["sections"].as_array().map(|v| v.iter().filter_map(|s| {
        let (start, end) = (ms(&s["start_ms"])?, ms(&s["end_ms"])?);
        let stems: [f32; 4] = s["stems"].as_array().and_then(|v| (v.len() == 4).then(|| std::array::from_fn(|k| v[k].as_f64().unwrap_or(0.0) as f32)))
            .map(|x| { has_activity = true; x }).unwrap_or([0.0; 4]);
        Some(Section { kind: kind_of(s["kind"].as_str().unwrap_or("")), start, end, start_bar: grid.map(|g| g.bar_at(start).round() as i32).unwrap_or(0),
                       bars: s["bars"].as_u64().unwrap_or(0) as u32, energy: s["energy"].as_f64().unwrap_or(0.0) as f32, stems })
    }).collect()).unwrap_or_default();
    let own = |k: &str| t.get(k).and_then(Value::as_f64).or(a[k].as_f64()).map(|x| x / 1000.0);
    Some(Side {
        id, bpm, key, energy: t.get("energy").and_then(Value::as_f64).or(a["energy"].as_f64()),
        level: lufs.map(|l| l + usb::gain_db(Some(l))),
        mix_in: own("mix_in_ms").unwrap_or(0.0), mix_out: own("mix_out_ms").unwrap_or(0.0),
        duration: ms(&a["duration_ms"]).unwrap_or(0.0), grid, sections, has_activity,
    })
}

fn as_analysis(s: &Side) -> Analysis {
    Analysis { version: Analysis::CURRENT_VERSION, duration: s.duration, grid: s.grid, key: s.key, loudness_lufs: None, gain_db: None,
               energy: s.energy.map(|e| e as u8), sections: s.sections.clone(), cues: vec![], stems_model: None }
}

/// Mean stem activity (drums, bass, other, vocals) over `[t0, t1]`, weighted by overlap.
fn activity(s: &Side, t0: f64, t1: f64) -> [f32; 4] {
    let (mut acc, mut w) = ([0f32; 4], 0f64);
    for sec in &s.sections {
        let o = (sec.end.min(t1) - sec.start.max(t0)).max(0.0);
        if o > 0.0 { for k in 0..4 { acc[k] += sec.stems[k] * o as f32; } w += o; }
    }
    if w > 0.0 { acc.iter_mut().for_each(|a| *a /= w as f32); }
    acc
}

/// Bars from the nearest 8-bar phrase line (0 = on it).
fn phrase_off(g: &BeatGrid, t: f64) -> i32 {
    (g.bar_at(t).round() as i32).rem_euclid(8)
}

/// Score `from` → `to`: 0-100, a verdict and the reasons behind it.
pub fn score(from: &Side, to: &Side) -> Value {
    let mut reasons: Vec<Value> = vec![];
    let mut reason = |ok: bool, text: String| reasons.push(json!({"ok": ok, "text": text}));
    let mut hard_no = false;
    // Tempo (30): within 2 % is nothing to hear; the deck's sync covers 6 %.
    let tempo = match (from.bpm, to.bpm) {
        (Some(a), Some(b)) => {
            let r = bpm_distance(a, b) / a;
            let pct = (b / a - 1.0) * 100.0;
            if r <= 0.02 { reason(true, format!("Tempo {a:.0} → {b:.0} ({pct:+.1} %)")); 1.0 }
            else if r <= 0.06 { reason(true, format!("Tempo {a:.0} → {b:.0} ({pct:+.1} %): sync covers it")); 1.0 - (r - 0.02) / 0.08 }
            else { hard_no = true; reason(false, format!("Tempo {a:.0} → {b:.0} ({pct:+.1} %) is too far for sync")); 0.0 }
        }
        _ => { reason(false, "Tempo unknown (analyse both tracks)".into()); 0.3 }
    };
    // Key (30): Camelot rules, with the deck's key shift applied.
    let key = match (from.key, to.key) {
        (Some(a), Some(b)) => {
            let c = a.compatibility(b) as f64;
            let (ca, cb) = (a.camelot(), b.camelot());
            if c >= 0.99 { reason(true, format!("Same key ({ca})")); }
            else if c >= 0.85 { reason(true, format!("{ca} → {cb}: {}", if a.minor == b.minor { "one step on the wheel" } else { "relative major / minor" })); }
            else if c >= 0.55 { reason(true, format!("{ca} → {cb}: an energy-boost modulation")); }
            else if c > 0.0 { reason(false, format!("{ca} → {cb} is a stretch on the wheel")); }
            else { hard_no = true; reason(false, format!("Key clash {ca} → {cb}")); }
            c
        }
        _ => { reason(false, "Key unknown".into()); 0.3 }
    };
    // Energy (15): a step of one or two is a set that moves; more is a jump or a fall.
    let energy = match (from.energy, to.energy) {
        (Some(a), Some(b)) => {
            let d = b - a;
            if d.abs() <= 2.0 { reason(true, format!("Energy {a:.0} → {b:.0}")); 1.0 }
            else if d < 0.0 { reason(false, format!("Energy falls {a:.0} → {b:.0}")); 0.3 }
            else { reason(false, format!("Energy jumps {a:.0} → {b:.0}")); 0.5 }
        }
        _ => 0.7,
    };
    // Level (10): what the deck plays after the export's gain.
    let level = match (from.level, to.level) {
        (Some(a), Some(b)) => {
            let d = b - a;
            if d.abs() <= 1.5 { reason(true, format!("Level matches after gain ({d:+.1} dB)")); 1.0 }
            else { reason(false, format!("Level {} by {:.1} dB after gain", if d > 0.0 { "up" } else { "down" }, d.abs())); (1.0 - (d.abs() - 1.5) / 4.5).max(0.0) }
        }
        _ => 0.7,
    };
    // The window: the planner fits it to the room after the mix-out and before B's first drop.
    let beats = choose_transition_beats(&as_analysis(from), from.mix_out, &as_analysis(to), to.mix_in, None);
    let (len_out, len_in) = (from.grid.map(|g| beats as f64 * g.beat_len()).unwrap_or(0.0), to.grid.map(|g| beats as f64 * g.beat_len()).unwrap_or(0.0));
    // Phrase lines (-5 each): a mix point off the phrase makes the blend land between phrases.
    let mut phrase = 0.0;
    if let Some(g) = from.grid { let o = phrase_off(&g, from.mix_out); if o != 0 { phrase += 0.5; reason(false, format!("Mix-out is bar {} of 8 in its phrase", o + 1)); } }
    if let Some(g) = to.grid { let o = phrase_off(&g, to.mix_in); if o != 0 { phrase += 0.5; reason(false, format!("Mix-in is bar {} of 8 in its phrase", o + 1)); } }
    if phrase == 0.0 && from.grid.is_some() && to.grid.is_some() { reason(true, "Both mix points sit on phrase lines".into()); }
    // Clashes (15) inside the window, from the sections' stem activity.
    let clash = if from.has_activity && to.has_activity {
        let (a, b) = (activity(from, from.mix_out, from.mix_out + len_out), activity(to, to.mix_in, to.mix_in + len_in));
        if a[3] >= 0.45 && b[3] >= 0.45 { reason(false, format!("Vocals over vocals in the {}-bar window", beats / 4)); 0.0 }
        else if a[1] >= 0.6 && b[1] >= 0.6 { reason(false, format!("Bass over bass in the window: cut one")); 0.5 }
        else { reason(true, format!("No clash in the {}-bar window", beats / 4)); 1.0 }
    } else { 0.7 };
    let score = (30.0 * tempo + 30.0 * key + 15.0 * energy + 10.0 * level + 15.0 * clash - 5.0 * phrase).round().clamp(0.0, 100.0);
    let verdict = if hard_no || score < 50.0 { "no" } else if score >= 75.0 { "good" } else { "ok" };
    json!({"id": to.id, "score": score, "verdict": verdict, "beats": beats, "out_ms": from.mix_out * 1000.0, "in_ms": to.mix_in * 1000.0, "reasons": reasons})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(id: i64, bpm: f64, key: &str, energy: f64, vocals: f32) -> Side {
        let g = BeatGrid::new(bpm, 0.0);
        let bar = g.bar_len();
        let sec = |kind, b0: f64, b1: f64, v| Section { kind, start: b0 * bar, end: b1 * bar, start_bar: b0 as i32, bars: (b1 - b0) as u32, energy: 0.5, stems: [0.8, 0.5, 0.5, v] };
        Side { id, bpm: Some(bpm), key: Key::from_camelot(key), energy: Some(energy), level: Some(-9.0), mix_in: 0.0, mix_out: 96.0 * bar, duration: 128.0 * bar, grid: Some(g),
               sections: vec![sec(SectionKind::Intro, 0.0, 16.0, vocals), sec(SectionKind::Drop, 16.0, 96.0, vocals), sec(SectionKind::Outro, 96.0, 128.0, vocals)], has_activity: true }
    }

    /// Same key, close tempo, no vocals in the window: good. Vocals over vocals pull the score down with a
    /// reason; a key clash or a tempo 10 % off is a "no" whatever else is right.
    #[test]
    fn scores_with_reasons() {
        let a = side(1, 128.0, "8A", 6.0, 0.0);
        let good = score(&a, &side(2, 126.0, "9A", 7.0, 0.0));
        assert_eq!(good["verdict"], "good", "{good}");
        assert!(good["reasons"].as_array().unwrap().iter().any(|r| r["text"].as_str().unwrap().contains("one step on the wheel")), "{good}");
        let vocal = score(&side(1, 128.0, "8A", 6.0, 0.8), &side(2, 128.0, "8A", 6.0, 0.8));
        assert!(vocal["score"].as_f64().unwrap() < good["score"].as_f64().unwrap(), "{vocal}");
        assert!(vocal["reasons"].as_array().unwrap().iter().any(|r| r["text"].as_str().unwrap().starts_with("Vocals over vocals")), "{vocal}");
        assert_eq!(score(&a, &side(3, 128.0, "3B", 6.0, 0.0))["verdict"], "no");
        assert_eq!(score(&a, &side(4, 141.0, "8A", 6.0, 0.0))["verdict"], "no");
        // Half time is the same tempo to a DJ.
        assert_eq!(score(&a, &side(5, 64.0, "8A", 6.0, 0.0))["verdict"], "good");
    }
}
