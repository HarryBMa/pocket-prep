//! Stem quality: what a separation produced, judged from the stems themselves, after every run.
//!
//! Three things go wrong with stems on stage, and each is measurable without a reference:
//! * **Clipping**: a stem can peak above the master (Demucs doesn't limit), and the 16-bit stem file
//!   hard-clips it. Share of samples at or over full scale, per stem.
//! * **Vocal bleed**: the vocal stem carries leakage of synths and hats 10-25 dB down. ClubCraft's
//!   vocal map says, per 16th, how surely the stem really sings; the share of the stem's energy that
//!   falls where nothing is sung is the bleed. A track with no vocals has a quiet vocal stem and is
//!   fine; that is reported as "no vocals", not as bleed.
//! * **Reconstruction**: how much of the mix the four stems fail to add back up to (dB). Large
//!   residuals mean the model struggled (distorted or very dense material).
//!
//! The grade is diagnostic: `good`, `usable` (a note to read) or `poor` (don't rely on these stems
//! live). Saved as JSON in `tracks.stem_quality`; the readiness ring and the open track show it.

use clubcraft_analysis::StemsRef;
use clubcraft_audio::AudioBuffer;
use clubcraft_core::BeatGrid;
use serde_json::{json, Value};

use crate::stems::{StemAudio, STEM_NAMES};

const CLIP_POOR: f64 = 0.005;     // 0.5 % of samples clipped: audible crackle on peaks
const CLIP_NOTE: f64 = 0.0005;
const BLEED_POOR: f32 = 0.5;      // half the vocal stem's energy is where nothing is sung
const BLEED_NOTE: f32 = 0.25;
const NO_VOCALS_DB: f64 = -30.0;  // vocal stem this far under the mix: an instrumental

fn energy(l: &[f32], r: &[f32]) -> f64 {
    l.iter().zip(r).map(|(a, b)| (*a as f64).powi(2) + (*b as f64).powi(2)).sum::<f64>() / (2.0 * l.len().max(1) as f64)
}

/// The track's beat grid as the analysis (with the DJ's nudge) has it, for the bleed measure.
pub fn grid_for(track_id: i64) -> Option<BeatGrid> {
    let t = crate::db::track(track_id).ok().flatten()?;
    let a = crate::analysis::user_grid(&crate::analysis::load_analysis(track_id)?, &t);
    let (bpm, period, first, down) = (a["bpm"].as_f64()?, a["period_ms"].as_f64()?, a["first_beat_ms"].as_f64()?, a["downbeat_index"].as_f64().unwrap_or(0.0));
    if bpm <= 0.0 || period <= 0.0 { return None; }
    Some(BeatGrid::new(bpm, (first + down * period) / 1000.0))
}

/// Assess `stems` of `mix` (both 44.1 kHz, same length). `grid` enables the vocal-bleed measure.
pub fn assess(mix: &AudioBuffer, stems: &StemAudio, grid: Option<BeatGrid>) -> Value {
    let n = stems[0].0.len().min(mix.left.len());
    let mut notes: Vec<String> = vec![];
    let mut grade = 0u8;                                           // 0 good, 1 usable, 2 poor
    let mut bump = |g: u8| grade = grade.max(g);
    // Clipping per stem.
    let mut clip = serde_json::Map::new();
    for (k, (l, r)) in stems.iter().enumerate() {
        let over = l[..n].iter().chain(&r[..n]).filter(|x| x.abs() >= 1.0).count() as f64 / (2 * n).max(1) as f64;
        clip.insert(STEM_NAMES[k].into(), json!((over * 1e4).round() / 1e4));
        if over >= CLIP_POOR { bump(2); notes.push(format!("{} clips ({:.1} % of samples)", STEM_NAMES[k], over * 100.0)); }
        else if over >= CLIP_NOTE { bump(1); notes.push(format!("{} touches full scale ({:.2} %)", STEM_NAMES[k], over * 100.0)); }
    }
    // Reconstruction: the mix minus the sum of the stems, relative to the mix.
    let mix_e = energy(&mix.left[..n], &mix.right[..n]);
    let (mut res, mut m) = (0f64, 0f64);
    for i in 0..n {
        let sl: f32 = stems.iter().map(|s| s.0[i]).sum();
        let sr: f32 = stems.iter().map(|s| s.1[i]).sum();
        res += ((mix.left[i] - sl) as f64).powi(2) + ((mix.right[i] - sr) as f64).powi(2);
        m += (mix.left[i] as f64).powi(2) + (mix.right[i] as f64).powi(2);
    }
    let recon_db = if m > 0.0 && res > 0.0 { 10.0 * (res / m).log10() } else { -120.0 };
    if recon_db > -6.0 { bump(2); notes.push(format!("The stems don't add up to the mix ({recon_db:.0} dB residual)")); }
    else if recon_db > -12.0 { bump(1); notes.push(format!("{recon_db:.0} dB of the mix is left over after the stems")); }
    // Vocal bleed, where there are vocals at all.
    let voc_e = energy(&stems[3].0[..n], &stems[3].1[..n]);
    let voc_db = if mix_e > 0.0 && voc_e > 0.0 { 10.0 * (voc_e / mix_e).log10() } else { -120.0 };
    let mut bleed: Option<f32> = None;
    if voc_db <= NO_VOCALS_DB {
        notes.push("No vocals (the vocal stem is near silent)".into());
    } else if let Some(g) = grid {
        let sr = StemsRef::new(std::array::from_fn(|k| (&stems[k].0[..n], &stems[k].1[..n])));
        let map = clubcraft_analysis::vocals::compute(&sr, g, n as f64 / 44100.0);
        // Energy of the vocal stem per 16th, split by whether the map says it sings there.
        let step_s = 60.0 / g.bpm / 4.0;
        let (mut sung_e, mut total_e) = (0f64, 0f64);
        for (j, &s) in map.sung.iter().enumerate() {
            let step = map.first_step + j as i32;
            let t0 = g.time_of_beat(step as f64 / 4.0).max(0.0);
            let (a, b) = (((t0) * 44100.0) as usize, (((t0 + step_s) * 44100.0) as usize).min(n));
            if b <= a { continue; }
            let e = energy(&stems[3].0[a..b], &stems[3].1[a..b]) * (b - a) as f64;
            total_e += e;
            if s >= 0.25 { sung_e += e; }
        }
        if total_e > 0.0 {
            let b = (1.0 - sung_e / total_e) as f32;
            bleed = Some(b);
            if b >= BLEED_POOR { bump(2); notes.push(format!("The vocal stem is mostly bleed ({:.0} % of its energy where nothing is sung)", b * 100.0)); }
            else if b >= BLEED_NOTE { bump(1); notes.push(format!("Some bleed in the vocal stem ({:.0} %)", b * 100.0)); }
        }
    }
    let grade_name = ["good", "usable", "poor"][grade as usize];
    json!({
        "grade": grade_name,
        "clip": clip, "reconstruction_db": (recon_db * 10.0).round() / 10.0,
        "vocal_db": (voc_db * 10.0).round() / 10.0, "vocal_bleed": bleed.map(|b| (b * 100.0).round() / 100.0),
        "notes": notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, f: f64, amp: f32) -> Vec<f32> {
        (0..n).map(|i| (amp as f64 * (2.0 * std::f64::consts::PI * f * i as f64 / 44100.0).sin()) as f32).collect()
    }

    /// Four clean stems that add up to the mix grade "good"; a stem pushed over full scale grades "poor"
    /// and names the stem; a near-silent vocal stem is "no vocals", not bleed.
    #[test]
    fn grades_clipping_and_reconstruction() {
        let n = 44100 * 2;
        let parts: Vec<Vec<f32>> = [60.0, 120.0, 440.0, 0.0].iter().map(|f| if *f > 0.0 { tone(n, *f, 0.2) } else { vec![0.0; n] }).collect();
        let mixv: Vec<f32> = (0..n).map(|i| parts.iter().map(|p| p[i]).sum()).collect();
        let mix = AudioBuffer::new(44100, mixv.clone(), mixv.clone());
        let stems: StemAudio = std::array::from_fn(|k| (parts[k].clone(), parts[k].clone()));
        let q = assess(&mix, &stems, None);
        assert_eq!(q["grade"], "good", "{q}");
        assert!(q["notes"].as_array().unwrap().iter().any(|s| s.as_str().unwrap().starts_with("No vocals")), "{q}");
        let mut loud = stems.clone();
        loud[0].0.iter_mut().for_each(|x| *x *= 8.0);           // drums far over full scale
        loud[0].1.iter_mut().for_each(|x| *x *= 8.0);
        let q = assess(&mix, &loud, None);
        assert_eq!(q["grade"], "poor", "{q}");
        assert!(q["notes"].as_array().unwrap().iter().any(|s| s.as_str().unwrap().starts_with("drums clips")), "{q}");
        // A vocal stem with real energy goes through the bleed measure with a grid: it reports a bleed share.
        let mut voc = stems.clone();
        voc[3] = (tone(n, 300.0, 0.2), tone(n, 300.0, 0.2));
        let q = assess(&mix, &voc, Some(BeatGrid::new(126.0, 0.0)));
        assert!(q["vocal_bleed"].is_number(), "{q}");
        assert!(q["grade"].is_string(), "{q}");
    }
}
