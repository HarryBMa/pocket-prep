//! Time signature: how many beats make a bar, from the accents on the beat grid.
//!
//! The same idea as madmom's `DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4])` and Dixon's meter
//! estimation, without a model: the onset strength at every beat of the grid is a sequence, and a
//! bar is the period that sequence repeats with. For each candidate length `m` and phase, the mean
//! accent on the downbeats is compared with the mean off them, and the autocorrelation of the accent
//! sequence at lag `m` (and `2m`, so 3 isn't mistaken for 6 or 4 for 8) backs it up. 4/4 is the
//! default and 3/4 has to win clearly: nearly everything a DJ plays is 4/4, and a wrong 3/4 costs
//! more than a missed one. The DJ can set it by hand in the open track (`tracks.beats_per_bar`,
//! kept through re-analysis like an edited tempo).
//!
//! Only 3 and 4 are detected. 6/8 reads as 3 (two groups of three) or as 4 when the accents sit on
//! the half bar; 5 and 7 are not looked for.

use crate::analysis::COLS_PER_S;

/// How clearly 3 must beat 4 to be chosen.
const MARGIN: f64 = 1.25;

/// Onset strength per column: the positive change of the log energy of the three bands, summed.
fn onsets(lo: &[f64], mid: &[f64], hi: &[f64]) -> Vec<f64> {
    let n = lo.len().min(mid.len()).min(hi.len());
    let db = |x: f64| 20.0 * (x.max(1e-6)).log10();
    (0..n).map(|i| if i == 0 { 0.0 } else {
        [lo, mid, hi].iter().map(|b| (db(b[i]) - db(b[i - 1])).max(0.0)).sum::<f64>()
    }).collect()
}

/// Accent of each beat: the strongest onset within an eighth of a beat around it.
fn accents(onset: &[f64], first_beat_s: f64, period_s: f64) -> Vec<f64> {
    let half = (period_s / 8.0 * COLS_PER_S).max(1.0) as usize;
    let mut out = vec![];
    let mut t = first_beat_s;
    while (t * COLS_PER_S) as usize + half < onset.len() {
        let c = (t * COLS_PER_S) as usize;
        let a = c.saturating_sub(half);
        out.push(onset[a..(c + half).min(onset.len())].iter().cloned().fold(0.0, f64::max));
        t += period_s;
    }
    out
}

fn autocorr(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if n <= lag + 2 { return 0.0; }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    if var <= 0.0 { return 0.0; }
    (0..n - lag).map(|i| (x[i] - mean) * (x[i + lag] - mean)).sum::<f64>() / var
}

/// Downbeat contrast for bar length `m`: the best phase and how much louder its beats are than the rest
/// (in standard deviations of the accent sequence).
fn contrast(acc: &[f64], m: usize) -> (usize, f64) {
    let n = acc.len();
    let mean = acc.iter().sum::<f64>() / n.max(1) as f64;
    let sd = (acc.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1) as f64).sqrt().max(1e-9);
    (0..m).map(|p| {
        let (mut on, mut k) = (0.0, 0usize);
        for (i, v) in acc.iter().enumerate() { if i % m == p { on += v; k += 1; } }
        let on = on / k.max(1) as f64;
        let off = (mean * n as f64 - on * k as f64) / (n - k).max(1) as f64;
        (p, (on - off) / sd)
    }).fold((0, f64::MIN), |b, c| if c.1 > b.1 { c } else { b })
}

/// What the detector found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meter {
    pub beats_per_bar: u32,
    /// Beat offset (0..beats_per_bar) of the downbeat from the first beat of the grid.
    pub downbeat_index: u32,
    /// How clearly the chosen length beat the other (1 = a toss-up).
    pub confidence: f64,
}

/// Beats per bar of a track from its band envelopes (150 columns per second, as `band_envelopes` makes them)
/// and its grid (seconds). Needs at least 8 bars of music to say anything but 4/4.
pub fn detect(lo: &[f64], mid: &[f64], hi: &[f64], first_beat_s: f64, period_s: f64) -> Meter {
    let four = |acc: &[f64]| Meter { beats_per_bar: 4, downbeat_index: contrast(acc, 4).0 as u32, confidence: 1.0 };
    if period_s <= 0.0 { return Meter { beats_per_bar: 4, downbeat_index: 0, confidence: 1.0 }; }
    let acc = accents(&onsets(lo, mid, hi), first_beat_s, period_s);
    if acc.len() < 32 { return four(&acc); }
    // 3 wins only when every reading agrees: the downbeat contrast, the periodicity at one bar, at two bars and
    // (with enough music) at four bars, and the summed score by a margin. Dance music is 4/4 with 4- and 8-bar
    // phrases, so one strong 3-beat motif must not be enough.
    let (c4, c3) = (contrast(&acc, 4).1, contrast(&acc, 3).1);
    let lag = |m: usize| autocorr(&acc, m);
    // One bar, two bars and four bars (the phrase: chord cycles and fills repeat there).
    let score = |m: usize, c: f64| c.max(0.0) + 2.0 * lag(m).max(0.0) + lag(2 * m).max(0.0) + lag(4 * m).max(0.0);
    let (s4, s3) = (score(4, c4), score(3, c3));
    let agree = c3 > c4 && lag(3) > lag(4) && lag(6) > lag(8) && (acc.len() < 64 || lag(12) >= lag(16));
    if agree && s3 > s4 * MARGIN && s3 > 0.5 {
        Meter { beats_per_bar: 3, downbeat_index: contrast(&acc, 3).0 as u32, confidence: s3 / s4.max(1e-9) }
    } else {
        Meter { confidence: s4 / s3.max(1e-9), ..four(&acc) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Band envelopes of a click track: a strong click on the downbeat, weaker ones on the other beats.
    fn clicks(beats_per_bar: usize, bars: usize, bpm: f64) -> [Vec<f64>; 3] {
        let period = 60.0 / bpm;
        let n = ((bars * beats_per_bar) as f64 * period * COLS_PER_S) as usize + 30;
        let mut lo = vec![0.001; n]; let mut mid = vec![0.001; n]; let mut hi = vec![0.001; n];
        for b in 0..bars * beats_per_bar {
            let c = (b as f64 * period * COLS_PER_S) as usize;
            let amp = if b % beats_per_bar == 0 { 1.0 } else { 0.3 };
            for j in 0..6 { if c + j < n { lo[c + j] = amp; mid[c + j] = amp * 0.8; hi[c + j] = amp * 0.5; } }
        }
        [lo, mid, hi]
    }

    #[test]
    fn hears_three_and_four() {
        for (bpb, want) in [(4usize, 4u32), (3, 3)] {
            let [lo, mid, hi] = clicks(bpb, 24, 128.0);
            let m = detect(&lo, &mid, &hi, 0.0, 60.0 / 128.0);
            assert_eq!((m.beats_per_bar, m.downbeat_index), (want, 0), "{bpb}: {m:?}");
        }
    }

    /// Accents on every beat alike (no bar structure) stay 4/4; a short track stays 4/4.
    #[test]
    fn defaults_to_four() {
        let n = 150 * 60;
        let flat = vec![0.5; n];
        assert_eq!(detect(&flat, &flat, &flat, 0.0, 0.5).beats_per_bar, 4);
        let [lo, mid, hi] = clicks(3, 4, 128.0);
        assert_eq!(detect(&lo, &mid, &hi, 0.0, 60.0 / 128.0).beats_per_bar, 4);
    }

    /// Synthetic music through the real analysis: a kick on every beat and a chord per bar reads as 4/4 with four
    /// beats to the bar and as 3/4 with three.
    #[test]
    fn through_the_analysis() {
        for (bpb, want) in [(4usize, 4i64), (3, 3)] {
            let mono = crate::testaudio::music_in(44100, 50.0, 126.0, 0.0, crate::testaudio::A_MINOR, bpb, 0.75);
            let audio = clubcraft_audio::decode::AudioBuffer::new(44100, mono.clone(), mono);
            let a = crate::analysis::analyse_audio(&audio, None, (60.0, 200.0));
            assert_eq!(a["beats_per_bar"].as_i64(), Some(want), "{bpb} beats to the bar: {}", a["beats_per_bar"]);
        }
    }

    /// A waltz whose downbeat is the second grid beat: 3/4 with the phase found.
    #[test]
    fn finds_the_downbeat_phase() {
        let [lo, mid, hi] = clicks(3, 24, 120.0);
        let shift = (60.0 / 120.0 * COLS_PER_S) as usize;
        let shifted = |v: &Vec<f64>| { let mut o = vec![0.001; shift]; o.extend_from_slice(v); o };
        let m = detect(&shifted(&lo), &shifted(&mid), &shifted(&hi), 0.0, 0.5);
        assert_eq!((m.beats_per_bar, m.downbeat_index), (3, 1), "{m:?}");
    }
}

#[cfg(test)]
mod debug {
    /// `METER_DEBUG=<file>[,<file>...] cargo test --release meter::debug -- --ignored --nocapture` prints the scores.
    #[test]
    #[ignore]
    fn scores() {
        let Ok(files) = std::env::var("METER_DEBUG") else { return };
        for f in files.split(',') {
            let audio = clubcraft_audio::decode::decode_native(std::path::Path::new(f)).unwrap();
            let mono: Vec<f32> = audio.left.iter().zip(&audio.right).map(|(l, r)| (l + r) / 2.0).collect();
            let a = crate::analysis::analyse_audio(&audio, None, (60.0, 200.0));
            let [_, lo, mid, hi] = crate::analysis::band_envelopes_pub(&mono, audio.sample_rate);
            let (first, period) = (a["first_beat_ms"].as_f64().unwrap() / 1000.0, a["period_ms"].as_f64().unwrap() / 1000.0);
            let acc = super::accents(&super::onsets(&lo, &mid, &hi), first, period);
            let score = |m: usize| (super::contrast(&acc, m), super::autocorr(&acc, m), super::autocorr(&acc, 2 * m));
            println!("{f}: bpm {} beats {} | 4: {:?} | 3: {:?} -> {:?}", a["bpm"], acc.len(), score(4), score(3), super::detect(&lo, &mid, &hi, first, period));
            println!("  accents: {}", acc.iter().take(36).map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "));
        }
    }
}
