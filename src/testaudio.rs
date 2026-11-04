//! Synthetic test music with a known tempo and key (tests only).

/// Stereo audio: a kick on every beat at `bpm`, plus one chord per bar from `chords`
/// (MIDI note numbers), each note with a few harmonics. First kick at `offset_s`.
pub fn music(sr: u32, seconds: f64, bpm: f64, offset_s: f64, chords: &[&[u8]]) -> Vec<f32> {
    let n = (sr as f64 * seconds) as usize;
    let mut y = vec![0.0f32; n];
    let beat = 60.0 / bpm;
    // Kick: a short 55 Hz -> 40 Hz thump.
    let mut t = offset_s;
    while t < seconds {
        let start = (t * sr as f64) as usize;
        for j in 0..(0.12 * sr as f64) as usize {
            if start + j >= n { break; }
            let tt = j as f64 / sr as f64;
            y[start + j] += (0.8 * (2.0 * std::f64::consts::PI * (40.0 + 15.0 * (-tt * 30.0).exp()) * tt).sin() * (-tt * 25.0).exp()) as f32;
        }
        t += beat;
    }
    // Chords: one per bar, starting with the first kick.
    let bar = 4.0 * beat;
    for i in 0..n {
        let ts = i as f64 / sr as f64;
        if ts < offset_s { continue; }
        let k = ((ts - offset_s) / bar) as usize % chords.len();
        let mut v = 0.0;
        for &note in chords[k] {
            let f = 440.0 * 2f64.powf((note as f64 - 69.0) / 12.0);
            for h in 1..=4 {
                v += (2.0 * std::f64::consts::PI * f * h as f64 * ts).sin() / (h * h) as f64;
            }
        }
        y[i] += (0.06 * v) as f32;
    }
    y
}

/// A minor, clearly minor: Am - Dm - E - Am (the E chord's G# marks harmonic minor).
pub const A_MINOR: &[&[u8]] = &[&[45, 57, 60, 64], &[50, 62, 65, 69], &[40, 56, 59, 64], &[45, 57, 60, 64]];

/// 16-bit stereo WAV bytes (the deck-readable layout from stems::wav_header).
pub fn wav(sr: u32, mono: &[f32]) -> Vec<u8> {
    let mut out = crate::stems::wav_header(2, sr, (mono.len() * 4) as u32);
    for &s in mono {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
