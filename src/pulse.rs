//! Pulse: the beat and downbeat tracker that the DECK001 firmware runs (patch 2.2.118h3).
//!
//! The C++ lives in `native/pulse/` (copies of the firmware's files, which must stay identical to it), and
//! `pulse_ffi.cpp` gives it a plain C ABI. This module is the safe wrapper: it interleaves the stereo
//! audio, calls Pulse, and returns a grid only when Pulse says the tempo is steady.
//!
//! Because the grid comes from the same code as the deck, a prepared stick carries the grid the deck
//! would compute for the same track.
use std::os::raw::{c_double, c_int};

/// Snap a fitted tempo to a whole BPM when it is within this distance (same as the firmware).
const INTEGER_SNAP_BPM: f64 = 0.02;

/// Mirror of `PulseGridC` in `native/pulse/pulse_ffi.cpp`. Field order and types must match exactly.
#[repr(C)]
#[derive(Default)]
struct PulseGridC {
    ok: c_int,
    steady: c_int,
    integer_tempo: c_int,
    beat_peaks: c_int,
    bpm: c_double,
    raw_bpm: c_double,
    first_downbeat_s: c_double,
    bar_share: c_double,
    on_grid: c_double,
    analysed_s: c_double,
}

extern "C" {
    fn pulse_analyze(
        interleaved: *const f32,
        frames: usize,
        channels: c_int,
        rate: c_int,
        snap_bpm: c_double,
        threads: c_int,
        out: *mut PulseGridC,
    ) -> c_int;
}

/// A grid found by Pulse. `bpm` is a whole number when the tempo snapped to one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseGrid {
    /// Grid tempo in BPM (whole when `integer_tempo`).
    pub bpm: f64,
    /// The fitted tempo before any snap.
    pub raw_bpm: f64,
    /// A beat on the grid, in seconds from the start. It is the bar line (beat 1) when downbeats are clear.
    pub first_downbeat_s: f64,
    /// Share of detected downbeats that fall on the chosen bar position (0..1).
    pub bar_share: f64,
    /// Share of beat peaks within 35 ms of the grid (0..1).
    pub on_grid: f64,
    pub beat_peaks: i32,
    pub analysed_s: f64,
    pub integer_tempo: bool,
}

/// Run Pulse on stereo audio at `rate` Hz. `threads` 0 means "cores - 1" (at most 3); the caller passes
/// 1 when several analyses already run at once.
///
/// Returns `None` unless Pulse found a grid and the tempo is steady, so the caller can fall back to the
/// previous grid.
pub fn analyse(left: &[f32], right: &[f32], rate: u32, threads: i32) -> Option<PulseGrid> {
    if left.is_empty() || left.len() != right.len() || rate == 0 {
        return None;
    }
    let mut interleaved = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right) {
        interleaved.push(*l);
        interleaved.push(*r);
    }
    let mut out = PulseGridC::default();
    // SAFETY: `interleaved` holds `frames * 2` samples and outlives the call; `out` is a valid, writable
    // PulseGridC with the layout the C side expects. The C side catches all exceptions.
    let steady = unsafe {
        pulse_analyze(
            interleaved.as_ptr(),
            left.len(),
            2,
            rate as c_int,
            INTEGER_SNAP_BPM,
            threads,
            &mut out,
        )
    };
    if steady != 1 || out.ok != 1 || out.steady != 1 {
        return None;
    }
    Some(PulseGrid {
        bpm: out.bpm,
        raw_bpm: out.raw_bpm,
        first_downbeat_s: out.first_downbeat_s,
        bar_share: out.bar_share,
        on_grid: out.on_grid,
        beat_peaks: out.beat_peaks,
        analysed_s: out.analysed_s,
        integer_tempo: out.integer_tempo == 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// The synthetic test signal from the brief: a 128 BPM click track at 22.05 kHz, 20 s, with beat 1
    /// louder, a hi-hat-like tick off the beat and a bass line that changes every bar.
    fn synthetic_128_bpm() -> Vec<f32> {
        let rate = 22050.0;
        let beat = 60.0 / 128.0;
        let n = (rate * 20.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / rate;
                let tb = t % beat;
                let one = ((t / beat).floor() as i64).rem_euclid(4) == 0;
                let off = (t + beat / 2.0) % beat;
                let bar = ((t / (4.0 * beat)).floor() as i64).rem_euclid(4);
                let x = (2.0 * PI * 55.0 * tb).sin() * (-tb / 0.12).exp() * if one { 0.8 } else { 0.55 }
                    + 0.15 * (2.0 * PI * 8000.0 * off).sin() * (-off / 0.015).exp()
                    + 0.25 * (2.0 * PI * 55.0 * 2f64.powf(bar as f64 * 3.0 / 12.0) * t).sin();
                x as f32
            })
            .collect()
    }

    #[test]
    fn finds_128_bpm_and_the_bar_line_on_the_synthetic_signal() {
        let x = synthetic_128_bpm();
        let g = analyse(&x, &x, 22050, 1).expect("Pulse should find a steady grid");
        assert!((g.bpm - 128.0).abs() < 0.05, "bpm {}", g.bpm);
        let bar = 4.0 * 60.0 / 128.0;
        let r = g.first_downbeat_s.rem_euclid(bar);
        let off = r.min(bar - r);
        assert!(off <= 0.020, "first downbeat {} s is {} s off a bar line", g.first_downbeat_s, off);
    }

    /// Real library tracks for the comparison with the firmware's `pulse_cli` (not in git: copyright). Writes
    /// each track as interleaved float32 (what `pulse_cli` reads) and prints DKPRP's Pulse result:
    ///   PULSE_DUMP_DIR=<folder> [PULSE_DUMP_N=5] cargo test --release -- --ignored dump_library_for_pulse_cli --nocapture
    #[test]
    #[ignore]
    fn dump_library_for_pulse_cli() {
        let dir = std::path::PathBuf::from(std::env::var("PULSE_DUMP_DIR").expect("set PULSE_DUMP_DIR"));
        eprintln!("dump dir: {:?}", dir);
        let n: i64 = std::env::var("PULSE_DUMP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        std::fs::create_dir_all(&dir).unwrap();
        let rows = crate::db::query(
            "SELECT id, title, path FROM tracks WHERE deleted=0 AND analysed_at IS NOT NULL AND path NOT LIKE '%.venv%' ORDER BY id LIMIT ?",
            &[serde_json::json!(n)],
        )
        .unwrap();
        for r in rows {
            let id = r["id"].as_i64().unwrap();
            let path = r["path"].as_str().unwrap();
            let audio = match clubcraft_audio::decode::decode_native(std::path::Path::new(path)) {
                Ok(a) => a,
                Err(e) => {
                    println!("{id}\tSKIP\t{path}\t{e:?}");
                    continue;
                }
            };
            let mut bytes = Vec::with_capacity(audio.left.len() * 8);
            for (l, rr) in audio.left.iter().zip(&audio.right) {
                bytes.extend_from_slice(&l.to_le_bytes());
                bytes.extend_from_slice(&rr.to_le_bytes());
            }
            let file = dir.join(format!("{id}.f32"));
            std::fs::write(&file, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", file.display()));
            let t = std::time::Instant::now();
            let g = analyse(&audio.left, &audio.right, audio.sample_rate, 0);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("{id}\t{}\t{}\t{}\t{:?}\t{ms:.1}", audio.sample_rate, file.display(), r["title"].as_str().unwrap_or(""), g);
        }
    }

    #[test]
    fn rejects_empty_and_mismatched_input() {
        assert!(analyse(&[], &[], 22050, 1).is_none());
        assert!(analyse(&[0.0; 10], &[0.0; 9], 22050, 1).is_none());
        assert!(analyse(&[0.0; 10], &[0.0; 10], 0, 1).is_none());
    }
}