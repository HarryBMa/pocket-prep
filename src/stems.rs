//! Stem separation: drums, bass, other (melody), vocals.
//!
//! The separation is ClubCraft's `clubcraft-stems` crate (MIT OR Apache-2.0): Meta's
//! HTDemucs (`htdemucs`, ClubCraft's `Quality::Fast`) reproducing `demucs` exactly, on
//! CoreML (Mac GPU, when the .mlmodelc is present), DirectML (Windows GPU, self-checked
//! against the CPU and skipped if it disagrees) or the processor. Model files: models/
//! (htdemucs.json + htdemucs.onnx, made by tools/export_demucs_onnx.py).
//!
//! Output: one 8-channel 16-bit WAV per track at 44.1 kHz, Demucs's own rate,
//! the format the deck reads (see docs/STEMS_FORMAT.md):
//!   channels 1-2 drums, 3-4 bass, 5-6 other, 7-8 vocals (each L, R)
//!   the data chunk starts at byte 4096, so the deck reads whole SD sectors

use anyhow::{anyhow, Result};
use clubcraft_stems::{Quality, Separator};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::{config, db};

pub const STEM_NAMES: [&str; 4] = ["drums", "bass", "other", "vocals"];
pub const RATE: u32 = 44100;
pub const DATA_OFFSET: usize = 4096;

/// Planar stereo per stem, in STEM_NAMES order.
pub type StemAudio = [(Vec<f32>, Vec<f32>); 4];

/// The model, loaded once (takes a few seconds) and reused for every track.
static SEPARATOR: Mutex<Option<Separator>> = Mutex::new(None);

pub fn stem_file(track_id: i64) -> PathBuf {
    config::stems_dir().join(format!("{track_id}.wav"))
}

// ── Files ────────────────────────────────────────────────────────────────────

/// WAV header for 16-bit PCM, padded with a JUNK chunk so samples start at DATA_OFFSET.
pub fn wav_header(channels: u16, rate: u32, data_bytes: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(DATA_OFFSET);
    let junk = (DATA_OFFSET - 12 - 24 - 8 - 8) as u32;
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(DATA_OFFSET as u32 - 8 + data_bytes).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());                               // plain PCM
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());       // bytes per second
    h.extend_from_slice(&(channels * 2).to_le_bytes());                     // block align
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"JUNK");
    h.extend_from_slice(&junk.to_le_bytes());
    h.resize(h.len() + junk as usize, 0);
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_bytes.to_le_bytes());
    debug_assert_eq!(h.len(), DATA_OFFSET);
    h
}

/// Write the stems interleaved as 16-bit with TPDF dither. Atomic: temp file, then rename.
fn write_stems(path: &Path, stems: &StemAudio) -> Result<()> {
    let frames = stems[0].0.len();
    let bytes = frames as u64 * 16;
    if bytes > u32::MAX as u64 - DATA_OFFSET as u64 {
        return Err(anyhow!("track too long for a WAV file"));
    }
    let tmp = path.with_extension("wav.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(&wav_header(8, RATE, bytes as u32))?;
    let mut seed = 0x1234_5678u32;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed as f32 / u32::MAX as f32 };
    for i in 0..frames {
        for (l, r) in stems {
            for c in [l, r] {
                let d = rnd() - rnd();                                      // ±1 LSB triangular dither
                let v = (c[i] * 32767.0 + d).round().clamp(-32768.0, 32767.0) as i16;
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a track's 8-channel stem file back as floats (for analysis).
pub fn read_stems(track_id: i64) -> Result<StemAudio> {
    let mut raw = vec![];
    std::fs::File::open(stem_file(track_id))?.read_to_end(&mut raw)?;
    let data = raw.get(DATA_OFFSET..).ok_or_else(|| anyhow!("stem file too short"))?;
    let mut out: StemAudio = Default::default();
    for frame in data.chunks_exact(16) {
        for (s, (l, r)) in out.iter_mut().enumerate() {
            let v = |i: usize| i16::from_le_bytes([frame[i], frame[i + 1]]) as f32 / 32768.0;
            l.push(v(s * 4));
            r.push(v(s * 4 + 2));
        }
    }
    Ok(out)
}

fn preview_file(track_id: i64, name: &str) -> PathBuf {
    config::stems_dir().join("preview").join(format!("{track_id}_{name}.wav"))
}

/// One stem from the 8-channel file as a stereo WAV on disk, for listening in the app.
/// A real file (not built in memory) so the player can seek in it: hot cues need that.
/// Made on first use, and again whenever the stems are newer.
pub fn stem_preview(track_id: i64, name: &str) -> Result<PathBuf> {
    let idx = STEM_NAMES.iter().position(|n| *n == name).ok_or_else(|| anyhow!("no such stem"))?;
    let src = stem_file(track_id);
    let out_path = preview_file(track_id, name);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if modified(&out_path).is_some() && modified(&out_path) >= modified(&src) {
        return Ok(out_path);
    }
    let mut f = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&src)?);
    let total = f.get_ref().metadata()?.len() as usize;
    f.seek(SeekFrom::Start(DATA_OFFSET as u64))?;
    let frames = (total - DATA_OFFSET) / 16;
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    let tmp = out_path.with_extension("wav.tmp");
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
    out.write_all(&wav_header(2, RATE, (frames * 4) as u32))?;
    let mut frame = [0u8; 16];
    for _ in 0..frames {
        f.read_exact(&mut frame)?;
        out.write_all(&frame[idx * 4..idx * 4 + 4])?;
    }
    out.into_inner().map_err(|e| e.into_error())?;
    std::fs::rename(&tmp, &out_path)?;
    Ok(out_path)
}

// ── Separation ───────────────────────────────────────────────────────────────

pub fn separate(track_id: i64, progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<()> {
    let t = db::track(track_id)?.ok_or_else(|| anyhow!("unknown track"))?;
    db::tx(|c| Ok(c.execute("UPDATE tracks SET stems='running' WHERE id=?", [track_id])?))?;
    let res = (|| -> Result<()> {
        progress(0.01, "Loading audio");
        // Demucs only works at 44.1 kHz; other rates are converted for it (the card keeps the original file).
        let audio = clubcraft_audio::decode_file(Path::new(t["path"].as_str().unwrap_or("")), RATE)?;
        let mut g = SEPARATOR.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            progress(0.03, "Loading Demucs");
            *g = Some(Separator::with_models_dir(Quality::Fast, &clubcraft_core::paths::models_dir())?);
        }
        let sep = g.as_ref().unwrap();
        let backend = sep.backend_name();
        let stems = sep.separate(&audio, &mut |p| {
            progress(0.05 + 0.9 * p as f64, &format!("Separating ({backend})"));
            !cancel()
        })?;
        drop(g);
        progress(0.97, "Writing stems");
        write_stems(&stem_file(track_id), &stems.stems)
    })();
    let state = if res.is_ok() { "done" } else { "failed" };
    db::tx(|c| Ok(c.execute("UPDATE tracks SET stems=? WHERE id=?", rusqlite::params![state, track_id])?))?;
    res?;
    progress(1.0, "Done");
    Ok(())
}

pub fn remove(track_id: i64) -> Result<()> {
    let _ = std::fs::remove_file(stem_file(track_id));
    for name in STEM_NAMES {
        let _ = std::fs::remove_file(preview_file(track_id, name));
    }
    db::tx(|c| Ok(c.execute("UPDATE tracks SET stems='none' WHERE id=?", [track_id])?))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_puts_data_at_4096() {
        let h = wav_header(8, RATE, 1600);
        assert_eq!(h.len(), 4096);
        assert_eq!(&h[4088..4092], b"data");
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 16);
    }

    #[test]
    fn stems_round_trip_through_the_file() {
        let dir = std::env::temp_dir().join("pocketprep_stemtest");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.wav");
        let s: StemAudio = std::array::from_fn(|k| {
            let ch = |c: f32| (0..1000).map(|i| (i as f32 * 0.01 * (k as f32 + 1.0)).sin() * 0.5 * c).collect::<Vec<f32>>();
            (ch(1.0), ch(-0.5))
        });
        write_stems(&p, &s).unwrap();
        let raw = std::fs::read(&p).unwrap();
        assert_eq!(raw.len(), DATA_OFFSET + 1000 * 16);
        // Frame 10, bass right channel (channel 4): within 2 LSB of the original.
        let at = DATA_OFFSET + 10 * 16 + 3 * 2;
        let v = i16::from_le_bytes([raw[at], raw[at + 1]]) as f32 / 32768.0;
        assert!((v - s[1].1[10]).abs() < 2.0 / 32768.0);
    }
}
