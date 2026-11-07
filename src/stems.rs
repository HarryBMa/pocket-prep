//! Stem separation: drums, bass, other (melody), vocals.
//!
//! The separation is ClubCraft's `clubcraft-stems` crate (MIT OR Apache-2.0): Meta's
//! HTDemucs (`htdemucs`, ClubCraft's `Quality::Fast`) reproducing `demucs` exactly, on
//! CoreML (Mac GPU, when the .mlmodelc is present), DirectML (Windows GPU, self-checked
//! against the CPU and skipped if it disagrees) or the processor. Model files: models/
//! (htdemucs.json + htdemucs.onnx, made by tools/export_demucs_onnx.py).
//!
//! Output: one 8-channel 16-bit FLAC per track at 44.1 kHz, Demucs's own rate, the
//! format the deck reads (docs/STEMS_FORMAT.md, version 3):
//!   channels 1-2 drums, 3-4 bass, 5-6 other, 7-8 vocals (each L, R)
//!   fixed 4410-sample blocks (100 ms) and a SEEKTABLE point at every block, so the
//!   deck can jump to any 100 ms boundary without scanning the file
//! Encoding is flacenc (same crate ClubCraft uses); decoding here is claxon.

use anyhow::{anyhow, Context, Result};
use clubcraft_stems::{Quality, Separator};
use flacenc::component::{BitRepr, MetadataBlockData};
use flacenc::error::Verify;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::{config, db};

pub const STEM_NAMES: [&str; 4] = ["drums", "bass", "other", "vocals"];
pub const RATE: u32 = 44100;
pub const CHANNELS: usize = 8;
/// FLAC block size = seek interval: 4410 samples = 100 ms at 44.1 kHz.
pub const SEEK_FRAMES: usize = 4410;
/// Where WAV previews put their samples (see wav_header).
pub const DATA_OFFSET: usize = 4096;

/// Planar stereo per stem, in STEM_NAMES order.
pub type StemAudio = [(Vec<f32>, Vec<f32>); 4];

/// The model, loaded once (takes a few seconds) and reused for every track.
static SEPARATOR: Mutex<Option<Separator>> = Mutex::new(None);

pub fn stem_file(track_id: i64) -> PathBuf {
    config::stems_dir().join(format!("{track_id}.flac"))
}

/// Stems from Pocket Prep 0.2 (8-channel WAV, data at byte 4096), converted by migrate_wav_stems.
fn old_wav_file(track_id: i64) -> PathBuf {
    config::stems_dir().join(format!("{track_id}.wav"))
}

pub fn has_stems(track_id: i64) -> bool {
    stem_file(track_id).exists() || old_wav_file(track_id).exists()
}

// ── WAV (previews in the app) ────────────────────────────────────────────────

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

// ── FLAC stem files ──────────────────────────────────────────────────────────

/// VORBIS_COMMENT block content naming the channel order.
fn vorbis_comment() -> Vec<u8> {
    let vendor = concat!("Pocket Prep ", env!("CARGO_PKG_VERSION"));
    let tags = ["POCKETDECK_STEMS=drums,bass,other,vocals"];
    let mut b = Vec::new();
    b.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    b.extend_from_slice(vendor.as_bytes());
    b.extend_from_slice(&(tags.len() as u32).to_le_bytes());
    for t in tags {
        b.extend_from_slice(&(t.len() as u32).to_le_bytes());
        b.extend_from_slice(t.as_bytes());
    }
    b
}

/// Encode interleaved 16-bit samples (8 channels) as FLAC with fixed 4410-sample blocks
/// and a SEEKTABLE point at every block. Atomic: temp file, then rename.
fn write_flac(path: &Path, interleaved: &[i32]) -> Result<()> {
    let frames = interleaved.len() / CHANNELS;
    let mut config = flacenc::config::Encoder::default();
    config.block_size = SEEK_FRAMES;
    let config = config.into_verified().map_err(|e| anyhow!("flac config: {e:?}"))?;
    let source = flacenc::source::MemSource::from_samples(interleaved, CHANNELS, 16, RATE as usize);
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, SEEK_FRAMES)
        .map_err(|e| anyhow!("flac encode: {e:?}"))?;
    // SEEKTABLE: one point per FLAC frame. Each point is 18 bytes, big-endian: first sample
    // number (u64), byte offset of the frame from the first frame's header (u64), samples (u16).
    let mut seek = Vec::new();
    let mut offset = 0u64;
    let mut n = 0;
    while let Some(f) = stream.frame(n) {
        seek.extend_from_slice(&((n * SEEK_FRAMES) as u64).to_be_bytes());
        seek.extend_from_slice(&offset.to_be_bytes());
        seek.extend_from_slice(&((frames - n * SEEK_FRAMES).min(SEEK_FRAMES) as u16).to_be_bytes());
        offset += (f.count_bits() / 8) as u64;
        n += 1;
    }
    stream.add_metadata_block(MetadataBlockData::new_unknown(3, &seek).map_err(|e| anyhow!("{e:?}"))?);
    stream.add_metadata_block(MetadataBlockData::new_unknown(4, &vorbis_comment()).map_err(|e| anyhow!("{e:?}"))?);
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| anyhow!("flac write: {e:?}"))?;
    let mut bytes = sink.into_inner();
    // flacenc records the short last block as STREAMINFO's minimum block size; the spec
    // excludes the last block, so a fixed-block stream has min = max = 4410.
    if bytes.len() > 12 && &bytes[..4] == b"fLaC" {
        bytes[8..10].copy_from_slice(&(SEEK_FRAMES as u16).to_be_bytes());
        bytes[10..12].copy_from_slice(&(SEEK_FRAMES as u16).to_be_bytes());
    }
    let tmp = path.with_extension("flac.tmp");
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Quantise the stems to 16-bit with TPDF dither (±1 LSB) and write the FLAC.
pub(crate) fn write_stems(path: &Path, stems: &StemAudio) -> Result<()> {
    let frames = stems[0].0.len();
    let mut seed = 0x1234_5678u32;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed as f32 / u32::MAX as f32 };
    let mut pcm = Vec::with_capacity(frames * CHANNELS);
    for i in 0..frames {
        for (l, r) in stems {
            for c in [l, r] {
                let d = rnd() - rnd();
                pcm.push((c[i] * 32767.0 + d).round().clamp(-32768.0, 32767.0) as i32);
            }
        }
    }
    write_flac(path, &pcm)
}

/// Decode a stem file to interleaved 16-bit samples (8 channels).
fn read_flac(path: &Path) -> Result<Vec<i32>> {
    let mut r = claxon::FlacReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let info = r.streaminfo();
    if info.channels as usize != CHANNELS || info.bits_per_sample != 16 || info.sample_rate != RATE {
        return Err(anyhow!("{} isn't an 8-channel 16-bit 44.1 kHz stem file", path.display()));
    }
    r.samples().map(|s| s.map_err(|e| anyhow!("decoding {}: {e}", path.display()))).collect()
}

/// Read a track's stems back as floats (for analysis).
pub fn read_stems(track_id: i64) -> Result<StemAudio> {
    let pcm = read_flac(&stem_file(track_id))?;
    let mut out: StemAudio = Default::default();
    for frame in pcm.chunks_exact(CHANNELS) {
        for (s, (l, r)) in out.iter_mut().enumerate() {
            l.push(frame[2 * s] as f32 / 32768.0);
            r.push(frame[2 * s + 1] as f32 / 32768.0);
        }
    }
    Ok(out)
}

/// Peak envelope of each stem for drawing: `cols_per_s` columns per second, each the largest
/// absolute sample of either channel in that column, scaled so the loudest column of all four
/// stems is 255 (so the stems keep their loudness relative to each other).
pub fn peaks(track_id: i64, cols_per_s: f64) -> Result<[Vec<u8>; 4]> {
    Ok(peaks_of(&read_stems(track_id)?, cols_per_s))
}

fn peaks_of(audio: &StemAudio, cols_per_s: f64) -> [Vec<u8>; 4] {
    let hop = (RATE as f64 / cols_per_s).max(1.0);
    let cols: Vec<Vec<f32>> = audio.iter().map(|(l, r)| {
        let n = (l.len() as f64 / hop).ceil() as usize;
        (0..n).map(|c| {
            let (a, b) = ((c as f64 * hop) as usize, (((c + 1) as f64 * hop) as usize).min(l.len()));
            l[a..b].iter().zip(&r[a..b]).fold(0f32, |m, (x, y)| m.max(x.abs()).max(y.abs()))
        }).collect()
    }).collect();
    let top = cols.iter().flatten().fold(1e-6f32, |m, v| m.max(*v));
    std::array::from_fn(|i| cols[i].iter().map(|v| (v / top * 255.0).round() as u8).collect())
}

/// Total samples per channel, from the stem file's STREAMINFO.
pub fn stem_frames(track_id: i64) -> Result<u64> {
    let r = claxon::FlacReader::open(stem_file(track_id))?;
    r.streaminfo().samples.ok_or_else(|| anyhow!("stem file has no length"))
}

/// Convert stems saved by Pocket Prep 0.2 (8-channel WAV) to FLAC, losslessly: the 16-bit
/// samples are re-encoded as they are and checked by decoding before the WAV is deleted.
pub fn migrate_wav_stems() -> Result<usize> {
    let mut done = 0;
    for e in std::fs::read_dir(config::stems_dir())?.flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "wav") {
            continue;
        }
        let Some(id) = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<i64>().ok()) else { continue };
        let raw = std::fs::read(&p)?;
        let pcm: Vec<i32> = raw.get(DATA_OFFSET..).unwrap_or_default().chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32).collect();
        write_flac(&stem_file(id), &pcm)?;
        if read_flac(&stem_file(id))? != pcm {
            let _ = std::fs::remove_file(stem_file(id));
            return Err(anyhow!("stems for track {id} didn't survive conversion; kept the WAV"));
        }
        std::fs::remove_file(&p)?;
        done += 1;
    }
    Ok(done)
}

fn preview_file(track_id: i64, name: &str) -> PathBuf {
    config::stems_dir().join("preview").join(format!("{track_id}_{name}.wav"))
}

/// One stem as a stereo WAV on disk, for listening in the app. A real file (not built in
/// memory) so the player can seek in it: hot cues need that. All four are made on first
/// use (one decode), and again whenever the stems are newer.
pub fn stem_preview(track_id: i64, name: &str) -> Result<PathBuf> {
    if !STEM_NAMES.contains(&name) {
        return Err(anyhow!("no such stem"));
    }
    let src = stem_file(track_id);
    let out_path = preview_file(track_id, name);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if modified(&out_path).is_some() && modified(&out_path) >= modified(&src) {
        return Ok(out_path);
    }
    let pcm = read_flac(&src)?;
    let frames = pcm.len() / CHANNELS;
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    for (k, stem) in STEM_NAMES.iter().enumerate() {
        let dst = preview_file(track_id, stem);
        let tmp = dst.with_extension("wav.tmp");
        let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
        out.write_all(&wav_header(2, RATE, (frames * 4) as u32))?;
        for f in pcm.chunks_exact(CHANNELS) {
            out.write_all(&(f[2 * k] as i16).to_le_bytes())?;
            out.write_all(&(f[2 * k + 1] as i16).to_le_bytes())?;
        }
        out.into_inner().map_err(|e| e.into_error())?;
        std::fs::rename(&tmp, &dst)?;
    }
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

    /// One column per 1/150 s, the loudest stem scaled to 255, quieter stems in proportion.
    #[test]
    fn stem_peaks() {
        let n = RATE as usize;                               // one second
        let tone = |amp: f32| (0..n).map(|i| amp * ((i as f32) * 0.05).sin()).collect::<Vec<f32>>();
        let audio: StemAudio = [(tone(0.8), tone(0.8)), (tone(0.4), vec![0.0; n]), (vec![0.0; n], vec![0.0; n]), (tone(0.2), tone(0.1))];
        let p = peaks_of(&audio, 150.0);
        assert_eq!(p[0].len(), 150);
        assert_eq!(*p[0].iter().max().unwrap(), 255);
        assert!((p[1][10] as i32 - 128).abs() <= 2, "{}", p[1][10]);
        assert!(p[2].iter().all(|&v| v == 0));
        assert!((p[3][10] as i32 - 64).abs() <= 2, "{}", p[3][10]);
    }

    #[test]
    fn header_puts_data_at_4096() {
        let h = wav_header(8, RATE, 1600);
        assert_eq!(h.len(), 4096);
        assert_eq!(&h[4088..4092], b"data");
        assert_eq!(u16::from_le_bytes([h[32], h[33]]), 16);
    }

    fn sample_stems(frames: usize) -> StemAudio {
        std::array::from_fn(|k| {
            let ch = |c: f32| (0..frames).map(|i| (i as f32 * 0.01 * (k as f32 + 1.0)).sin() * 0.5 * c).collect::<Vec<f32>>();
            (ch(1.0), ch(-0.5))
        })
    }

    /// Lossless: the decoded file holds exactly the 16-bit samples, in channel order.
    #[test]
    fn flac_round_trip() {
        let dir = std::env::temp_dir().join(format!("pocketprep_flac_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = sample_stems(10_000);
        let pcm: Vec<i32> = (0..10_000).flat_map(|i| s.iter().flat_map(move |(l, r)| [l[i], r[i]]))
            .map(|v| (v * 32767.0).round() as i32).collect();
        let p = dir.join("a.flac");
        write_flac(&p, &pcm).unwrap();
        assert_eq!(read_flac(&p).unwrap(), pcm);
        // Dithered writer: bass right channel within 2 LSB of the float input.
        write_stems(&p, &s).unwrap();
        let back = read_flac(&p).unwrap();
        assert!((back[10 * CHANNELS + 3] as f32 / 32768.0 - s[1].1[10]).abs() < 2.0 / 32768.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The SEEKTABLE has a point every 4410 samples (100 ms), each pointing at a FLAC frame
    /// header (sync code 0xFFF8) whose frame number is that point's index.
    #[test]
    fn seektable_every_100_ms() {
        let dir = std::env::temp_dir().join(format!("pocketprep_seek_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let frames = 3 * RATE as usize + 1234;                       // 3.03 s: 31 blocks, last one short
        let p = dir.join("s.flac");
        write_stems(&p, &sample_stems(frames)).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..4], b"fLaC");
        // Walk the metadata blocks.
        let (mut pos, mut seek) = (4usize, None);
        loop {
            let (last, kind) = (b[pos] & 0x80 != 0, b[pos] & 0x7f);
            let len = u32::from_be_bytes([0, b[pos + 1], b[pos + 2], b[pos + 3]]) as usize;
            if kind == 0 {
                assert_eq!(u16::from_be_bytes([b[pos + 4], b[pos + 5]]), 4410, "min block size");
            }
            if kind == 3 {
                seek = Some(b[pos + 4..pos + 4 + len].to_vec());
            }
            pos += 4 + len;
            if last { break; }
        }
        let first_frame = pos;
        let seek = seek.expect("no SEEKTABLE");
        let points: Vec<(u64, u64, u16)> = seek.chunks_exact(18).map(|c| (
            u64::from_be_bytes(c[0..8].try_into().unwrap()), u64::from_be_bytes(c[8..16].try_into().unwrap()),
            u16::from_be_bytes([c[16], c[17]]))).collect();
        assert_eq!(points.len(), frames.div_ceil(SEEK_FRAMES));
        for (n, &(sample, offset, count)) in points.iter().enumerate() {
            assert_eq!(sample, (n * SEEK_FRAMES) as u64);
            assert_eq!(count as usize, (frames - n * SEEK_FRAMES).min(SEEK_FRAMES));
            let at = first_frame + offset as usize;
            assert_eq!(&b[at..at + 2], &[0xFF, 0xF8], "point {n} isn't a frame header");
            // Fixed-blocksize frames carry their frame number (UTF-8 coded) at byte 4.
            let num = if b[at + 4] < 0x80 { b[at + 4] as usize } else { (((b[at + 4] & 0x1F) as usize) << 6) | (b[at + 5] & 0x3F) as usize };
            assert_eq!(num, n, "point {n} leads to frame {num}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
