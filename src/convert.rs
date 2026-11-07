//! "Convert for the deck": a copy of a track that the DECK001 can play.
//!
//! The deck decodes MP3, FLAC and 16-bit PCM WAV, so a 24-bit WAV, an AIFF or an M4A can't be played on it (CDJs play
//! them). The original is never touched or replaced in the library. Converting makes a 16-bit FLAC copy at the track's
//! own sample rate (TPDF dither, lossless from there on) in `converted/<track id>/<name>.flac` in the data folder, and
//! the export writes that copy to the drive instead of the original. Removing the copy puts the original back.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::{config, db};

pub fn dir() -> PathBuf {
    config::data_dir().join("converted")
}

/// Where the copy of a track lives (it keeps the original file's name, with a .flac ending).
pub fn copy_path(track_id: i64, original: &Path) -> PathBuf {
    let stem = original.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| track_id.to_string());
    dir().join(track_id.to_string()).join(format!("{stem}.flac"))
}

/// The converted copy, if there is one that is still newer than the original.
pub fn existing(track_id: i64, original: &Path) -> Option<PathBuf> {
    let p = copy_path(track_id, original);
    let (c, o) = (std::fs::metadata(&p).ok()?, std::fs::metadata(original).ok());
    let stale = match (c.modified().ok(), o.and_then(|o| o.modified().ok())) {
        (Some(c), Some(o)) => o > c,
        _ => false,
    };
    (c.len() > 0 && !stale).then_some(p)
}

pub fn remove(track_id: i64) {
    let _ = std::fs::remove_dir_all(dir().join(track_id.to_string()));
}

/// Makes the copy of one track. `progress` gets 0..1 and a message; `cancel` stops it between the steps.
pub fn convert(track_id: i64, progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<PathBuf> {
    let t = db::track(track_id)?.ok_or_else(|| anyhow!("unknown track"))?;
    let original = PathBuf::from(t["path"].as_str().unwrap_or(""));
    if !original.exists() {
        return Err(anyhow!("The file is missing: {}", original.display()));
    }
    progress(0.02, "Reading the file");
    let audio = clubcraft_audio::decode::decode_native(&original).with_context(|| format!("decoding {}", original.display()))?;
    if cancel() {
        return Err(anyhow!("cancelled"));
    }
    progress(0.4, "Reducing to 16 bit and encoding");
    let bytes = flac_bytes(&audio.left, &audio.right, audio.sample_rate as usize)?;
    drop(audio);
    if cancel() {
        return Err(anyhow!("cancelled"));
    }
    let out = copy_path(track_id, &original);
    std::fs::create_dir_all(out.parent().unwrap())?;
    let tmp = out.with_extension("flac.tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &out)?;
    progress(1.0, "Done");
    Ok(out)
}

/// 16-bit stereo FLAC from float samples: triangular dither (plus or minus one step), then lossless.
pub fn flac_bytes(left: &[f32], right: &[f32], rate: usize) -> Result<Vec<u8>> {
    let mut seed = 0x1234_5678u32;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed as f32 / u32::MAX as f32 };
    let mut pcm = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right) {
        for c in [l, r] {
            let d = rnd() - rnd();
            pcm.push((c * 32767.0 + d).round().clamp(-32768.0, 32767.0) as i32);
        }
    }
    let mut config = flacenc::config::Encoder::default();
    config.block_size = 4096;
    let config = flacenc::error::Verify::into_verified(config).map_err(|e| anyhow!("flac config: {e:?}"))?;
    let source = flacenc::source::MemSource::from_samples(&pcm, 2, 16, rate);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, 4096).map_err(|e| anyhow!("flac encode: {e:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    flacenc::component::BitRepr::write(&stream, &mut sink).map_err(|e| anyhow!("flac write: {e:?}"))?;
    Ok(sink.into_inner())
}

/// What the deck gets for a track: the converted copy when there is one, otherwise the original.
pub fn deck_file(track_id: i64, original: &Path) -> PathBuf {
    existing(track_id, original).unwrap_or_else(|| original.to_path_buf())
}

/// For the track list and panel: why the deck can't play this track (None when it can, or when a copy exists) and
/// whether a copy exists. The WAV header check is cached by file size and time, because the list asks for every track.
pub fn state(track_id: i64, original: &Path) -> Value {
    if existing(track_id, original).is_some() {
        return json!({"deck_problem": Value::Null, "deck_copy": true});
    }
    use std::sync::Mutex;
    static CACHE: Mutex<Option<std::collections::HashMap<(PathBuf, u64, u64), Option<String>>>> = Mutex::new(None);
    let key = std::fs::metadata(original).ok().map(|m| (original.to_path_buf(), m.len(),
        m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0)));
    let problem = match key {
        None => None,
        Some(k) => {
            let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
            g.get_or_insert_with(Default::default).entry(k).or_insert_with(|| crate::export::usb::deck_problem(original)).clone()
        }
    };
    json!({"deck_problem": problem, "deck_copy": false})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The copy is 16-bit FLAC at the same rate, and within two steps of the original samples.
    #[test]
    fn flac_copy_is_16_bit_and_faithful() {
        let rate = 48_000usize;
        let left: Vec<f32> = (0..rate).map(|i| 0.5 * (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin()).collect();
        let right: Vec<f32> = left.iter().map(|v| -v * 0.5).collect();
        let bytes = flac_bytes(&left, &right, rate).unwrap();
        let mut r = claxon::FlacReader::new(std::io::Cursor::new(bytes)).unwrap();
        let info = r.streaminfo();
        assert_eq!((info.channels, info.bits_per_sample, info.sample_rate, info.samples), (2, 16, rate as u32, Some(rate as u64)));
        let samples: Vec<i32> = r.samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), rate * 2);
        for i in (0..rate).step_by(97) {
            assert!((samples[2 * i] as f32 - left[i] * 32767.0).abs() <= 2.0, "left {i}");
            assert!((samples[2 * i + 1] as f32 - right[i] * 32767.0).abs() <= 2.0, "right {i}");
        }
    }
}
