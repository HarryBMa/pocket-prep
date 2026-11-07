//! Needle Tags: the mood embedding of a track (docs/MOOD.md).
//!
//! HTDemucs (the network that makes the stems) cut after its cross-transformer and pooled into 2,048 numbers,
//! run on the middle two minutes of the track. `models/htdemucs_embed.onnx` is made by
//! `tools/export_demucs_onnx.py --embedding`; the code is `clubcraft_stems::embed`. The embedding is saved with the
//! track (`mood_emb`) and a small head (mood.rs) turns it into a mood, so the model runs once per track.
//! Without the model file everything still works: the mood then comes from the nine measured numbers.

use anyhow::{anyhow, Result};
use clubcraft_audio::decode::AudioBuffer;
use clubcraft_stems::embed::Embedder;
use rusqlite::params;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::{db, jobs, mood};

static EMBEDDER: Mutex<Option<Embedder>> = Mutex::new(None);

fn model_dir() -> PathBuf {
    clubcraft_core::paths::models_dir()
}

/// The embedding model is installed (and the stem model, whose geometry it reads).
pub fn available() -> bool {
    let d = model_dir();
    d.join("htdemucs_embed.onnx").exists() && d.join("htdemucs.json").exists()
}

/// DirectML (the GPU, Windows) is about eight times faster than the processor, but a GPU path can silently give wrong
/// numbers (see the stems backend), so it is used only after it agrees with the processor on a test signal.
/// POCKET_PREP_EMBED_CPU forces the processor.
fn load_embedder() -> Result<Embedder> {
    #[cfg(windows)]
    if std::env::var_os("POCKET_PREP_EMBED_CPU").is_none() {
        match Embedder::load_with(&model_dir(), true) {
            Ok(gpu) => match gpu_agrees(&gpu) {
                Ok(true) => return Ok(gpu),
                Ok(false) => eprintln!("mood embedding: DirectML disagrees with the processor; using the processor"),
                Err(e) => eprintln!("mood embedding: DirectML check failed ({e:#}); using the processor"),
            },
            Err(e) => eprintln!("mood embedding: DirectML unavailable ({e:#}); using the processor"),
        }
    }
    Embedder::load(&model_dir())
}

/// A tone with noise, 8.5 s: one segment through the GPU and through a temporary processor session.
#[cfg(windows)]
fn gpu_agrees(gpu: &Embedder) -> Result<bool> {
    let mut x = 12345u32;
    let mut noise = move || {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let n = (8.5 * 44_100.0) as usize;
    let tone = |i: usize, f: f32| (i as f32 * f * std::f32::consts::TAU / 44_100.0).sin();
    let left: Vec<f32> = (0..n).map(|i| 0.3 * tone(i, 110.0) + 0.05 * tone(i, 3000.0) + 0.1 * noise()).collect();
    let right: Vec<f32> = (0..n).map(|i| 0.3 * tone(i, 110.0) + 0.05 * tone(i, 2500.0) + 0.1 * noise()).collect();
    let audio = AudioBuffer::new(44_100, left, right);
    let a = gpu.embed(&audio, &mut |_| true)?;
    let b = Embedder::load(&model_dir())?.embed(&audio, &mut |_| true)?;
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| (*x * *y) as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x * *x) as f64).sum::<f64>().sqrt();
    Ok(dot / (norm(&a) * norm(&b)) > 0.999)
}

/// The embedding of already decoded audio. The model is loaded on first use (a few seconds) and kept.
pub fn embedding(audio: &AudioBuffer, progress: &mut dyn FnMut(f32) -> bool) -> Result<Vec<f32>> {
    let mut g = EMBEDDER.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        *g = Some(load_embedder()?);
    }
    g.as_ref().unwrap().embed(audio, progress)
}

/// Computes and saves the embedding of one track from its file.
pub fn embed_track(track_id: i64, cancel: &dyn Fn() -> bool) -> Result<()> {
    let t = db::track(track_id)?.ok_or_else(|| anyhow!("unknown track"))?;
    let audio = clubcraft_audio::decode::decode_native(std::path::Path::new(t["path"].as_str().unwrap_or("")))?;
    let e = embedding(&audio, &mut |_| !cancel())?;
    db::tx(|c| Ok(c.execute("UPDATE tracks SET mood_emb=? WHERE id=?", params![mood::embedding_to_blob(&e), track_id])?))?;
    Ok(())
}

/// Embeddings for every analysed track that has none yet (older libraries, or tracks analysed before the model was
/// installed), then every automatic mood is worked out again. Tracks whose file can't be read are skipped.
pub fn backfill(h: &jobs::Handle) -> Result<Value> {
    if !available() {
        return Ok(json!({"skipped": "no embedding model"}));
    }
    let ids: Vec<i64> = db::query("SELECT id FROM tracks WHERE deleted=0 AND mood_emb IS NULL AND analysed_at IS NOT NULL ORDER BY id", &[])?
        .iter().filter_map(|r| r["id"].as_i64()).collect();
    let (mut done, mut failed) = (0, 0);
    for (i, id) in ids.iter().enumerate() {
        if h.cancelled() {
            break;
        }
        h.progress(i as f64 / ids.len().max(1) as f64, &format!("{} of {}", i + 1, ids.len()));
        match embed_track(*id, &|| h.cancelled()) {
            Ok(()) => done += 1,
            Err(e) => {
                failed += 1;
                eprintln!("mood embedding for track {id}: {e:#}");
            }
        }
        if done > 0 && done % 25 == 0 {
            db::tx(mood::refresh)?;
        }
    }
    db::tx(mood::refresh)?;
    Ok(json!({"embedded": done, "failed": failed}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedding computed here must match the one the head was trained on (tools/needle/mood_embed.py, PyTorch).
    /// Needs the models and real tracks; POCKET_PREP_EMBED_REF is a JSON file {path: [2048 numbers]}.
    ///   POCKET_PREP_EMBED_REF=ref_emb.json cargo test --release embedding_matches_pytorch -- --ignored --nocapture
    #[test]
    #[ignore]
    fn embedding_matches_pytorch() {
        let refs: Value = serde_json::from_slice(&std::fs::read(std::env::var("POCKET_PREP_EMBED_REF").expect("set POCKET_PREP_EMBED_REF")).unwrap()).unwrap();
        for (path, want) in refs.as_object().unwrap() {
            let audio = clubcraft_audio::decode::decode_native(std::path::Path::new(path)).unwrap();
            let t0 = std::time::Instant::now();
            let got = embedding(&audio, &mut |_| true).unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let want: Vec<f32> = want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let dot: f64 = got.iter().zip(&want).map(|(a, b)| (*a * *b) as f64).sum();
            let na: f64 = got.iter().map(|a| (*a * *a) as f64).sum::<f64>().sqrt();
            let nb: f64 = want.iter().map(|a| (*a * *a) as f64).sum::<f64>().sqrt();
            let cos = dot / (na * nb);
            let ranking = mood::rank_embedding(&got, &[]);
            let ranking_ref = mood::rank_embedding(&want, &[]);
            println!("{path}: cosine {cos:.5}, {secs:.1} s, mood {} ({:.0} %) vs {} ({:.0} %) from PyTorch",
                     mood::MOODS[ranking[0].0].id, ranking[0].1 * 100.0, mood::MOODS[ranking_ref[0].0].id, ranking_ref[0].1 * 100.0);
            assert!(cos > 0.995, "{path}: cosine {cos}");
            assert_eq!(ranking[0].0, ranking_ref[0].0);
        }
    }
}
