//! Needle Tags: the mood embedding of a track (docs/MOOD.md).
//!
//! The convolutional encoders of HTDemucs (the network that makes the stems), pooled into 2,304 numbers, run on the
//! middle two minutes of the track. `models/htdemucs_mood.onnx` (12 MB) is made by
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
    d.join("htdemucs_mood.onnx").exists() && d.join("htdemucs.json").exists()
}

/// The embedding of already decoded audio, on the processor (the encoders are small: under a second per track). The
/// model is loaded on first use and kept.
pub fn embedding(audio: &AudioBuffer, progress: &mut dyn FnMut(f32) -> bool) -> Result<Vec<f32>> {
    let mut g = EMBEDDER.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        *g = Some(Embedder::load(&model_dir())?);
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
    let ids: Vec<i64> = db::query("SELECT id FROM tracks WHERE deleted=0 AND (mood_emb IS NULL OR length(mood_emb) <> ?) AND analysed_at IS NOT NULL ORDER BY id",
                         &[json!(mood::EMB_LEN * 4)])?
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
