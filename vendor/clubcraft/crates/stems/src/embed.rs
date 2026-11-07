//! Mood embedding (added by Pocket Prep / DKPRP, not part of upstream ClubCraft).
//!
//! The convolutional encoders of the same HTDemucs network as the separation (not the transformer, 75 % of the
//! weights), pooled into 2,304 numbers (`tools/export_demucs_onnx.py --embedding` writes `models/htdemucs_mood.onnx`,
//! 12 MB). A small head on top turns the
//! vector into a mood (docs/MOOD.md, "Needle Tags"). The pipeline mirrors `tools/needle/mood_embed.py`, which the
//! head was trained with: the middle two minutes, 7.8 s segments without overlap, every segment normalised by the
//! mean and standard deviation of its mono mix (as `demucs.apply_model` does), the vectors averaged.

use crate::backend::onnx::{builder, opt_level, threads};
use crate::model::Manifest;
use crate::stft::Spectro;
use anyhow::{bail, Context, Result};
use clubcraft_audio::decode::AudioBuffer;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// Length of the embedding: mean and standard deviation of encoder layers 2 and 3, both branches (192 * 4 + 384 * 4).
pub const EMBEDDING_LEN: usize = 2304;
/// How much of the track is used, from its middle, in seconds.
pub const EXCERPT_SECONDS: f64 = 120.0;

pub struct Embedder {
    session: Session,
    spectro: Spectro,
    segment: usize,
    freqs: usize,
    frames: usize,
}

impl Embedder {
    /// Loads `<dir>/htdemucs_mood.onnx` (CPU), using the geometry of `<dir>/htdemucs.json`.
    pub fn load(dir: &Path) -> Result<Embedder> {
        let man = Manifest::load(dir, "htdemucs")?;
        let path = dir.join("htdemucs_mood.onnx");
        if !path.exists() {
            bail!("missing {}", path.display());
        }
        let session = builder(false)?
            .with_optimization_level(opt_level())?
            .with_intra_threads(threads())?
            .commit_from_file(&path)
            .with_context(|| format!("load {}", path.display()))?;
        Ok(Embedder { session, spectro: Spectro::new(man.nfft, man.hop_length), segment: man.segment_samples, freqs: man.freqs(), frames: man.spec_frames })
    }

    /// The embedding of a track; `progress` gets 0..1 and returns false to cancel (the error is then "cancelled").
    pub fn embed(&self, audio: &AudioBuffer, progress: &mut dyn FnMut(f32) -> bool) -> Result<Vec<f32>> {
        if audio.sample_rate != 44_100 {
            return self.embed(&clubcraft_audio::resample::resample(audio, 44_100), progress);
        }
        let seg = self.segment;
        let len = audio.left.len();
        if len < seg {
            bail!("track shorter than {:.1} s", seg as f64 / 44_100.0);
        }
        let want = ((EXCERPT_SECONDS * 44_100.0) as usize).min(len);
        let start = (len - want) / 2;
        let chunks = want / seg;
        let mut sum = vec![0f64; EMBEDDING_LEN];
        let mut mix = vec![0f32; 2 * seg];
        let mut spec = vec![0f32; 4 * self.freqs * self.frames];
        for k in 0..chunks {
            let from = start + k * seg;
            let (l, r) = (&audio.left[from..from + seg], &audio.right[from..from + seg]);
            // mean / unbiased standard deviation of the mono mix, like torch's `ref.mean()` / `ref.std()`
            let n = seg as f64;
            let mono = |i: usize| (l[i] as f64 + r[i] as f64) * 0.5;
            let mean = (0..seg).map(mono).sum::<f64>() / n;
            let var = (0..seg).map(|i| (mono(i) - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let sd = var.sqrt() + 1e-8;
            for i in 0..seg {
                mix[i] = ((l[i] as f64 - mean) / sd) as f32;
                mix[seg + i] = ((r[i] as f64 - mean) / sd) as f32;
            }
            self.spectro.spec(&[&mix[..seg], &mix[seg..]], &mut spec);
            let mix_t = Tensor::from_array(([1usize, 2, seg], mix.as_slice()))?;
            let spec_t = Tensor::from_array(([1usize, 4, self.freqs, self.frames], spec.as_slice()))?;
            let outputs = self.session.run(ort::inputs!["mix" => mix_t, "spec" => spec_t]?)?;
            let (shape, data) = outputs["emb"].try_extract_raw_tensor::<f32>()?;
            if shape != [1, EMBEDDING_LEN as i64] {
                bail!("unexpected embedding shape {shape:?}");
            }
            sum.iter_mut().zip(data).for_each(|(s, &v)| *s += v as f64);
            if !progress((k + 1) as f32 / chunks as f32) {
                bail!("cancelled");
            }
        }
        if chunks == 0 {
            bail!("nothing to embed");
        }
        Ok(sum.iter().map(|s| (s / chunks as f64) as f32).collect())
    }
}
