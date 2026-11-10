//! The whole model.

use burn::prelude::*;
use burn::tensor::DType;
use watershed_dsp::{Chunking, Stft};

use crate::config::Config;
use crate::layers::{BandSplit, MaskEstimator, RmsNorm, Rotary, Transformer};
use crate::weights::{Source, Weights, WeightsError};

/// How the model computes. The band split, where the quietest signals are, always stays
/// in f32: their squares vanish in f16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// f32 throughout: matches PyTorch to about 1e-5, for validation.
    Full,
    /// f16 after the band split: ~1.6× faster and half the memory on Metal. Masks stay
    /// within 3e-3 of PyTorch's (relative RMS).
    Half,
}

/// Weights that [`BsRoformer::load`] reads as f32 even in [`Precision::Half`]: the
/// band split, and the rotary frequencies, which become angles. All others are cast to
/// f16, so a file can store them that way (see [`crate::convert_to_half`]).
pub(crate) fn stays_f32(name: &str) -> bool {
    name.starts_with("band_split.") || name.ends_with(".rotary_embed.freqs")
}

pub struct BsRoformer {
    band_split: BandSplit,
    layers: Vec<(Transformer, Transformer)>,
    final_norm: RmsNorm,
    pub(crate) mask_estimators: Vec<MaskEstimator>,
    time_freqs: Vec<f32>,
    freq_freqs: Vec<f32>,
    bands: usize,
    dtype: DType,
    pub(crate) stft: Stft,
    pub(crate) chunking: Chunking,
    pub(crate) device: Device,
}

impl BsRoformer {
    pub fn load(
        weights: &Weights,
        config: &Config,
        precision: Precision,
        device: &Device,
    ) -> Result<Self, WeightsError> {
        let source = |dtype| Source {
            weights,
            device,
            dtype,
        };
        let dtype = match precision {
            Precision::Full => DType::F32,
            Precision::Half => DType::F16,
        };
        let inner = source(dtype);
        let layers = (0..config.depth)
            .map(|i| {
                let time = format!("layers.{i}.0");
                let freq = format!("layers.{i}.1");
                Ok((
                    Transformer::load(&inner, &time, config.time_transformer_depth, config)?,
                    Transformer::load(&inner, &freq, config.freq_transformer_depth, config)?,
                ))
            })
            .collect::<Result<_, WeightsError>>()?;
        Ok(Self {
            band_split: BandSplit::load(&source(DType::F32), config)?,
            layers,
            final_norm: RmsNorm::load(&inner, "final_norm")?,
            mask_estimators: (0..config.num_stems)
                .map(|stem| MaskEstimator::load(&inner, stem, config))
                .collect::<Result<_, _>>()?,
            time_freqs: weights.read("layers.0.0.layers.0.0.rotary_embed.freqs")?.0,
            freq_freqs: weights.read("layers.0.1.layers.0.0.rotary_embed.freqs")?.0,
            bands: config.freqs_per_bands.len(),
            dtype,
            stft: Stft::new(config.stft_n_fft, config.stft_hop_length),
            chunking: Chunking {
                size: config.chunk_size,
                overlap: config.num_overlap,
            },
            device: device.clone(),
        })
    }

    /// One complex mask per stem.
    ///
    /// `stft` is (batch, frequency * channel, time, 2). The result is
    /// (batch, stems, frequency * channel, time, 2).
    pub fn forward(&self, stft: Tensor<4>) -> Tensor<5> {
        self.forward_traced(stft, &mut |_, _| {})
    }

    /// [`Self::forward`], handing each layer's output to `trace`.
    pub fn forward_traced(
        &self,
        stft: Tensor<4>,
        trace: &mut dyn FnMut(&str, Tensor<4>),
    ) -> Tensor<5> {
        let [batch, freqs, frames, _] = stft.dims();
        let device = stft.device();
        let x = stft.swap_dims(1, 2).reshape([batch, frames, freqs * 2]);

        let x = self.band_split.forward(x);
        trace("band_split", x.clone());

        let [_, _, _, dim] = x.dims();
        let time_rotary = Rotary::new(&self.time_freqs, frames, &device, self.dtype);
        let freq_rotary = Rotary::new(&self.freq_freqs, self.bands, &device, self.dtype);

        let mut x = x.cast(self.dtype);
        for (i, (time, freq)) in self.layers.iter().enumerate() {
            let y = x.swap_dims(1, 2).reshape([batch * self.bands, frames, dim]);
            let y: Tensor<4> = time
                .forward(y, &time_rotary)
                .reshape([batch, self.bands, frames, dim]);
            trace(&format!("time_{i}"), y.clone());

            let y = y.swap_dims(1, 2).reshape([batch * frames, self.bands, dim]);
            x = freq
                .forward(y, &freq_rotary)
                .reshape([batch, frames, self.bands, dim]);
            trace(&format!("freq_{i}"), x.clone());
        }

        let x = self.final_norm.forward(x);
        trace("final_norm", x.clone());

        let masks = self
            .mask_estimators
            .iter()
            .map(|estimator| estimator.forward(&x))
            .collect();
        let masks: Tensor<4> = Tensor::stack(masks, 1);
        let stems = self.mask_estimators.len();
        masks
            .reshape([batch, stems, frames, freqs, 2])
            .swap_dims(2, 3)
    }
}
