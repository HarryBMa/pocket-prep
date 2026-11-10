//! Model parameters, named as in the checkpoint's `.yaml`.

use std::path::Path;

use serde::Deserialize;

/// What the checkpoint's weights do not say by themselves.
#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub depth: usize,
    pub num_stems: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub time_transformer_depth: usize,
    pub freq_transformer_depth: usize,
    /// STFT bins per band, for one channel.
    pub freqs_per_bands: Vec<usize>,
    pub mask_estimator_depth: usize,
    pub mlp_expansion_factor: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    /// Frames of audio the model takes at a time.
    pub chunk_size: usize,
    pub num_overlap: usize,
    pub sample_rate: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read the model's config: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model config: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    #[error("the model needs an option this backend lacks: {0}")]
    Unsupported(&'static str),
}

/// The parts of the `.yaml` that matter here. Its `audio.hop_length` and `audio.dim_t`
/// are not used by the model; see `model.stft_hop_length`.
#[derive(Deserialize)]
struct YamlFile {
    audio: YamlAudio,
    model: YamlModel,
    inference: YamlInference,
}

#[derive(Deserialize)]
struct YamlAudio {
    chunk_size: usize,
    sample_rate: u32,
}

#[derive(Deserialize)]
struct YamlInference {
    /// Overrides `audio.chunk_size`, as in MSST's `demix`.
    chunk_size: Option<usize>,
    num_overlap: usize,
}

const fn yes() -> bool {
    true
}

const fn four() -> usize {
    4
}

/// Options without a value in the `.yaml` take the model's defaults.
#[derive(Deserialize)]
struct YamlModel {
    dim: usize,
    depth: usize,
    num_stems: usize,
    heads: usize,
    dim_head: usize,
    time_transformer_depth: usize,
    freq_transformer_depth: usize,
    freqs_per_bands: Vec<usize>,
    mask_estimator_depth: usize,
    #[serde(default = "four")]
    mlp_expansion_factor: usize,
    stft_n_fft: usize,
    stft_hop_length: usize,
    stft_win_length: usize,
    #[serde(default)]
    stft_normalized: bool,
    #[serde(default)]
    stereo: bool,
    #[serde(default)]
    linear_transformer_depth: usize,
    #[serde(default)]
    skip_connection: bool,
    #[serde(default = "yes")]
    zero_dc: bool,
}

impl Config {
    /// Reads a checkpoint's `.yaml`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }

    /// Fails on options the model here does not implement, instead of loading a model
    /// that would compute something else.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let YamlFile {
            audio,
            model,
            inference,
        } = serde_yaml_ng::from_str(yaml)?;
        let unsupported = [
            (!model.stereo, "mono"),
            (model.linear_transformer_depth != 0, "linear transformers"),
            (model.skip_connection, "skip connections"),
            (!model.zero_dc, "a DC bin that is kept"),
            (model.stft_normalized, "a normalized STFT"),
            (
                model.stft_win_length != model.stft_n_fft,
                "an STFT window of another size",
            ),
        ];
        if let Some((_, option)) = unsupported.iter().find(|(found, _)| *found) {
            return Err(ConfigError::Unsupported(option));
        }
        Ok(Self {
            dim: model.dim,
            depth: model.depth,
            num_stems: model.num_stems,
            heads: model.heads,
            dim_head: model.dim_head,
            time_transformer_depth: model.time_transformer_depth,
            freq_transformer_depth: model.freq_transformer_depth,
            freqs_per_bands: model.freqs_per_bands,
            mask_estimator_depth: model.mask_estimator_depth,
            mlp_expansion_factor: model.mlp_expansion_factor,
            stft_n_fft: model.stft_n_fft,
            stft_hop_length: model.stft_hop_length,
            chunk_size: inference.chunk_size.unwrap_or(audio.chunk_size),
            num_overlap: inference.num_overlap,
            sample_rate: audio.sample_rate,
        })
    }

    /// Size of each band in the model's input: complex values of both channels.
    pub(crate) fn band_sizes(&self) -> Vec<usize> {
        self.freqs_per_bands.iter().map(|f| f * 2 * 2).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real file's layout: a tuple tag, a hop that must not be used, other sections.
    const YAML: &str = "
audio:
  chunk_size: 588800 #882000
  hop_length: 441 # don't work (use in model)
  sample_rate: 44100
model:
  dim: 256
  depth: 12
  stereo: true
  num_stems: 6
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
    - 2
    - 2
    - 4
  dim_head: 64
  heads: 8
  stft_n_fft: 2048
  stft_hop_length: 512
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
training:
  instruments: ['bass', 'drums']
inference:
  batch_size: 1
  num_overlap: 2
";

    #[test]
    fn reads_the_yaml() {
        let config = Config::from_yaml(YAML).unwrap();
        assert_eq!(config.freqs_per_bands, [2, 2, 4]);
        assert_eq!(config.stft_hop_length, 512);
        assert_eq!(config.chunk_size, 588800);
        assert_eq!(config.num_overlap, 2);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.mlp_expansion_factor, 4);
    }

    #[test]
    fn the_inference_section_can_override_the_chunk() {
        let yaml = YAML.replace("  batch_size: 1", "  chunk_size: 1000");
        assert_eq!(Config::from_yaml(&yaml).unwrap().chunk_size, 1000);
    }

    #[test]
    fn options_the_model_lacks_are_refused() {
        for (from, to) in [
            ("stereo: true", "stereo: false"),
            ("linear_transformer_depth: 0", "linear_transformer_depth: 1"),
            ("stft_normalized: false", "stft_normalized: true"),
            ("stft_win_length: 2048", "stft_win_length: 1024"),
            ("heads: 8", "heads: 8\n  skip_connection: true"),
            ("heads: 8", "heads: 8\n  zero_dc: false"),
        ] {
            let result = Config::from_yaml(&YAML.replace(from, to));
            assert!(matches!(result, Err(ConfigError::Unsupported(_))), "{to}");
        }
    }

    #[test]
    fn a_broken_file_is_an_error() {
        assert!(matches!(
            Config::from_yaml("audio: [1"),
            Err(ConfigError::Yaml(_))
        ));
    }

    #[test]
    fn band_sizes_count_both_channels_and_the_complex_parts() {
        let config = Config::from_yaml(YAML).unwrap();
        assert_eq!(config.band_sizes(), [8, 8, 16]);
    }
}
