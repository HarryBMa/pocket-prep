//! The building blocks, as in `bs_roformer.py`.

use std::f32::consts::FRAC_1_SQRT_2;

use burn::prelude::*;
use burn::tensor::activation::{glu, sigmoid, softmax};
use burn::tensor::{DType, TensorData};

use crate::config::Config;
use crate::kernels::{self, Kernel, MAX_SCORES, RotaryTables, matmul};
use crate::weights::{Source, WeightsError};

type Result<T> = std::result::Result<T, WeightsError>;

/// `nn.Linear`. PyTorch stores the weight as (out, in); burn wants (in, out).
pub struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}

impl Linear {
    pub fn load(source: &Source, name: &str) -> Result<Self> {
        let bias = format!("{name}.bias");
        Ok(Self {
            weight: source
                .tensor::<2>(&format!("{name}.weight"))?
                .swap_dims(0, 1),
            bias: if source.weights.has(&bias) {
                Some(source.tensor(&bias)?)
            } else {
                None
            },
        })
    }

    /// Applied to all leading dimensions as one matrix: burn runs a 3D input as a
    /// batched matmul, which with many short batches took 1.7× as long on Vulkan.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let mut dims = x.dims();
        let features = dims[D - 1];
        let rows = x.shape().num_elements() / features;
        let outputs = self.weight.dims()[1];
        let kernel = Kernel::for_shape(rows, features, outputs);
        let y = matmul(x.reshape([rows, features]), self.weight.clone(), kernel);
        let y = match &self.bias {
            Some(bias) => y + bias.clone().unsqueeze(),
            None => y,
        };
        dims[D - 1] = outputs;
        y.reshape(dims)
    }
}

/// Unit length over the last dimension, then a learned gain per feature.
pub struct RmsNorm {
    /// The gain, times the root of the dimension as PyTorch's norm is scaled by.
    gamma: Tensor<1>,
    epsilon: f32,
}

impl RmsNorm {
    pub fn load(source: &Source, name: &str) -> Result<Self> {
        let gamma: Tensor<1> = source.tensor(&format!("{name}.gamma"))?;
        let scale = (gamma.dims()[0] as f32).sqrt();
        // PyTorch's 1e-12 is zero in f16.
        let epsilon = if source.dtype == DType::F16 {
            1e-6
        } else {
            1e-12
        };
        Ok(Self {
            gamma: gamma * scale,
            epsilon,
        })
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let norm = x
            .clone()
            .square()
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(self.epsilon);
        x / norm * self.gamma.clone().unsqueeze()
    }
}

/// The rotary position embedding for one sequence length: every pair of features
/// turns by an angle that grows with the position.
pub struct Rotary {
    /// (1, 1, sequence, pairs, 2): `[cos, cos]` and `[-sin, sin]`, so that a pair
    /// `(a, b)` turns into `(a, b) * cos + (b, a) * sin`.
    cos: Tensor<5>,
    sin: Tensor<5>,
    /// The same angles for our own kernel, when the device runs it.
    tables: Option<RotaryTables>,
}

impl Rotary {
    pub fn new(freqs: &[f32], length: usize, device: &Device, dtype: DType) -> Self {
        let pairs = length * freqs.len();
        let (mut cos, mut sin) = (Vec::with_capacity(pairs * 2), Vec::with_capacity(pairs * 2));
        let (mut cos_table, mut sin_table) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
        for position in 0..length {
            for freq in freqs {
                let (s, c) = (position as f32 * freq).sin_cos();
                cos.extend([c, c]);
                sin.extend([-s, s]);
                cos_table.push(c);
                sin_table.push(s);
            }
        }
        let shape = [1, 1, length, freqs.len(), 2];
        let table =
            |values: Vec<f32>| Tensor::<1>::from_data(TensorData::new(values, [pairs]), device);
        Self {
            cos: Tensor::<5>::from_data(TensorData::new(cos, shape), device).cast(dtype),
            sin: Tensor::<5>::from_data(TensorData::new(sin, shape), device).cast(dtype),
            tables: RotaryTables::new(&table(cos_table), &table(sin_table)),
        }
    }

    /// `x` is (batch, heads, sequence, dim).
    fn rotate(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, heads, sequence, dim] = x.dims();
        let x: Tensor<5> = x.reshape([batch, heads, sequence, dim / 2, 2]);
        (x.clone().flip([4]) * self.sin.clone() + x * self.cos.clone())
            .reshape([batch, heads, sequence, dim])
    }
}

/// Multi-head self-attention with one learned gate per head.
struct Attention {
    norm: RmsNorm,
    to_qkv: Linear,
    to_gates: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl Attention {
    fn load(source: &Source, name: &str, config: &Config) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(source, &format!("{name}.norm"))?,
            to_qkv: Linear::load(source, &format!("{name}.to_qkv"))?,
            to_gates: Linear::load(source, &format!("{name}.to_gates"))?,
            to_out: Linear::load(source, &format!("{name}.to_out.0"))?,
            heads: config.heads,
            dim_head: config.dim_head,
        })
    }

    /// `x` is (batch, sequence, dim).
    fn forward(&self, x: Tensor<3>, rotary: &Rotary) -> Tensor<3> {
        let x = self.norm.forward(x);
        let qkv = self.to_qkv.forward(x.clone());
        let gates = self.to_gates.forward(x);
        let fused = rotary
            .tables
            .as_ref()
            .and_then(|tables| kernels::attention(&qkv, &gates, tables, self.heads));
        let out = fused.unwrap_or_else(|| self.attention(qkv, gates, rotary));
        self.to_out.forward(out)
    }

    /// The same as [`kernels::attention`], with burn's ops.
    fn attention(&self, qkv: Tensor<3>, gates: Tensor<3>, rotary: &Rotary) -> Tensor<3> {
        let [batch, sequence, _] = qkv.dims();
        let qkv: Tensor<5> = qkv.reshape([batch, sequence, 3, self.heads, self.dim_head]);
        let part = |i| {
            qkv.clone()
                .narrow(2, i, 1)
                .squeeze_dim::<4>(2)
                .swap_dims(1, 2)
        };
        let (q, k, v) = (rotary.rotate(part(0)), rotary.rotate(part(1)), part(2));
        let step = (MAX_SCORES / (self.heads * sequence * sequence)).clamp(1, batch);
        let kernel = Kernel::for_attention(sequence);
        let out = (0..batch)
            .step_by(step)
            .map(|start| {
                // Not burn's `attention`: without autotune it ends in burn's matmul, which
                // can panic on GPUs without matrix instructions.
                let slice = |t: &Tensor<4>| t.clone().narrow(0, start, step.min(batch - start));
                let scale = (self.dim_head as f32).sqrt().recip();
                let scores = matmul(slice(&q) * scale, slice(&k).swap_dims(2, 3), kernel);
                let weights = kernels::softmax(&scores).unwrap_or_else(|| softmax(scores, 3));
                matmul(weights, slice(&v), kernel)
            })
            .collect();
        let out = Tensor::cat(out, 0);
        let gates = sigmoid(gates).swap_dims(1, 2).unsqueeze_dim(3);
        (out * gates)
            .swap_dims(1, 2)
            .reshape([batch, sequence, self.heads * self.dim_head])
    }
}

/// A norm, then two linear layers with a GELU between them.
struct FeedForward {
    norm: RmsNorm,
    expand: Linear,
    contract: Linear,
}

impl FeedForward {
    /// GELU(y) = y (1 + erf(y / sqrt 2)) / 2 costs five GPU kernels. With z = y / sqrt 2
    /// it is z (1 + erf z) / sqrt 2, and both divisions fit in the weights around it.
    fn load(source: &Source, name: &str) -> Result<Self> {
        let mut expand = Linear::load(source, &format!("{name}.net.1"))?;
        expand.weight = expand.weight * FRAC_1_SQRT_2;
        expand.bias = expand.bias.map(|bias| bias * FRAC_1_SQRT_2);
        let mut contract = Linear::load(source, &format!("{name}.net.4"))?;
        contract.weight = contract.weight * FRAC_1_SQRT_2;
        Ok(Self {
            norm: RmsNorm::load(source, &format!("{name}.net.0"))?,
            expand,
            contract,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let z = self.expand.forward(self.norm.forward(x));
        self.contract.forward(z.clone() * (z.erf() + 1.0))
    }
}

/// Attention and feed-forward layers, each added to its input.
pub struct Transformer {
    layers: Vec<(Attention, FeedForward)>,
}

impl Transformer {
    pub fn load(source: &Source, name: &str, depth: usize, config: &Config) -> Result<Self> {
        let layers = (0..depth)
            .map(|i| {
                Ok((
                    Attention::load(source, &format!("{name}.layers.{i}.0"), config)?,
                    FeedForward::load(source, &format!("{name}.layers.{i}.1"))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, mut x: Tensor<3>, rotary: &Rotary) -> Tensor<3> {
        for (attention, feed_forward) in &self.layers {
            x = attention.forward(x.clone(), rotary) + x;
            x = feed_forward.forward(x.clone()) + x;
        }
        x
    }
}

/// Turns each frequency band into one feature vector.
pub struct BandSplit {
    bands: Vec<(usize, RmsNorm, Linear)>,
}

impl BandSplit {
    pub fn load(source: &Source, config: &Config) -> Result<Self> {
        let bands = config
            .band_sizes()
            .into_iter()
            .enumerate()
            .map(|(i, size)| {
                let name = format!("band_split.to_features.{i}");
                Ok((
                    size,
                    RmsNorm::load(source, &format!("{name}.0"))?,
                    Linear::load(source, &format!("{name}.1"))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self { bands })
    }

    /// (batch, time, features) to (batch, time, bands, dim).
    pub fn forward(&self, x: Tensor<3>) -> Tensor<4> {
        let mut start = 0;
        let bands = self
            .bands
            .iter()
            .map(|(size, norm, linear)| {
                let band = x.clone().narrow(2, start, *size);
                start += size;
                linear.forward(norm.forward(band))
            })
            .collect();
        Tensor::stack(bands, 2)
    }
}

/// Turns each band's features into its part of one stem's complex mask.
pub struct MaskEstimator {
    /// Per band, the layers of its network, with a tanh between them.
    bands: Vec<Vec<Linear>>,
}

impl MaskEstimator {
    pub fn load(source: &Source, stem: usize, config: &Config) -> Result<Self> {
        let bands = (0..config.freqs_per_bands.len())
            .map(|band| {
                (0..config.mask_estimator_depth)
                    .map(|layer| {
                        // Sequential(Linear, Tanh, Linear, ...)
                        let name =
                            format!("mask_estimators.{stem}.to_freqs.{band}.0.{}", layer * 2);
                        Linear::load(source, &name)
                    })
                    .collect()
            })
            .collect::<Result<_>>()?;
        Ok(Self { bands })
    }

    /// (batch, time, bands, dim) to (batch, time, mask values).
    pub fn forward(&self, x: &Tensor<4>) -> Tensor<3> {
        let bands = self
            .bands
            .iter()
            .enumerate()
            .map(|(i, layers)| {
                let mut band: Tensor<3> = x.clone().narrow(2, i, 1).squeeze_dim(2);
                for (j, layer) in layers.iter().enumerate() {
                    if j > 0 {
                        band = band.tanh();
                    }
                    band = layer.forward(band);
                }
                glu(band, 2)
            })
            .collect();
        Tensor::cat(bands, 2)
    }
}
