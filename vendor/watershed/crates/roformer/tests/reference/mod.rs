//! Compares the model with PyTorch's, from a dump of `tools/reference/dump_tensors.py`.

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use safetensors::SafeTensors;
use watershed_roformer::BsRoformer;

pub fn values(dump: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = dump.tensor(name).with_context(|| name.to_string())?;
    let values = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((values, view.shape().to_vec()))
}

/// The model's input.
pub fn stft(dump: &SafeTensors, device: &Device) -> Result<Tensor<4>> {
    let (values, shape) = values(dump, "stft")?;
    let shape: [usize; 4] = shape.as_slice().try_into()?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

pub fn floats<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .cast(DType::F32)
        .into_data()
        .try_to_vec()
        .expect("cast to f32")
}

/// RMS error relative to the reference's RMS.
pub fn error(got: &[f32], reference: &[f32]) -> f64 {
    let energy: f64 = reference.iter().map(|v| (*v as f64).powi(2)).sum();
    let error: f64 = got
        .iter()
        .zip(reference)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    (error / energy).sqrt()
}

/// The relative error of each layer's output, in order, and last of the masks.
pub fn layer_errors(
    model: &BsRoformer,
    dump: &SafeTensors,
    stems: usize,
    device: &Device,
) -> Result<Vec<(String, f64)>> {
    let input = stft(dump, device)?;
    let [_, freqs, frames, _] = input.dims();
    let mut errors = Vec::new();
    let masks = floats(model.forward_traced(input, &mut |name, tensor| {
        let (reference, _) = values(dump, name).expect("in the dump");
        errors.push((name.to_string(), error(&floats(tensor), &reference)));
    }));

    // PyTorch's masks are (time, frequency * 2), ours (frequency, time, 2).
    let mut reference = Vec::with_capacity(masks.len());
    for stem in 0..stems {
        let (mask, _) = values(dump, &format!("mask_{stem}"))?;
        for f in 0..freqs {
            for t in 0..frames {
                reference.extend_from_slice(&mask[(t * freqs + f) * 2..][..2]);
            }
        }
    }
    errors.push(("masks".to_string(), error(&masks, &reference)));
    Ok(errors)
}
