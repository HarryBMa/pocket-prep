//! Reads weights from a PyTorch checkpoint, without running its pickle code, or from
//! the safetensors file [`convert_to_half`] makes of one.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use half::f16;
use pytorch_reader::PytorchReader;
use safetensors::tensor::{Metadata, TensorView};

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("cannot read the checkpoint: {0}")]
    Read(#[from] pytorch_reader::PytorchError),
    #[error("cannot read a tensor: {0}")]
    Io(#[from] std::io::Error),
    #[error("the weights file is damaged: {0}")]
    Format(String),
    #[error("the checkpoint has no tensor {0}")]
    Missing(String),
    #[error("{name} is {got:?}, expected f32 with {expected} dimensions")]
    Shape {
        name: String,
        got: Vec<usize>,
        expected: usize,
    },
}

/// A weights file open for reading; tensors are read from the file on demand.
pub struct Weights {
    reader: Reader,
}

enum Reader {
    Checkpoint(PytorchReader),
    SafeTensors {
        file: Mutex<File>,
        /// Where the tensor data starts, after the header.
        start: u64,
        header: Metadata,
    },
}

impl Weights {
    /// Opens a `.safetensors` file as such, anything else as a PyTorch checkpoint.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightsError> {
        let path = path.as_ref();
        let reader = if path.extension().is_some_and(|e| e == "safetensors") {
            open_safetensors(path)?
        } else {
            Reader::Checkpoint(PytorchReader::new(path)?)
        };
        Ok(Self { reader })
    }

    pub fn has(&self, name: &str) -> bool {
        match &self.reader {
            Reader::Checkpoint(reader) => reader.get(name).is_some(),
            Reader::SafeTensors { header, .. } => header.info(name).is_some(),
        }
    }

    /// The tensor's values and shape. Values stored in f16 come back as f32, exactly.
    pub fn read(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), WeightsError> {
        let missing = || WeightsError::Missing(name.into());
        let (bytes, shape, half) = match &self.reader {
            Reader::Checkpoint(reader) => {
                let tensor = reader.get(name).ok_or_else(missing)?;
                if tensor.dtype() != pytorch_reader::DType::F32 {
                    return Err(not_f32(name, tensor.shape()));
                }
                (tensor.read()?, tensor.shape().to_vec(), false)
            }
            Reader::SafeTensors {
                file,
                start,
                header,
            } => {
                let info = header.info(name).ok_or_else(missing)?;
                let half = match info.dtype {
                    safetensors::Dtype::F32 => false,
                    safetensors::Dtype::F16 => true,
                    _ => return Err(not_f32(name, &info.shape)),
                };
                let (from, to) = info.data_offsets;
                let mut bytes = vec![0; to - from];
                let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
                file.seek(SeekFrom::Start(start + from as u64))?;
                file.read_exact(&mut bytes)?;
                (bytes, info.shape.clone(), half)
            }
        };
        let values = if half {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b).to_f32())
                .collect()
        } else {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        Ok((values, shape))
    }
}

fn not_f32(name: &str, shape: &[usize]) -> WeightsError {
    WeightsError::Shape {
        name: name.into(),
        got: shape.to_vec(),
        expected: shape.len(),
    }
}

/// Reads only the header: the tensors are read when asked for.
fn open_safetensors(path: &Path) -> Result<Reader, WeightsError> {
    let format = |e: &dyn std::fmt::Display| WeightsError::Format(e.to_string());
    let mut file = File::open(path)?;
    let mut length = [0; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > 100 << 20 {
        return Err(format(&"header too large"));
    }
    let mut json = vec![0; length as usize];
    file.read_exact(&mut json)?;
    let header: Metadata = serde_json::from_slice(&json).map_err(|e| format(&e))?;
    let start = 8 + length;
    let data = header
        .tensors()
        .values()
        .map(|info| info.data_offsets.1)
        .max()
        .unwrap_or(0);
    if file.metadata()?.len() != start + data as u64 {
        return Err(format(&"the file is not as long as its header says"));
    }
    Ok(Reader::SafeTensors {
        file: Mutex::new(file),
        start,
        header,
    })
}

/// Writes the checkpoint's tensors to `out` as safetensors, in f16 except those
/// [`crate::model::stays_f32`] names. The model in [`crate::Precision::Half`] then
/// loads the same values from either file, with half the size on disk.
///
/// The output depends only on the checkpoint, so its hash can be known in advance.
/// The whole output is held in memory before writing; for BS-RoFormer SW that took
/// 0.4 s and 388 MB on an M5 Pro.
pub fn convert_to_half(checkpoint: &Path, out: &Path) -> Result<(), WeightsError> {
    let weights = Weights::open(checkpoint)?;
    let Reader::Checkpoint(reader) = &weights.reader else {
        return Err(WeightsError::Format("not a PyTorch checkpoint".into()));
    };
    let mut names = reader.keys();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        let (values, shape) = weights.read(&name)?;
        let (dtype, bytes) = if crate::model::stays_f32(&name) {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (safetensors::Dtype::F32, bytes)
        } else {
            let bytes = values
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                .collect::<Vec<u8>>();
            (safetensors::Dtype::F16, bytes)
        };
        tensors.push((name, dtype, shape, bytes));
    }
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.as_str(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| WeightsError::Format(e.to_string()))?;
    safetensors::serialize_to_file(views, None::<HashMap<String, String>>, out)
        .map_err(|e| WeightsError::Format(e.to_string()))
}

/// Where a part of the model loads from, and on what device and precision it runs.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub weights: &'a Weights,
    pub device: &'a Device,
    pub dtype: DType,
}

impl Source<'_> {
    pub fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, WeightsError> {
        let (values, shape) = self.weights.read(name)?;
        if shape.len() != D {
            return Err(WeightsError::Shape {
                name: name.into(),
                got: shape,
                expected: D,
            });
        }
        let tensor: Tensor<D> = Tensor::from_data(TensorData::new(values, shape), self.device);
        Ok(tensor.cast(self.dtype))
    }
}
