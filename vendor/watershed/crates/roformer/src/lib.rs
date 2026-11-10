//! BS-RoFormer on `burn` + `wgpu`, the native separation backend.
//!
//! [`BsRoformer::forward`] takes a spectrogram and returns masks;
//! [`BsRoformer::separate`] does the rest, from a whole song to its stems. Inference
//! only: the layers are plain tensors, with no dropout and no gradients.

mod config;
mod kernels;
mod layers;
mod model;
mod separate;
mod weights;

pub use config::{Config, ConfigError};
pub use kernels::diagnostics;
pub use model::{BsRoformer, Precision};
pub use separate::Cancelled;
pub use weights::{Weights, WeightsError, convert_to_half};

use burn::prelude::Device;
use burn::tensor::DeviceKind;
use cubecl_environment::stream::{StreamPolicy, set_policy};

/// The GPU, on the native graphics API: Metal on macOS, Vulkan elsewhere.
pub fn default_device() -> Device {
    // By default each thread gets a stream with a staging buffer of its own, as big as
    // the masks of a chunk, which no cleanup frees. The app separates on a new thread
    // each time, so that would cost 108 MiB per song for as long as it runs.
    set_policy(StreamPolicy::Single);
    #[cfg(target_os = "macos")]
    return Device::metal(DeviceKind::DefaultDevice);
    #[cfg(not(target_os = "macos"))]
    return Device::vulkan(DeviceKind::DefaultDevice);
}
