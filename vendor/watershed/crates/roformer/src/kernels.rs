//! GPU kernels of our own, for the attention, where burn's generic ops spent most of the
//! time on an RTX 3050: its softmax is five passes over memory, its rotary flip and head
//! reshapes are copies, and its flash attention ran at about a tenth of the card's speed.
//!
//! Each entry point returns `None` where it cannot run (another backend, such as `flex`
//! in the tests, or a GPU without plane operations of one fixed size), and the caller
//! then takes burn's ops.

use std::sync::atomic::{AtomicUsize, Ordering};

use burn::prelude::{Device, Tensor};
use burn::tensor::{DType, f16};
use burn_cubecl::Cube;
use burn_cubecl::cubecl::features::Plane;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind};
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim, TensorArg};
use burn_cubecl::tensor::CubeTensor;
use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek::std::InputBinding;

/// Most attention scores held at once: 128 MiB in f16. Longer batches go in slices,
/// which costs no measurable time and keeps the GPU memory down.
pub(crate) const MAX_SCORES: usize = 1 << 26;

/// Rows handled by each cube in the softmax, one plane per row.
const ROWS_PER_CUBE: u32 = 4;

/// Units per cube in the element-wise kernels.
const UNITS_PER_CUBE: u32 = 256;

/// How often each path ran, for [`diagnostics`]: on another GPU the fast kernels may
/// not run, and a test log should say so.
static FUSED_ATTENTION: AtomicUsize = AtomicUsize::new(0);
static FALLBACK_ATTENTION: AtomicUsize = AtomicUsize::new(0);
static FAST_MATMUL: AtomicUsize = AtomicUsize::new(0);
static FALLBACK_MATMUL: AtomicUsize = AtomicUsize::new(0);
static BURN_MATMUL: AtomicUsize = AtomicUsize::new(0);

fn count(counter: &AtomicUsize) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// The GPU behind `device` and which of this module's kernels have run on it so far, as
/// lines of text for a test log.
pub fn diagnostics(device: &Device) -> String {
    let mut lines = Vec::new();
    let probe = Tensor::<1>::zeros([1], device).try_into_primitive::<Cube>();
    match probe {
        Ok(tensor) => {
            let properties = tensor.client.properties();
            let hardware = &properties.hardware;
            let features = &properties.features;
            lines.push(format!("GPU: {}", properties.identity.name));
            lines.push(format!(
                "plane size: {}..{}, plane operations: {}",
                hardware.plane_size_min,
                hardware.plane_size_max,
                features.plane.contains(Plane::Ops)
            ));
            lines.push(format!(
                "matrix instructions (cmma) configurations: {}",
                features.matmul.cmma.len()
            ));
        }
        Err(_) => lines.push("GPU: none (not a cubecl device)".to_string()),
    }
    let get = |counter: &AtomicUsize| counter.load(Ordering::Relaxed);
    lines.push(format!(
        "attention: {} fast, {} with burn's ops",
        get(&FUSED_ATTENTION),
        get(&FALLBACK_ATTENTION)
    ));
    lines.push(format!(
        "matmul: {} with matrix instructions, {} without, {} with burn's",
        get(&FAST_MATMUL),
        get(&FALLBACK_MATMUL),
        get(&BURN_MATMUL)
    ));
    lines.join(
        "
",
    )
}

mod gpu {
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::prelude::*;

    /// The index of this unit in a grid laid out by [`super::grid`].
    #[cube]
    fn position() -> u32 {
        (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X
    }

    /// Softmax of each row of `input`, `length` values long, with one plane per row:
    /// read once for the maximum, once for the sum and once more to write. The row stays
    /// in the cache, so that is one trip to memory each way. Sums in f32.
    #[cube(launch)]
    pub fn softmax_rows<F: Float>(
        input: &Tensor<F>,
        output: &mut Tensor<F>,
        rows: u32,
        length: u32,
    ) {
        let row = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_Y + UNIT_POS_Y;
        if row < rows {
            let start = (row * length) as usize;
            let mut max = f32::new(-3.0e38_f32);
            for i in range_stepped(UNIT_POS_X, length, PLANE_DIM) {
                max = f32::max(max, f32::cast_from(input[start + i as usize]));
            }
            let max = plane_max(max);
            let mut sum = f32::new(0.0_f32);
            for i in range_stepped(UNIT_POS_X, length, PLANE_DIM) {
                sum += f32::exp(f32::cast_from(input[start + i as usize]) - max);
            }
            let scale = f32::new(1.0_f32) / plane_sum(sum);
            for i in range_stepped(UNIT_POS_X, length, PLANE_DIM) {
                let value = f32::exp(f32::cast_from(input[start + i as usize]) - max) * scale;
                output[start + i as usize] = F::cast_from(value);
            }
        }
    }

    /// From `to_qkv`'s output, (batch, sequence, 3, heads, dim), to q, k and v as
    /// (batch, heads, sequence, dim), with the rotary embedding on q and k and the
    /// attention's scale on q. One unit per pair of features of one head.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn split_heads<F: Float>(
        qkv: &Tensor<F>,
        cos: &Tensor<f32>,
        sin: &Tensor<f32>,
        q: &mut Tensor<F>,
        k: &mut Tensor<F>,
        v: &mut Tensor<F>,
        pairs: u32,
        sequence: u32,
        heads: u32,
        scale: f32,
    ) {
        let index = position();
        if index < pairs {
            let half = q.shape(3) as u32 / 2;
            let pair = index % half;
            let head_position = index / half;
            let head = head_position % heads;
            let token = head_position / heads;
            let step = token % sequence;
            let item = token / sequence;
            let dim = half * 2;

            let source = ((token * 3 * heads + head) * dim + pair * 2) as usize;
            let target = (((item * heads + head) * sequence + step) * dim + pair * 2) as usize;
            let angle = (step * half + pair) as usize;
            let (c, s) = (cos[angle], sin[angle]);
            let stride = (heads * dim) as usize;

            let a = f32::cast_from(qkv[source]);
            let b = f32::cast_from(qkv[source + 1]);
            q[target] = F::cast_from((a * c - b * s) * scale);
            q[target + 1] = F::cast_from((b * c + a * s) * scale);
            let a = f32::cast_from(qkv[source + stride]);
            let b = f32::cast_from(qkv[source + stride + 1]);
            k[target] = F::cast_from(a * c - b * s);
            k[target + 1] = F::cast_from(b * c + a * s);
            v[target] = qkv[source + 2 * stride];
            v[target + 1] = qkv[source + 2 * stride + 1];
        }
    }

    /// Attention output for some batch items, (items, heads, sequence, dim), times the
    /// sigmoid of each head's gate, into its place in (batch, sequence, heads * dim).
    /// `first` is the batch item the slice starts at.
    #[cube(launch)]
    pub fn merge_heads<F: Float>(
        attention: &Tensor<F>,
        gates: &Tensor<F>,
        output: &mut Tensor<F>,
        values: u32,
        sequence: u32,
        heads: u32,
        first: u32,
    ) {
        let index = position();
        if index < values {
            let dim = attention.shape(3) as u32;
            let feature = index % dim;
            let row = index / dim;
            let step = row % sequence;
            let item_head = row / sequence;
            let head = item_head % heads;
            let item = item_head / heads + first;

            // Gates and output share their rows: one per item, step and head.
            let out_row = (item * sequence + step) * heads + head;
            let gate = f32::cast_from(gates[out_row as usize]);
            let gate = f32::new(1.0_f32) / (f32::new(1.0_f32) + f32::exp(-gate));
            let value = f32::cast_from(attention[index as usize]) * gate;
            output[(out_row * dim + feature) as usize] = F::cast_from(value);
        }
    }
}

/// The tensor's cubecl handle, if it is one this module's kernels can read: on a cubecl
/// device, contiguous, in f16 or f32.
fn primitive<const D: usize>(tensor: &Tensor<D>) -> Option<CubeTensor> {
    let tensor = tensor.clone().try_into_primitive::<Cube>().ok()?;
    (tensor.is_contiguous() && matches!(tensor.dtype, DType::F16 | DType::F32)).then_some(tensor)
}

/// Whether the device has plane operations, all of one size, as the softmax needs.
fn has_planes(tensor: &CubeTensor) -> bool {
    let hardware = &tensor.client.properties().hardware;
    tensor.client.features().plane.contains(Plane::Ops)
        && hardware.plane_size_min == hardware.plane_size_max
}

/// A new contiguous tensor on `like`'s device.
fn empty(like: &CubeTensor, shape: &[usize], dtype: DType) -> CubeTensor {
    let size = shape.iter().product::<usize>() * dtype.size();
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.to_vec().into(),
        like.client.empty(size),
        dtype,
    )
}

fn arg(tensor: &CubeTensor) -> TensorArg {
    // SAFETY: the shape and strides are the tensor's own.
    unsafe {
        TensorArg::from_raw_parts(
            tensor.handle.clone(),
            tensor.meta.strides().clone(),
            tensor.meta.shape().clone(),
        )
    }
}

/// Cubes for `units` units, `per_cube` each, in two dimensions, since one dimension
/// is limited to 65535 cubes.
fn grid(units: usize, per_cube: u32) -> CubeCount {
    let cubes = (units as u32).div_ceil(per_cube);
    let x = cubes.clamp(1, 32768);
    CubeCount::Static(x, cubes.div_ceil(x), 1)
}

/// Runs the generic kernel `$kernel` for the element type of `$dtype`.
macro_rules! launch {
    ($kernel:ident, $dtype:expr, $($arg:expr),* $(,)?) => {
        match $dtype {
            DType::F16 => gpu::$kernel::launch::<f16>($($arg),*),
            _ => gpu::$kernel::launch::<f32>($($arg),*),
        }
    };
}

/// Softmax over the last dimension.
pub fn softmax<const D: usize>(x: &Tensor<D>) -> Option<Tensor<D>> {
    let input = primitive(x).filter(has_planes)?;
    let length = x.dims()[D - 1];
    let rows = x.shape().num_elements() / length;
    let output = empty(&input, &x.dims(), input.dtype);
    let plane = input.client.properties().hardware.plane_size_max;
    launch!(
        softmax_rows,
        input.dtype,
        &input.client,
        grid(rows, ROWS_PER_CUBE),
        CubeDim::new_2d(plane, ROWS_PER_CUBE),
        arg(&input),
        arg(&output),
        rows as u32,
        length as u32,
    );
    Some(Tensor::from_primitive::<Cube>(output))
}

/// Which accelerated matmul to run. Picked by hand from what burn's autotune chose on an
/// RTX 3050, so that no run pays the ~30 s of tuning, nor the ~1.6× time without it.
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    /// Small or narrow products.
    Simple,
    /// Large products whose output is narrower than the input.
    Ordered,
    /// Large products, double buffered.
    Double,
}

impl Kernel {
    /// The kernel for a product of `rows` by `inner` by `columns`.
    pub fn for_shape(rows: usize, inner: usize, columns: usize) -> Self {
        if rows < 16384 || columns <= 64 {
            Self::Simple
        } else if columns >= 1024 && inner <= 256 {
            Self::Double
        } else {
            Self::Ordered
        }
    }

    /// The kernel for attention scores over `sequence` steps. Each product is small, but
    /// there are many: what the autotune chose for the time layers' 1151 steps and the
    /// frequency layers' 62 bands.
    pub fn for_attention(sequence: usize) -> Self {
        if sequence > 256 {
            Self::Double
        } else {
            Self::Simple
        }
    }

    fn strategy(self) -> cubek::matmul::strategy::Strategy {
        use cubek::matmul::multi_level::Strategy;
        use cubek::matmul::multi_level::components::tile::TileMatmulKind;
        use cubek::matmul::multi_level::routines::batch::double_buffering::DoubleBufferingArgs;
        use cubek::matmul::multi_level::routines::batch::ordered_double_buffering::OrderedSelectionArgs;
        use cubek::matmul::multi_level::routines::batch::simple::SimpleArgs;
        use cubek::matmul::routine::BlueprintStrategy;

        let tile_matmul = TileMatmulKind::Cmma;
        match self {
            Self::Simple => Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                tile_matmul,
            })),
            Self::Ordered => {
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                    tile_matmul,
                }))
            }
            Self::Double => {
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                    tile_matmul,
                }))
            }
        }
        .into()
    }
}

/// Matmuls that need no matrix instructions, tried in order when `Kernel`'s cannot run:
/// without autotune, burn's own matmul takes one fixed kernel, which panicked on a Radeon
/// Vega for asking more shared memory than it has. On the RTX 3050 with its matrix
/// instructions left out, `double_unit` with small tiles was the fastest of these at
/// every shape of the model, 5-13× ahead of `simple_unit`. The last needs no shared
/// memory at all.
fn fallback_strategies() -> [cubek::matmul::strategy::Strategy; 4] {
    use cubek::matmul::multi_level::Strategy;
    use cubek::matmul::multi_level::routines::TileSizeSelection;
    use cubek::matmul::multi_level::routines::batch::double_unit::DoubleUnitSelectionArgs;
    use cubek::matmul::multi_level::routines::batch::simple_unit::SimpleUnitSelectionArgs;
    use cubek::matmul::routine::BlueprintStrategy;

    let double = |tile_size| {
        Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
            tile_size,
        }))
        .into()
    };
    [
        double(TileSizeSelection::MinTileSize),
        double(TileSizeSelection::MaxTileSize),
        Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        }))
        .into(),
        Strategy::Naive.into(),
    ]
}

/// `lhs @ rhs` with the given kernel, or the first of [`fallback_strategies`] that runs
/// where it cannot (a GPU without matrix instructions). burn's matmul only runs off
/// cubecl, as on `flex` in the tests.
pub fn matmul<const D: usize>(lhs: Tensor<D>, rhs: Tensor<D>, kernel: Kernel) -> Tensor<D> {
    let (Ok(l), Ok(r)) = (
        lhs.clone().try_into_primitive::<Cube>(),
        rhs.clone().try_into_primitive::<Cube>(),
    ) else {
        count(&BURN_MATMUL);
        return lhs.matmul(rhs);
    };
    let mut shape = lhs.dims();
    shape[D - 1] = rhs.dims()[D - 1];
    let out = empty(&l, &shape, l.dtype);
    let elem = |dtype| match dtype {
        DType::F16 => ElemType::Float(FloatKind::F16),
        _ => ElemType::Float(FloatKind::F32),
    };
    let (lhs_elem, rhs_elem, out_elem) = (elem(l.dtype), elem(r.dtype), elem(l.dtype));
    let strategies = std::iter::once(kernel.strategy()).chain(fallback_strategies());
    for (attempt, strategy) in strategies.enumerate() {
        let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: lhs_elem,
            rhs: rhs_elem,
            out: out_elem,
        });
        let launched = cubek::matmul::launch::launch_ref(
            &strategy,
            &l.client,
            InputBinding::new(l.clone().binding(), lhs_elem),
            InputBinding::new(r.clone().binding(), rhs_elem),
            out.clone().binding(),
            &mut dtypes,
        );
        if launched.is_ok() {
            count(if attempt == 0 {
                &FAST_MATMUL
            } else {
                &FALLBACK_MATMUL
            });
            return Tensor::from_primitive::<Cube>(out);
        }
    }
    panic!(
        "no matmul kernel runs on this GPU for {:?} @ {:?}",
        lhs.dims(),
        rhs.dims()
    );
}

/// The rotary embedding's angles as f32 tables of (sequence, pairs), for [`attention`].
pub struct RotaryTables {
    cos: CubeTensor,
    sin: CubeTensor,
}

impl RotaryTables {
    pub fn new(cos: &Tensor<1>, sin: &Tensor<1>) -> Option<Self> {
        Some(Self {
            cos: primitive(cos).filter(|t| t.dtype == DType::F32)?,
            sin: primitive(sin).filter(|t| t.dtype == DType::F32)?,
        })
    }
}

/// Gated multi-head attention from `to_qkv`'s output, (batch, sequence, 3 * heads *
/// dim), and `to_gates`' output, (batch, sequence, heads), to (batch, sequence, heads *
/// dim), ready for `to_out`.
pub fn attention(
    qkv: &Tensor<3>,
    gates: &Tensor<3>,
    rotary: &RotaryTables,
    heads: usize,
) -> Option<Tensor<3>> {
    let out = fused_attention(qkv, gates, rotary, heads);
    count(match out {
        Some(_) => &FUSED_ATTENTION,
        None => &FALLBACK_ATTENTION,
    });
    out
}

fn fused_attention(
    qkv: &Tensor<3>,
    gates: &Tensor<3>,
    rotary: &RotaryTables,
    heads: usize,
) -> Option<Tensor<3>> {
    let qkv_handle = primitive(qkv).filter(has_planes)?;
    let gates_handle = primitive(gates).filter(|g| g.dtype == qkv_handle.dtype)?;
    let [batch, sequence, features] = qkv.dims();
    let dim = features / 3 / heads;
    let dtype = qkv_handle.dtype;
    let client = qkv_handle.client.clone();

    let shape = [batch, heads, sequence, dim];
    let (q, k, v) = (
        empty(&qkv_handle, &shape, dtype),
        empty(&qkv_handle, &shape, dtype),
        empty(&qkv_handle, &shape, dtype),
    );
    let pairs = batch * heads * sequence * dim / 2;
    launch!(
        split_heads,
        dtype,
        &client,
        grid(pairs, UNITS_PER_CUBE),
        CubeDim::new_1d(UNITS_PER_CUBE),
        arg(&qkv_handle),
        arg(&rotary.cos),
        arg(&rotary.sin),
        arg(&q),
        arg(&k),
        arg(&v),
        pairs as u32,
        sequence as u32,
        heads as u32,
        (dim as f32).sqrt().recip(),
    );
    let [q, k, v] = [q, k, v].map(Tensor::<4>::from_primitive::<Cube>);

    let output = empty(&qkv_handle, &[batch, sequence, heads * dim], dtype);
    let kernel = Kernel::for_attention(sequence);
    let step = (MAX_SCORES / (heads * sequence * sequence)).clamp(1, batch);
    for first in (0..batch).step_by(step) {
        let size = step.min(batch - first);
        let slice = |t: &Tensor<4>| t.clone().narrow(0, first, size);
        let scores = matmul(slice(&q), slice(&k).swap_dims(2, 3), kernel);
        let weights = softmax(&scores)?;
        let out = matmul(weights, slice(&v), kernel);
        let out = primitive(&out)?;
        let values = size * heads * sequence * dim;
        launch!(
            merge_heads,
            dtype,
            &client,
            grid(values, UNITS_PER_CUBE),
            CubeDim::new_1d(UNITS_PER_CUBE),
            arg(&out),
            arg(&gates_handle),
            arg(&output),
            values as u32,
            sequence as u32,
            heads as u32,
            first as u32,
        );
    }
    Some(Tensor::from_primitive::<Cube>(output))
}
