//! RoFormer separation: Watershed's BS-RoFormer SW (github.com/edarbieto/watershed, vendored in vendor/watershed) as a
//! second stems engine beside htdemucs.
//!
//! Why: a larger model, so cleaner vocals and a 6-stem split (bass, drums, other, vocals, guitar, piano). The deck's file
//! has four stems in a fixed order, so guitar and piano are folded into "other" there; the three of them (other, guitar,
//! piano) are also kept apart in `<id>.extra.flac` for the practice player's mixer.
//!
//! Cost: it runs on the GPU through burn + wgpu (Vulkan on Windows and Linux, Metal on macOS), needs a download of 700 MB
//! that is converted to 350 MB on the user's machine, and takes 1-2 minutes for a song on a laptop GPU. It is a choice in
//! Settings ("Stems engine"); htdemucs stays the default. The checkpoint declares no licence, so it is never bundled or
//! hosted by us: the app fetches it from its original source, checks size and SHA-256, converts it and deletes the original.
//!
//! Without the `roformer` feature (a lean build) every function says it isn't built.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::{jobs, models};

/// The stems that come out apart: (other, guitar, piano), planar stereo each.
pub type Extra = [(Vec<f32>, Vec<f32>); 3];

pub struct Split {
    /// Drums, bass, other (with guitar and piano), vocals: the deck's layout.
    pub four: crate::stems::StemAudio,
    /// Other, guitar and piano apart, when the model has them.
    pub extra: Option<Extra>,
}

#[cfg(feature = "roformer")]
mod imp {
    use super::*;
    use watershed_separation::{catalog, Separator, StereoBuffer};

    pub fn built() -> bool { true }

    pub fn installed() -> bool {
        catalog::default_model().is_installed(&models::dir())
    }

    pub fn name() -> String { catalog::default_model().name.clone() }

    pub fn bytes() -> u64 { catalog::default_model().size }

    pub fn license() -> String { catalog::default_model().license.clone() }

    /// Download (resumable, mirrors, SHA-256), convert to f16, delete the checkpoint.
    pub fn install(progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<()> {
        let entry = catalog::default_model();
        progress(0.0, "Connecting");
        entry.install(&models::dir(), &|done, total| {
            let f = if total > 0 { done as f64 / total as f64 } else { 0.0 };
            progress(f * 0.95, &format!("{} of {} MB", done >> 20, total >> 20));
        }, cancel).map_err(|e| anyhow!("{e}"))?;
        progress(1.0, "Done");
        Ok(())
    }

    /// Separate a 44.1 kHz stereo buffer. `progress` is 0..1; `cancel` is polled.
    pub fn separate(audio: &clubcraft_audio::AudioBuffer, progress: &(dyn Fn(f32) + Sync), cancel: &(dyn Fn() -> bool + Sync)) -> Result<Split> {
        let entry = catalog::default_model();
        let mut sep = entry.load(&models::dir()).map_err(|e| anyhow!("{e}"))?;
        let (names, stems) = run(&mut sep, audio, progress, cancel)?;
        split(&names, stems)
    }

    /// Any separator on a buffer: the model's stem names and one interleaved stereo signal per stem.
    pub(super) fn run(sep: &mut dyn Separator, audio: &clubcraft_audio::AudioBuffer, progress: &(dyn Fn(f32) + Sync), cancel: &(dyn Fn() -> bool + Sync))
                      -> Result<(Vec<String>, Vec<Vec<f32>>)> {
        if sep.sample_rate() != audio.sample_rate {
            return Err(anyhow!("the model needs {} Hz, got {} Hz", sep.sample_rate(), audio.sample_rate));
        }
        let mut samples = Vec::with_capacity(audio.left.len() * 2);
        for (l, r) in audio.left.iter().zip(&audio.right) { samples.push(*l); samples.push(*r); }
        let input = StereoBuffer::new(audio.sample_rate, samples);
        let out = sep.separate(&input, &|f| progress(f), &|| cancel()).map_err(|e| anyhow!("{e}"))?;
        Ok((sep.stem_names().to_vec(), out.into_iter().map(|b| b.samples).collect()))
    }
}

#[cfg(not(feature = "roformer"))]
mod imp {
    use super::*;
    pub fn built() -> bool { false }
    pub fn installed() -> bool { false }
    pub fn name() -> String { "BS-RoFormer SW".into() }
    pub fn bytes() -> u64 { 0 }
    pub fn license() -> String { "unknown".into() }
    pub fn install(_: &dyn Fn(f64, &str), _: &dyn Fn() -> bool) -> Result<()> { Err(anyhow!("This build doesn't include the RoFormer engine")) }
    pub fn separate(_: &clubcraft_audio::AudioBuffer, _: &(dyn Fn(f32) + Sync), _: &(dyn Fn() -> bool + Sync)) -> Result<Split> { Err(anyhow!("This build doesn't include the RoFormer engine")) }
}

pub use imp::{built, installed, separate};

fn deinterleave(s: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (mut l, mut r) = (Vec::with_capacity(s.len() / 2), Vec::with_capacity(s.len() / 2));
    for f in s.chunks_exact(2) { l.push(f[0]); r.push(f[1]); }
    (l, r)
}

/// Arrange the model's stems (any order, by name, interleaved stereo each) as the deck's four, plus the extras.
/// Guitar and piano (when the model has them) are added to "other" for the deck.
pub fn split(names: &[String], stems: Vec<Vec<f32>>) -> Result<Split> {
    let find = |n: &str| names.iter().position(|x| x == n).map(|i| deinterleave(&stems[i]));
    let (drums, bass, vocals) = (find("drums"), find("bass"), find("vocals"));
    let (Some(drums), Some(bass), Some(vocals)) = (drums, bass, vocals) else { return Err(anyhow!("the model's stems are {names:?}: drums, bass and vocals are needed")) };
    let other = find("other").unwrap_or_else(|| (vec![0.0; drums.0.len()], vec![0.0; drums.0.len()]));
    let (guitar, piano) = (find("guitar"), find("piano"));
    let mut folded = other.clone();
    for extra in [&guitar, &piano].into_iter().flatten() {
        for (a, b) in folded.0.iter_mut().zip(&extra.0) { *a += b; }
        for (a, b) in folded.1.iter_mut().zip(&extra.1) { *a += b; }
    }
    let n = drums.0.len();
    let zero = || (vec![0.0f32; n], vec![0.0f32; n]);
    let extra = (guitar.is_some() || piano.is_some()).then(|| [other, guitar.unwrap_or_else(zero), piano.unwrap_or_else(zero)]);
    Ok(Split { four: [drums, bass, folded, vocals], extra })
}

/// What Settings shows.
pub fn status() -> Value {
    let job = jobs::listing().into_iter().find(|j| j.kind == "roformer" && matches!(j.status.as_str(), "queued" | "running"))
        .map(|j| json!({"id": j.id, "progress": j.progress, "message": j.message}));
    json!({"built": built(), "present": installed(), "name": imp::name(), "bytes": imp::bytes(), "license": imp::license(), "job": job})
}

/// Fetch the model as a job on the io lane.
pub fn fetch() -> Result<i64> {
    if !built() { return Err(anyhow!("This build doesn't include the RoFormer engine")); }
    if installed() { return Err(anyhow!("The RoFormer model is already here")); }
    if status()["job"].is_object() { return Err(anyhow!("Already downloading")); }
    Ok(jobs::submit("roformer", "Stems model (BS-RoFormer SW)", "io", None, move |h| {
        imp::install(&|f, m| h.progress(f, m), &|| h.cancelled())?;
        Ok(Some(json!({"installed": true})))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model's six stems (any order) become the deck's four plus the three kept apart; "other" for the deck is
    /// other + guitar + piano. A model without guitar and piano gives no extras.
    #[test]
    fn six_stems_fold_into_the_decks_four() {
        let names: Vec<String> = ["bass", "drums", "other", "vocals", "guitar", "piano"].iter().map(|s| s.to_string()).collect();
        let stem = |v: f32| vec![v, -v, v, -v];                       // two frames, interleaved
        let s = split(&names, vec![stem(0.1), stem(0.2), stem(0.3), stem(0.4), stem(0.05), stem(0.02)]).unwrap();
        assert_eq!(s.four[0].0[0], 0.2);                              // drums
        assert_eq!(s.four[1].0[0], 0.1);                              // bass
        assert!((s.four[2].0[0] - 0.37).abs() < 1e-6, "{}", s.four[2].0[0]);   // other + guitar + piano
        assert_eq!(s.four[3].1[0], -0.4);                             // vocals, right channel
        let e = s.extra.unwrap();
        assert_eq!((e[0].0[0], e[1].0[0], e[2].0[0]), (0.3, 0.05, 0.02));
        let four_only: Vec<String> = ["drums", "bass", "other", "vocals"].iter().map(|s| s.to_string()).collect();
        let s = split(&four_only, vec![stem(0.2), stem(0.1), stem(0.3), stem(0.4)]).unwrap();
        assert!(s.extra.is_none() && s.four[2].0[0] == 0.3);
        assert!(split(&["vocals".to_string()], vec![stem(0.1)]).is_err());
    }

    /// Whether a graphics device the model can use is there: a Vulkan driver file on Linux, or `POCKET_PREP_GPU_TEST=1`
    /// where we can't tell (Windows, macOS: set it on a machine with a GPU). GitHub's runners have none, and their
    /// Mesa software driver crashes the test process (SIGSEGV), so CI skips this test; it was run with a newer Mesa.
    #[cfg(feature = "roformer")]
    fn gpu_present() -> bool {
        std::env::var_os("POCKET_PREP_GPU_TEST").is_some()
            || ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"].iter()
                .any(|d| std::fs::read_dir(d).is_ok_and(|mut r| r.next().is_some()))
    }

    /// The vendored model code runs here, end to end, on the tiny test model through the GPU path (Vulkan: a software
    /// device on machines without a GPU): a second of tone in, two stems of the same length out, finite. Without a
    /// graphics device the test says so and returns: it needs one, and a runner without one can't tell us anything.
    #[cfg(feature = "roformer")]
    #[test]
    fn the_tiny_model_separates_through_the_gpu_path() {
        use watershed_separation::BurnSeparator;
        if !gpu_present() {
            eprintln!("skipped: no Vulkan device found (set POCKET_PREP_GPU_TEST=1 to run it anyway)");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pocketprep_roformer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let yaml = dir.join("tiny.yaml");
        std::fs::write(&yaml, "audio: {chunk_size: 496, sample_rate: 44100}\nmodel: {dim: 16, depth: 2, num_stems: 2, heads: 2, dim_head: 8, \
            time_transformer_depth: 1, freq_transformer_depth: 1, freqs_per_bands: [2, 2, 4, 8, 17], mask_estimator_depth: 2, \
            mlp_expansion_factor: 4, stft_n_fft: 64, stft_hop_length: 16, stft_win_length: 64, stereo: true}\ninference: {num_overlap: 2}\n").unwrap();
        let ckpt = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/watershed/crates/roformer/tests/fixtures/tiny.ckpt");
        let mut sep = BurnSeparator::load(&ckpt, &yaml, vec!["a".into(), "b".into()]).expect("the tiny model loads");
        let n = 44100;
        let tone: Vec<f32> = (0..n).map(|i| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() as f32).collect();
        let audio = clubcraft_audio::AudioBuffer::new(44100, tone.clone(), tone);
        let seen = std::sync::atomic::AtomicU32::new(0);
        let (names, stems) = imp::run(&mut sep, &audio, &|f| { seen.store((f * 100.0) as u32, std::sync::atomic::Ordering::Relaxed); }, &|| false).expect("separates");
        assert_eq!(names, ["a", "b"]);
        assert!(stems.len() == 2 && stems.iter().all(|s| s.len() == n * 2 && s.iter().all(|v| v.is_finite())), "{:?}", stems.iter().map(Vec::len).collect::<Vec<_>>());
        assert!(seen.load(std::sync::atomic::Ordering::Relaxed) >= 90, "progress reached {}", seen.load(std::sync::atomic::Ordering::Relaxed));
        // Cancelling stops it.
        assert!(imp::run(&mut sep, &audio, &|_| {}, &|| true).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
