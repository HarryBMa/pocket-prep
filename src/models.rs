//! The models, and the one that isn't in the installer.
//!
//! `htdemucs.onnx` (174 MB, the stems model) is not bundled on Windows and Linux: it is fetched
//! once, when the user asks, into `<data>/models/`, and checked against the SHA-256 that the build
//! pinned in `models/manifest.json` (CI writes that file from the exported model it also publishes
//! as a release asset; in a development checkout it is `{}` and the model is used from `models/`).
//! On macOS the CoreML model is still bundled. The small files (`htdemucs.json`, the 12 MB mood
//! model) ship in the app and are copied next to the fetched one, so ClubCraft reads one folder.
//! After that first fetch nothing needs a network.

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::{config, jobs};

const MANIFEST: &str = include_str!("../models/manifest.json");
pub const STEMS_MODEL: &str = "htdemucs.onnx";
const SMALL: &[&str] = &["htdemucs.json", "htdemucs_mood.onnx"];

pub fn manifest() -> Value {
    serde_json::from_str(MANIFEST).unwrap_or(json!({}))
}

/// The folder ClubCraft reads: `<data>/models`.
pub fn dir() -> PathBuf {
    config::data_dir().join("models")
}

/// The models folder in the app bundle (or `models/` in a development checkout).
fn bundled() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())).unwrap_or_default();
    [exe_dir.join("models"), exe_dir.join("../Resources/models"), PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")]
        .into_iter().find(|d| d.join("htdemucs.json").exists())
}

fn same_size(a: &Path, b: &Path) -> bool {
    match (std::fs::metadata(a), std::fs::metadata(b)) { (Ok(x), Ok(y)) => x.len() == y.len() && x.len() > 0, _ => false }
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() { copy_dir(&e.path(), &to)?; } else if !same_size(&e.path(), &to) { std::fs::copy(e.path(), &to)?; }
    }
    Ok(())
}

/// Called once at start-up, before any thread: fill `<data>/models` from the bundle (the small
/// files always; the big ones when the bundle has them) and point ClubCraft at it.
pub fn prepare() -> Result<()> {
    let d = dir();
    std::fs::create_dir_all(&d)?;
    if let Some(b) = bundled() {
        for name in SMALL.iter().chain([STEMS_MODEL].iter()) {
            let (from, to) = (b.join(name), d.join(name));
            if from.exists() && !same_size(&from, &to) {
                std::fs::copy(&from, &to).map_err(|e| anyhow!("copying {name} into the data folder: {e}"))?;
            }
        }
        let ml = b.join("htdemucs.mlmodelc");
        if ml.is_dir() { copy_dir(&ml, &d.join("htdemucs.mlmodelc"))?; }
    }
    std::env::set_var("CLUBCRAFT_MODELS", &d);
    Ok(())
}

/// Whether stems can be made: the ONNX model, or the CoreML one on macOS.
pub fn stems_ready() -> bool {
    let d = dir();
    d.join(STEMS_MODEL).exists() || d.join("htdemucs.mlmodelc").is_dir()
}

/// What the Settings page shows: present, the size and address the build pinned, and the job in flight.
pub fn status() -> Value {
    let m = manifest();
    let entry = &m[STEMS_MODEL];
    let job = jobs::listing().into_iter().find(|j| j.kind == "models" && matches!(j.status.as_str(), "queued" | "running")).map(|j| json!({"id": j.id, "progress": j.progress, "message": j.message}));
    json!({
        "stems_model": {"present": stems_ready(), "bytes": entry["bytes"], "url": entry["url"],
                        "fetchable": entry["url"].is_string() && entry["sha256"].is_string(), "job": job, "dir": dir()},
    })
}

/// Fetch the stems model as a job (the io lane); the file counts only once its hash matches.
pub fn fetch() -> Result<i64> {
    let m = manifest();
    let (Some(url), Some(sha)) = (m[STEMS_MODEL]["url"].as_str(), m[STEMS_MODEL]["sha256"].as_str()) else {
        bail!("This build has no address for the stems model. Put htdemucs.onnx in {}", dir().display());
    };
    if stems_ready() { bail!("The stems model is already here"); }
    if status()["stems_model"]["job"].is_object() { bail!("Already downloading"); }
    let (url, sha) = (url.to_string(), sha.to_string());
    let bytes = m[STEMS_MODEL]["bytes"].as_u64();
    Ok(jobs::submit("models", "Stems model (htdemucs)", "io", None, move |h| {
        download(&url, &sha, bytes, &dir().join(STEMS_MODEL), &|f, msg| h.progress(f, msg), &|| h.cancelled())?;
        Ok(Some(json!({"file": STEMS_MODEL})))
    }))
}

fn download(url: &str, sha: &str, expect: Option<u64>, to: &Path, progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<()> {
    progress(0.0, "Connecting");
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut r = client.get(url).send()?.error_for_status()?;
    let total = r.content_length().or(expect).unwrap_or(0);
    let part = to.with_extension("part");
    let mut f = std::fs::File::create(&part)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut got = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 { break; }
        if cancel() { drop(f); let _ = std::fs::remove_file(&part); bail!("Cancelled"); }
        f.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        got += n as u64;
        if total > 0 { progress((got as f64 / total as f64) * 0.98, &format!("{} of {} MB", got >> 20, total >> 20)); }
    }
    f.flush()?; drop(f);
    let hex = format!("{:x}", hasher.finalize());
    if hex != sha {
        let _ = std::fs::remove_file(&part);
        bail!("The downloaded file doesn't match the build's hash (got {}…, wanted {}…). Nothing was kept.", &hex[..12], &sha[..12]);
    }
    progress(0.99, "Checking");
    std::fs::rename(&part, to)?;
    Ok(())
}
