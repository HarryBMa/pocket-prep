//! Talking to the deck: status, signed firmware updates, wired recovery.
//!
//! Wireless (normal): the deck runs Pajoniiir's update service. Turn on
//! "Wi-Fi Remote" on the deck, join its Wi-Fi network, then this module can
//! read /api/status and /api/firmware and upload a signed .ddjota bundle to
//! /api/ota/p4. The deck checks the signature itself before it erases anything.
//!
//! Wired (recovery): espflash writes an ESP-IDF build folder over USB using its
//! flash_args file. Use this for first installs or a deck that won't boot.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::config;

fn base() -> String {
    config::load_settings()["deck_address"].as_str().unwrap_or("http://192.168.4.1").trim_end_matches('/').to_string()
}

/// Ask the deck what it's running. Short timeouts: the deck may simply be off.
pub fn status() -> Value {
    let mut out = json!({"reachable": false, "address": base()});
    let c = reqwest::blocking::Client::builder().timeout(Duration::from_millis(2500)).build().unwrap();
    match c.get(base() + "/api/status").send() {
        Ok(r) => {
            out["reachable"] = json!(r.status().is_success());
            let text = r.text().unwrap_or_default();
            out["status"] = serde_json::from_str(&text).unwrap_or_else(|_| json!(text.chars().take(2000).collect::<String>()));
            if let Ok(f) = c.get(base() + "/api/firmware").send() {
                if f.status().is_success() {
                    if let Ok(v) = serde_json::from_str::<Value>(&f.text().unwrap_or_default()) {
                        out["firmware"] = v;
                    }
                }
            }
        }
        Err(e) => out["error"] = json!(e.to_string()),
    }
    out
}

/// Counts bytes as reqwest reads the upload, so the caller can report progress.
struct Counting {
    data: std::io::Cursor<Vec<u8>>,
    sent: Arc<AtomicU64>,
}

impl Read for Counting {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let max = buf.len().min(64 * 1024);
        let n = self.data.read(&mut buf[..max])?;
        self.sent.store(self.data.position(), Ordering::Relaxed);
        Ok(n)
    }
}

/// Send a signed .ddjota bundle. The deck verifies it before installing.
pub fn ota_upload(bundle: &Path, progress: &dyn Fn(f64, &str)) -> Result<Value> {
    if !bundle.extension().is_some_and(|e| e.eq_ignore_ascii_case("ddjota")) {
        return Err(anyhow!("Only signed .ddjota bundles can be sent wirelessly"));
    }
    let data = std::fs::read(bundle)?;
    let size = data.len() as u64;
    let sent = Arc::new(AtomicU64::new(0));
    let body = reqwest::blocking::Body::sized(Counting { data: std::io::Cursor::new(data), sent: sent.clone() }, size);
    let req = reqwest::blocking::Client::builder().connect_timeout(Duration::from_secs(5)).timeout(Duration::from_secs(120)).build()?
        .post(base() + "/api/ota/p4")
        .header("X-DDJ-Control", "1").header("X-DDJ-OTA", "p4").header("Content-Type", "application/octet-stream")
        .body(body);
    // Upload on a helper thread; report how far it has got until it finishes.
    let worker = std::thread::spawn(move || -> Result<(u16, String)> {
        let r = req.send()?;
        Ok((r.status().as_u16(), r.text().unwrap_or_default().chars().take(1000).collect()))
    });
    while !worker.is_finished() {
        let s = sent.load(Ordering::Relaxed);
        progress(0.9 * s as f64 / size.max(1) as f64, &format!("Uploading {} / {} KB", s / 1024, size / 1024));
        std::thread::sleep(Duration::from_millis(200));
    }
    let (code, text) = worker.join().map_err(|_| anyhow!("upload thread crashed"))??;
    if !(200..300).contains(&code) {
        return Err(anyhow!("Deck refused the update ({code}): {}", text.chars().take(300).collect::<String>()));
    }
    progress(1.0, "Uploaded. The deck restarts and confirms the new firmware itself.");
    Ok(json!({"http": code, "reply": text}))
}

pub fn serial_ports() -> Vec<Value> {
    serialport::available_ports().unwrap_or_default().into_iter().map(|p| {
        let (desc, espressif) = match &p.port_type {
            serialport::SerialPortType::UsbPort(u) => (u.product.clone().unwrap_or_else(|| "USB serial".into()), u.vid == 0x303A),
            _ => ("Serial port".into(), false),
        };
        json!({"port": p.port_name, "description": desc, "espressif": espressif})
    }).collect()
}

/// Address/file pairs from an ESP-IDF flash_args file ("0x10000 app.bin" lines).
fn parse_flash_args(build_dir: &Path) -> Result<Vec<(u32, Vec<u8>)>> {
    let text = std::fs::read_to_string(build_dir.join("flash_args"))
        .map_err(|_| anyhow!("That folder has no flash_args file; pick the ESP-IDF build folder"))?;
    let mut out = vec![];
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(addr), Some(file)) = (parts.next(), parts.next()) else { continue };
        let Some(hex) = addr.strip_prefix("0x") else { continue };       // "--flash_mode dio ..." lines
        let addr = u32::from_str_radix(hex, 16).with_context(|| format!("bad address in flash_args: {addr}"))?;
        out.push((addr, std::fs::read(build_dir.join(file)).with_context(|| format!("reading {file}"))?));
    }
    if out.is_empty() {
        return Err(anyhow!("flash_args lists no images"));
    }
    Ok(out)
}

/// espflash reports blocks per image; turn that into one bar over all images (by bytes).
struct FlashProgress<'a> {
    report: &'a dyn Fn(f64, &str),
    sizes: Vec<usize>,
    image: usize,
    blocks: usize,
    addr: u32,
}

impl FlashProgress<'_> {
    fn frac(&self, within: f64) -> f64 {
        let total: usize = self.sizes.iter().sum();
        let done: usize = self.sizes[..self.image.min(self.sizes.len())].iter().sum();
        let cur = self.sizes.get(self.image).copied().unwrap_or(0) as f64;
        ((done as f64 + within * cur) / total.max(1) as f64).min(0.99)
    }
}

impl espflash::target::ProgressCallbacks for FlashProgress<'_> {
    fn init(&mut self, addr: u32, total: usize) {
        (self.addr, self.blocks) = (addr, total.max(1));
    }
    fn update(&mut self, current: usize) {
        (self.report)(self.frac(current as f64 / self.blocks as f64), &format!("Writing 0x{:X}", self.addr));
    }
    fn verifying(&mut self) {
        (self.report)(self.frac(1.0), "Verifying");
    }
    fn finish(&mut self, _skipped: bool) {
        self.image += 1;
    }
}

/// Flash an ESP-IDF build folder (needs flash_args) over USB.
pub fn wired_flash(build_dir: &Path, port: &str, progress: &dyn Fn(f64, &str)) -> Result<Value> {
    use espflash::connection::{Connection, ResetAfterOperation, ResetBeforeOperation};
    let images = parse_flash_args(build_dir)?;
    let info = serialport::available_ports()?.into_iter().find(|p| p.port_name == port)
        .and_then(|p| match p.port_type { serialport::SerialPortType::UsbPort(u) => Some(u), _ => None })
        .ok_or_else(|| anyhow!("{port} isn't a USB serial port"))?;
    progress(0.0, "Connecting");
    let serial = serialport::new(port, 115_200).flow_control(serialport::FlowControl::None).open_native()?;
    let conn = Connection::new(serial, info, ResetAfterOperation::HardReset, ResetBeforeOperation::DefaultReset, 115_200);
    let chip = espflash::target::Chip::Esp32p4;
    let mut flasher = espflash::flasher::Flasher::connect(conn, true, true, false, Some(chip), Some(460_800))?;
    let segments: Vec<_> = images.iter().map(|(a, d)| espflash::image_format::Segment::new(*a, d)).collect();
    let mut cb = FlashProgress { report: progress, sizes: images.iter().map(|(_, d)| d.len()).collect(), image: 0, blocks: 1, addr: 0 };
    flasher.write_bins_to_flash(&segments, &mut cb)?;
    flasher.connection().reset_after(true, chip)?;
    progress(1.0, "Done. The deck restarts.");
    Ok(json!({"log": images.iter().map(|(a, d)| format!("0x{a:X}: {} bytes", d.len())).collect::<Vec<_>>()}))
}
