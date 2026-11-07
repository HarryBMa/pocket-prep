//! Where Pocket Prep keeps its files, and the user's settings.

use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Library database, analysis files and stems live here (not in the repo).
/// Windows: %APPDATA%\PocketPrep. macOS: ~/Library/Application Support/PocketPrep.
/// Override with the POCKET_PREP_DATA environment variable.
pub fn data_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| match std::env::var_os("POCKET_PREP_DATA") {
        Some(d) => PathBuf::from(d),
        None => dirs::data_dir().unwrap_or_else(|| dirs::home_dir().unwrap_or_default()).join("PocketPrep"),
    })
}

pub fn db_path() -> PathBuf { data_dir().join("library.db") }
pub fn analysis_dir() -> PathBuf { data_dir().join("analysis") }
pub fn stems_dir() -> PathBuf { data_dir().join("stems") }
pub fn artwork_dir() -> PathBuf { data_dir().join("artwork") }
fn settings_path() -> PathBuf { data_dir().join("settings.json") }

pub const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "aiff", "aif", "m4a"];
/// What the deck firmware can decode (minimp3, dr_flac, 16-bit PCM WAV).
pub const DECK_EXTENSIONS: &[&str] = &["mp3", "wav", "flac"];

pub fn defaults() -> Value {
    json!({
        "auto_analyse": true,           // analyse tracks as soon as they are imported
        "deck_address": "http://192.168.4.1",
        "bpm_min": 85,                  // BPM range used when halving/doubling detected tempo
        "bpm_max": 175,
        "target_lufs": -9.0,            // loudness the deck should play tracks at; the export writes the gain to get there
    })
}

pub fn ensure_dirs() -> std::io::Result<()> {
    for d in [data_dir().clone(), analysis_dir(), stems_dir(), artwork_dir()] {
        std::fs::create_dir_all(d)?;
    }
    Ok(())
}

pub fn load_settings() -> Map<String, Value> {
    let mut out = defaults().as_object().unwrap().clone();
    if let Ok(text) = std::fs::read_to_string(settings_path()) {
        if let Ok(Value::Object(saved)) = serde_json::from_str::<Value>(&text) {
            for (k, v) in saved {
                if out.contains_key(&k) {
                    out.insert(k, v);
                }
            }
        }
    }
    out
}

pub fn save_settings(values: &Map<String, Value>) -> std::io::Result<Map<String, Value>> {
    let mut merged = load_settings();
    for (k, v) in values {
        if merged.contains_key(k) {
            merged.insert(k.clone(), v.clone());
        }
    }
    std::fs::write(settings_path(), serde_json::to_string_pretty(&merged)?)?;
    Ok(merged)
}

pub fn setting_f64(key: &str) -> f64 {
    load_settings().get(key).and_then(Value::as_f64).unwrap_or(0.0)
}
