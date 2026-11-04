//! Importing tracks and organising them: tags, artwork, playlists, cues.

use anyhow::Result;
use lofty::prelude::*;
use lofty::tag::ItemKey;
use rusqlite::params;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::config;
use crate::db::{self, Row};

pub fn ext(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

pub fn is_audio(p: &Path) -> bool {
    config::AUDIO_EXTENSIONS.contains(&ext(p).as_str())
}

#[derive(Default)]
pub struct Tags {
    pub title: String,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub comment: Option<String>,
    pub year: Option<u32>,
    pub track_number: Option<u32>,
    pub duration: Option<f64>,
    pub sample_rate: Option<u32>,
    pub bitrate: Option<u32>,
}

/// Title, artist etc. from the file's own tags (falls back to the filename).
pub fn read_tags(path: &Path) -> Tags {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let mut info = Tags { title: stem.clone(), ..Default::default() };
    let Ok(f) = lofty::read_from_path(path) else { return info };
    let p = f.properties();
    info.duration = Some(p.duration().as_secs_f64());
    info.sample_rate = p.sample_rate();
    info.bitrate = p.audio_bitrate().or(p.overall_bitrate());
    if let Some(t) = f.primary_tag().or_else(|| f.first_tag()) {
        let s = |x: Option<std::borrow::Cow<str>>| x.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        info.title = s(t.title()).unwrap_or(stem);
        info.artist = s(t.artist()).or_else(|| t.get_string(&ItemKey::AlbumArtist).map(str::to_string));
        info.album = s(t.album());
        info.genre = s(t.genre());
        info.comment = s(t.comment());
        info.year = t.year();
        info.track_number = t.track();
    }
    info
}

/// Save embedded cover art as <artwork dir>/<id>.jpg; returns the path.
fn extract_artwork(path: &Path, track_id: i64) -> Option<String> {
    let f = lofty::read_from_path(path).ok()?;
    let data = f.tags().iter().flat_map(|t| t.pictures()).next()?.data().to_vec();
    let img = image::load_from_memory(&data).ok()?.thumbnail(500, 500).to_rgb8();
    let out = config::artwork_dir().join(format!("{track_id}.jpg"));
    save_jpeg(&img, &out, 88).ok()?;
    Some(out.to_string_lossy().to_string())
}

pub fn save_jpeg(img: &image::RgbImage, out: &Path, quality: u8) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, quality).encode_image(img)?;
    Ok(())
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if is_audio(&p) {
            out.push(p);
        }
    }
}

/// Add files, or every audio file inside folders. Returns new track ids.
pub fn import_paths(paths: &[String]) -> Result<Vec<i64>> {
    let mut files = Vec::new();
    for p in paths {
        let p = PathBuf::from(p);
        if p.is_dir() {
            walk(&p, &mut files);
        } else if is_audio(&p) && p.exists() {
            files.push(p);
        }
    }
    files.sort();
    let mut new_ids = Vec::new();
    for f in files {
        let f = std::fs::canonicalize(&f).unwrap_or(f);
        let path = clean_path(&f);
        if let Some(existing) = db::one("SELECT id, deleted FROM tracks WHERE path=?", &[json!(path)])? {
            if existing["deleted"].as_i64() == Some(1) {
                db::tx(|c| Ok(c.execute("UPDATE tracks SET deleted=0 WHERE id=?", [existing["id"].as_i64()])?))?;
            }
            continue;
        }
        let t = read_tags(&f);
        let size = std::fs::metadata(&f)?.len() as i64;
        let tid = db::tx(|c| {
            c.execute("INSERT INTO tracks(path, title, artist, album, genre, comment, year, track_number,
                       duration, sample_rate, bitrate, file_size, added_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      params![path, t.title, t.artist, t.album, t.genre, t.comment, t.year, t.track_number,
                              t.duration, t.sample_rate, t.bitrate, size, db::now()])?;
            Ok(c.last_insert_rowid())
        })?;
        if let Some(art) = extract_artwork(&f, tid) {
            db::tx(|c| Ok(c.execute("UPDATE tracks SET artwork=? WHERE id=?", params![art, tid])?))?;
        }
        new_ids.push(tid);
    }
    if !new_ids.is_empty() {
        db::tx(|c| db::log(c, "import", json!({"count": new_ids.len()}), Some(json!({"kind": "none"}))))?;
    }
    Ok(new_ids)
}

/// canonicalize() gives \\?\C:\... on Windows; store the plain form, as before.
fn clean_path(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

/// Hide tracks from the library. Files are never deleted. Undoable.
pub fn remove_tracks(ids: &[i64]) -> Result<i64> {
    db::tx(|c| {
        c.execute(&format!("UPDATE tracks SET deleted=1 WHERE id IN ({})", db::marks(ids.len())),
                  rusqlite::params_from_iter(ids))?;
        db::log(c, "remove_tracks", json!({"ids": ids}), Some(json!({"kind": "undelete", "ids": ids})))
    })
}

// ── Playlists ────────────────────────────────────────────────────────────────

pub fn playlist_tree() -> Result<Vec<Row>> {
    let mut rows = db::query("SELECT * FROM playlists ORDER BY parent_id, sort, name", &[])?;
    let counts = db::query("SELECT playlist_id, COUNT(*) n FROM playlist_tracks GROUP BY playlist_id", &[])?;
    for r in &mut rows {
        let n = counts.iter().find(|c| c["playlist_id"] == r["id"]).map(|c| c["n"].clone()).unwrap_or(json!(0));
        r.insert("count".into(), n);
    }
    Ok(rows)
}

pub fn create_playlist(name: &str, parent_id: i64, is_folder: bool) -> Result<Row> {
    let name = if name.trim().is_empty() { "Untitled" } else { name.trim() };
    let id = db::tx(|c| {
        let sort: i64 = c.query_row("SELECT COALESCE(MAX(sort),0)+1 FROM playlists WHERE parent_id=?", [parent_id], |r| r.get(0))?;
        c.execute("INSERT INTO playlists(name, parent_id, is_folder, sort) VALUES (?,?,?,?)",
                  params![name, parent_id, is_folder as i64, sort])?;
        let id = c.last_insert_rowid();
        db::log(c, "create_playlist", json!({"id": id, "name": name}), None)?;
        Ok(id)
    })?;
    Ok(db::one("SELECT * FROM playlists WHERE id=?", &[json!(id)])?.unwrap_or_default())
}

pub fn rename_playlist(pid: i64, name: &str) -> Result<()> {
    db::tx(|c| Ok(c.execute("UPDATE playlists SET name=? WHERE id=?", params![name, pid])?))?;
    Ok(())
}

/// Deletes a playlist (and anything inside a folder). Tracks stay in the library.
pub fn delete_playlist(pid: i64) -> Result<()> {
    db::tx(|c| {
        let (mut ids, mut todo) = (vec![], vec![pid]);
        while let Some(p) = todo.pop() {
            ids.push(p);
            let mut st = c.prepare("SELECT id FROM playlists WHERE parent_id=?")?;
            todo.extend(st.query_map([p], |r| r.get::<_, i64>(0))?.flatten());
        }
        let q = db::marks(ids.len());
        c.execute(&format!("DELETE FROM playlist_tracks WHERE playlist_id IN ({q})"), rusqlite::params_from_iter(&ids))?;
        c.execute(&format!("DELETE FROM playlists WHERE id IN ({q})"), rusqlite::params_from_iter(&ids))?;
        db::log(c, "delete_playlist", json!({"ids": ids}), None)
    })?;
    Ok(())
}

pub fn playlist_track_ids(pid: i64) -> Result<Vec<i64>> {
    Ok(db::query("SELECT track_id FROM playlist_tracks WHERE playlist_id=? ORDER BY position", &[json!(pid)])?
        .iter().filter_map(|r| r["track_id"].as_i64()).collect())
}

/// Replace the playlist's order (used for add, remove and reorder). Undoable.
pub fn set_playlist_tracks(pid: i64, track_ids: &[i64], action: &str) -> Result<i64> {
    let before = playlist_track_ids(pid)?;
    db::tx(|c| {
        c.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", [pid])?;
        for (i, t) in track_ids.iter().enumerate() {
            c.execute("INSERT INTO playlist_tracks VALUES (?,?,?)", params![pid, t, i as i64])?;
        }
        db::log(c, action, json!({"playlist_id": pid, "count": track_ids.len()}),
                Some(json!({"kind": "playlist_restore", "playlist_id": pid, "tracks": before})))
    })
}

pub fn add_to_playlist(pid: i64, track_ids: &[i64]) -> Result<i64> {
    let mut ids = playlist_track_ids(pid)?;
    for t in track_ids {
        if !ids.contains(t) {
            ids.push(*t);
        }
    }
    set_playlist_tracks(pid, &ids, "add_to_playlist")
}

// ── Cues ─────────────────────────────────────────────────────────────────────

pub fn cues(track_id: i64) -> Result<Vec<Row>> {
    db::query("SELECT * FROM cues WHERE track_id=? ORDER BY slot=0, slot, start_ms", &[json!(track_id)])
}

/// Replace a track's cues. Each cue: slot (0-8), start_ms, optional end_ms/label.
pub fn set_cues(track_id: i64, new: &[Value]) -> Result<i64> {
    let before = cues(track_id)?;
    let mut seen = vec![];
    let mut clean = vec![];
    for q in new {
        let slot = q["slot"].as_i64().unwrap_or(0);
        if slot != 0 && seen.contains(&slot) {
            continue;
        }
        seen.push(slot);
        let start = q["start_ms"].as_f64().ok_or_else(|| anyhow::anyhow!("cue needs start_ms"))? as i64;
        clean.push((slot, start, q["end_ms"].as_f64().map(|v| v as i64), q["label"].as_str().map(str::to_string),
                    q["auto"].as_i64().unwrap_or(0)));
    }
    db::tx(|c| {
        c.execute("DELETE FROM cues WHERE track_id=?", [track_id])?;
        for (slot, start, end, label, auto) in &clean {
            c.execute("INSERT INTO cues(track_id, slot, start_ms, end_ms, label, auto) VALUES (?,?,?,?,?,?)",
                      params![track_id, slot, start, end, label, auto])?;
        }
        db::log(c, "edit_cues", json!({"track_id": track_id, "count": clean.len()}),
                Some(json!({"kind": "cues_restore", "track_id": track_id, "cues": before})))
    })
}
