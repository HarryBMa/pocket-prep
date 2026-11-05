//! Export the library to an SD card or USB stick.
//!
//! Layout written (rekordbox-compatible, so CDJs can read it too):
//!
//!   Contents/<Artist>/<Album>/<file>                  the audio files, copied byte for byte
//!   PIONEER/rekordbox/export.pdb                      track database
//!   PIONEER/USBANLZ/Pxxx/xxxxxxxx/ANLZ0000.DAT|.EXT   beat grid, waveforms, cues
//!   PIONEER/Artwork/00001/a<id>.jpg (+ _m.jpg)        cover art
//!
//! Pocket Deck extensions (other players ignore these; see docs/STEMS_FORMAT.md):
//!
//!   POCKETDECK/export.json                 what wrote the stick, when, and any warnings
//!   POCKETDECK/stems.json                  which stem file belongs to which track
//!   POCKETDECK/STEMS/xxxxxxxx.flac         8-channel FLAC stems: drums, bass, other, vocals
//!   POCKETDECK/analysis/xxxxxxxx.json      phrases, energy, key (for autopilot)

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{anlz, pdb};
use crate::db::{self, Row};
use crate::{analysis, config, library, stems};

pub fn safe(name: Option<&str>, fallback: &str) -> String {
    let cleaned: String = name.unwrap_or("").trim().chars()
        .map(|c| if "<>:\"/\\|?*".contains(c) || (c as u32) < 0x20 { '_' } else { c }).collect();
    let cleaned = cleaned.trim_end_matches(['.', ' ']);
    let s = if cleaned.is_empty() { fallback } else { cleaned };
    s.chars().take(80).collect()
}

fn file_type(ext: &str) -> u16 {
    match ext { "m4a" => 4, "flac" => 5, "wav" => 0x0B, "aiff" | "aif" => 0x0C, _ => 1 }
}

/// Why the deck can't play this file, or None if it can. The deck decodes
/// MP3, FLAC and 16-bit PCM WAV (mono or stereo) at any sample rate.
pub fn deck_problem(path: &Path) -> Option<String> {
    let ext = library::ext(path);
    if !config::DECK_EXTENSIONS.contains(&ext.as_str()) {
        return Some(format!("{} files can't be played on the deck (MP3, WAV or FLAC only)", ext.to_uppercase()));
    }
    if ext != "wav" {
        return None;
    }
    // Walk the RIFF chunks to the fmt chunk, as the firmware does.
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).ok()?;
    loop {
        let mut ch = [0u8; 8];
        if f.read_exact(&mut ch).is_err() {
            return Some("WAV file has no format chunk".into());
        }
        let size = u32::from_le_bytes(ch[4..8].try_into().unwrap()) as u64;
        if &ch[..4] == b"fmt " {
            let mut fmt = [0u8; 16];
            f.read_exact(&mut fmt).ok()?;
            let (tag, chans, bits) = (u16::from_le_bytes([fmt[0], fmt[1]]), u16::from_le_bytes([fmt[2], fmt[3]]), u16::from_le_bytes([fmt[14], fmt[15]]));
            return if tag != 1 || bits != 16 {
                Some(format!("{bits}-bit WAV can't be played on the deck (16-bit PCM only)"))
            } else if !(1..=2).contains(&chans) {
                Some(format!("{chans}-channel WAV can't be played on the deck"))
            } else {
                None
            };
        }
        std::io::Seek::seek(&mut f, std::io::SeekFrom::Current((size + (size & 1)) as i64)).ok()?;
    }
}

/// The system drive (Windows: %SystemDrive%, macOS: / and its system volumes). Never an export target.
fn is_system(mount: &Path) -> bool {
    if cfg!(windows) {
        let sys = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".into());
        mount.to_string_lossy().trim_end_matches('\\').eq_ignore_ascii_case(&sys)
    } else {
        let m = mount.to_string_lossy();
        m == "/" || m.starts_with("/System/") || m.starts_with("/private/") || m.starts_with("/dev")
    }
}

/// The deck's SD/USB stack (FatFs) reads FAT32 and exFAT only.
fn deck_filesystem_problem(fs: &str) -> Option<String> {
    match fs.to_ascii_lowercase().as_str() {
        "fat32" | "fat" | "fat16" | "vfat" | "msdos" | "exfat" => None,
        _ => Some(format!("{fs} drive: the deck reads FAT32 and exFAT only. Format it as exFAT first (this erases it).")),
    }
}

/// If `root` holds a rekordbox library DKPRP (formerly Pocket Prep) didn't write, why we think so. Ours means
/// POCKETDECK/export.json says Pocket Prep wrote it and its recorded hash still matches
/// export.pdb (exports from before the hash was recorded count as ours). Anything else (a
/// rekordbox export, or one rekordbox changed after us) must not be overwritten.
pub fn foreign_library(root: &Path) -> Option<String> {
    let pdb = root.join("PIONEER/rekordbox/export.pdb");
    if !pdb.exists() {
        return None;
    }
    let ours: Option<Value> = std::fs::read(root.join("POCKETDECK/export.json")).ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .filter(|j: &Value| j["app"] == "Pocket Prep");   // internal marker: keep, or every card we wrote looks foreign
    let Some(ours) = ours else {
        return Some("This drive already has a rekordbox library that DKPRP didn't write.".into());
    };
    let Some(hash) = ours["pdb_blake3"].as_str() else { return None };
    match std::fs::read(&pdb) {
        Ok(b) if blake3::hash(&b).to_hex().as_str() == hash => None,
        Ok(_) => Some("This drive's rekordbox library was changed by another program after DKPRP wrote it.".into()),
        Err(e) => Some(format!("Can't read this drive's rekordbox library ({e}).")),
    }
}

/// Drives to offer for export: removable ones only unless `all`, and never the system drive.
pub fn list_drives(all: bool) -> Vec<Value> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut out: Vec<Value> = disks.list().iter().filter(|d| !is_system(d.mount_point()) && (all || d.is_removable())).map(|d| {
        let fs = d.file_system().to_string_lossy().to_string();
        json!({
            "root": d.mount_point().to_string_lossy(), "label": d.name().to_string_lossy(), "filesystem": fs,
            "removable": d.is_removable(), "free": d.available_space(), "total": d.total_space(),
            "has_export": d.mount_point().join("PIONEER/rekordbox/export.pdb").exists(),
            "problem": deck_filesystem_problem(&fs),
        })
    }).collect();
    out.sort_by_key(|d| (!d["removable"].as_bool().unwrap_or(false), d["root"].as_str().unwrap_or("").to_string()));
    out
}

/// Why `root` can't be an export target, if it can't: the system drive, or a filesystem
/// the deck can't read. POCKET_PREP_ANY_TARGET=1 lifts this for tests/e2e.sh.
pub fn target_problem(root: &Path) -> Option<String> {
    if std::env::var_os("POCKET_PREP_ANY_TARGET").is_some() {
        return None;
    }
    let mut probe = root.to_path_buf();
    while !probe.exists() && probe.pop() {}
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let Some(d) = disks.list().iter().filter(|d| probe.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len()) else {
        return Some("Can't find the drive for that folder".into());
    };
    if is_system(d.mount_point()) {
        return Some("That's the system drive. Choose an SD card or USB stick.".into());
    }
    deck_filesystem_problem(&d.file_system().to_string_lossy())
}

fn anlz_rel(tid: u32) -> String {
    format!("PIONEER/USBANLZ/P{:03X}/{tid:08X}", (tid >> 8) & 0xFFF)
}

/// Tracks and playlists to export. None = the whole library.
fn collect(playlist_ids: Option<&[i64]>) -> Result<(Vec<Row>, Vec<Row>)> {
    let mut pls = library::playlist_tree()?;
    let Some(chosen_ids) = playlist_ids.filter(|p| !p.is_empty()) else {
        return Ok((db::query("SELECT * FROM tracks WHERE deleted=0 ORDER BY artist, title", &[])?, pls));
    };
    let id = |p: &Row, k: &str| p[k].as_i64().unwrap_or(0);
    let mut chosen = vec![];
    let mut todo = chosen_ids.to_vec();
    while let Some(p) = todo.pop() {                  // include everything inside folders
        chosen.push(p);
        todo.extend(pls.iter().filter(|x| id(x, "parent_id") == p).map(|x| id(x, "id")));
    }
    for p in chosen.clone() {                         // and the folders above them
        let mut parent = pls.iter().find(|x| id(x, "id") == p).map(|x| id(x, "parent_id")).unwrap_or(0);
        while parent != 0 && !chosen.contains(&parent) {
            chosen.push(parent);
            parent = pls.iter().find(|x| id(x, "id") == parent).map(|x| id(x, "parent_id")).unwrap_or(0);
        }
    }
    pls.retain(|p| chosen.contains(&id(p, "id")));
    let mut ids: Vec<i64> = vec![];
    for p in &pls {
        for t in library::playlist_track_ids(id(p, "id"))? {
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
    }
    let tracks = ids.iter().filter_map(|t| db::one("SELECT * FROM tracks WHERE id=? AND deleted=0", &[json!(t)]).ok().flatten()).collect();
    Ok((tracks, pls))
}

fn free_space(root: &Path) -> u64 {
    let mut probe = root.to_path_buf();
    while !probe.exists() && probe.pop() {}
    let disks = sysinfo::Disks::new_with_refreshed_list();
    // The disk whose mount point is the longest prefix of the target.
    disks.list().iter().filter(|d| probe.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len()).map(|d| d.available_space()).unwrap_or(0)
}

fn track_path(t: &Row) -> PathBuf {
    PathBuf::from(t["path"].as_str().unwrap_or(""))
}

fn warnings(tracks: &[Row]) -> Vec<Value> {
    tracks.iter().filter_map(|t| {
        let p = track_path(t);
        p.exists().then(|| deck_problem(&p)).flatten()
            .map(|why| json!({"track_id": t["id"], "title": t["title"], "problem": why}))
    }).collect()
}

/// What an export would copy, whether it fits, and which tracks the deck can't play.
pub fn plan(root: &str, playlist_ids: Option<&[i64]>, include_stems: bool) -> Result<Value> {
    let (tracks, pls) = collect(playlist_ids)?;
    let mut need = 0u64;
    for t in &tracks {
        need += t["file_size"].as_u64().unwrap_or(0);
        if include_stems {
            need += std::fs::metadata(stems::stem_file(t["id"].as_i64().unwrap_or(0))).map(|m| m.len()).unwrap_or(0);
        }
    }
    let free = free_space(Path::new(root));
    Ok(json!({
        "tracks": tracks.len(), "playlists": pls.len(), "bytes": need, "free": free, "fits": (need as f64) < free as f64 * 0.98,
        "unanalysed": tracks.iter().filter(|t| t["analysed_at"].is_null()).count(),
        "with_stems": tracks.iter().filter(|t| t["stems"] == "done").count(),
        "warnings": warnings(&tracks), "target_problem": target_problem(Path::new(root)),
        "foreign_library": foreign_library(Path::new(root)),
    }))
}

fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool> {
    let (sm, dm) = (std::fs::metadata(src)?, std::fs::metadata(dst).ok());
    if dm.is_some_and(|d| d.len() == sm.len()) {
        return Ok(false);
    }
    std::fs::create_dir_all(dst.parent().unwrap())?;
    let tmp = dst.with_extension("part");
    std::fs::copy(src, &tmp)?;
    std::fs::rename(&tmp, dst)?;
    Ok(true)
}

pub fn export(root: &str, playlist_ids: Option<&[i64]>, include_stems: bool,
              progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<Value> {
    if let Some(why) = target_problem(Path::new(root)) {
        return Err(anyhow!(why));
    }
    export_to(root, playlist_ids, include_stems, progress, cancel)
}

/// The export itself, without the target-drive check (tests write to a temp folder).
fn export_to(root: &str, playlist_ids: Option<&[i64]>, include_stems: bool,
             progress: &dyn Fn(f64, &str), cancel: &dyn Fn() -> bool) -> Result<Value> {
    let root = PathBuf::from(root);
    // Before anything is written: never overwrite a rekordbox library someone else made.
    if let Some(why) = foreign_library(&root) {
        return Err(anyhow!("{why} DKPRP won't overwrite it. Use another drive, or move that library first."));
    }
    std::fs::create_dir_all(&root)?;
    let (tracks, pls) = collect(playlist_ids)?;
    if tracks.is_empty() {
        return Err(anyhow!("Nothing to export"));
    }
    let p = plan(&root.to_string_lossy(), playlist_ids, include_stems)?;
    if p["fits"] != json!(true) {
        return Err(anyhow!("Not enough space: needs {:.1} GB, {:.1} GB free",
                           p["bytes"].as_f64().unwrap_or(0.0) / 1e9, p["free"].as_f64().unwrap_or(0.0) / 1e9));
    }

    let mut database = pdb::Database::new();
    // name -> (id, artist id for albums), in first-seen order
    let mut names: HashMap<&str, Vec<(String, u32, u32)>> = ["artist", "album", "genre", "key"].iter().map(|k| (*k, vec![])).collect();
    let mut name_id = |kind: &'static str, value: Option<&str>, artist: u32| -> u32 {
        let Some(v) = value.filter(|v| !v.is_empty()) else { return 0 };
        let table = names.get_mut(kind).unwrap();
        if let Some(e) = table.iter().find(|e| e.0 == v) {
            return e.1;
        }
        let id = table.len() as u32 + 1;
        table.push((v.to_string(), id, artist));
        id
    };

    let mut stem_manifest = Map::new();
    let (mut copied, mut skipped) = (0, 0);
    let art_dir = root.join("PIONEER/Artwork/00001");
    let mut artwork_rows = vec![];
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let mut exported = vec![];

    for (n, t) in tracks.iter().enumerate() {
        if cancel() {
            return Err(anyhow!("cancelled"));
        }
        let tid = t["id"].as_i64().unwrap_or(0);
        let s = |k: &str| t[k].as_str().map(str::to_string);
        progress(0.9 * n as f64 / tracks.len() as f64, &format!("{} – {}", s("artist").unwrap_or_default(), s("title").unwrap_or_default()));
        let src = track_path(t);
        if !src.exists() {
            skipped += 1;
            continue;
        }
        let a = match analysis::load_analysis(tid) {
            Some(a) => a,
            None => analysis::analyse(tid, &|_, _| {})?,
        };
        let t = db::track(tid)?.unwrap_or_else(|| t.clone());
        let s = |k: &str| t[k].as_str().map(str::to_string);

        // 1. Audio file, copied byte for byte (never converted or resampled)
        let fname = src.file_name().map(|f| f.to_string_lossy().to_string());
        let rel_audio = format!("Contents/{}/{}/{}", safe(s("artist").as_deref(), "Unknown Artist"),
                                safe(s("album").as_deref(), "Unknown Album"),
                                safe(fname.as_deref(), &format!("{tid}.{}", library::ext(&src))));
        let dst = root.join(&rel_audio);
        if copy_if_changed(&src, &dst)? {
            copied += 1;
        }
        let audio_path = format!("/{rel_audio}");

        // 2. Analysis files
        let cue_rows = library::cues(tid)?;
        let cue = |q: &Row| anlz::Cue { start_ms: q["start_ms"].as_u64().unwrap_or(0) as u32,
                                        end_ms: q["end_ms"].as_u64().map(|v| v as u32), hot: q["slot"].as_u64().unwrap_or(0) as u32 };
        let ta = anlz::TrackAnalysis {
            path: audio_path.clone(), beats: analysis::beats_from_grid(&a),
            pwav: analysis::waveform_bytes(&a, "pwav"), pwv2: analysis::waveform_bytes(&a, "pwv2"),
            pwv3: analysis::waveform_bytes(&a, "pwv3"), pwv4: analysis::waveform_bytes(&a, "pwv4"),
            pwv5: analysis::waveform_bytes(&a, "pwv5"),
            hot_cues: cue_rows.iter().filter(|q| (1..=8).contains(&q["slot"].as_i64().unwrap_or(0))).map(cue).collect(),
            memory_cues: cue_rows.iter().filter(|q| q["slot"].as_i64() == Some(0)).map(cue).collect(),
        };
        let adir = root.join(anlz_rel(tid as u32));
        std::fs::create_dir_all(&adir)?;
        std::fs::write(adir.join("ANLZ0000.DAT"), anlz::build_dat(&ta))?;
        std::fs::write(adir.join("ANLZ0000.EXT"), anlz::build_ext(&ta))?;

        // 3. Artwork (80 px and 240 px, as rekordbox writes them)
        let mut art_id = 0;
        if let Some(img) = s("artwork").and_then(|p| image::open(p).ok()) {
            std::fs::create_dir_all(&art_dir)?;
            for (suffix, size) in [("", 80), ("_m", 240)] {
                library::save_jpeg(&img.thumbnail(size, size).to_rgb8(), &art_dir.join(format!("a{tid}{suffix}.jpg")), 85)?;
            }
            art_id = artwork_rows.len() as u32 + 1;
            artwork_rows.push((art_id, format!("/PIONEER/Artwork/00001/a{tid}.jpg")));
        }

        // 4. Database row: rate, bit depth and bitrate from the file itself
        let tags = library::read_tags(&src);
        let depth = lofty::read_from_path(&src).ok().and_then(|f| lofty::prelude::AudioFile::properties(&f).bit_depth()).unwrap_or(16);
        let artist_id = name_id("artist", s("artist").as_deref(), 0);
        let added = t["added_at"].as_f64().and_then(|v| chrono::DateTime::from_timestamp(v as i64, 0))
            .map(|d| d.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string()).unwrap_or_else(|| today.clone());
        let num = |k: &str| t[k].as_f64().unwrap_or(0.0);
        let row = pdb::TrackRow {
            id: tid as u32, title: s("title").unwrap_or_default(), file_path: audio_path.clone(),
            filename: dst.file_name().unwrap().to_string_lossy().to_string(), anlz_path: format!("/{}/ANLZ0000.DAT", anlz_rel(tid as u32)),
            artist_id, album_id: name_id("album", s("album").as_deref(), artist_id),
            genre_id: name_id("genre", s("genre").as_deref(), 0), key_id: name_id("key", s("musical_key").as_deref(), 0),
            artwork_id: art_id, color_id: num("color") as u8, bpm: num("bpm"),
            duration_s: (a["duration_ms"].as_f64().unwrap_or(num("duration") * 1000.0) / 1000.0).round() as u16,
            sample_rate: tags.sample_rate.unwrap_or(t["sample_rate"].as_u64().unwrap_or(0) as u32),
            bitrate: tags.bitrate.unwrap_or(0), file_size: std::fs::metadata(&src)?.len() as u32,
            sample_depth: depth as u16, year: num("year") as u16, rating: num("rating") as u8,
            track_number: num("track_number") as u32, comment: s("comment").unwrap_or_default(),
            date_added: added, analyze_date: today.clone(), file_type: file_type(&library::ext(&src)),
        };
        database.add(pdb::T_TRACKS, pdb::track_row(&row, n));
        exported.push(tid);

        // 5. Pocket Deck extras: stems and phrase data
        let ext_id = format!("{tid:08X}");
        let sf = stems::stem_file(tid);
        if include_stems && sf.exists() {
            let rel = format!("POCKETDECK/STEMS/{ext_id}.flac");
            copy_if_changed(&sf, &root.join(&rel))?;
            let _ = std::fs::remove_file(root.join(format!("POCKETDECK/STEMS/{ext_id}.wav")));   // format 2 leftover
            stem_manifest.insert(audio_path.clone(), json!({
                "file": format!("/{rel}"), "format": "flac", "sample_rate": stems::RATE, "channels": stems::CHANNELS,
                "frames": stems::stem_frames(tid)?, "seek_interval_frames": stems::SEEK_FRAMES, "order": stems::STEM_NAMES}));
        }
        let extra = root.join("POCKETDECK/analysis");
        std::fs::create_dir_all(&extra)?;
        std::fs::write(extra.join(format!("{ext_id}.json")), json!({
            "file_path": audio_path, "bpm": a["bpm"], "key": a["key"], "camelot": a["camelot"],
            "energy": a["energy"], "phrases": a["phrases"]}).to_string())?;
    }

    // Name tables
    for (i, (name, id, _)) in names["artist"].iter().enumerate() {
        database.add(pdb::T_ARTISTS, pdb::artist_row(*id, name, i));
    }
    for (i, (name, id, artist)) in names["album"].iter().enumerate() {
        database.add(pdb::T_ALBUMS, pdb::album_row(*id, name, *artist, i));
    }
    for (name, id, _) in &names["genre"] {
        database.add(pdb::T_GENRES, pdb::simple_name_row(*id, name));
    }
    for (name, id, _) in &names["key"] {
        database.add(pdb::T_KEYS, pdb::key_row(*id, name));
    }
    for (id, path) in &artwork_rows {
        database.add(pdb::T_ARTWORK, pdb::artwork_row(*id, path));
    }

    // Playlists
    for pl in &pls {
        let pid = pl["id"].as_i64().unwrap_or(0);
        let folder = pl["is_folder"].as_i64() == Some(1);
        database.add(pdb::T_PLAYLIST_TREE, pdb::playlist_tree_row(pid as u32, pl["parent_id"].as_i64().unwrap_or(0) as u32,
                                                                  pl["sort"].as_i64().unwrap_or(0) as u32, pl["name"].as_str().unwrap_or(""), folder));
        if !folder {
            for (i, t) in library::playlist_track_ids(pid)?.iter().filter(|t| exported.contains(t)).enumerate() {
                database.add(pdb::T_PLAYLIST_ENTRIES, pdb::playlist_entry_row(i as u32 + 1, *t as u32, pid as u32));
            }
        }
    }

    progress(0.95, "Writing database");
    let out = root.join("PIONEER/rekordbox");
    std::fs::create_dir_all(&out)?;
    let tmp = out.join("export.pdb.tmp");
    let pdb_bytes = database.write()?;
    std::fs::write(&tmp, &pdb_bytes)?;
    std::fs::rename(&tmp, out.join("export.pdb"))?;            // never leave a half-written database
    let pd = root.join("POCKETDECK");
    std::fs::create_dir_all(&pd)?;
    let n_stems = stem_manifest.len();
    std::fs::write(pd.join("stems.json"), serde_json::to_string_pretty(&json!({"version": 3, "tracks": stem_manifest}))?)?;
    let warn = warnings(&tracks);
    let summary = json!({
        "app": "Pocket Prep", "version": env!("CARGO_PKG_VERSION"),
        "exported_at": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        "tracks": tracks.len() - skipped, "playlists": pls.len(), "copied": copied,
        "missing_files": skipped, "stems": n_stems, "warnings": warn,
        // Lets the next export tell whether export.pdb is still ours (see foreign_library).
        "pdb_blake3": blake3::hash(&pdb_bytes).to_hex().to_string(), "pdb_size": pdb_bytes.len()});
    std::fs::write(pd.join("export.json"), serde_json::to_string_pretty(&summary)?)?;
    progress(1.0, "Done");
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_files_the_deck_cant_play() {
        let dir = std::env::temp_dir().join("pocketprep_deckcheck");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = |bits: u16, name: &str| {
            let mut h = crate::stems::wav_header(2, 44100, 0);
            h[34..36].copy_from_slice(&bits.to_le_bytes());
            let p = dir.join(name);
            std::fs::write(&p, h).unwrap();
            p
        };
        assert!(deck_problem(&wav(16, "ok.wav")).is_none());
        assert!(deck_problem(&wav(24, "hi.wav")).unwrap().contains("24-bit"));
        assert!(deck_problem(Path::new("x.m4a")).is_some());
        assert!(deck_problem(Path::new("x.flac")).is_none());
    }
}

#[cfg(test)]
mod drive_tests {
    use super::*;

    #[test]
    fn only_fat_and_exfat_pass() {
        for ok in ["FAT32", "exFAT", "msdos", "exfat"] {
            assert!(deck_filesystem_problem(ok).is_none(), "{ok}");
        }
        for bad in ["NTFS", "apfs", "hfs", "ReFS"] {
            assert!(deck_filesystem_problem(bad).unwrap().contains("FAT32 and exFAT"), "{bad}");
        }
    }

    #[test]
    fn system_drive_is_never_a_target() {
        let sys = if cfg!(windows) { format!("{}\\", std::env::var("SystemDrive").unwrap_or_else(|_| "C:".into())) } else { "/".into() };
        assert!(is_system(Path::new(&sys)));
        assert!(target_problem(&Path::new(&sys).join("PocketPrepTest")).unwrap().contains("system drive"));
        assert!(list_drives(true).iter().all(|d| !is_system(Path::new(d["root"].as_str().unwrap()))));
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use binrw::BinRead;
    use fallible_iterator::FallibleIterator;
    use rekordcrate::anlz::{Content, CueListType, ANLZ};
    use rekordcrate::pdb::io::Database as Reader;
    use rekordcrate::pdb::{DatabaseType, PlaylistEntry, PlaylistTreeNode, Track};

    /// The whole pipeline on a throwaway library: import a synthetic track, analyse it,
    /// set a hot cue, put it in a playlist, export, then read the card back with
    /// rekordcrate and follow every reference (PDB -> audio file, PDB -> ANLZ files).
    /// The only test that touches the library database (its location is process-wide).
    #[test]
    fn full_export_reads_back_with_rekordcrate() {
        let tmp = std::env::temp_dir().join(format!("pocketprep_export_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("POCKET_PREP_DATA", tmp.join("data"));
        std::fs::create_dir_all(tmp.join("music")).unwrap();
        let song = tmp.join("music").join("Testartist - Testsång.wav");
        let y = crate::testaudio::music(44100, 40.0, 128.0, 0.1, crate::testaudio::A_MINOR);
        std::fs::write(&song, crate::testaudio::wav(44100, &y)).unwrap();

        let ids = library::import_paths(&[song.to_string_lossy().to_string()]).unwrap();
        let tid = ids[0];
        analysis::analyse(tid, &|_, _| {}).unwrap();
        library::set_cues(tid, &[json!({"slot": 1, "start_ms": 1000, "label": "A"}), json!({"slot": 0, "start_ms": 5000})]).unwrap();
        let pl = library::create_playlist("Fredag", 0, false).unwrap();
        library::add_to_playlist(pl["id"].as_i64().unwrap(), &[tid]).unwrap();

        // Stems: the track itself in all four slots is enough to check the file and manifest.
        let mono: Vec<f32> = y.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        crate::stems::write_stems(&crate::stems::stem_file(tid), &std::array::from_fn(|_| (mono.clone(), mono.clone()))).unwrap();

        let card = tmp.join("card");
        let summary = export_to(&card.to_string_lossy(), None, true, &|_, _| {}, &|| false).unwrap();
        assert_eq!((summary["tracks"].as_i64(), summary["stems"].as_i64()), (Some(1), Some(1)));

        // export.pdb: the track, its playlist, and the entry linking them.
        let pdb = std::fs::read(card.join("PIONEER/rekordbox/export.pdb")).unwrap();
        let mut r = Reader::open_non_persistent(std::io::Cursor::new(pdb.as_slice()), DatabaseType::Plain).unwrap();
        let tracks: Vec<(u32, String, String)> = r.iter_rows::<Track>().unwrap()
            .map(|t| Ok((t.id.0, t.offsets.inner.title.clone().into_string().unwrap(),
                         t.offsets.inner.file_path.clone().into_string().unwrap())))
            .collect().unwrap();
        assert_eq!(tracks.len(), 1);
        let (pdb_id, title, path) = &tracks[0];
        assert_eq!(title, "Testartist - Testsång");
        assert!(card.join(path.trim_start_matches('/')).exists(), "audio missing at {path}");
        assert_eq!(std::fs::read(card.join(path.trim_start_matches('/'))).unwrap(), std::fs::read(&song).unwrap(), "audio not byte-identical");
        let names: Vec<String> = r.iter_rows::<PlaylistTreeNode>().unwrap().map(|p| Ok(p.name.clone().into_string().unwrap())).collect().unwrap();
        assert_eq!(names, ["Fredag"]);
        assert_eq!(r.iter_rows::<PlaylistEntry>().unwrap().count().unwrap(), 1);

        // ANLZ files: path matches the PDB, beat grid at 128 BPM, our two cues.
        let adir = card.join(anlz_rel(*pdb_id));
        for name in ["ANLZ0000.DAT", "ANLZ0000.EXT"] {
            let bytes = std::fs::read(adir.join(name)).unwrap();
            let anlz = ANLZ::read(&mut std::io::Cursor::new(&bytes)).unwrap();
            for s in &anlz.sections {
                match &s.content {
                    Content::Path(p) => assert_eq!(&p.path.to_string(), path),
                    Content::BeatGrid(g) => {
                        assert!(g.beats.len() > 80, "{} beats", g.beats.len());
                        assert!(g.beats.iter().all(|b| (b.tempo as i32 - 12800).abs() <= 5));
                    }
                    Content::CueList(c) if name.ends_with("DAT") => match c.list_type {
                        CueListType::HotCues => assert_eq!(c.cues.iter().map(|q| (q.hot_cue, q.time)).collect::<Vec<_>>(), [(1, 1000)]),
                        CueListType::MemoryCues => assert_eq!(c.cues.iter().map(|q| q.time).collect::<Vec<_>>(), [5000]),
                    },
                    _ => {}
                }
            }
        }

        // stems.json: keyed by the PDB path, pointing at a decodable 8-channel FLAC of the right length.
        let manifest: Value = serde_json::from_slice(&std::fs::read(card.join("POCKETDECK/stems.json")).unwrap()).unwrap();
        assert_eq!(manifest["version"], 3);
        let e = &manifest["tracks"][path.as_str()];
        assert_eq!((e["format"].as_str(), e["channels"].as_i64(), e["seek_interval_frames"].as_i64()), (Some("flac"), Some(8), Some(4410)));
        assert_eq!(e["frames"].as_u64(), Some(y.len() as u64));
        let flac = claxon::FlacReader::open(card.join(e["file"].as_str().unwrap().trim_start_matches('/'))).unwrap();
        assert_eq!((flac.streaminfo().channels, flac.streaminfo().samples), (8, Some(y.len() as u64)));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod interop_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn copy_dir(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).unwrap();
        for e in std::fs::read_dir(src).unwrap().flatten() {
            let (s, d) = (e.path(), dst.join(e.file_name()));
            if s.is_dir() { copy_dir(&s, &d) } else { std::fs::copy(&s, &d).unwrap(); }
        }
    }

    /// Every file under `dir` with its bytes.
    fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut todo = vec![dir.to_path_buf()];
        while let Some(d) = todo.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() { todo.push(p) } else { out.insert(p.strip_prefix(dir).unwrap().to_path_buf(), std::fs::read(&p).unwrap()); }
            }
        }
        out
    }

    /// A real rekordbox export (tests/fixtures) must survive an export from us unchanged:
    /// we refuse before writing anything, and every file keeps its bytes.
    #[test]
    fn real_rekordbox_export_is_left_untouched() {
        let card = std::env::temp_dir().join(format!("pocketprep_rb_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&card);
        copy_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rekordbox_demo_tracks"), &card);
        let before = snapshot(&card);
        assert!(before.len() >= 12);
        assert!(foreign_library(&card).unwrap().contains("didn't write"));
        let err = export_to(&card.to_string_lossy(), None, true, &|_, _| {}, &|| false).unwrap_err();
        assert!(err.to_string().contains("won't overwrite"), "{err}");
        assert_eq!(snapshot(&card), before, "the rekordbox library was changed");
        let _ = std::fs::remove_dir_all(&card);
    }

    /// Ours = export.json from Pocket Prep whose hash matches export.pdb.
    #[test]
    fn ownership_follows_the_recorded_hash() {
        let card = std::env::temp_dir().join(format!("pocketprep_own_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&card);
        std::fs::create_dir_all(card.join("PIONEER/rekordbox")).unwrap();
        std::fs::create_dir_all(card.join("POCKETDECK")).unwrap();
        assert!(foreign_library(&card).is_none(), "an empty drive is fine");
        let pdb = b"pretend export.pdb".to_vec();
        std::fs::write(card.join("PIONEER/rekordbox/export.pdb"), &pdb).unwrap();
        let info = |hash: Option<String>| std::fs::write(card.join("POCKETDECK/export.json"),
            json!({"app": "Pocket Prep", "pdb_blake3": hash}).to_string()).unwrap();
        info(Some(blake3::hash(&pdb).to_hex().to_string()));
        assert!(foreign_library(&card).is_none(), "ours, unchanged");
        info(Some(blake3::hash(b"something else").to_hex().to_string()));
        assert!(foreign_library(&card).unwrap().contains("changed by another program"));
        info(None);
        assert!(foreign_library(&card).is_none(), "written before hashes were recorded");
        std::fs::write(card.join("POCKETDECK/export.json"), r#"{"app": "Other"}"#).unwrap();
        assert!(foreign_library(&card).unwrap().contains("didn't write"));
        let _ = std::fs::remove_dir_all(&card);
    }
}
