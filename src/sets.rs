//! Sets: a playlist read as the order it will be played in. The set report lines the tracks up with
//! the transition between each pair (transitions.rs), the energy curve, the running time and a load
//! estimate; "order by" hands the tracks to ClubCraft's planner (`order_tracks`) for a key, tempo,
//! energy or balanced order. The export writes `POCKETDECK/sets/<playlist id>.json` from the same
//! report so the deck can show "next in N bars" without working anything out.

use anyhow::Result;
use clubcraft_core::planner::{order_tracks, OrderItem, OrderOptions};
use clubcraft_core::Key;
use serde_json::{json, Value};

use crate::{analysis, db, library, server::TRACK_COLUMNS, transitions};

/// Bytes a DECK001 reads per second when it loads a track (SD card, class 10): the load estimate's base.
const LOAD_BYTES_PER_S: f64 = 20e6;

/// The playlist's tracks in order, as rows with the shifted key.
fn rows(pid: i64) -> Result<Vec<db::Row>> {
    let all = db::query(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0"), &[])?;
    let mut out: Vec<db::Row> = library::playlist_track_ids(pid)?.iter().filter_map(|i| all.iter().find(|r| r["id"] == json!(i)).cloned()).collect();
    out.iter_mut().for_each(analysis::add_shifted_key);
    for t in out.iter_mut() {
        let missing = !std::path::Path::new(t["path"].as_str().unwrap_or("")).exists();
        t.insert("file_missing".into(), json!(missing));
    }
    Ok(out)
}

/// The set report for a playlist: tracks in order, the transition between each consecutive pair (null where a
/// side isn't analysed), running time, the energy curve and the load estimate.
pub fn report(pid: i64) -> Result<Value> {
    let pl = db::one("SELECT id, name, is_folder FROM playlists WHERE id=?", &[json!(pid)])?.ok_or_else(|| anyhow::anyhow!("No such playlist"))?;
    let tracks = rows(pid)?;
    let sides: Vec<Option<transitions::Side>> = tracks.iter().map(transitions::side).collect();
    let trans: Vec<Value> = sides.windows(2).map(|w| match (&w[0], &w[1]) {
        (Some(a), Some(b)) => transitions::score(a, b),
        _ => Value::Null,
    }).collect();
    let duration: f64 = tracks.iter().map(|t| t["duration"].as_f64().unwrap_or(0.0)).sum();
    let size = |p: &str| std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0);
    let bytes: f64 = tracks.iter().map(|t| size(t["path"].as_str().unwrap_or("")) + if t["stems"] == "done" { crate::stems::stem_file(t["id"].as_i64().unwrap_or(0)).metadata().map(|m| m.len() as f64).unwrap_or(0.0) } else { 0.0 }).sum();
    let count = |v: &str| trans.iter().filter(|t| t["verdict"] == v).count();
    Ok(json!({
        "playlist": {"id": pid, "name": pl["name"]},
        "tracks": tracks, "transitions": trans,
        "duration_s": duration, "bytes": bytes, "load_s": bytes / LOAD_BYTES_PER_S,
        "energy": tracks.iter().map(|t| t["energy"].clone()).collect::<Vec<_>>(),
        "verdicts": {"good": count("good"), "ok": count("ok"), "no": count("no"), "unknown": trans.iter().filter(|t| t.is_null()).count()},
    }))
}

/// Reorder the playlist with ClubCraft's planner. `by`: "key" (harmonic), "tempo", "energy" (a rising arc) or
/// "auto" (balanced). Tracks without analysis keep a neutral cost. Returns the undo id.
pub fn order(pid: i64, by: &str) -> Result<i64> {
    let tracks = rows(pid)?;
    let items: Vec<OrderItem> = tracks.iter().map(|t| OrderItem {
        id: t["id"].as_i64().unwrap_or(0),
        bpm: t["bpm"].as_f64().filter(|b| *b > 0.0),
        key: t["shifted_camelot"].as_str().or(t["camelot"].as_str()).and_then(Key::from_camelot),
        energy: t["energy"].as_u64().map(|e| e as u8),
    }).collect();
    let o = match by {
        "key" => OrderOptions { key_weight: 0.9, energy_weight: 0.1, ..Default::default() },
        "tempo" => OrderOptions { key_weight: 0.1, energy_weight: 0.1, ..Default::default() },
        "energy" => OrderOptions { key_weight: 0.5, energy_weight: 1.5, ..Default::default() },
        _ => OrderOptions::default(),
    };
    let o = OrderOptions { iterations: 20_000.max(items.len() * items.len() * 50).min(200_000), ..o };
    let ids = order_tracks(&items, &o);
    library::set_playlist_tracks(pid, &ids, "order_set")
}

/// What the card gets for a set: the order with each track's file and mix points, and the window between
/// consecutive tracks (beats, out / in, verdict, reasons). Paths are the card's (`/Contents/...`).
pub fn card_json(pid: i64, card_path: &dyn Fn(i64) -> Option<String>) -> Result<Value> {
    let r = report(pid)?;
    let tracks: Vec<Value> = r["tracks"].as_array().unwrap().iter().filter_map(|t| {
        let id = t["id"].as_i64()?;
        Some(json!({"track_id": id, "file_path": card_path(id)?, "title": t["title"], "artist": t["artist"], "bpm": t["bpm"],
                    "camelot": t["shifted_camelot"].as_str().or(t["camelot"].as_str()), "energy": t["energy"]}))
    }).collect();
    Ok(json!({"version": 1, "playlist_id": pid, "name": r["playlist"]["name"], "duration_s": r["duration_s"], "tracks": tracks,
              "transitions": r["transitions"].as_array().unwrap().iter().map(|t| if t.is_null() { Value::Null } else {
                  json!({"beats": t["beats"], "out_ms": t["out_ms"], "in_ms": t["in_ms"], "verdict": t["verdict"], "score": t["score"],
                         "reasons": t["reasons"].as_array().unwrap().iter().map(|x| x["text"].clone()).collect::<Vec<_>>()}) }).collect::<Vec<_>>()}))
}
