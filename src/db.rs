//! The library database (SQLite). One file, no server. Same file and schema as
//! the old Python app, so an existing library carries over.
//!
//! Tables
//!   tracks          one row per audio file
//!   cues            hot cues (slot 1-8) and memory cues (slot 0) per track
//!   playlists       folders and playlists (a tree via parent_id)
//!   playlist_tracks which tracks are in which playlist, in order
//!   history         every edit, so changes can be undone and audited

use anyhow::Result;
use rusqlite::types::{Value as Sql, ValueRef};
use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Map, Value};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::config;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  title TEXT, artist TEXT, album TEXT, genre TEXT, comment TEXT,
  year INTEGER, track_number INTEGER,
  duration REAL, sample_rate INTEGER, bitrate INTEGER, file_size INTEGER,
  bpm REAL, musical_key TEXT, camelot TEXT, energy INTEGER,
  rating INTEGER DEFAULT 0, color INTEGER DEFAULT 0,
  artwork TEXT,
  added_at REAL, analysed_at REAL,
  stems TEXT DEFAULT 'none',          -- none | queued | running | done | failed
  deleted INTEGER DEFAULT 0           -- soft delete, so it can be undone
);
CREATE TABLE IF NOT EXISTS cues (
  id INTEGER PRIMARY KEY,
  track_id INTEGER NOT NULL REFERENCES tracks(id),
  slot INTEGER NOT NULL,              -- 1-8 hot cue A-H, 0 memory cue
  start_ms INTEGER NOT NULL,
  end_ms INTEGER,                     -- set for loops
  label TEXT,
  auto INTEGER DEFAULT 0              -- placed by analysis, not by the user
);
CREATE TABLE IF NOT EXISTS playlists (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  parent_id INTEGER DEFAULT 0,
  is_folder INTEGER DEFAULT 0,
  sort INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS playlist_tracks (
  playlist_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY,
  at REAL, action TEXT, detail TEXT, undo TEXT
);
CREATE INDEX IF NOT EXISTS cues_track ON cues(track_id);
CREATE INDEX IF NOT EXISTS pt_playlist ON playlist_tracks(playlist_id, position);
";

pub type Row = Map<String, Value>;

static CONN: OnceLock<Mutex<Connection>> = OnceLock::new();

/// The one shared connection. Holding the guard is the transaction lock.
pub fn conn() -> MutexGuard<'static, Connection> {
    CONN.get_or_init(|| {
        config::ensure_dirs().expect("can't create the data folder");
        let c = Connection::open(config::db_path()).expect("can't open the library database");
        // Exclusive locking before WAL: SQLite then keeps the WAL index in ordinary memory
        // instead of shared memory, which network and redirected folders can't map
        // ("disk I/O error"). The app has one connection, and this also stops a second copy
        // of the app from using the same library at the same time.
        c.pragma_update(None, "locking_mode", "EXCLUSIVE").expect("can't open the library database");
        c.pragma_update(None, "journal_mode", "WAL").expect("can't open the library database");
        c.execute_batch(SCHEMA).unwrap();
        // Stems from the Python app were separate MP3s; the new format is one 8-channel WAV.
        // MP3 encoder delay makes the old files drift from the track, so they must be remade.
        let mut st = c.prepare("SELECT id FROM tracks WHERE stems='done'").unwrap();
        let ids: Vec<i64> = st.query_map([], |r| r.get(0)).unwrap().flatten().collect();
        drop(st);
        for id in ids {
            if !crate::stems::has_stems(id) {
                c.execute("UPDATE tracks SET stems='none' WHERE id=?", [id]).unwrap();
            }
        }
        c.execute("UPDATE tracks SET stems='none' WHERE stems IN ('queued','running')", []).unwrap();
        // A "0" comment came from the tag, not the user (see library::read_tags).
        c.execute("UPDATE tracks SET comment=NULL WHERE trim(comment)='0'", []).unwrap();
        Mutex::new(c)
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

/// Run a block of writes as one transaction.
pub fn tx<T>(f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
    let mut c = conn();
    let t = c.transaction()?;
    let out = f(&t)?;
    t.commit()?;
    Ok(out)
}

pub fn json_to_sql(v: &Value) -> Sql {
    match v {
        Value::Null => Sql::Null,
        Value::Bool(b) => Sql::Integer(*b as i64),
        Value::Number(n) => n.as_i64().map(Sql::Integer).unwrap_or_else(|| Sql::Real(n.as_f64().unwrap_or(0.0))),
        Value::String(s) => Sql::Text(s.clone()),
        other => Sql::Text(other.to_string()),
    }
}

pub fn query_on(c: &Connection, sql: &str, args: &[Value]) -> Result<Vec<Row>> {
    let mut st = c.prepare(sql)?;
    let names: Vec<String> = st.column_names().iter().map(|s| s.to_string()).collect();
    let mut rows = st.query(params_from_iter(args.iter().map(json_to_sql)))?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let mut m = Map::new();
        for (i, n) in names.iter().enumerate() {
            let v = match r.get_ref(i)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(x) => json!(x),
                ValueRef::Real(x) => json!(x),
                ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
                ValueRef::Blob(_) => Value::Null,
            };
            m.insert(n.clone(), v);
        }
        out.push(m);
    }
    Ok(out)
}

pub fn query(sql: &str, args: &[Value]) -> Result<Vec<Row>> {
    query_on(&conn(), sql, args)
}

pub fn one(sql: &str, args: &[Value]) -> Result<Option<Row>> {
    Ok(query(sql, args)?.into_iter().next())
}

pub fn track(id: i64) -> Result<Option<Row>> {
    one("SELECT * FROM tracks WHERE id=?", &[json!(id)])
}

pub fn log(c: &Connection, action: &str, detail: Value, undo: Option<Value>) -> Result<i64> {
    c.execute("INSERT INTO history(at, action, detail, undo) VALUES (?,?,?,?)",
              params![now(), action, detail.to_string(), undo.map(|u| u.to_string())])?;
    Ok(c.last_insert_rowid())
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

/// "?,?,?" for an IN (...) list.
pub fn marks(n: usize) -> String {
    vec!["?"; n].join(",")
}

const TRACK_EDITABLE: &[&str] = &["title", "artist", "album", "genre", "comment", "year", "rating", "color", "bpm", "musical_key"];

pub fn update_track(id: i64, fields: &Row) -> Result<()> {
    let fields: Vec<(&String, &Value)> = fields.iter().filter(|(k, _)| TRACK_EDITABLE.contains(&k.as_str())).collect();
    if fields.is_empty() {
        return Ok(());
    }
    tx(|c| {
        let before = query_on(c, "SELECT * FROM tracks WHERE id=?", &[json!(id)])?.pop().unwrap_or_default();
        // Column names come from TRACK_EDITABLE only, so formatting them in is safe.
        let sets = fields.iter().map(|(k, _)| format!("{k}=?")).collect::<Vec<_>>().join(", ");
        let mut args: Vec<Value> = fields.iter().map(|(_, v)| (*v).clone()).collect();
        args.push(json!(id));
        c.execute(&format!("UPDATE tracks SET {sets} WHERE id=?"), params_from_iter(args.iter().map(json_to_sql)))?;
        let mut detail = Map::new();
        detail.insert("id".into(), json!(id));
        let mut old = Map::new();
        for (k, v) in &fields {
            detail.insert((*k).clone(), (*v).clone());
            old.insert((*k).clone(), before.get(*k).cloned().unwrap_or(Value::Null));
        }
        log(c, "edit_track", Value::Object(detail), Some(json!({"kind": "track_fields", "id": id, "fields": old})))
    })?;
    Ok(())
}

/// Reverse one logged change. The reversal is itself logged.
pub fn undo(history_id: i64) -> Result<bool> {
    let Some(row) = one("SELECT * FROM history WHERE id=?", &[json!(history_id)])? else { return Ok(false) };
    let Some(u) = row["undo"].as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()) else { return Ok(false) };
    tx(|c| {
        match u["kind"].as_str().unwrap_or("") {
            "track_fields" => {
                let f = u["fields"].as_object().cloned().unwrap_or_default();
                let f: Vec<_> = f.iter().filter(|(k, _)| TRACK_EDITABLE.contains(&k.as_str())).collect();
                if !f.is_empty() {
                    let sets = f.iter().map(|(k, _)| format!("{k}=?")).collect::<Vec<_>>().join(", ");
                    let mut args: Vec<Sql> = f.iter().map(|(_, v)| json_to_sql(v)).collect();
                    args.push(json_to_sql(&u["id"]));
                    c.execute(&format!("UPDATE tracks SET {sets} WHERE id=?"), params_from_iter(args))?;
                }
            }
            "undelete" => {
                let ids = u["ids"].as_array().cloned().unwrap_or_default();
                c.execute(&format!("UPDATE tracks SET deleted=0 WHERE id IN ({})", marks(ids.len())),
                          params_from_iter(ids.iter().map(json_to_sql)))?;
            }
            "playlist_restore" => {
                let pid = u["playlist_id"].as_i64().unwrap_or(0);
                c.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", [pid])?;
                for (i, t) in u["tracks"].as_array().cloned().unwrap_or_default().iter().enumerate() {
                    c.execute("INSERT INTO playlist_tracks VALUES (?,?,?)", params![pid, t.as_i64(), i as i64])?;
                }
            }
            "cues_restore" => {
                let tid = u["track_id"].as_i64().unwrap_or(0);
                c.execute("DELETE FROM cues WHERE track_id=?", [tid])?;
                for q in u["cues"].as_array().cloned().unwrap_or_default() {
                    c.execute("INSERT INTO cues(track_id, slot, start_ms, end_ms, label, auto) VALUES (?,?,?,?,?,?)",
                              params![tid, q["slot"].as_i64(), q["start_ms"].as_i64(), q["end_ms"].as_i64(),
                                      q["label"].as_str(), q["auto"].as_i64().unwrap_or(0)])?;
                }
            }
            _ => {}
        }
        c.execute("UPDATE history SET undo=NULL WHERE id=?", [history_id])?;
        log(c, "undo", json!({"history_id": history_id, "action": row["action"]}), None)?;
        Ok(true)
    })
}
