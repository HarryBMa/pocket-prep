"""The library database (SQLite). One file, no server.

Tables
  tracks          one row per audio file
  cues            hot cues (slot 1-8) and memory cues (slot 0) per track
  playlists       folders and playlists (a tree via parent_id)
  playlist_tracks which tracks are in which playlist, in order
  history         every edit, so changes can be undone and audited
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
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
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.ensure_dirs()
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
    return _conn


@contextmanager
def tx():
    """Run a block of writes as one transaction (thread-safe)."""
    with _lock:
        c = connect()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise


def query(sql: str, args: tuple = ()) -> list[dict]:
    with _lock:
        return [dict(r) for r in connect().execute(sql, args).fetchall()]


def one(sql: str, args: tuple = ()) -> dict | None:
    rows = query(sql, args)
    return rows[0] if rows else None


def log(c: sqlite3.Connection, action: str, detail: dict, undo: dict | None = None) -> int:
    cur = c.execute("INSERT INTO history(at, action, detail, undo) VALUES (?,?,?,?)",
                    (time.time(), action, json.dumps(detail), json.dumps(undo) if undo else None))
    return cur.lastrowid


TRACK_EDITABLE = {"title", "artist", "album", "genre", "comment", "year", "rating", "color", "bpm", "musical_key"}


def update_track(track_id: int, fields: dict) -> dict:
    fields = {k: v for k, v in fields.items() if k in TRACK_EDITABLE}
    if not fields:
        return one("SELECT * FROM tracks WHERE id=?", (track_id,))
    with tx() as c:
        before = dict(c.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone())
        sets = ", ".join(f"{k}=?" for k in fields)
        c.execute(f"UPDATE tracks SET {sets} WHERE id=?", (*fields.values(), track_id))
        log(c, "edit_track", {"id": track_id, **fields},
            {"kind": "track_fields", "id": track_id, "fields": {k: before[k] for k in fields}})
    return one("SELECT * FROM tracks WHERE id=?", (track_id,))


def undo(history_id: int) -> bool:
    """Reverse one logged change. The reversal is itself logged."""
    row = one("SELECT * FROM history WHERE id=?", (history_id,))
    if not row or not row["undo"]:
        return False
    u = json.loads(row["undo"])
    with tx() as c:
        if u["kind"] == "track_fields":
            sets = ", ".join(f"{k}=?" for k in u["fields"])
            c.execute(f"UPDATE tracks SET {sets} WHERE id=?", (*u["fields"].values(), u["id"]))
        elif u["kind"] == "undelete":
            c.execute(f"UPDATE tracks SET deleted=0 WHERE id IN ({','.join('?' * len(u['ids']))})", u["ids"])
        elif u["kind"] == "playlist_restore":
            c.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", (u["playlist_id"],))
            c.executemany("INSERT INTO playlist_tracks VALUES (?,?,?)",
                          [(u["playlist_id"], t, i) for i, t in enumerate(u["tracks"])])
        elif u["kind"] == "cues_restore":
            c.execute("DELETE FROM cues WHERE track_id=?", (u["track_id"],))
            c.executemany("INSERT INTO cues(track_id, slot, start_ms, end_ms, label, auto) VALUES (?,?,?,?,?,?)",
                          [(u["track_id"], q["slot"], q["start_ms"], q["end_ms"], q["label"], q["auto"]) for q in u["cues"]])
        c.execute("UPDATE history SET undo=NULL WHERE id=?", (history_id,))
        log(c, "undo", {"history_id": history_id, "action": row["action"]})
    return True
