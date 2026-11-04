"""Importing tracks and organising them: tags, artwork, playlists, cues."""
from __future__ import annotations

import time
from pathlib import Path

import mutagen

from . import config, db


def _first(tags, *keys):
    for k in keys:
        v = tags.get(k) if tags else None
        if v:
            return str(v[0]) if isinstance(v, list) else str(v)
    return None


def _int(text):
    try:
        return int(str(text).split("/")[0])
    except (TypeError, ValueError):
        return None


def read_tags(path: Path) -> dict:
    """Title, artist etc. from the file's own tags (falls back to the filename)."""
    info = {"title": path.stem, "artist": None, "album": None, "genre": None, "comment": None,
            "year": None, "track_number": None, "duration": None, "sample_rate": None, "bitrate": None}
    try:
        f = mutagen.File(path, easy=True)
    except Exception:
        f = None
    if f is not None:
        t = f.tags
        info.update(
            title=_first(t, "title") or path.stem,
            artist=_first(t, "artist", "albumartist"),
            album=_first(t, "album"),
            genre=_first(t, "genre"),
            comment=_first(t, "comment"),
            year=_int(_first(t, "date", "year")),
            track_number=_int(_first(t, "tracknumber")),
        )
        if f.info:
            info["duration"] = getattr(f.info, "length", None)
            info["sample_rate"] = getattr(f.info, "sample_rate", None)
            br = getattr(f.info, "bitrate", None)
            info["bitrate"] = int(br / 1000) if br else None
    return info


def extract_artwork(path: Path, track_id: int) -> str | None:
    """Save embedded cover art as <artwork dir>/<id>.jpg; returns the path."""
    try:
        f = mutagen.File(path)
    except Exception:
        return None
    data = None
    if f is None:
        return None
    tags = f.tags
    if tags is not None:
        if hasattr(tags, "getall"):                     # ID3 (mp3, aiff, wav)
            pics = tags.getall("APIC")
            data = pics[0].data if pics else None
        elif "covr" in tags:                            # m4a
            data = bytes(tags["covr"][0])
    if data is None and getattr(f, "pictures", None):   # flac
        data = f.pictures[0].data
    if not data:
        return None
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(data)).convert("RGB")
        img.thumbnail((500, 500))
        out = config.ARTWORK_DIR / f"{track_id}.jpg"
        img.save(out, "JPEG", quality=88)
        return str(out)
    except Exception:
        return None


def import_paths(paths: list[str]) -> list[int]:
    """Add files, or every audio file inside folders. Returns new track ids."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.suffix.lower() in config.AUDIO_EXTENSIONS]
        elif p.suffix.lower() in config.AUDIO_EXTENSIONS and p.exists():
            files.append(p)
    new_ids = []
    for f in sorted(files):
        path = str(f.resolve())
        existing = db.one("SELECT id, deleted FROM tracks WHERE path=?", (path,))
        if existing:
            if existing["deleted"]:
                with db.tx() as c:
                    c.execute("UPDATE tracks SET deleted=0 WHERE id=?", (existing["id"],))
            continue
        tags = read_tags(f)
        with db.tx() as c:
            cur = c.execute(
                """INSERT INTO tracks(path, title, artist, album, genre, comment, year, track_number,
                   duration, sample_rate, bitrate, file_size, added_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (path, tags["title"], tags["artist"], tags["album"], tags["genre"], tags["comment"],
                 tags["year"], tags["track_number"], tags["duration"], tags["sample_rate"],
                 tags["bitrate"], f.stat().st_size, time.time()))
            tid = cur.lastrowid
        art = extract_artwork(f, tid)
        if art:
            with db.tx() as c:
                c.execute("UPDATE tracks SET artwork=? WHERE id=?", (art, tid))
        new_ids.append(tid)
    if new_ids:
        with db.tx() as c:
            db.log(c, "import", {"count": len(new_ids)}, {"kind": "none"})
    return new_ids


def remove_tracks(ids: list[int]) -> int:
    """Hide tracks from the library. Files are never deleted. Undoable."""
    with db.tx() as c:
        c.execute(f"UPDATE tracks SET deleted=1 WHERE id IN ({','.join('?' * len(ids))})", ids)
        return db.log(c, "remove_tracks", {"ids": ids}, {"kind": "undelete", "ids": ids})


# ── Playlists ────────────────────────────────────────────────────────────────

def playlist_tree() -> list[dict]:
    rows = db.query("SELECT * FROM playlists ORDER BY parent_id, sort, name")
    counts = {r["playlist_id"]: r["n"] for r in
              db.query("SELECT playlist_id, COUNT(*) n FROM playlist_tracks GROUP BY playlist_id")}
    for r in rows:
        r["count"] = counts.get(r["id"], 0)
    return rows


def create_playlist(name: str, parent_id: int = 0, is_folder: bool = False) -> dict:
    with db.tx() as c:
        sort = c.execute("SELECT COALESCE(MAX(sort),0)+1 FROM playlists WHERE parent_id=?", (parent_id,)).fetchone()[0]
        cur = c.execute("INSERT INTO playlists(name, parent_id, is_folder, sort) VALUES (?,?,?,?)",
                        (name.strip() or "Untitled", parent_id, int(is_folder), sort))
        db.log(c, "create_playlist", {"id": cur.lastrowid, "name": name})
    return db.one("SELECT * FROM playlists WHERE id=?", (cur.lastrowid,))


def rename_playlist(pid: int, name: str) -> None:
    with db.tx() as c:
        c.execute("UPDATE playlists SET name=? WHERE id=?", (name, pid))


def delete_playlist(pid: int) -> None:
    """Deletes a playlist (and anything inside a folder). Tracks stay in the library."""
    with db.tx() as c:
        ids, todo = [], [pid]
        while todo:
            p = todo.pop()
            ids.append(p)
            todo += [r[0] for r in c.execute("SELECT id FROM playlists WHERE parent_id=?", (p,))]
        q = ",".join("?" * len(ids))
        c.execute(f"DELETE FROM playlist_tracks WHERE playlist_id IN ({q})", ids)
        c.execute(f"DELETE FROM playlists WHERE id IN ({q})", ids)
        db.log(c, "delete_playlist", {"ids": ids})


def playlist_track_ids(pid: int) -> list[int]:
    return [r["track_id"] for r in db.query(
        "SELECT track_id FROM playlist_tracks WHERE playlist_id=? ORDER BY position", (pid,))]


def set_playlist_tracks(pid: int, track_ids: list[int], action: str = "edit_playlist") -> int:
    """Replace the playlist's order (used for add, remove and reorder). Undoable."""
    before = playlist_track_ids(pid)
    with db.tx() as c:
        c.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", (pid,))
        c.executemany("INSERT INTO playlist_tracks VALUES (?,?,?)", [(pid, t, i) for i, t in enumerate(track_ids)])
        return db.log(c, action, {"playlist_id": pid, "count": len(track_ids)},
                      {"kind": "playlist_restore", "playlist_id": pid, "tracks": before})


def add_to_playlist(pid: int, track_ids: list[int]) -> int:
    current = playlist_track_ids(pid)
    return set_playlist_tracks(pid, current + [t for t in track_ids if t not in current], "add_to_playlist")


# ── Cues ─────────────────────────────────────────────────────────────────────

def cues(track_id: int) -> list[dict]:
    return db.query("SELECT * FROM cues WHERE track_id=? ORDER BY slot=0, slot, start_ms", (track_id,))


def set_cues(track_id: int, new: list[dict]) -> int:
    """Replace a track's cues. Each cue: slot (0-8), start_ms, optional end_ms/label."""
    before = cues(track_id)
    seen = set()
    clean = []
    for q in new:
        slot = int(q.get("slot", 0))
        if slot and slot in seen:
            continue
        seen.add(slot)
        clean.append((track_id, slot, int(q["start_ms"]), q.get("end_ms"), q.get("label"), int(q.get("auto", 0))))
    with db.tx() as c:
        c.execute("DELETE FROM cues WHERE track_id=?", (track_id,))
        c.executemany("INSERT INTO cues(track_id, slot, start_ms, end_ms, label, auto) VALUES (?,?,?,?,?,?)", clean)
        return db.log(c, "edit_cues", {"track_id": track_id, "count": len(clean)},
                      {"kind": "cues_restore", "track_id": track_id, "cues": before})
