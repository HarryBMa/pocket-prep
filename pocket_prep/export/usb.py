"""Export the library to an SD card or USB stick.

Layout written (rekordbox-compatible, so CDJs can read it too):

  Contents/<Artist>/<Album>/<file>                  the audio files
  PIONEER/rekordbox/export.pdb                      track database
  PIONEER/USBANLZ/Pxxx/xxxxxxxx/ANLZ0000.DAT|.EXT   beat grid, waveforms, cues
  PIONEER/Artwork/00001/a<id>.jpg (+ _m.jpg)        cover art

Pocket Deck extensions (other players ignore these):

  POCKETDECK/export.json                 what wrote the stick, and when
  POCKETDECK/stems.json                  which stem files belong to which track
  POCKETDECK/STEMS/xxxxxxxx/<stem>.mp3   drums, bass, other, vocals
  POCKETDECK/analysis/xxxxxxxx.json      phrases, energy, key (for autopilot)
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import string
import time
from pathlib import Path

from .. import analysis, config, db, library, stems
from . import anlz, pdb

FILE_TYPES = {".mp3": 1, ".m4a": 4, ".flac": 5, ".wav": 0x0B, ".aiff": 0x0C, ".aif": 0x0C}
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe(name: str | None, fallback: str) -> str:
    name = _BAD.sub("_", (name or "").strip()).rstrip(". ")
    return (name or fallback)[:80]


def list_drives() -> list[dict]:
    """Drives Windows can see, removable ones first."""
    out = []
    if os.name != "nt":
        return out
    k32 = ctypes.windll.kernel32
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        kind = k32.GetDriveTypeW(root)
        if kind not in (2, 3):          # 2 removable, 3 fixed
            continue
        label = ctypes.create_unicode_buffer(256)
        fs = ctypes.create_unicode_buffer(64)
        if not k32.GetVolumeInformationW(root, label, 256, None, None, None, fs, 64):
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        out.append({"root": root, "label": label.value, "filesystem": fs.value, "removable": kind == 2,
                    "free": usage.free, "total": usage.total,
                    "has_export": Path(root, "PIONEER", "rekordbox", "export.pdb").exists()})
    return sorted(out, key=lambda d: (not d["removable"], d["root"]))


def _anlz_rel(tid: int) -> str:
    return f"PIONEER/USBANLZ/P{(tid >> 8) & 0xFFF:03X}/{tid:08X}"


def _collect(playlist_ids: list[int] | None):
    """Tracks and playlists to export. None = the whole library."""
    pls = library.playlist_tree()
    if playlist_ids:
        chosen = set()
        todo = list(playlist_ids)
        while todo:                                  # include everything inside folders
            p = todo.pop()
            chosen.add(p)
            todo += [x["id"] for x in pls if x["parent_id"] == p]
        by_id = {p["id"]: p for p in pls}
        for p in list(chosen):                       # and the folders above them
            parent = by_id.get(p, {}).get("parent_id", 0)
            while parent and parent in by_id:
                chosen.add(parent)
                parent = by_id[parent]["parent_id"]
        pls = [p for p in pls if p["id"] in chosen]
        ids = []
        for p in pls:
            ids += [t for t in library.playlist_track_ids(p["id"]) if t not in ids]
        tracks = [db.one("SELECT * FROM tracks WHERE id=? AND deleted=0", (t,)) for t in ids]
        tracks = [t for t in tracks if t]
    else:
        tracks = db.query("SELECT * FROM tracks WHERE deleted=0 ORDER BY artist, title")
    return tracks, pls


def plan(root: str, playlist_ids: list[int] | None, include_stems: bool) -> dict:
    """What an export would copy, and whether it fits."""
    tracks, pls = _collect(playlist_ids)
    need = 0
    for t in tracks:
        need += t["file_size"] or 0
        if include_stems:
            need += sum(f.stat().st_size for f in stems.stem_files(t["id"]).values())
    probe = Path(root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return {"tracks": len(tracks), "playlists": len(pls), "bytes": need, "free": free, "fits": need < free * 0.98,
            "unanalysed": sum(1 for t in tracks if not t["analysed_at"]),
            "with_stems": sum(1 for t in tracks if t["stems"] == "done")}


def export(root: str, playlist_ids: list[int] | None = None, include_stems: bool = True,
           progress=lambda f, msg="": None, cancel=lambda: False) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tracks, pls = _collect(playlist_ids)
    if not tracks:
        raise ValueError("Nothing to export")
    p = plan(str(root), playlist_ids, include_stems)
    if not p["fits"]:
        raise ValueError(f"Not enough space: needs {p['bytes'] / 1e9:.1f} GB, {p['free'] / 1e9:.1f} GB free")

    database = pdb.Database()
    pdb.add_colors(database)
    names = {"artist": {}, "album": {}, "genre": {}, "key": {}}

    def name_id(kind, value, **extra):
        if not value:
            return 0
        table = names[kind]
        if value not in table:
            table[value] = (len(table) + 1, extra)
        return table[value][0]

    stem_manifest, copied, skipped = {}, 0, 0
    art_dir = root / "PIONEER" / "Artwork" / "00001"
    artwork_rows = []
    today = time.strftime("%Y-%m-%d")

    for n, t in enumerate(tracks):
        if cancel():
            raise RuntimeError("cancelled")
        tid = t["id"]
        progress(0.9 * n / len(tracks), f"{t['artist'] or ''} – {t['title']}")
        src = Path(t["path"])
        if not src.exists():
            skipped += 1
            continue
        a = analysis.load_analysis(tid)
        if a is None:
            a = analysis.analyse(tid)
            t = db.one("SELECT * FROM tracks WHERE id=?", (tid,))

        # 1. Audio file
        rel_audio = f"Contents/{safe(t['artist'], 'Unknown Artist')}/{safe(t['album'], 'Unknown Album')}/{safe(src.name, f'{tid}{src.suffix}')}"
        dst = root / rel_audio
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        audio_path = "/" + rel_audio

        # 2. Analysis files
        cue_rows = library.cues(tid)
        hot = [anlz.Cue(q["start_ms"], q["end_ms"], hot=q["slot"]) for q in cue_rows if 1 <= q["slot"] <= 8]
        mem = [anlz.Cue(q["start_ms"], q["end_ms"]) for q in cue_rows if q["slot"] == 0]
        ta = anlz.TrackAnalysis(
            path=audio_path, beats=analysis.beats_from_grid(a),
            pwav=analysis.waveform_bytes(a, "pwav"), pwv2=analysis.waveform_bytes(a, "pwv2"),
            pwv3=analysis.waveform_bytes(a, "pwv3"), pwv4=analysis.waveform_bytes(a, "pwv4"),
            pwv5=analysis.waveform_bytes(a, "pwv5"), hot_cues=hot, memory_cues=mem)
        adir = root / _anlz_rel(tid)
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "ANLZ0000.DAT").write_bytes(anlz.build_dat(ta))
        (adir / "ANLZ0000.EXT").write_bytes(anlz.build_ext(ta))

        # 3. Artwork (80 px and 240 px, as rekordbox writes them)
        art_id = 0
        if t["artwork"] and Path(t["artwork"]).exists():
            from PIL import Image
            art_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(t["artwork"]).convert("RGB")
            for suffix, size in (("", 80), ("_m", 240)):
                im = img.copy()
                im.thumbnail((size, size))
                im.save(art_dir / f"a{tid}{suffix}.jpg", "JPEG", quality=85)
            art_id = len(artwork_rows) + 1
            artwork_rows.append((art_id, f"/PIONEER/Artwork/00001/a{tid}.jpg"))

        # 4. Database row
        artist_id = name_id("artist", t["artist"])
        row = pdb.TrackRow(
            id=tid, title=t["title"] or src.stem, file_path=audio_path, filename=dst.name,
            anlz_path=f"/{_anlz_rel(tid)}/ANLZ0000.DAT",
            artist_id=artist_id, album_id=name_id("album", t["album"], artist=artist_id),
            genre_id=name_id("genre", t["genre"]), key_id=name_id("key", t["musical_key"]),
            artwork_id=art_id, color_id=t["color"] or 0, bpm=t["bpm"] or 0,
            duration_s=int(round(t["duration"] or 0)), sample_rate=t["sample_rate"] or 44100,
            bitrate=t["bitrate"] or 0, file_size=src.stat().st_size, year=t["year"] or 0,
            rating=t["rating"] or 0, track_number=t["track_number"] or 0, comment=t["comment"] or "",
            date_added=time.strftime("%Y-%m-%d", time.localtime(t["added_at"] or time.time())),
            analyze_date=today, file_type=FILE_TYPES.get(src.suffix.lower(), 1))
        database.add(pdb.T_TRACKS, pdb.track_row(row, n))

        # 5. Pocket Deck extras: stems and phrase data
        ext_id = f"{tid:08X}"
        if include_stems:
            files = stems.stem_files(tid)
            if len(files) == 4:
                sdir = root / "POCKETDECK" / "STEMS" / ext_id
                sdir.mkdir(parents=True, exist_ok=True)
                entry = {}
                for name, f in files.items():
                    d = sdir / f.name
                    if not (d.exists() and d.stat().st_size == f.stat().st_size):
                        shutil.copy2(f, d)
                    entry[name] = f"/POCKETDECK/STEMS/{ext_id}/{f.name}"
                stem_manifest[audio_path] = entry
        extra = root / "POCKETDECK" / "analysis"
        extra.mkdir(parents=True, exist_ok=True)
        (extra / f"{ext_id}.json").write_text(json.dumps({
            "file_path": audio_path, "bpm": a["bpm"], "key": a["key"], "camelot": a["camelot"],
            "energy": a["energy"], "phrases": a["phrases"]}), "utf-8")

    # Name tables
    for i, (name, (nid, _)) in enumerate(names["artist"].items()):
        database.add(pdb.T_ARTISTS, pdb.artist_row(nid, name, i))
    for i, (name, (nid, extra)) in enumerate(names["album"].items()):
        database.add(pdb.T_ALBUMS, pdb.album_row(nid, name, extra.get("artist", 0), i))
    for name, (nid, _) in names["genre"].items():
        database.add(pdb.T_GENRES, pdb.simple_name_row(nid, name))
    for name, (nid, _) in names["key"].items():
        database.add(pdb.T_KEYS, pdb.key_row(nid, name))
    for aid, path in artwork_rows:
        database.add(pdb.T_ARTWORK, pdb.artwork_row(aid, path))

    # Playlists
    exported = {t["id"] for t in tracks}
    for pl in pls:
        database.add(pdb.T_PLAYLIST_TREE, pdb.playlist_tree_row(pl["id"], pl["parent_id"], pl["sort"], pl["name"], bool(pl["is_folder"])))
        if not pl["is_folder"]:
            for i, t in enumerate(x for x in library.playlist_track_ids(pl["id"]) if x in exported):
                database.add(pdb.T_PLAYLIST_ENTRIES, pdb.playlist_entry_row(i + 1, t, pl["id"]))

    progress(0.95, "Writing database")
    out = root / "PIONEER" / "rekordbox"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "export.pdb.tmp"
    tmp.write_bytes(database.write())
    os.replace(tmp, out / "export.pdb")            # never leave a half-written database
    pd = root / "POCKETDECK"
    pd.mkdir(exist_ok=True)
    (pd / "stems.json").write_text(json.dumps({"version": 1, "tracks": stem_manifest}, indent=1), "utf-8")
    from .. import __version__
    summary = {"app": "Pocket Prep", "version": __version__, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "tracks": len(tracks) - skipped, "playlists": len(pls), "copied": copied,
               "missing_files": skipped, "stems": len(stem_manifest)}
    (pd / "export.json").write_text(json.dumps(summary, indent=1), "utf-8")
    progress(1.0, "Done")
    return summary
