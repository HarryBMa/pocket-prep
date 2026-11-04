"""The local web server behind the app window (FastAPI).

Only listens on 127.0.0.1, so nothing on the network can reach it.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, analysis, config, db, deck, jobs, library, stems
from .export import usb

app = FastAPI(title="Pocket Prep", version=__version__)
STATIC = Path(__file__).parent / "static"
WINDOW = {"w": None}          # set by __main__ when running in a desktop window

TRACK_COLUMNS = ("id, path, title, artist, album, genre, year, duration, bpm, musical_key, camelot, energy, "
                 "rating, color, comment, stems, analysed_at, added_at, artwork IS NOT NULL AS has_artwork")


def _track(tid: int) -> dict:
    t = db.one(f"SELECT {TRACK_COLUMNS} FROM tracks WHERE id=?", (tid,))
    if not t:
        raise HTTPException(404, "No such track")
    return t


# ── Jobs ─────────────────────────────────────────────────────────────────────

def queue_analysis(ids: list[int]) -> list[int]:
    out = []
    for tid in ids:
        if jobs.pending_for(tid, "analyse"):
            continue
        t = db.one("SELECT title FROM tracks WHERE id=?", (tid,))
        j = jobs.submit("analyse", f"Analyse: {t['title'] if t else tid}",
                        lambda job, tid=tid: (analysis.analyse(tid, lambda f, m="": jobs.job_progress(job, f, m)), None)[1],
                        "cpu", tid)
        out.append(j.id)
    return out


def queue_stems(ids: list[int]) -> list[int]:
    out = []
    for tid in ids:
        if jobs.pending_for(tid, "stems"):
            continue
        t = db.one("SELECT title FROM tracks WHERE id=?", (tid,))
        with db.tx() as c:
            c.execute("UPDATE tracks SET stems='queued' WHERE id=?", (tid,))

        def run(job, tid=tid):
            stems.separate(tid, lambda f, m="": jobs.job_progress(job, f, m), lambda: job.cancel_requested)

        out.append(jobs.submit("stems", f"Stems: {t['title'] if t else tid}", run, "gpu", tid).id)
    return out


@app.get("/api/jobs")
def get_jobs():
    return jobs.listing()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    return {"ok": jobs.cancel(job_id)}


@app.post("/api/jobs/clear")
def clear_jobs():
    jobs.clear_finished()
    return {"ok": True}


# ── Tracks ───────────────────────────────────────────────────────────────────

@app.get("/api/tracks")
def list_tracks(q: str = "", playlist: int = 0):
    if playlist:
        ids = library.playlist_track_ids(playlist)
        rows = {r["id"]: r for r in db.query(f"SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0")}
        tracks = [rows[i] for i in ids if i in rows]
    else:
        tracks = db.query(f"SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0 ORDER BY added_at DESC, id DESC")
    if q:
        words = q.lower().split()
        tracks = [t for t in tracks if all(
            w in " ".join(str(t[k] or "") for k in ("title", "artist", "album", "genre", "camelot", "musical_key", "comment")).lower()
            for w in words)]
    return tracks


@app.get("/api/tracks/{tid}")
def get_track(tid: int):
    t = _track(tid)
    a = analysis.load_analysis(tid)
    t["cues"] = library.cues(tid)
    t["phrases"] = a["phrases"] if a else None
    t["grid"] = {k: a[k] for k in ("bpm", "first_beat_ms", "period_ms", "downbeat_index", "duration_ms")} if a else None
    t["stem_files"] = sorted(stems.stem_files(tid))
    t["playlists"] = [r["playlist_id"] for r in db.query("SELECT DISTINCT playlist_id FROM playlist_tracks WHERE track_id=?", (tid,))]
    return t


@app.get("/api/tracks/{tid}/waveform")
def get_waveform(tid: int):
    a = analysis.load_analysis(tid)
    if not a:
        raise HTTPException(404, "Not analysed yet")
    return {"overview": a["waveforms"]["pwv4"], "detail": a["waveforms"]["pwv5"], "duration_ms": a["duration_ms"]}


@app.get("/api/tracks/{tid}/audio")
def get_audio(tid: int):
    t = db.one("SELECT path FROM tracks WHERE id=?", (tid,))
    if not t or not Path(t["path"]).exists():
        raise HTTPException(404, "File missing")
    return FileResponse(t["path"])


@app.get("/api/tracks/{tid}/stem/{name}")
def get_stem(tid: int, name: str):
    f = stems.stem_files(tid).get(name)
    if not f:
        raise HTTPException(404, "No such stem")
    return FileResponse(f)


@app.get("/api/tracks/{tid}/artwork")
def get_artwork(tid: int):
    t = db.one("SELECT artwork FROM tracks WHERE id=?", (tid,))
    if not t or not t["artwork"] or not Path(t["artwork"]).exists():
        raise HTTPException(404, "No artwork")
    return FileResponse(t["artwork"])


@app.patch("/api/tracks/{tid}")
def patch_track(tid: int, fields: dict = Body(...)):
    _track(tid)
    db.update_track(tid, fields)
    last = db.one("SELECT id FROM history ORDER BY id DESC LIMIT 1")
    return {"track": _track(tid), "undo": last["id"] if last else None}


@app.put("/api/tracks/{tid}/cues")
def put_cues(tid: int, cues: list[dict] = Body(...)):
    _track(tid)
    hid = library.set_cues(tid, cues)
    return {"cues": library.cues(tid), "undo": hid}


@app.post("/api/import")
def import_files(paths: list[str] = Body(..., embed=True)):
    ids = library.import_paths(paths)
    if ids and config.load_settings()["auto_analyse"]:
        queue_analysis(ids)
    return {"imported": len(ids), "ids": ids}


@app.post("/api/tracks/remove")
def remove_tracks(ids: list[int] = Body(..., embed=True)):
    return {"undo": library.remove_tracks(ids)}


@app.post("/api/analyse")
def analyse_tracks(ids: list[int] = Body(..., embed=True)):
    return {"jobs": queue_analysis(ids)}


@app.post("/api/stems")
def make_stems(ids: list[int] = Body(..., embed=True)):
    return {"jobs": queue_stems(ids)}


@app.delete("/api/tracks/{tid}/stems")
def delete_stems(tid: int):
    stems.remove(tid)
    return {"ok": True}


# ── Playlists ────────────────────────────────────────────────────────────────

@app.get("/api/playlists")
def get_playlists():
    return library.playlist_tree()


@app.post("/api/playlists")
def new_playlist(name: str = Body(...), parent_id: int = Body(0), is_folder: bool = Body(False)):
    return library.create_playlist(name, parent_id, is_folder)


@app.patch("/api/playlists/{pid}")
def rename(pid: int, name: str = Body(..., embed=True)):
    library.rename_playlist(pid, name)
    return {"ok": True}


@app.delete("/api/playlists/{pid}")
def remove_playlist(pid: int):
    library.delete_playlist(pid)
    return {"ok": True}


@app.put("/api/playlists/{pid}/tracks")
def order_playlist(pid: int, track_ids: list[int] = Body(..., embed=True)):
    return {"undo": library.set_playlist_tracks(pid, track_ids)}


@app.post("/api/playlists/{pid}/add")
def add_tracks(pid: int, track_ids: list[int] = Body(..., embed=True)):
    return {"undo": library.add_to_playlist(pid, track_ids)}


# ── Undo / history ───────────────────────────────────────────────────────────

@app.post("/api/undo/{hid}")
def undo(hid: int):
    if not db.undo(hid):
        raise HTTPException(409, "That change can't be undone")
    return {"ok": True}


@app.get("/api/history")
def history():
    return db.query("SELECT id, at, action, detail, undo IS NOT NULL AS can_undo FROM history ORDER BY id DESC LIMIT 200")


# ── Export ───────────────────────────────────────────────────────────────────

@app.get("/api/drives")
def drives():
    return usb.list_drives()


@app.post("/api/export/plan")
def export_plan(root: str = Body(...), playlist_ids: list[int] | None = Body(None), include_stems: bool = Body(True)):
    return usb.plan(root, playlist_ids, include_stems)


@app.post("/api/export")
def export(root: str = Body(...), playlist_ids: list[int] | None = Body(None), include_stems: bool = Body(True)):
    def run(job):
        job.result = usb.export(root, playlist_ids, include_stems,
                                lambda f, m="": jobs.job_progress(job, f, m), lambda: job.cancel_requested)
        return job.result
    j = jobs.submit("export", f"Export to {root}", run, "io")
    return {"job": j.id}


# ── Deck ─────────────────────────────────────────────────────────────────────

@app.get("/api/deck/status")
def deck_status():
    return deck.status()


@app.post("/api/deck/ota")
async def deck_ota(file: UploadFile):
    tmp = Path(tempfile.mkdtemp()) / Path(file.filename or "update.ddjota").name
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    j = jobs.submit("ota", f"Update deck: {tmp.name}",
                    lambda job: deck.ota_upload(tmp, lambda fr, m="": jobs.job_progress(job, fr, m)), "io")
    return {"job": j.id}


@app.get("/api/deck/ports")
def deck_ports():
    return deck.serial_ports()


@app.post("/api/deck/flash")
def deck_flash(build_dir: str = Body(...), port: str = Body(...)):
    j = jobs.submit("flash", f"Flash over USB ({port})",
                    lambda job: deck.wired_flash(Path(build_dir), port, lambda f, m="": jobs.job_progress(job, f, m)), "io")
    return {"job": j.id}


# ── Settings, system, file pickers ───────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return config.load_settings()


@app.put("/api/settings")
def put_settings(values: dict = Body(...)):
    return config.save_settings(values)


@app.get("/api/system")
def system():
    info = {"version": __version__, "data_dir": str(config.DATA_DIR), "window": WINDOW["w"] is not None}
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
    except Exception:
        info["cuda"] = False
    return info


@app.post("/api/pick")
def pick(kind: str = Body("folder", embed=True)):
    """Native file dialogs, only available in the desktop window."""
    w = WINDOW["w"]
    if w is None:
        return {"paths": [], "unavailable": True}
    import webview
    if kind == "folder":
        r = w.create_file_dialog(webview.FOLDER_DIALOG)
    elif kind == "bundle":
        r = w.create_file_dialog(webview.OPEN_DIALOG, file_types=("Signed update (*.ddjota)",))
    else:
        r = w.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True,
                                 file_types=("Audio (*.mp3;*.wav;*.flac;*.aiff;*.aif;*.m4a)",))
    return {"paths": list(r or [])}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
