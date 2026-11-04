"""Stem separation with Demucs: drums, bass, vocals and "other" (melody).

Demucs runs as a separate process (python -m demucs) so a crash or an
out-of-memory error on the GPU can't take the app down, and so its
progress bar can be read line by line.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config, db

STEM_NAMES = ["drums", "bass", "other", "vocals"]
_PCT = re.compile(rb"(\d{1,3})%\|")


def stem_dir(track_id: int) -> Path:
    return config.STEMS_DIR / str(track_id)


def stem_files(track_id: int) -> dict[str, Path]:
    d = stem_dir(track_id)
    return {n: d / f"{n}.mp3" for n in STEM_NAMES if (d / f"{n}.mp3").exists()}


def separate(track_id: int, progress=lambda f, msg="": None, cancel=lambda: False) -> dict[str, Path]:
    t = db.one("SELECT * FROM tracks WHERE id=?", (track_id,))
    if not t:
        raise ValueError("unknown track")
    s = config.load_settings()
    with db.tx() as c:
        c.execute("UPDATE tracks SET stems='running' WHERE id=?", (track_id,))
    tmp = Path(tempfile.mkdtemp(prefix="pocketprep_"))
    cmd = [sys.executable, "-m", "demucs", "-n", s["stem_model"], "-d", s["stem_device"],
           "--mp3", "--mp3-bitrate", str(s["stem_bitrate"]), "-o", str(tmp), t["path"]]
    progress(0.02, f"Starting Demucs ({s['stem_model']} on {s['stem_device']})")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = b""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            tail = (tail + chunk)[-4000:]
            found = _PCT.findall(chunk)
            if found:
                progress(0.05 + 0.9 * int(found[-1]) / 100, "Separating")
            if cancel():
                proc.kill()
                raise RuntimeError("cancelled")
        if proc.wait() != 0:
            raise RuntimeError("Demucs failed: " + tail.decode(errors="replace")[-600:])
        produced = list(tmp.rglob("*.mp3"))
        if len(produced) < 4:
            raise RuntimeError("Demucs produced no stems")
        out = stem_dir(track_id)
        out.mkdir(parents=True, exist_ok=True)
        for f in produced:
            shutil.move(str(f), out / f.name)
        with db.tx() as c:
            c.execute("UPDATE tracks SET stems='done' WHERE id=?", (track_id,))
        progress(1.0, "Done")
        return stem_files(track_id)
    except Exception:
        with db.tx() as c:
            c.execute("UPDATE tracks SET stems='failed' WHERE id=?", (track_id,))
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def remove(track_id: int) -> None:
    shutil.rmtree(stem_dir(track_id), ignore_errors=True)
    with db.tx() as c:
        c.execute("UPDATE tracks SET stems='none' WHERE id=?", (track_id,))
