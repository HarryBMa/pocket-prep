"""Download Creative Commons music from Jamendo for teacher separation (distill.py / teacher2.py).

Jamendo hosts CC-licensed music with an API that filters by genre tag and reports the licence of
every track, so the training set stays accountable. Only tracks whose artist allows downloads
are fetched (audiodownload_allowed), and the licence URL of each track is written to
manifest.csv next to the files.

    set JAMENDO_CLIENT_ID=...         (free: https://devportal.jamendo.com)
    python jamendo_fetch.py --out E:\\needle-music\\jamendo --per-tag 600 --tags house techno deephouse techhouse

Resumable: existing files are skipped. Full-length tracks (3-10 minutes) at mp32 (VBR, the
best quality the API offers for downloads; FLAC is requested first and used when available).
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.jamendo.com/v3.0/tracks/"
DEFAULT_TAGS = ["house", "techno", "deephouse", "techhouse", "electro", "minimal", "progressivehouse", "electronic"]


def api(params, client_id):
    q = {"client_id": client_id, "format": "json", **params}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "needle-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_tracks(tag, n, client_id, min_s=180, max_s=600):
    """Up to n downloadable full-length tracks for a tag, 200 per request."""
    out, offset = [], 0
    while len(out) < n:
        d = api({"tags": tag, "limit": 200, "offset": offset, "durationbetween": f"{min_s}_{max_s}",
                 "audiodlformat": "flac", "include": "licenses", "type": "single+albumtrack", "order": "popularity_total"},
                client_id)
        rows = d.get("results", [])
        if not rows:                                   # the API sometimes returns an empty page: retry twice
            empties = empties + 1 if "empties" in dir() else 1
            if empties > 2:
                break
            time.sleep(3)
            continue
        empties = 0
        for t in rows:
            if t.get("audiodownload_allowed") and t.get("audiodownload"):
                out.append(t)
        offset += 200
        time.sleep(0.5)
    return out[:n]


def safe(s):
    return re.sub(r'[\\/:*?"<>|]+', "_", s).strip()[:80]


def fetch(t, folder):
    url = t["audiodownload"]
    ext = ".flac" if "flac" in url.lower() else ".mp3"
    path = folder / f"{t['id']} - {safe(t['artist_name'])} - {safe(t['name'])}{ext}"
    if path.exists() and path.stat().st_size > 100_000:
        return path, "exists"
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "needle-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                ctype = r.headers.get("Content-Type", "")
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if "mpeg" in ctype and ext == ".flac":                      # server fell back to mp3
                path = path.with_suffix(".mp3")
            os.replace(tmp, path)
            return path, "ok"
        except Exception as e:                                           # noqa: BLE001
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return path, f"failed {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--list-only", action="store_true", help="only count what would be downloaded")
    a = ap.parse_args()
    cid = os.environ.get("JAMENDO_CLIENT_ID")
    if not cid:
        raise SystemExit("set JAMENDO_CLIENT_ID (free at https://devportal.jamendo.com)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for tag in a.tags:
        tracks = list_tracks(tag, a.per_tag, cid)
        new = [t for t in tracks if t["id"] not in seen]
        for t in new:
            seen[t["id"]] = (tag, t)
        hours = sum(int(t["duration"]) for t in new) / 3600
        print(f"{tag}: {len(tracks)} downloadable tracks, {len(new)} new, {hours:.1f} h", flush=True)
    total_h = sum(int(t["duration"]) for _, t in seen.values()) / 3600
    print(f"total {len(seen)} tracks, {total_h:.1f} h", flush=True)
    if a.list_only:
        return
    manifest = out / "manifest.csv"
    have = set()
    if manifest.exists():
        have = {r["id"] for r in csv.DictReader(open(manifest, encoding="utf-8"))}
    with open(manifest, "a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if not have:
            w.writerow(["id", "tag", "artist", "title", "duration_s", "license", "file", "status"])
        jobs = [(tag, t) for tag, t in seen.values()]
        for tag in a.tags:
            (out / tag).mkdir(exist_ok=True)
        with ThreadPoolExecutor(a.threads) as ex:
            for i, ((tag, t), (path, status)) in enumerate(zip(jobs, ex.map(lambda j: fetch(j[1], out / j[0]), jobs))):
                if t["id"] not in have:
                    w.writerow([t["id"], tag, t["artist_name"], t["name"], t["duration"],
                                t.get("license_ccurl", ""), str(path.relative_to(out)), status])
                    mf.flush()
                if (i + 1) % 25 == 0 or status.startswith("failed"):
                    print(f"[{i + 1}/{len(jobs)}] {status}: {path.name}", flush=True)


if __name__ == "__main__":
    main()
