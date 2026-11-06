"""Real stems for training: convert multitrack sources into the layout distill.py writes
(<out>/<id>/{drums,bass,other,vocals}.flac + done), so train.py can load them with --stems.

    # Native Instruments Stems files (.stem.mp4: master + 4 named stems)
    python import_stems.py ni E:\\needle-data\\stems_raw\\ni_free_stems --out E:\\needle-data\\stems\\ni ^
        --test-out E:\\needle-data\\stems\\ni_test --musdb E:\\needle-data\\musdb18hq

    # one folder per song with one file per instrument (e.g. mshoxxDB: <song>/<song>_<instrument>.flac)
    python import_stems.py folder E:\\needle-data\\stems_raw\\mshoxxDB --out E:\\needle-data\\stems\\mshoxx --musdb ...

    # sample packs (.zip): construction kits become songs, loop libraries single-stem tracks
    python import_stems.py pack "C:\\Users\\Harry\\Downloads\\Ghosthack_...zip" ... --out E:\\needle-data\\stems\\packs

How a stem is assigned to drums / bass / other / vocals: by its name (NI stores names such as
"Drums", "Sub", "Chords", "Vox" inside the file; folder packs use file names). Anything that
isn't clearly drums, bass or vocals is "other", as in MUSDB18.

When one stem holds two kinds of sound ("Vox+FX", "Sub+Clap", "Bass+More") or an unknown one
("Chops"), there is no right answer for it, so the whole song is skipped rather than teaching
the model something wrong. Songs that also exist in MUSDB18 are skipped too: training on a
MUSDB test song would make the test scores meaningless.

--test-out puts every --test-every'th usable song in a separate folder that is never trained on:
a test set with real stems in the deck's own genres.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from data import SAMPLE_RATE, read_audio
from needle import SOURCES

PATTERNS = [("vocals", r"vox|vocal|voice|choir|acapell|\bsing(er|ing)?\b|chant|harmony (left|right)"),
            ("drums", r"drum|perc|kick|snare|\bhats?\b|hi-?hats?|clap|cymbal|breakbeat|\brim|\btoms?\b|shaker|tamb|\bride\b|crash|open half"),
            ("bass", r"bass|\bsub\b|808")]
UNKNOWN = r"chops|metal|more|\bmix\b"                  # can't tell what these hold


def category(name):
    """Stem name -> one of SOURCES, or None when it is ambiguous."""
    parts = [p.strip() for p in re.split(r"\+|&|\band\b", name.lower()) if p.strip()]
    cats = set()
    for p in parts:
        if re.search(UNKNOWN, p):
            return None
        cats.add(next((c for c, rx in PATTERNS if re.search(rx, p)), "other"))
    return cats.pop() if len(cats) == 1 else None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def track_id(rel: str):
    """A folder name that is readable, unique and stable for the same source (as in distill.py)."""
    slug = re.sub(r"[^\w\-]+", "_", rel).strip("_")[-80:]
    return f"{slug}-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:8]}"


def musdb_titles(root):
    """Normalised song titles of all 150 MUSDB18 tracks ("Artist - Title" -> "title")."""
    if not root:
        return set()
    return {norm(p.name.split(" - ", 1)[-1]) for sub in ("train", "test") for p in (Path(root) / sub).iterdir() if p.is_dir()}


# ── Sources ──────────────────────────────────────────────────────────────────

def ni_songs(root):
    """NI .stem.mp4: audio stream 0 is the master, streams 1-4 the stems; their names are in
    a JSON block inside the file."""
    for p in sorted(Path(root).rglob("*.stem.mp4")):
        m = re.search(rb'\{"(?:mastering_dsp|stems|version)".*?\}\]\}', p.read_bytes(), re.S)
        names = [s["name"] for s in json.loads(m.group(0))["stems"]] if m else []
        title = p.name.split(".stem")[0].split("_", 1)[-1]           # "PR_HappyDaze" -> "HappyDaze"

        def load(p=p, n=len(names)):
            out = []
            for k in range(1, n + 1):
                raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(p), "-map", f"0:a:{k}", "-f", "f32le",
                                      "-ac", "2", "-ar", str(SAMPLE_RATE), "-"], capture_output=True, check=True).stdout
                out.append(np.frombuffer(raw, np.float32).reshape(-1, 2).T)
            return out
        yield p.name.split(".stem")[0], title, names, load, str(p)


AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff")
MIXDOWN = r"\bmaster\b|\bmixdown\b|full ?mix"                     # the finished mix, not a stem


def folder_songs(root):
    """Multitrack folders anywhere under root: a song is a folder whose audio files start with
    the folder's name, one file per instrument, e.g. mshoxxDB <song>/<song>_drums.flac or a
    sample pack's "Stems/Battery/Battery 165 Bpm D# Min Kick.wav". The song's own mix
    (<song>.flac, "... Master.wav") is not a stem and is ignored; loose loops are ignored."""
    for d in sorted({f.parent for f in Path(root).rglob("*") if f.suffix.lower() in AUDIO_EXT}):
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in AUDIO_EXT and f.stem.startswith(d.name)
                 and f.stem != d.name and not re.search(MIXDOWN, f.stem, re.I)]
        if len(files) < 2:
            continue
        names = [re.sub(rf"^{re.escape(d.name)}[_ -]*", "", f.stem) for f in files]
        yield d.name, d.name.rsplit("_v", 1)[0], names, (lambda files=files: [read_audio(f, dtype="float32") for f in files]), str(d)


PACK_SKIP = r"one[ -]?shots?|oneshot|\bfx\b|foley|demo|midi|__macosx|^\._"
PACK_KIT = r"construction kit|(^|/)stems$|wav loops|(^|/)(wet|dry)$"
PACK_LOOPS = r"loop|phrase"


def pack_songs(zip_paths):
    """Sample packs, read straight from their .zip files.

    - Construction kits (a folder of stems of one short song: ".../STEMS", "Construction Kit ...",
      "WAV Loops", ".../Wet") become one song; shorter loops are repeated to the kit's length when
      they fit a whole number of times, as a producer would loop them. A "Dry" folder is skipped
      when a "Wet" copy of the same kit exists.
    - Loop libraries ("Bass Loops", "Vocal Phrases", ...) are unrelated loops of one instrument
      each; every loop becomes its own single-stem track (remixing pairs it with other stems).
    - One-shots, FX, foley, demo mixes and MIDI are skipped (a hit shorter than a training crop
      teaches little, and a demo is a mix, not a stem)."""
    import io
    import zipfile
    for zp in zip_paths:
        z = zipfile.ZipFile(zp)
        pack = Path(zp).stem
        audio = [n for n in z.namelist() if n.lower().endswith((".wav", ".aif", ".aiff", ".flac"))]
        folders = {}
        for n in audio:
            folders.setdefault(n.rsplit("/", 1)[0] if "/" in n else "", []).append(n)
        for folder, files in sorted(folders.items()):
            low = folder.lower()
            if any(re.search(PACK_SKIP, seg) for seg in low.split("/")):
                continue
            files = [f for f in files if not re.search(MIXDOWN + "|demo", f.rsplit("/", 1)[-1], re.I)]
            vocal_folder = re.search(r"vocal|acapell|\bvox\b", low)

            def read(name):
                return read_audio(io.BytesIO(z.read(name)), dtype="float32")

            def stem_name(f):
                return re.sub(r"\.(wav|aiff?|flac)$", "", f.rsplit("/", 1)[-1], flags=re.I)

            if re.search(PACK_KIT, low):
                if low.endswith("/dry") and any(k.lower() == low[:-3] + "wet" for k in folders):
                    continue
                if len(files) < 2:
                    continue
                names = [("vocals " if vocal_folder else "") + stem_name(f) for f in files]

                def load(files=files):
                    xs = [read(f) for f in files]
                    n = max(x.shape[-1] for x in xs)
                    out = []
                    for x in xs:
                        k = n / x.shape[-1]
                        if round(k) > 1 and abs(k - round(k)) < 0.01 * k:   # a loop that fits a whole number of times
                            x = np.tile(x, (1, int(round(k))))[:, :n]
                        out.append(np.pad(x, ((0, 0), (0, n - x.shape[-1]))))   # anything else plays once
                    return out
                kit = folder.split("/")[-2] if folder.lower().endswith(("/stems", "/wet", "/dry", "wav loops")) else folder.split("/")[-1]
                yield track_id(f"{pack}/{kit}"), kit, names, load, f"{zp}::{folder}"
            elif re.search(PACK_LOOPS, low) or vocal_folder:
                for f in files:
                    nm = stem_name(f)
                    cat = category(f"{folder.split('/')[-1]} {nm}") or "other"
                    yield track_id(f"{pack}/{f}"), nm, [f"{cat}: {nm}"], (lambda f=f: [read(f)]), f"{zp}::{f}"


def zipstems_songs(zip_paths):
    """Multitrack datasets packed as <root>/<song>/... in a .zip, read without unpacking:
    - MoisesDB: <song>/<stem>/<track id>.wav, where <stem> is vocals, bass, drums, percussion,
      guitar, piano, other_keys, bowed_strings, wind, other_plucked or other; drums and
      percussion -> drums, everything that isn't vocals, bass or drums -> other.
    - SDXDB23 (bleeding / label noise): <song>/{bass,drums,other,vocals}.wav."""
    import io
    import posixpath
    import zipfile
    for zp in zip_paths:
        z = zipfile.ZipFile(zp)
        entries = z.namelist()
        wavs = [n for n in entries if n.lower().endswith(".wav")]
        # A song folder holds MoisesDB's data.json; without any (SDX) it's the folder of the wavs.
        roots = sorted({posixpath.dirname(n) for n in entries if n.endswith("data.json")}) or \
            sorted({posixpath.dirname(n) for n in wavs})
        songs = {}
        for n in wavs:
            root = next((r for r in roots if n.startswith(r + "/")), None)
            if root is not None:
                songs.setdefault(root, []).append(n)
        for root, files in sorted(songs.items()):
            song = posixpath.basename(root)
            if len(files) > 60:                       # a layout we don't understand: never sum a whole dataset
                yield song, song, [], (lambda: []), f"{zp}::{root} ({len(files)} files: unexpected layout)"
                continue
            # the stem's name: its folder inside the song (MoisesDB), or the file name (SDX)
            rel = [f[len(root) + 1:] for f in files]
            names = [r.split("/")[0] if "/" in r else r[:-4] for r in rel]
            load = (lambda files=files, z=z: [read_audio(io.BytesIO(z.read(f)), dtype="float32") for f in files])
            yield song, song, names, load, f"{zp}::{root}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["ni", "folder", "pack", "zipstems"])
    ap.add_argument("root", nargs="+", help="a folder (ni, folder) or .zip files (pack, zipstems)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-out", help="every --test-every'th usable song goes here instead (never trained on)")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--musdb", help="MUSDB18-HQ root, to skip songs that are also in MUSDB")
    ap.add_argument("--redo", action="store_true", help="rewrite songs that were already imported (e.g. after a mapping fix)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many songs (a quick test)")
    a = ap.parse_args()

    skip_titles = musdb_titles(a.musdb)
    songs = {"ni": lambda: ni_songs(a.root[0]), "folder": lambda: folder_songs(a.root[0]),
             "pack": lambda: pack_songs(a.root), "zipstems": lambda: zipstems_songs(a.root)}[a.kind]()
    report, usable = [], 0
    for sid, title, names, load, src in songs:
        if a.limit and len(report) >= a.limit:
            break
        cats = [category(n) for n in names]
        if norm(title) in skip_titles:
            report.append((sid, "skipped: also in MUSDB18", names, cats))
            continue
        if not names:
            report.append((sid, f"skipped: no usable stems in {src}", names, cats))
            continue
        if None in cats:
            report.append((sid, "skipped: ambiguous stem " + ", ".join(n for n, c in zip(names, cats) if c is None), names, cats))
            continue
        usable += 1
        is_test = a.test_out and usable % a.test_every == 0
        d = Path(a.test_out if is_test else a.out) / sid
        if (d / "done").exists() and not a.redo:
            report.append((sid, "test (already done)" if is_test else "train (already done)", names, cats))
            continue
        audio = load()
        n = min(x.shape[-1] for x in audio)
        if n < 2 * SAMPLE_RATE:
            report.append((sid, f"skipped: only {n / SAMPLE_RATE:.1f} s", names, cats))
            continue
        stems = np.zeros((len(SOURCES), 2, n), np.float32)
        for x, c in zip(audio, cats):
            stems[SOURCES.index(c)] += x[:, :n]
        peak = np.abs(stems.sum(0)).max()
        if peak > 0.99:                                     # keep the mix from clipping; relative levels unchanged
            stems *= 0.99 / peak
        d.mkdir(parents=True, exist_ok=True)
        for s, x in zip(SOURCES, stems):
            sf.write(d / f"{s}.part", x.T, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            (d / f"{s}.part").replace(d / f"{s}.flac")
        (d / "source.txt").write_text(f"{src}\nseconds\t{n / SAMPLE_RATE:.2f}\n" +
                                      "".join(f"stem\t{nm}\t{c}\n" for nm, c in zip(names, cats)), encoding="utf-8")
        (d / "done").touch()
        report.append((sid, "test" if is_test else "train", names, cats))

    lines = [f"{sid}\t{status}\t" + ", ".join(f"{n}->{c or '?'}" for n, c in zip(names, cats)) for sid, status, names, cats in report]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    count = lambda word: sum(1 for r in report if r[1].startswith(word))
    print(f"\n{count('train')} train, {count('test')} test, {count('skipped')} skipped -> {a.out}")


if __name__ == "__main__":
    main()
