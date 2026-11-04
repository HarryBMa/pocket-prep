"""Where Pocket Prep keeps its files, and the user's settings."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Library database, analysis files and stems live here (not in the repo).
DATA_DIR = Path(os.environ.get("POCKET_PREP_DATA", Path(os.environ.get("APPDATA", Path.home())) / "PocketPrep"))
DB_PATH = DATA_DIR / "library.db"
ANALYSIS_DIR = DATA_DIR / "analysis"
STEMS_DIR = DATA_DIR / "stems"
ARTWORK_DIR = DATA_DIR / "artwork"
SETTINGS_PATH = DATA_DIR / "settings.json"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}

DEFAULTS = {
    "stem_model": "htdemucs",       # Demucs model; htdemucs_ft is slower but cleaner
    "stem_bitrate": 192,            # kbps per stem MP3
    "stem_device": "cuda",          # "cuda" or "cpu"
    "auto_analyse": True,           # analyse tracks as soon as they are imported
    "deck_address": "http://192.168.4.1",
    "bpm_min": 85,                  # BPM range used when halving/doubling detected tempo
    "bpm_max": 175,
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, ANALYSIS_DIR, STEMS_DIR, ARTWORK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    try:
        return {**DEFAULTS, **json.loads(SETTINGS_PATH.read_text("utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(values: dict) -> dict:
    merged = {**load_settings(), **{k: v for k, v in values.items() if k in DEFAULTS}}
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2), "utf-8")
    return merged
