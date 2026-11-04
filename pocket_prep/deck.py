"""Talking to the deck: status, signed firmware updates, wired recovery.

Wireless (normal): the deck runs Pajoniiir's update service. Turn on
"Wi-Fi Remote" on the deck, join its Wi-Fi network, then this module can
read /api/status and /api/firmware and upload a signed .ddjota bundle to
/api/ota/p4. The deck checks the signature itself before it erases anything.

Wired (recovery): esptool writes an ESP-IDF build folder over USB using its
flash_args file. Use this for first installs or a deck that won't boot.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx

from . import config

CONTROL = {"X-DDJ-Control": "1"}


def _base() -> str:
    return config.load_settings()["deck_address"].rstrip("/")


def status() -> dict:
    """Ask the deck what it's running. Short timeouts: the deck may simply be off."""
    out = {"reachable": False, "address": _base()}
    try:
        with httpx.Client(timeout=2.5) as c:
            r = c.get(_base() + "/api/status")
            out["status"] = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:2000]
            out["reachable"] = r.status_code == 200
            f = c.get(_base() + "/api/firmware")
            if f.status_code == 200:
                out["firmware"] = f.json()
    except (httpx.HTTPError, ValueError) as e:
        out["error"] = str(e)
    return out


def ota_upload(bundle: Path, progress=lambda f, msg="": None) -> dict:
    """Send a signed .ddjota bundle. The deck verifies it before installing."""
    if bundle.suffix.lower() != ".ddjota":
        raise ValueError("Only signed .ddjota bundles can be sent wirelessly")
    data = bundle.read_bytes()
    size = len(data)

    def chunks():
        sent = 0
        step = 64 * 1024
        while sent < size:
            yield data[sent:sent + step]
            sent += step
            progress(0.9 * sent / size, f"Uploading {sent // 1024} / {size // 1024} KB")

    headers = {**CONTROL, "X-DDJ-OTA": "p4", "Content-Type": "application/octet-stream", "Content-Length": str(size)}
    with httpx.Client(timeout=httpx.Timeout(120, connect=5)) as c:
        r = c.post(_base() + "/api/ota/p4", content=chunks(), headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"Deck refused the update ({r.status_code}): {r.text[:300]}")
    progress(1.0, "Uploaded. The deck restarts and confirms the new firmware itself.")
    return {"http": r.status_code, "reply": r.text[:1000]}


def serial_ports() -> list[dict]:
    from serial.tools import list_ports
    return [{"port": p.device, "description": p.description, "espressif": (p.vid == 0x303A)}
            for p in list_ports.comports()]


def wired_flash(build_dir: Path, port: str, progress=lambda f, msg="": None) -> dict:
    """Flash an ESP-IDF build folder (needs flash_args) over USB with esptool."""
    if not (build_dir / "flash_args").exists():
        raise ValueError("That folder has no flash_args file; pick the ESP-IDF build folder")
    cmd = [sys.executable, "-m", "esptool", "--chip", "esp32p4", "-p", port, "-b", "460800",
           "--before", "default-reset", "--after", "hard-reset", "write-flash", "@flash_args"]
    proc = subprocess.Popen(cmd, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), text=True)
    log = []
    for line in proc.stdout:
        log.append(line.rstrip())
        if "%" in line:
            try:
                pct = float(line.split("(")[-1].split("%")[0])
                progress(pct / 100, line.strip()[:120])
            except ValueError:
                pass
    if proc.wait() != 0:
        raise RuntimeError("esptool failed:\n" + "\n".join(log[-15:]))
    return {"log": log[-30:]}
