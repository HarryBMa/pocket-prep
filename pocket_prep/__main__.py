"""Start Pocket Prep: the server in a thread, the UI in a desktop window.

    uv run pocket-prep            desktop window
    uv run pocket-prep --browser  open in your web browser instead
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser

import uvicorn

from . import config, db, jobs, server


def _free_port(preferred: int = 8765) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser(prog="pocket-prep")
    ap.add_argument("--browser", action="store_true", help="use the web browser instead of a window")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    config.ensure_dirs()
    db.connect()
    jobs.start()
    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    print(f"Pocket Prep running at {url}  (data: {config.DATA_DIR})")

    if not args.browser:
        try:
            import webview
            w = webview.create_window("Pocket Prep", url, width=1400, height=880, min_size=(980, 640))
            server.WINDOW["w"] = w
            webview.start()
            return
        except Exception as e:  # no WebView2 runtime, etc.
            print("Window unavailable, opening the browser instead:", e)
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
