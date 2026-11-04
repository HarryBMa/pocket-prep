"""Writer for rekordbox analysis files (ANLZ0000.DAT and ANLZ0000.EXT).

One pair of files per track under PIONEER/USBANLZ/. They hold the beat
grid, waveforms and cue points so the player doesn't have to analyse the
track itself. Everything is big-endian.

Each file is a PMAI header followed by tagged sections:
    4 bytes  tag, e.g. b"PQTZ"
    4 bytes  header size (counted from the tag)
    4 bytes  total section size (header + data)
    ...      rest of the header, then the data

References: Deep Symmetry's ANLZ notes and Pajoniiir's rekordbox_anlz.c.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


def _section(tag: bytes, header_rest: bytes, data: bytes) -> bytes:
    hdr = 12 + len(header_rest)
    return tag + struct.pack(">II", hdr, hdr + len(data)) + header_rest + data


def _file(sections: list[bytes]) -> bytes:
    body = b"".join(sections)
    # PMAI: header size 28, total = whole file, then four constants.
    return b"PMAI" + struct.pack(">IIIIII", 28, 28 + len(body), 1, 0x10000, 0x10000, 0) + body


def ppth(path: str) -> bytes:
    """Path of the audio file on the stick, UTF-16 BE with a trailing NUL."""
    data = (path + "\x00").encode("utf-16-be")
    return _section(b"PPTH", struct.pack(">I", len(data)), data)


def pqtz(beats: list[tuple[int, int, float]]) -> bytes:
    """Beat grid: (beat number 1-4, time ms, bpm) for every beat."""
    data = b"".join(struct.pack(">HHI", n, int(round(bpm * 100)), int(ms)) for n, ms, bpm in beats)
    return _section(b"PQTZ", struct.pack(">III", 0, 0x80000, len(beats)), data)


def pwav(cols: bytes) -> bytes:
    """400-column monochrome preview: bits 0-4 height, 5-7 whiteness."""
    return _section(b"PWAV", struct.pack(">II", len(cols), 0x10000), cols)


def pwv2(cols: bytes) -> bytes:
    """100-column tiny preview (4-bit heights) used by older players."""
    return _section(b"PWV2", struct.pack(">II", len(cols), 0x10000), cols)


def pwv3(cols: bytes) -> bytes:
    """Scrolling monochrome detail, 150 columns per second, 1 byte each."""
    return _section(b"PWV3", struct.pack(">III", 1, len(cols), 0x00960000), cols)


def pwv4(entries: bytes) -> bytes:
    """Colour preview: 1200 columns × 6 bytes (height, height, 0, low, mid, high)."""
    return _section(b"PWV4", struct.pack(">III", 6, len(entries) // 6, 0), entries)


def pwv5(entries: bytes) -> bytes:
    """Colour detail, 150 columns per second, 2 bytes each (RGB 3+3+3 bits, height 5)."""
    return _section(b"PWV5", struct.pack(">III", 2, len(entries) // 2, 0x00960305), entries)


@dataclass
class Cue:
    start_ms: int
    end_ms: int | None = None   # set for a loop
    hot: int = 0                # 1..8 = hot cue A..H, 0 = memory cue


def _pcpt(c: Cue) -> bytes:
    is_loop = c.end_ms is not None and c.end_ms > c.start_ms
    rest = struct.pack(
        ">IIIHHBBHII16x",
        c.hot,                               # 0x0C hot cue number (0 = memory)
        0 if c.hot == 0 else 4,              # 0x10 status
        0x10000,                             # 0x14
        0xFFFF, 0xFFFF,                      # 0x18 order
        2 if is_loop else 1,                 # 0x1C type: 1 cue, 2 loop
        0,                                   # 0x1D
        1000,                                # 0x1E
        c.start_ms,                          # 0x20
        c.end_ms if is_loop else 0xFFFFFFFF, # 0x24
    )
    # PCPT has a 28-byte header and is 56 bytes in total.
    return b"PCPT" + struct.pack(">II", 28, 56) + rest


def pcob(cues: list[Cue], hot: bool) -> bytes:
    data = b"".join(_pcpt(c) for c in cues)
    rest = struct.pack(">IHHI", 1 if hot else 0, 0, len(cues), 0xFFFFFFFF)
    return _section(b"PCOB", rest, data)


@dataclass
class TrackAnalysis:
    path: str                                  # "/Contents/..." on the stick
    beats: list[tuple[int, int, float]]        # (1-4, ms, bpm)
    pwav: bytes                                # 400 bytes
    pwv2: bytes                                # 100 bytes
    pwv3: bytes                                # 150/s
    pwv4: bytes                                # 1200 × 6
    pwv5: bytes                                # 150/s × 2
    hot_cues: list[Cue] = field(default_factory=list)
    memory_cues: list[Cue] = field(default_factory=list)


def build_dat(a: TrackAnalysis) -> bytes:
    hot = sorted((c for c in a.hot_cues if 1 <= c.hot <= 8), key=lambda c: c.hot)
    return _file([
        ppth(a.path),
        pqtz(a.beats),
        pwav(a.pwav),
        pwv2(a.pwv2),
        pcob(hot, hot=True),
        pcob(a.memory_cues, hot=False),
    ])


def build_ext(a: TrackAnalysis) -> bytes:
    return _file([
        ppth(a.path),
        pwv3(a.pwv3),
        pcob([], hot=True),
        pcob([], hot=False),
        pwv4(a.pwv4),
        pwv5(a.pwv5),
    ])
