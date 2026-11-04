"""Writer for rekordbox's device database, PIONEER/rekordbox/export.pdb.

This is the file CDJs and the Pocket Deck firmware (Pajoniiir) read to list
tracks. The format is reverse-engineered; the references are
  https://djl-analysis.deepsymmetry.org/rekordbox-export-analysis/exports.html
  Pajoniiir: firmware/p4-core/components/library/rekordbox_pdb.c

Shape of the file
-----------------
The file is a row of fixed-size pages (4096 bytes). Page 0 is the file
header: it lists every table (tracks, artists, ...) and the first and last
page of each table. Each table is a linked list of pages. A page has a
40-byte header, then a "heap" where rows are written front to back, and at
the very end a slot index that grows backwards: groups of up to 16 row
offsets plus a bitmask saying which slots hold a live row.

Everything is little-endian. Strings use the "DeviceSQL" encoding below.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

PAGE_SIZE = 4096
HEAP_START = 0x28

# Table type ids (the order rekordbox itself uses in the header).
T_TRACKS, T_GENRES, T_ARTISTS, T_ALBUMS, T_LABELS, T_KEYS, T_COLORS = 0, 1, 2, 3, 4, 5, 6
T_PLAYLIST_TREE, T_PLAYLIST_ENTRIES = 7, 8
T_ARTWORK, T_COLUMNS, T_HISTORY_PLAYLISTS, T_HISTORY_ENTRIES, T_HISTORY = 13, 16, 17, 18, 19
ALL_TABLES = list(range(20))

# The eight track colours rekordbox shows (id 1..8).
COLORS = ["Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"]


# ── DeviceSQL strings ────────────────────────────────────────────────────────

def devicesql(text: str | None) -> bytes:
    """Encode a string the way the PDB stores it.

    * short ASCII (< 127 bytes): one flag byte ((len+1) << 1 | 1) then the bytes
    * long ASCII: 0x40, uint16 total length, pad byte, bytes
    * anything non-ASCII: 0x90 (UTF-16 little-endian), uint16 length, pad, data
    """
    text = text or ""
    try:
        raw = text.encode("ascii")
        if len(raw) <= 126:
            return bytes([((len(raw) + 1) << 1) | 1]) + raw
        return struct.pack("<BHB", 0x40, len(raw) + 4, 0) + raw
    except UnicodeEncodeError:
        raw = text.encode("utf-16-le")
        return struct.pack("<BHB", 0x90, len(raw) + 4, 0) + raw


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


# ── Rows ─────────────────────────────────────────────────────────────────────

@dataclass
class TrackRow:
    id: int
    title: str
    file_path: str            # "/Contents/Artist/track.mp3"
    filename: str
    anlz_path: str            # "/PIONEER/USBANLZ/P000/00000001/ANLZ0000.DAT"
    artist_id: int = 0
    album_id: int = 0
    genre_id: int = 0
    key_id: int = 0
    artwork_id: int = 0
    color_id: int = 0
    bpm: float = 0.0
    duration_s: int = 0
    sample_rate: int = 44100
    bitrate: int = 0
    file_size: int = 0
    sample_depth: int = 16
    year: int = 0
    rating: int = 0
    track_number: int = 0
    comment: str = ""
    date_added: str = ""      # "YYYY-MM-DD"
    analyze_date: str = ""
    file_type: int = 1        # 1 mp3, 4 m4a, 5 flac, 0x0b wav, 0x0c aiff


def track_row(t: TrackRow, index: int) -> bytes:
    # 21 strings, in the order of the string-offset table.
    strings = [""] * 21
    strings[7] = "ON"                 # autoload hot cues
    strings[10] = t.date_added
    strings[14] = t.anlz_path
    strings[15] = t.analyze_date
    strings[16] = t.comment
    strings[17] = t.title
    strings[19] = t.filename
    strings[20] = t.file_path

    fixed = struct.pack(
        "<HHIIIIIHHIIIIIIIIIIIIHHHHHHBBHH",
        0x24,                       # 0x00 subtype: track row
        index * 0x20,               # 0x02 index shift
        0x000C0700,                 # 0x04 bitmask seen in real exports
        t.sample_rate,              # 0x08
        0,                          # 0x0C composer id
        t.file_size,                # 0x10
        0,                          # 0x14 unknown
        0x5FE9, 0x6C2A,             # 0x18, 0x1A unknown constants
        t.artwork_id,               # 0x1C
        t.key_id,                   # 0x20
        0,                          # 0x24 original artist
        0,                          # 0x28 label
        0,                          # 0x2C remixer
        t.bitrate,                  # 0x30 kbps
        t.track_number,             # 0x34
        int(round(t.bpm * 100)),    # 0x38 tempo × 100
        t.genre_id,                 # 0x3C
        t.album_id,                 # 0x40
        t.artist_id,                # 0x44
        t.id,                       # 0x48
        0,                          # 0x4C disc number
        0,                          # 0x4E play count
        t.year,                     # 0x50
        t.sample_depth,             # 0x52
        t.duration_s,               # 0x54
        0x29,                       # 0x56 unknown (always 41)
        t.color_id,                 # 0x58
        t.rating,                   # 0x59
        t.file_type,                # 0x5A
        3,                          # 0x5C unknown
    )
    assert len(fixed) == 0x5E, len(fixed)
    heap = b""
    offsets = []
    base = 0x5E + 21 * 2
    for s in strings:
        offsets.append(base + len(heap))
        heap += devicesql(s)
    row = fixed + struct.pack("<21H", *offsets) + heap
    return _pad4(row)


def artist_row(id: int, name: str, index: int) -> bytes:
    # subtype 0x60, index shift, id, 0x03, offset-to-name (0x0A), name
    return _pad4(struct.pack("<HHIBB", 0x60, index * 0x20, id, 0x03, 0x0A) + devicesql(name))


def album_row(id: int, name: str, artist_id: int, index: int) -> bytes:
    # subtype 0x80, index shift, unknown, artist id, id, unknown, 0x03, name offset (0x16), name
    return _pad4(struct.pack("<HHIIIIBB", 0x80, index * 0x20, 0, artist_id, id, 0, 0x03, 0x16)
                 + devicesql(name))


def simple_name_row(id: int, name: str) -> bytes:
    """Genres and labels: id then the name."""
    return _pad4(struct.pack("<I", id) + devicesql(name))


def key_row(id: int, name: str) -> bytes:
    """Keys: id, id again, name."""
    return _pad4(struct.pack("<II", id, id) + devicesql(name))


def color_row(id: int, name: str) -> bytes:
    """Colours: 4 unknown bytes, unknown byte, id (u16), unknown byte, name."""
    return _pad4(struct.pack("<IBHB", 0, 0, id, 0) + devicesql(name))


def playlist_tree_row(id: int, parent_id: int, sort: int, name: str, is_folder: bool) -> bytes:
    return _pad4(struct.pack("<IIIII", parent_id, 0, sort, id, 1 if is_folder else 0) + devicesql(name))


def playlist_entry_row(entry_index: int, track_id: int, playlist_id: int) -> bytes:
    return struct.pack("<III", entry_index, track_id, playlist_id)


def artwork_row(id: int, path: str) -> bytes:
    return _pad4(struct.pack("<I", id) + devicesql(path))


# ── Pages ────────────────────────────────────────────────────────────────────

def _slot_bytes(n_rows: int) -> int:
    """Bytes the backwards slot index needs for n rows (groups of 16)."""
    full, rest = divmod(n_rows, 16)
    return full * 36 + (4 + 2 * rest if rest else 0)


def _build_page(page_index: int, table: int, next_page: int, rows: list[bytes]) -> bytes:
    page = bytearray(PAGE_SIZE)
    heap = bytearray()
    offsets = []
    for r in rows:
        offsets.append(len(heap))
        heap += r
    page[HEAP_START:HEAP_START + len(heap)] = heap

    # Slot groups, written backwards from the end of the page.
    ptr = PAGE_SIZE
    for g in range(0, len(rows), 16):
        group = offsets[g:g + 16]
        m = len(group)
        present = (1 << m) - 1
        struct.pack_into("<H", page, ptr - 2, 0)          # transaction flags
        struct.pack_into("<H", page, ptr - 4, present)    # live-row bitmask
        for i, off in enumerate(group):
            struct.pack_into("<H", page, ptr - 4 - 2 * (i + 1), off)
        ptr -= 4 + 2 * m

    n = len(rows)
    used = len(heap)
    free = PAGE_SIZE - HEAP_START - used - _slot_bytes(n)
    packed = (n & 0x1FFF) | ((n & 0x7FF) << 13) | (0x34 << 24)   # 0x34 = data page
    struct.pack_into("<IIIIII", page, 0, 0, page_index, table, next_page, 1, 0)
    struct.pack_into("<IHHHHHH", page, 0x18, packed, free, used, 1 if n else 0,
                     n if n > 0xFF else 0, 0, 0)
    return bytes(page)


def _paginate(rows: list[bytes]) -> list[list[bytes]]:
    """Split rows into pages that fit heap + slot index."""
    pages, cur, cur_bytes = [], [], 0
    room = PAGE_SIZE - HEAP_START
    for r in rows:
        if len(r) + _slot_bytes(1) > room:
            raise ValueError("row too large for a page")
        if cur and cur_bytes + len(r) + _slot_bytes(len(cur) + 1) > room:
            pages.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += len(r)
    pages.append(cur)          # always at least one (possibly empty) page
    return pages


@dataclass
class Database:
    """Collect rows per table, then write() the whole file."""
    tables: dict[int, list[bytes]] = field(default_factory=lambda: {t: [] for t in ALL_TABLES})

    def add(self, table: int, row: bytes) -> None:
        self.tables[table].append(row)

    def write(self) -> bytes:
        out_pages: list[bytes] = []
        pointers = []
        next_index = 1                       # page 0 is the file header
        for t in ALL_TABLES:
            chunks = _paginate(self.tables[t])
            first = next_index
            for i, chunk in enumerate(chunks):
                idx = next_index
                nxt = idx + 1 if i < len(chunks) - 1 else 0xFFFFFFFF
                out_pages.append(_build_page(idx, t, nxt, chunk))
                next_index += 1
            pointers.append((t, next_index, first, next_index - 1))

        header = bytearray(PAGE_SIZE)
        struct.pack_into("<IIIIIII", header, 0, 0, PAGE_SIZE, len(ALL_TABLES),
                         next_index, 5, 1, 0)
        for i, (t, empty, first, last) in enumerate(pointers):
            struct.pack_into("<IIII", header, 0x1C + 16 * i, t, empty, first, last)
        return bytes(header) + b"".join(out_pages)


def add_colors(db: Database) -> None:
    for i, name in enumerate(COLORS, start=1):
        db.add(T_COLORS, color_row(i, name))
