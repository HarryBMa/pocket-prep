//! Writer for rekordbox's device database, PIONEER/rekordbox/export.pdb.
//!
//! This is the file CDJs and the Pocket Deck firmware read to list
//! tracks. The format is reverse-engineered; the references are
//!   https://djl-analysis.deepsymmetry.org/rekordbox-export-analysis/exports.html
//!   the deck firmware's PDB reader (p4-core, components/library/rekordbox_pdb.c)
//!
//! Shape of the file
//! -----------------
//! The file is a row of fixed-size pages (4096 bytes). Page 0 is the file
//! header: it lists every table (tracks, artists, ...) and the first and last
//! page of each table. Each table is a linked list of pages. A page has a
//! 40-byte header, then a "heap" where rows are written front to back, and at
//! the very end a slot index that grows backwards: groups of up to 16 row
//! offsets plus a bitmask saying which slots hold a live row.
//!
//! Everything is little-endian. Strings use the "DeviceSQL" encoding below.

use anyhow::{anyhow, Result};

pub const PAGE_SIZE: usize = 4096;
const HEAP_START: usize = 0x28;

// Table type ids (the order rekordbox itself uses in the header).
pub const T_TRACKS: usize = 0;
pub const T_GENRES: usize = 1;
pub const T_ARTISTS: usize = 2;
pub const T_ALBUMS: usize = 3;
pub const T_KEYS: usize = 5;
pub const T_COLORS: usize = 6;
pub const T_PLAYLIST_TREE: usize = 7;
pub const T_PLAYLIST_ENTRIES: usize = 8;
pub const T_ARTWORK: usize = 13;
const N_TABLES: usize = 20;

/// The eight track colours rekordbox shows (id 1..8).
const COLORS: [&str; 8] = ["Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"];

// ── DeviceSQL strings ────────────────────────────────────────────────────────

/// Encode a string the way the PDB stores it.
///
/// * short ASCII (< 127 bytes): one flag byte ((len+1) << 1 | 1) then the bytes
/// * long ASCII: 0x40, uint16 total length, pad byte, bytes
/// * anything non-ASCII: 0x90 (UTF-16 little-endian), uint16 length, pad, data
pub fn devicesql(text: &str) -> Vec<u8> {
    if text.is_ascii() {
        let raw = text.as_bytes();
        if raw.len() <= 126 {
            return [&[(((raw.len() + 1) << 1) | 1) as u8], raw].concat();
        }
        let mut out = vec![0x40];
        out.extend_from_slice(&((raw.len() + 4) as u16).to_le_bytes());
        out.push(0);
        out.extend_from_slice(raw);
        return out;
    }
    let raw: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let mut out = vec![0x90];
    out.extend_from_slice(&((raw.len() + 4) as u16).to_le_bytes());
    out.push(0);
    out.extend_from_slice(&raw);
    out
}

fn pad4(mut b: Vec<u8>) -> Vec<u8> {
    b.resize(b.len().div_ceil(4) * 4, 0);
    b
}

/// Little-endian byte builder, so row layouts read like the format notes.
#[derive(Default)]
struct Le(Vec<u8>);

impl Le {
    fn u8(mut self, v: u8) -> Self { self.0.push(v); self }
    fn u16(mut self, v: u16) -> Self { self.0.extend_from_slice(&v.to_le_bytes()); self }
    fn u32(mut self, v: u32) -> Self { self.0.extend_from_slice(&v.to_le_bytes()); self }
    fn bytes(mut self, v: &[u8]) -> Self { self.0.extend_from_slice(v); self }
}

// ── Rows ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct TrackRow {
    pub id: u32,
    pub title: String,
    pub file_path: String,     // "/Contents/Artist/track.mp3"
    pub filename: String,
    pub anlz_path: String,     // "/PIONEER/USBANLZ/P000/00000001/ANLZ0000.DAT"
    pub artist_id: u32,
    pub album_id: u32,
    pub genre_id: u32,
    pub key_id: u32,
    pub artwork_id: u32,
    pub color_id: u8,
    pub bpm: f64,
    pub duration_s: u16,
    pub sample_rate: u32,      // the file's own rate, from its tags
    pub bitrate: u32,
    pub file_size: u32,
    pub sample_depth: u16,
    pub year: u16,
    pub rating: u8,
    pub track_number: u32,
    pub comment: String,
    pub date_added: String,    // "YYYY-MM-DD"
    pub analyze_date: String,
    pub file_type: u16,        // 1 mp3, 4 m4a, 5 flac, 0x0b wav, 0x0c aiff
}

pub fn track_row(t: &TrackRow, index: usize) -> Vec<u8> {
    // 21 strings, in the order of the string-offset table.
    let mut strings = vec![""; 21];
    strings[7] = "ON";                     // autoload hot cues
    strings[10] = &t.date_added;
    strings[14] = &t.anlz_path;
    strings[15] = &t.analyze_date;
    strings[16] = &t.comment;
    strings[17] = &t.title;
    strings[19] = &t.filename;
    strings[20] = &t.file_path;

    let fixed = Le::default()
        .u16(0x24)                         // 0x00 subtype: track row
        .u16((index * 0x20) as u16)        // 0x02 index shift
        .u32(0x000C_0700)                  // 0x04 bitmask seen in real exports
        .u32(t.sample_rate)                // 0x08
        .u32(0)                            // 0x0C composer id
        .u32(t.file_size)                  // 0x10
        .u32(0)                            // 0x14 unknown
        .u16(0x5FE9).u16(0x6C2A)           // 0x18, 0x1A unknown constants
        .u32(t.artwork_id)                 // 0x1C
        .u32(t.key_id)                     // 0x20
        .u32(0)                            // 0x24 original artist
        .u32(0)                            // 0x28 label
        .u32(0)                            // 0x2C remixer
        .u32(t.bitrate)                    // 0x30 kbps
        .u32(t.track_number)               // 0x34
        .u32((t.bpm * 100.0).round_ties_even() as u32) // 0x38 tempo × 100
        .u32(t.genre_id)                   // 0x3C
        .u32(t.album_id)                   // 0x40
        .u32(t.artist_id)                  // 0x44
        .u32(t.id)                         // 0x48
        .u16(0)                            // 0x4C disc number
        .u16(0)                            // 0x4E play count
        .u16(t.year)                       // 0x50
        .u16(t.sample_depth)               // 0x52
        .u16(t.duration_s)                 // 0x54
        .u16(0x29)                         // 0x56 unknown (always 41)
        .u8(t.color_id)                    // 0x58
        .u8(t.rating)                      // 0x59
        .u16(t.file_type)                  // 0x5A
        .u16(3)                            // 0x5C unknown
        .0;
    assert_eq!(fixed.len(), 0x5E);
    let mut heap = vec![];
    let mut offsets = Le::default();
    let base = 0x5E + 21 * 2;
    for s in strings {
        offsets = offsets.u16((base + heap.len()) as u16);
        heap.extend(devicesql(s));
    }
    pad4([fixed, offsets.0, heap].concat())
}

pub fn artist_row(id: u32, name: &str, index: usize) -> Vec<u8> {
    // subtype 0x60, index shift, id, 0x03, offset-to-name (0x0A), name
    pad4(Le::default().u16(0x60).u16((index * 0x20) as u16).u32(id).u8(3).u8(0x0A).bytes(&devicesql(name)).0)
}

pub fn album_row(id: u32, name: &str, artist_id: u32, index: usize) -> Vec<u8> {
    // subtype 0x80, index shift, unknown, artist id, id, unknown, 0x03, name offset (0x16), name
    pad4(Le::default().u16(0x80).u16((index * 0x20) as u16).u32(0).u32(artist_id).u32(id).u32(0).u8(3).u8(0x16)
        .bytes(&devicesql(name)).0)
}

/// Genres and labels: id then the name.
pub fn simple_name_row(id: u32, name: &str) -> Vec<u8> {
    pad4(Le::default().u32(id).bytes(&devicesql(name)).0)
}

/// Keys: id, id again, name.
pub fn key_row(id: u32, name: &str) -> Vec<u8> {
    pad4(Le::default().u32(id).u32(id).bytes(&devicesql(name)).0)
}

/// Colours: 4 unknown bytes, unknown byte, id (u16), unknown byte, name.
fn color_row(id: u16, name: &str) -> Vec<u8> {
    pad4(Le::default().u32(0).u8(0).u16(id).u8(0).bytes(&devicesql(name)).0)
}

pub fn playlist_tree_row(id: u32, parent_id: u32, sort: u32, name: &str, is_folder: bool) -> Vec<u8> {
    pad4(Le::default().u32(parent_id).u32(0).u32(sort).u32(id).u32(is_folder as u32).bytes(&devicesql(name)).0)
}

pub fn playlist_entry_row(entry_index: u32, track_id: u32, playlist_id: u32) -> Vec<u8> {
    Le::default().u32(entry_index).u32(track_id).u32(playlist_id).0
}

pub fn artwork_row(id: u32, path: &str) -> Vec<u8> {
    pad4(Le::default().u32(id).bytes(&devicesql(path)).0)
}

// ── Pages ────────────────────────────────────────────────────────────────────

/// Bytes the backwards slot index needs for n rows (groups of 16).
fn slot_bytes(n: usize) -> usize {
    let (full, rest) = (n / 16, n % 16);
    full * 36 + if rest > 0 { 4 + 2 * rest } else { 0 }
}

fn put16(p: &mut [u8], at: usize, v: u16) { p[at..at + 2].copy_from_slice(&v.to_le_bytes()); }
fn put32(p: &mut [u8], at: usize, v: u32) { p[at..at + 4].copy_from_slice(&v.to_le_bytes()); }

fn build_page(page_index: u32, table: u32, next_page: u32, rows: &[Vec<u8>]) -> Vec<u8> {
    let mut page = vec![0u8; PAGE_SIZE];
    let mut offsets = vec![];
    let mut used = 0;
    for r in rows {
        offsets.push(used as u16);
        page[HEAP_START + used..HEAP_START + used + r.len()].copy_from_slice(r);
        used += r.len();
    }
    // Slot groups, written backwards from the end of the page.
    let mut ptr = PAGE_SIZE;
    for group in offsets.chunks(16) {
        let m = group.len();
        put16(&mut page, ptr - 2, 0);                                   // transaction flags
        put16(&mut page, ptr - 4, ((1u32 << m) - 1) as u16);            // live-row bitmask
        for (i, off) in group.iter().enumerate() {
            put16(&mut page, ptr - 4 - 2 * (i + 1), *off);
        }
        ptr -= 4 + 2 * m;
    }
    let n = rows.len() as u32;
    let free = (PAGE_SIZE - HEAP_START - used - slot_bytes(rows.len())) as u16;
    let packed = (n & 0x1FFF) | ((n & 0x7FF) << 13) | (0x34 << 24);     // 0x34 = data page
    for (i, v) in [0, page_index, table, next_page, 1, 0].iter().enumerate() {
        put32(&mut page, i * 4, *v);
    }
    put32(&mut page, 0x18, packed);
    for (i, v) in [free, used as u16, (n > 0) as u16, if n > 0xFF { n as u16 } else { 0 }, 0, 0].iter().enumerate() {
        put16(&mut page, 0x1C + i * 2, *v);
    }
    page
}

/// Split rows into pages that fit heap + slot index.
fn paginate(rows: &[Vec<u8>]) -> Result<Vec<Vec<Vec<u8>>>> {
    let room = PAGE_SIZE - HEAP_START;
    let mut pages = vec![];
    let (mut cur, mut cur_bytes): (Vec<Vec<u8>>, usize) = (vec![], 0);
    for r in rows {
        if r.len() + slot_bytes(1) > room {
            return Err(anyhow!("row too large for a page"));
        }
        if !cur.is_empty() && cur_bytes + r.len() + slot_bytes(cur.len() + 1) > room {
            pages.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur_bytes += r.len();
        cur.push(r.clone());
    }
    pages.push(cur);          // always at least one (possibly empty) page
    Ok(pages)
}

/// Collect rows per table, then write() the whole file.
pub struct Database {
    tables: Vec<Vec<Vec<u8>>>,
}

impl Database {
    pub fn new() -> Self {
        let mut db = Database { tables: vec![vec![]; N_TABLES] };
        for (i, name) in COLORS.iter().enumerate() {
            db.add(T_COLORS, color_row(i as u16 + 1, name));
        }
        db
    }

    pub fn add(&mut self, table: usize, row: Vec<u8>) {
        self.tables[table].push(row);
    }

    pub fn write(&self) -> Result<Vec<u8>> {
        let mut out_pages = vec![];
        let mut pointers = vec![];
        let mut next_index = 1u32;                 // page 0 is the file header
        for (t, rows) in self.tables.iter().enumerate() {
            let chunks = paginate(rows)?;
            let first = next_index;
            for (i, chunk) in chunks.iter().enumerate() {
                let nxt = if i + 1 < chunks.len() { next_index + 1 } else { 0xFFFF_FFFF };
                out_pages.extend(build_page(next_index, t as u32, nxt, chunk));
                next_index += 1;
            }
            pointers.push((t as u32, next_index, first, next_index - 1));
        }
        let mut header = vec![0u8; PAGE_SIZE];
        for (i, v) in [0, PAGE_SIZE as u32, N_TABLES as u32, next_index, 5, 1, 0].iter().enumerate() {
            put32(&mut header, i * 4, *v);
        }
        for (i, (t, empty, first, last)) in pointers.iter().enumerate() {
            for (j, v) in [*t, *empty, *first, *last].iter().enumerate() {
                put32(&mut header, 0x1C + 16 * i + 4 * j, *v);
            }
        }
        Ok([header, out_pages].concat())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fallible_iterator::FallibleIterator;
    use rekordcrate::pdb::io::Database as Reader;
    use rekordcrate::pdb::{Album, Artist, DatabaseType, Key, PlaylistEntry, PlaylistTreeNode, Track};

    /// Parse our export.pdb with rekordcrate, an independent parser tested against real
    /// rekordbox exports: every table must read back with the rows and strings we wrote.
    #[test]
    fn rekordcrate_reads_our_database() {
        let mut db = Database::new();
        let n = 60;                                        // enough to span several pages
        for i in 0..n {
            let title = if i % 3 == 0 { format!("Låt nummer {i} – åäö") } else { format!("Track {i}") };
            let row = TrackRow {
                id: i + 1, title, file_path: format!("/Contents/A/B/{i}.mp3"), filename: format!("{i}.mp3"),
                anlz_path: format!("/PIONEER/USBANLZ/P000/{:08X}/ANLZ0000.DAT", i + 1),
                artist_id: 1, album_id: 1, key_id: 1, bpm: 128.0, sample_rate: 44100, file_type: 1,
                ..Default::default()
            };
            db.add(T_TRACKS, track_row(&row, i as usize));
        }
        db.add(T_ARTISTS, artist_row(1, "Sébastien", 0));
        db.add(T_ALBUMS, album_row(1, "Album", 1, 0));
        db.add(T_KEYS, key_row(1, "Am"));
        db.add(T_PLAYLIST_TREE, playlist_tree_row(1, 0, 1, "Friday", false));
        for i in 0..n {
            db.add(T_PLAYLIST_ENTRIES, playlist_entry_row(i + 1, i + 1, 1));
        }
        let bytes = db.write().unwrap();

        let mut r = Reader::open_non_persistent(std::io::Cursor::new(bytes.as_slice()), DatabaseType::Plain).unwrap();
        let tracks: Vec<(String, String)> = r.iter_rows::<Track>().unwrap()
            .map(|t| Ok((t.offsets.inner.title.clone().into_string().unwrap(), t.offsets.inner.file_path.clone().into_string().unwrap())))
            .collect().unwrap();
        assert_eq!(tracks.len(), n as usize);
        assert_eq!(tracks[3].0, "Låt nummer 3 – åäö");
        assert_eq!(tracks[59].1, "/Contents/A/B/59.mp3");
        assert_eq!(r.iter_rows::<Artist>().unwrap().count().unwrap(), 1);
        assert_eq!(r.iter_rows::<Album>().unwrap().count().unwrap(), 1);
        assert_eq!(r.iter_rows::<Key>().unwrap().count().unwrap(), 1);
        assert_eq!(r.iter_rows::<PlaylistTreeNode>().unwrap().count().unwrap(), 1);
        assert_eq!(r.iter_rows::<PlaylistEntry>().unwrap().count().unwrap(), n as usize);
    }
}
