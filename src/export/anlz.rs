//! Writer for rekordbox analysis files (ANLZ0000.DAT and ANLZ0000.EXT).
//!
//! One pair of files per track under PIONEER/USBANLZ/. They hold the beat
//! grid, waveforms and cue points so the player doesn't have to analyse the
//! track itself. Everything is big-endian.
//!
//! Each file is a PMAI header followed by tagged sections:
//!     4 bytes  tag, e.g. b"PQTZ"
//!     4 bytes  header size (counted from the tag)
//!     4 bytes  total section size (header + data)
//!     ...      rest of the header, then the data
//!
//! References: Deep Symmetry's ANLZ notes and Pajoniiir's rekordbox_anlz.c.

fn be32(v: u32) -> [u8; 4] { v.to_be_bytes() }

fn section(tag: &[u8; 4], header_rest: &[u8], data: &[u8]) -> Vec<u8> {
    let hdr = 12 + header_rest.len() as u32;
    [tag.as_slice(), &be32(hdr), &be32(hdr + data.len() as u32), header_rest, data].concat()
}

fn file(sections: Vec<Vec<u8>>) -> Vec<u8> {
    let body = sections.concat();
    // PMAI: header size 28, total = whole file, then four constants.
    let head: Vec<u8> = [28, 28 + body.len() as u32, 1, 0x10000, 0x10000, 0].iter().flat_map(|v| be32(*v)).collect();
    [b"PMAI".as_slice(), &head, &body].concat()
}

/// Path of the audio file on the stick, UTF-16 BE with a trailing NUL.
fn ppth(path: &str) -> Vec<u8> {
    let data: Vec<u8> = path.encode_utf16().chain([0]).flat_map(u16::to_be_bytes).collect();
    section(b"PPTH", &be32(data.len() as u32), &data)
}

/// Beat grid: (beat number 1-4, time ms, bpm) for every beat.
fn pqtz(beats: &[(u16, u32, f64)]) -> Vec<u8> {
    let data: Vec<u8> = beats.iter().flat_map(|&(n, ms, bpm)| {
        [n.to_be_bytes().as_slice(), &((bpm * 100.0).round_ties_even() as u16).to_be_bytes(), &be32(ms)].concat()
    }).collect();
    section(b"PQTZ", &[be32(0), be32(0x80000), be32(beats.len() as u32)].concat(), &data)
}

/// 400-column monochrome preview (PWAV) or 100-column tiny preview (PWV2).
fn preview(tag: &[u8; 4], cols: &[u8]) -> Vec<u8> {
    section(tag, &[be32(cols.len() as u32), be32(0x10000)].concat(), cols)
}

/// Scrolling monochrome detail, 150 columns per second, 1 byte each.
fn pwv3(cols: &[u8]) -> Vec<u8> {
    section(b"PWV3", &[be32(1), be32(cols.len() as u32), be32(0x0096_0000)].concat(), cols)
}

/// Colour preview: 1200 columns × 6 bytes (height, height, 0, low, mid, high).
fn pwv4(e: &[u8]) -> Vec<u8> {
    section(b"PWV4", &[be32(6), be32(e.len() as u32 / 6), be32(0)].concat(), e)
}

/// Colour detail, 150 columns per second, 2 bytes each (RGB 3+3+3 bits, height 5).
fn pwv5(e: &[u8]) -> Vec<u8> {
    section(b"PWV5", &[be32(2), be32(e.len() as u32 / 2), be32(0x0096_0305)].concat(), e)
}

#[derive(Clone)]
pub struct Cue {
    pub start_ms: u32,
    pub end_ms: Option<u32>,   // set for a loop
    pub hot: u32,              // 1..8 = hot cue A..H, 0 = memory cue
}

fn pcpt(c: &Cue) -> Vec<u8> {
    let is_loop = c.end_ms.is_some_and(|e| e > c.start_ms);
    let mut rest = [
        be32(c.hot).as_slice(),                                   // 0x0C hot cue number (0 = memory)
        &be32(if c.hot == 0 { 0 } else { 4 }),                    // 0x10 status
        &be32(0x10000),                                           // 0x14
        &[0xFF, 0xFF, 0xFF, 0xFF],                                // 0x18 order
        &[if is_loop { 2 } else { 1 }, 0],                        // 0x1C type: 1 cue, 2 loop
        &1000u16.to_be_bytes(),                                   // 0x1E
        &be32(c.start_ms),                                        // 0x20
        &be32(if is_loop { c.end_ms.unwrap() } else { 0xFFFF_FFFF }), // 0x24
    ].concat();
    rest.resize(rest.len() + 16, 0);
    // PCPT has a 28-byte header and is 56 bytes in total.
    [b"PCPT".as_slice(), &be32(28), &be32(56), &rest].concat()
}

fn pcob(cues: &[Cue], hot: bool) -> Vec<u8> {
    let data: Vec<u8> = cues.iter().flat_map(pcpt).collect();
    let rest = [be32(hot as u32).as_slice(), &0u16.to_be_bytes(), &(cues.len() as u16).to_be_bytes(), &be32(0xFFFF_FFFF)].concat();
    section(b"PCOB", &rest, &data)
}

pub struct TrackAnalysis {
    pub path: String,                    // "/Contents/..." on the stick
    pub beats: Vec<(u16, u32, f64)>,     // (1-4, ms, bpm)
    pub pwav: Vec<u8>,                   // 400 bytes
    pub pwv2: Vec<u8>,                   // 100 bytes
    pub pwv3: Vec<u8>,                   // 150/s
    pub pwv4: Vec<u8>,                   // 1200 × 6
    pub pwv5: Vec<u8>,                   // 150/s × 2
    pub hot_cues: Vec<Cue>,
    pub memory_cues: Vec<Cue>,
}

pub fn build_dat(a: &TrackAnalysis) -> Vec<u8> {
    let mut hot: Vec<Cue> = a.hot_cues.iter().filter(|c| (1..=8).contains(&c.hot)).cloned().collect();
    hot.sort_by_key(|c| c.hot);
    file(vec![ppth(&a.path), pqtz(&a.beats), preview(b"PWAV", &a.pwav), preview(b"PWV2", &a.pwv2),
              pcob(&hot, true), pcob(&a.memory_cues, false)])
}

pub fn build_ext(a: &TrackAnalysis) -> Vec<u8> {
    file(vec![ppth(&a.path), pwv3(&a.pwv3), pcob(&[], true), pcob(&[], false), pwv4(&a.pwv4), pwv5(&a.pwv5)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinRead, BinWrite};
    use rekordcrate::anlz::{Content, ANLZ};

    fn sample() -> TrackAnalysis {
        TrackAnalysis {
            path: "/Contents/Artist/Album/Låt.mp3".into(),
            beats: (0..64).map(|i| ((i % 4) as u16 + 1, 250 + i * 469, 128.0)).collect(),
            pwav: vec![0x25; 400], pwv2: vec![7; 100], pwv3: vec![0x3f; 1500],
            pwv4: vec![10; 1200 * 6], pwv5: vec![0x12; 3000],
            hot_cues: vec![Cue { start_ms: 250, end_ms: None, hot: 1 }, Cue { start_ms: 8000, end_ms: Some(12000), hot: 2 }],
            memory_cues: vec![Cue { start_ms: 30000, end_ms: None, hot: 0 }],
        }
    }

    /// rekordcrate must parse our DAT and EXT files and write them back byte for byte.
    #[test]
    fn rekordcrate_reads_and_rewrites_our_anlz() {
        let a = sample();
        for bytes in [build_dat(&a), build_ext(&a)] {
            let parsed = ANLZ::read(&mut std::io::Cursor::new(&bytes)).expect("rekordcrate can't parse it");
            let mut again = std::io::Cursor::new(Vec::new());
            parsed.write(&mut again).unwrap();
            assert_eq!(again.into_inner(), bytes, "rekordcrate rewrites it differently");
            for s in &parsed.sections {
                match &s.content {
                    Content::BeatGrid(g) => {
                        assert_eq!(g.beats.len(), 64);
                        assert_eq!((g.beats[1].beat_number, g.beats[1].tempo, g.beats[1].time), (2, 12800, 719));
                    }
                    Content::Path(p) => assert_eq!(p.path.to_string(), a.path),
                    _ => {}
                }
            }
        }
    }
}
