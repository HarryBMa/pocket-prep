//! Mood: one descriptive label per track that joins genre and the kind of track ("Deep",
//! "Peak time", "Mainstream pop", "Hard techno", ...), so the library can be grouped by it.
//!
//! The same audio always gets the same mood. Nothing is random and no model is downloaded:
//!
//! 1. **Measure.** The analysis measures nine numbers per track ([`FEATURES`]): tempo,
//!    loudness, how much of the sound is bass and how much is treble, how much of the track has
//!    drums, melody and vocals playing, how much of it is breakdowns and build-ups, and whether
//!    the key is minor. They are saved with the analysis (`mood_features`).
//! 2. **Compare.** Each mood has a typical value and a spread for every number ([`MOODS`]).
//!    The distance to a mood is how many spreads the track is away from its typical values,
//!    averaged over the numbers (a weighted, scaled Euclidean distance). The nearest mood wins.
//!    The confidence is how clearly it wins (the nearest mood's share of exp(-d²/2τ²) over all
//!    moods, τ = 0.35).
//! 3. **Learn.** When you set a track's mood by hand, that track pulls the mood's typical
//!    values towards itself: typical = (3 · built-in + sum of your tracks) / (3 + your tracks).
//!    Every automatic mood is then worked out again. This is a nearest-centroid classifier,
//!    seeded with built-in centroids, so the result depends only on the audio and your choices.
//!
//! Vocals are only known when the track has stems; without stems that number is skipped.

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::{json, Value};

// Every automatic mood is worked out again at startup, so changing MOODS needs nothing else;
// changing what is measured needs analysis::VERSION bumped.

/// (name, weight, what it means). The order is the order of every feature array.
pub const FEATURES: [(&str, f64, &str); 9] = [
    ("bpm", 2.0, "tempo"),
    ("lufs", 0.75, "loudness (LUFS)"),
    ("bass", 0.75, "share of the sound below 200 Hz"),
    ("bright", 0.5, "share of the sound above 2.5 kHz"),
    ("drums", 1.0, "share of the track with drums playing"),
    ("melody", 0.5, "share of the track with melody playing"),
    ("vocals", 1.5, "share of the track with vocals (needs stems)"),
    ("breaks", 0.5, "share of the track in breakdowns and build-ups"),
    ("minor", 0.5, "1 = minor key, 0 = major"),
];
pub const N: usize = FEATURES.len();
pub type Features = [Option<f64>; N];

pub struct Mood {
    pub id: &'static str,
    pub label: &'static str,
    pub about: &'static str,
    /// (typical value, spread) per feature, in FEATURES order.
    pub proto: [(f64, f64); N],
}

/// Built-in moods, in the order the library groups them (calm to hard, then the rest).
/// Deep, melodic, tech house and peak time are fitted on a DJ library (docs/MOOD.md); the other four are
/// still rules of thumb, because no labelled set could confirm them yet.
pub const MOODS: &[Mood] = &[
    Mood { id: "deep", label: "Deep", about: "Deep house and deep tech: warm, rolling, low-key, few vocals",
           proto: [(124.0, 2.0), (-8.871, 1.0), (0.447, 0.03), (0.018, 0.03), (0.924, 0.048), (0.762, 0.11), (0.15, 0.25), (0.109, 0.095), (1.0, 0.741)] },
    Mood { id: "melodic", label: "Melodic", about: "Melodic techno and progressive: big melodies, long breakdowns, minor keys",
           proto: [(126.0, 3.7), (-9.386, 1.0), (0.494, 0.046), (0.02, 0.03), (0.887, 0.119), (0.757, 0.158), (0.2, 0.3), (0.114, 0.071), (1.0, 0.741)] },
    Mood { id: "tech_house", label: "Tech house", about: "Groovy, percussive, bass-led, vocal snippets",
           proto: [(126.0, 2.22), (-8.646, 1.0), (0.438, 0.055), (0.025, 0.03), (0.904, 0.06), (0.738, 0.155), (0.3, 0.3), (0.139, 0.111), (1.0, 0.741)] },
    Mood { id: "peaktime", label: "Peak time", about: "Driving peak-time techno and house: loud, dense, full on",
           proto: [(126.0, 2.0), (-7.768, 1.04), (0.472, 0.033), (0.02, 0.03), (0.815, 0.145), (0.637, 0.121), (0.15, 0.25), (0.181, 0.039), (1.0, 0.741)] },
    Mood { id: "hard_techno", label: "Hard techno", about: "Hard, industrial and schranz: fast, loud, distorted kicks",
           proto: [(145.0, 8.0), (-6.5, 3.0), (0.5, 0.12), (0.03, 0.02), (0.92, 0.12), (0.65, 0.15), (0.1, 0.2), (0.06, 0.08), (0.8, 0.7)] },
    Mood { id: "dnb", label: "Drum & bass", about: "Drum & bass, jungle and breaks at 160-180 BPM",
           proto: [(172.0, 6.0), (-8.0, 3.0), (0.35, 0.12), (0.04, 0.02), (0.85, 0.12), (0.7, 0.15), (0.2, 0.3), (0.1, 0.08), (0.7, 0.7)] },
    Mood { id: "pop", label: "Mainstream pop", about: "Song-shaped pop and dance-pop: vocals most of the way, bright, often major",
           proto: [(120.0, 7.0), (-7.5, 2.5), (0.3, 0.12), (0.04, 0.02), (0.8, 0.12), (0.7, 0.15), (0.6, 0.25), (0.12, 0.08), (0.3, 0.7)] },
    Mood { id: "downtempo", label: "Downtempo", about: "Chill-out, downtempo and ambient: slow, soft, sparse drums",
           proto: [(98.0, 12.0), (-13.0, 3.0), (0.35, 0.12), (0.02, 0.02), (0.65, 0.15), (0.65, 0.15), (0.2, 0.3), (0.08, 0.08), (0.6, 0.7)] },
];

/// How sharply the confidence falls off with distance (see `rank`).
const TAU: f64 = 0.35;
/// Weight of the built-in typical values against your own hand-picked tracks.
const PRIOR: f64 = 3.0;

pub fn index(id: &str) -> Option<usize> {
    MOODS.iter().position(|m| m.id == id)
}

/// The mood features from an analysis (None when it predates them).
pub fn features_of(a: &Value) -> Option<Features> {
    let f = a.get("mood_features")?.as_object()?;
    Some(std::array::from_fn(|i| f.get(FEATURES[i].0).and_then(Value::as_f64)))
}

pub fn features_json(f: &Features) -> Value {
    Value::Object(FEATURES.iter().zip(f).map(|((k, _, _), v)| (k.to_string(), json!(v))).collect())
}

/// Built-in typical values, moved towards the tracks you gave each mood by hand.
pub fn prototypes(manual: &[(usize, Features)]) -> Vec<[(f64, f64); N]> {
    MOODS.iter().enumerate().map(|(m, mood)| {
        std::array::from_fn(|j| {
            let (c, s) = mood.proto[j];
            let xs: Vec<f64> = manual.iter().filter(|(k, _)| *k == m).filter_map(|(_, f)| f[j]).collect();
            ((PRIOR * c + xs.iter().sum::<f64>()) / (PRIOR + xs.len() as f64), s)
        })
    }).collect()
}

/// Distance (in spreads) from a track to each mood, nearest first, with confidences.
pub fn rank(f: &Features, protos: &[[(f64, f64); N]]) -> Vec<(usize, f64, f64)> {
    let d: Vec<f64> = protos.iter().map(|p| {
        let (mut sum, mut w) = (0.0, 0.0);
        for (j, x) in f.iter().enumerate() {
            if let Some(x) = x {
                let z = (x - p[j].0) / p[j].1;
                sum += FEATURES[j].1 * z * z;
                w += FEATURES[j].1;
            }
        }
        if w > 0.0 { (sum / w).sqrt() } else { f64::INFINITY }
    }).collect();
    // exp(-d²/2τ²) with τ = 0.35 spreads: a mood half a spread nearer than the next reads as
    // clearly ahead (a plain τ = 1 makes eight moods look almost equally likely).
    let like: Vec<f64> = d.iter().map(|d| (-d * d / (2.0 * TAU * TAU)).exp()).collect();
    let total: f64 = like.iter().sum::<f64>().max(1e-300);
    let mut out: Vec<(usize, f64, f64)> = (0..d.len()).map(|m| (m, d[m], like[m] / total)).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

/// Work out every automatic mood again from the saved features and the hand-picked moods.
pub fn refresh(c: &Connection) -> Result<()> {
    let mut st = c.prepare("SELECT id, mood, mood_manual, mood_features FROM tracks WHERE deleted=0 AND mood_features IS NOT NULL")?;
    let rows: Vec<(i64, Option<String>, i64, String)> =
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, Option<i64>>(2)?.unwrap_or(0), r.get(3)?)))?.flatten().collect();
    let parsed: Vec<(i64, Option<String>, bool, Features)> = rows.into_iter().filter_map(|(id, mood, manual, f)| {
        let v: Value = serde_json::from_str(&f).ok()?;
        Some((id, mood, manual == 1, features_of(&json!({"mood_features": v}))?))
    }).collect();
    let manual: Vec<(usize, Features)> = parsed.iter()
        .filter_map(|(_, mood, man, f)| if *man { Some((index(mood.as_deref()?)?, *f)) } else { None }).collect();
    let protos = prototypes(&manual);
    for (id, _, man, f) in &parsed {
        let r = rank(f, &protos);
        if *man {
            continue;
        }
        let (m, _, conf) = r[0];
        c.execute("UPDATE tracks SET mood=?, mood_conf=? WHERE id=?", params![MOODS[m].id, (conf * 100.0).round() / 100.0, id])?;
    }
    Ok(())
}

/// Why a track got its mood: the top three moods and the measured numbers.
pub fn explain(c: &Connection, track_id: i64) -> Result<Value> {
    let f: Option<String> = c.query_row("SELECT mood_features FROM tracks WHERE id=?", [track_id], |r| r.get(0))?;
    let Some(f) = f.and_then(|s| serde_json::from_str::<Value>(&s).ok()) else { return Ok(Value::Null) };
    let feats = features_of(&json!({"mood_features": f.clone()})).unwrap_or([None; N]);
    let mut st = c.prepare("SELECT mood, mood_features FROM tracks WHERE deleted=0 AND mood_manual=1 AND mood_features IS NOT NULL AND id<>?")?;
    let manual: Vec<(usize, Features)> = st.query_map([track_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?.flatten()
        .filter_map(|(m, f)| Some((index(&m)?, features_of(&json!({"mood_features": serde_json::from_str::<Value>(&f).ok()?}))?)))
        .collect();
    let ranking = rank(&feats, &prototypes(&manual));
    Ok(json!({
        "features": f,
        "ranking": ranking.iter().take(3).map(|&(m, d, p)| json!({"id": MOODS[m].id, "distance": (d * 100.0).round() / 100.0,
                                                                     "confidence": (p * 100.0).round() / 100.0})).collect::<Vec<_>>(),
    }))
}

pub fn list() -> Value {
    json!({
        "moods": MOODS.iter().map(|m| json!({"id": m.id, "label": m.label, "about": m.about})).collect::<Vec<_>>(),
        "features": FEATURES.iter().map(|(k, w, about)| json!({"id": k, "weight": w, "about": about})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(v: [f64; N]) -> Features {
        v.map(Some)
    }

    /// A track at a mood's typical values gets that mood, and the same input always ranks the same.
    #[test]
    fn typical_track_gets_its_mood() {
        let protos = prototypes(&[]);
        for (m, mood) in MOODS.iter().enumerate() {
            let f = feats(mood.proto.map(|(c, _)| c));
            let r = rank(&f, &protos);
            assert_eq!(MOODS[r[0].0].id, mood.id);
            assert_eq!(r[0].0, m);
            assert_eq!(r, rank(&f, &protos));
        }
    }

    /// Hand-picked tracks pull a mood's typical values towards them.
    #[test]
    fn manual_moods_move_the_centre() {
        let mut f = feats(MOODS[0].proto.map(|(c, _)| c));
        f[0] = Some(127.0);
        let p = prototypes(&[(0, f), (0, f), (0, f)]);
        assert!((p[0][0].0 - (3.0 * MOODS[0].proto[0].0 + 3.0 * 127.0) / 6.0).abs() < 1e-9);
        assert_eq!(p[1][0].0, MOODS[1].proto[0].0);
    }

    /// Without stems the vocals number is skipped, not read as "no vocals".
    #[test]
    fn missing_features_are_skipped() {
        let mut f = feats(MOODS[6].proto.map(|(c, _)| c));
        f[6] = None;
        assert!(rank(&f, &prototypes(&[]))[0].1.is_finite());
    }
}
