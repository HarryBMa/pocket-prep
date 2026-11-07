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

// ── Needle Tags: the embedding head ──────────────────────────────────────────

/// Length of the embedding (clubcraft_stems::embed::EMBEDDING_LEN).
pub const EMB_LEN: usize = 2048;

/// The trained head (tools/needle/mood_head_v1.json): standardise the embedding, one weight per number and mood,
/// softmax. Its moods are `MOODS` ids.
struct Head {
    classes: Vec<usize>, // MOODS index of each output
    mean: Vec<f32>,
    std: Vec<f32>,
    w: Vec<f32>, // [EMB_LEN][classes]
    b: Vec<f32>,
}

fn head() -> &'static Head {
    static HEAD: std::sync::OnceLock<Head> = std::sync::OnceLock::new();
    HEAD.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../tools/needle/mood_head_v1.json")).expect("mood head");
        let floats = |k: &str| -> Vec<f32> { v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let classes: Vec<usize> = v["moods"].as_array().unwrap().iter().map(|m| index(m.as_str().unwrap()).expect("mood id")).collect();
        let w: Vec<f32> = v["W"].as_array().unwrap().iter().flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32)).collect();
        assert_eq!(w.len(), EMB_LEN * classes.len());
        Head { classes, mean: floats("mean"), std: floats("std"), w, b: floats("b") }
    })
}

fn standardised(emb: &[f32]) -> Vec<f32> {
    let h = head();
    emb.iter().zip(h.mean.iter().zip(&h.std)).map(|(x, (m, s))| (x - m) / s).collect()
}

/// Softmax over the head's moods for one standardised embedding, with weights `w` and biases `b`.
fn probabilities(z: &[f32], w: &[f32], b: &[f32]) -> Vec<f64> {
    let k = b.len();
    let mut logit: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    for (d, &x) in z.iter().enumerate() {
        for c in 0..k {
            logit[c] += (x * w[d * k + c]) as f64;
        }
    }
    let top = logit.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = logit.iter().map(|l| (l - top).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// The head moved towards the tracks you gave a mood by hand (the embedding version of "learn"): a few gradient steps
/// of the cross-entropy over those tracks, with a pull back to the trained weights so one pick can't swing a mood.
fn adapted(manual: &[(usize, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
    let h = head();
    let (mut w, mut b) = (h.w.clone(), h.b.clone());
    let k = h.classes.len();
    let samples: Vec<(usize, Vec<f32>)> = manual.iter()
        .filter_map(|(m, e)| Some((h.classes.iter().position(|c| c == m)?, standardised(e)))).collect();
    if samples.is_empty() {
        return (w, b);
    }
    let step = 0.5 / EMB_LEN as f32; // the standardised numbers have unit size, so |z|² is about EMB_LEN
    for _ in 0..30 {
        let mut gw = vec![0f32; w.len()];
        let mut gb = vec![0f32; k];
        for (y, z) in &samples {
            let p = probabilities(z, &w, &b);
            for c in 0..k {
                let g = (p[c] - if c == *y { 1.0 } else { 0.0 }) as f32;
                gb[c] += g;
                for (d, &x) in z.iter().enumerate() {
                    gw[d * k + c] += g * x;
                }
            }
        }
        for i in 0..w.len() {
            w[i] -= step * gw[i] + 0.02 * (w[i] - h.w[i]);
        }
        for c in 0..k {
            b[c] -= 0.05 * gb[c] + 0.02 * (b[c] - h.b[c]);
        }
    }
    (w, b)
}

/// Moods for an embedding, most likely first: (index into `MOODS`, probability). `manual` are the hand-picked
/// tracks (mood index, embedding) the head is adapted to.
pub fn rank_embedding(emb: &[f32], manual: &[(usize, Vec<f32>)]) -> Vec<(usize, f64)> {
    let (w, b) = adapted(manual);
    let p = probabilities(&standardised(emb), &w, &b);
    let mut out: Vec<(usize, f64)> = head().classes.iter().cloned().zip(p).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

pub fn embedding_to_blob(e: &[f32]) -> Vec<u8> {
    e.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn blob_to_embedding(b: &[u8]) -> Option<Vec<f32>> {
    (b.len() == EMB_LEN * 4).then(|| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn features_from(json_text: &Option<String>) -> Option<Features> {
    let v: Value = serde_json::from_str(json_text.as_ref()?).ok()?;
    features_of(&json!({"mood_features": v}))
}

/// Work out every automatic mood again from the saved numbers and the hand-picked moods. A track with an embedding
/// (the Needle Tags model, see embed.rs) gets its mood from the head; a track without one from the nine numbers.
pub fn refresh(c: &Connection) -> Result<()> {
    type Row = (i64, Option<String>, i64, Option<String>, Option<Vec<u8>>);
    let mut st = c.prepare("SELECT id, mood, mood_manual, mood_features, mood_emb FROM tracks WHERE deleted=0 AND (mood_features IS NOT NULL OR mood_emb IS NOT NULL)")?;
    let rows: Vec<Row> = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, Option<i64>>(2)?.unwrap_or(0), r.get(3)?, r.get(4)?)))?.flatten().collect();
    // hand-picked moods: for the nine-number prototypes and for the head
    let mut by_feats: Vec<(usize, Features)> = vec![];
    let mut by_emb: Vec<(usize, Vec<f32>)> = vec![];
    for (_, mood, man, f, e) in &rows {
        let Some(m) = (*man == 1).then(|| mood.as_deref().and_then(index)).flatten() else { continue };
        if let Some(f) = features_from(f) {
            by_feats.push((m, f));
        }
        if let Some(e) = e.as_deref().and_then(blob_to_embedding) {
            by_emb.push((m, e));
        }
    }
    let protos = prototypes(&by_feats);
    let (w, b) = adapted(&by_emb);
    for (id, _, man, f, e) in rows {
        if man == 1 {
            continue;
        }
        let (m, conf) = if let Some(e) = e.as_deref().and_then(blob_to_embedding) {
            let p = probabilities(&standardised(&e), &w, &b);
            let best = (0..p.len()).max_by(|&x, &y| p[x].total_cmp(&p[y]).then(y.cmp(&x))).unwrap();
            (head().classes[best], p[best])
        } else if let Some(f) = features_from(&f) {
            let r = rank(&f, &protos);
            (r[0].0, r[0].2)
        } else {
            continue;
        };
        c.execute("UPDATE tracks SET mood=?, mood_conf=? WHERE id=?", params![MOODS[m].id, (conf * 100.0).round() / 100.0, id])?;
    }
    Ok(())
}

/// Why a track got its mood: the top three moods and the measured numbers.
pub fn explain(c: &Connection, track_id: i64) -> Result<Value> {
    let (f, e): (Option<String>, Option<Vec<u8>>) =
        c.query_row("SELECT mood_features, mood_emb FROM tracks WHERE id=?", [track_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let features = f.as_ref().and_then(|s| serde_json::from_str::<Value>(s).ok());
    if let Some(e) = e.as_deref().and_then(blob_to_embedding) {
        let mut st = c.prepare("SELECT mood, mood_emb FROM tracks WHERE deleted=0 AND mood_manual=1 AND mood_emb IS NOT NULL AND id<>?")?;
        let manual: Vec<(usize, Vec<f32>)> = st.query_map([track_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?.flatten()
            .filter_map(|(m, b)| Some((index(&m)?, blob_to_embedding(&b)?))).collect();
        let ranking = rank_embedding(&e, &manual);
        return Ok(json!({
            "source": "needle",
            "features": features,
            "ranking": ranking.iter().take(3).map(|&(m, p)| json!({"id": MOODS[m].id, "distance": Value::Null,
                                                                     "confidence": (p * 100.0).round() / 100.0})).collect::<Vec<_>>(),
        }));
    }
    let Some(feats) = features_from(&f) else { return Ok(Value::Null) };
    let mut st = c.prepare("SELECT mood, mood_features FROM tracks WHERE deleted=0 AND mood_manual=1 AND mood_features IS NOT NULL AND id<>?")?;
    let manual: Vec<(usize, Features)> = st.query_map([track_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?.flatten()
        .filter_map(|(m, f)| Some((index(&m)?, features_from(&Some(f))?))).collect();
    let ranking = rank(&feats, &prototypes(&manual));
    Ok(json!({
        "source": "numbers",
        "features": features,
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

    /// An embedding in the direction one mood's weights point gets that mood, from the trained head.
    fn embedding_strength(class: usize, strength: f32, seed: u32) -> Vec<f32> {
        let h = head();
        let k = h.classes.len();
        let col: Vec<f32> = (0..EMB_LEN).map(|d| h.w[d * k + class]).collect();
        let norm = col.iter().map(|v| v * v).sum::<f32>().sqrt();
        // a real track is mostly "everything else": unit-size noise in every number, the class direction on top
        let mut x = seed;
        let mut noise = || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 8) as f32 / (1u32 << 24) as f32 * 3.4641 - 1.7321 // uniform, unit variance
        };
        (0..EMB_LEN).map(|d| h.mean[d] + h.std[d] * (noise() + strength * col[d] / norm)).collect()
    }

    fn embedding_for(class: usize) -> Vec<f32> {
        embedding_strength(class, 40.0, 7)
    }

    /// The weakest embedding along a class direction that the head still calls that class with at least `p`.
    fn embedding_with_confidence(class: usize, p: f64, seed: u32) -> Vec<f32> {
        let mut s = 0.5f32;
        loop {
            let e = embedding_strength(class, s, seed);
            let r = rank_embedding(&e, &[]);
            if r[0].0 == head().classes[class] && r[0].1 >= p {
                return e;
            }
            s *= 1.1;
            assert!(s < 1000.0);
        }
    }

    #[test]
    fn the_head_names_the_mood_it_points_at() {
        let h = head();
        assert_eq!(h.classes.len(), MOODS.len());
        for (c, &m) in h.classes.iter().enumerate() {
            let r = rank_embedding(&embedding_for(c), &[]);
            assert_eq!(MOODS[r[0].0].id, MOODS[m].id);
            assert!((r.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-9);
            assert_eq!(r, rank_embedding(&embedding_for(c), &[]), "same input, same answer");
        }
    }

    /// Hand-picked tracks move the head: three tracks given as "peak time" that look just like a deep-looking track make
    /// it peak time too, a single pick doesn't change a similar but different track, and a track of another kind stays.
    /// (On real embeddings, correcting a few mistakes fixes some of the others and keeps 97 % of the rest right.)
    #[test]
    fn hand_picked_moods_move_the_head() {
        let (deep, peak, down) = (index("deep").unwrap(), index("peaktime").unwrap(), index("downtempo").unwrap());
        let class = |m: usize| head().classes.iter().position(|&c| c == m).unwrap();
        let like_deep = embedding_with_confidence(class(deep), 0.8, 999);
        assert_eq!(rank_embedding(&like_deep, &[])[0].0, deep);
        let picks: Vec<(usize, Vec<f32>)> = (0..3).map(|i| (peak, like_deep.iter().map(|v| v + 0.001 * i as f32).collect())).collect();
        assert_eq!(rank_embedding(&like_deep, &picks)[0].0, peak, "three picks move it");
        let cousin = embedding_with_confidence(class(deep), 0.8, 5);
        assert_eq!(rank_embedding(&cousin, &picks[..1])[0].0, deep, "one pick doesn't swing a similar track");
        let far = embedding_with_confidence(class(down), 0.8, 17);
        assert_eq!(rank_embedding(&far, &picks)[0].0, down, "a track of another kind stays");
    }

    #[test]
    fn embeddings_survive_the_database() {
        let e: Vec<f32> = (0..EMB_LEN).map(|i| i as f32 * 0.25 - 3.0).collect();
        assert_eq!(blob_to_embedding(&embedding_to_blob(&e)).unwrap(), e);
        assert!(blob_to_embedding(&[1, 2, 3]).is_none());
    }
}
