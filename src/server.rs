//! The local web server behind the app window (axum). Same /api/* routes as the
//! old Python app, so the UI in ui/ is unchanged.
//!
//! Only listens on 127.0.0.1, and rejects requests whose Host header isn't the
//! local address (stops websites using DNS rebinding to reach it).

use axum::body::Body;
use axum::extract::{Multipart, Path as P, Query, Request};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::OnceLock;
use tauri_plugin_dialog::DialogExt;
use tower::ServiceExt;

use crate::db::{self, Row};
use crate::export::usb;
use crate::{analysis, config, convert, deck, jobs, library, models, mood, player, roformer, sets, stems, transitions, vocals};

pub static APP: OnceLock<tauri::AppHandle> = OnceLock::new();

#[derive(rust_embed::Embed)]
#[folder = "ui/"]
struct Ui;

pub const TRACK_COLUMNS: &str = "id, path, title, artist, album, genre, year, duration, bpm, musical_key, camelot, energy, \
    rating, color, comment, stems, analysed_at, added_at, artwork IS NOT NULL AS has_artwork, grid_offset_ms, downbeat_shift, key_shift, \
    mood, mood_manual, mood_conf, edited, lufs, lead_s, tail_s, mix_in_ms, mix_out_ms, intent, grid_ok_at, \
    stems_range, stem_quality, clip_pct, first_vocal_ms, bitrate, sample_rate, beats_per_bar, practice, (SELECT COUNT(*) FROM cues WHERE cues.track_id = tracks.id AND slot > 0) AS hot_cues";

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors go back as {"detail": "..."}, which the UI shows in a toast.
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{:#}", e.into()))
    }
}

fn not_found(msg: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, msg.into())
}

type R = Result<Json<Value>, ApiError>;

/// Database and file work is blocking; keep it off the async threads.
async fn blocking(f: impl FnOnce() -> Result<Value, ApiError> + Send + 'static) -> R {
    Ok(Json(tokio::task::spawn_blocking(f).await.map_err(anyhow::Error::from)??))
}

fn ids(v: &Value, key: &str) -> Vec<i64> {
    v[key].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default()
}

fn track_or_404(tid: i64) -> Result<Row, ApiError> {
    let mut t = db::one(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE id=?"), &[json!(tid)])?.ok_or_else(|| not_found("No such track"))?;
    analysis::add_shifted_key(&mut t);
    Ok(t)
}

// ── Jobs ─────────────────────────────────────────────────────────────────────

fn title_of(tid: i64) -> String {
    db::track(tid).ok().flatten().and_then(|t| t["title"].as_str().map(str::to_string)).unwrap_or_else(|| tid.to_string())
}

fn queue_analysis(track_ids: &[i64]) -> Vec<i64> {
    track_ids.iter().filter(|t| !jobs::pending_for(**t, "analyse")).map(|&tid| {
        jobs::submit("analyse", &format!("Analyse: {}", title_of(tid)), "cpu", Some(tid), move |h| {
            analysis::analyse(tid, &|f, m| h.progress(f, m))?;
            Ok(None)
        })
    }).collect()
}

/// Startup housekeeping as a visible job: convert 0.2-era WAV stems to FLAC, then
/// re-analyse outdated tracks (after, so the analysis can use the converted stems).
pub fn startup_jobs() {
    jobs::submit("stems", "Convert stems to FLAC", "io", None, |h| {
        // Moods follow the current MOODS table and your hand-picked moods.
        db::tx(mood::refresh)?;
        h.progress(0.1, "Converting");
        let n = stems::migrate_wav_stems()?;
        queue_outdated_analysis();
        Ok(Some(json!({"converted": n})))
    });
    // Loudness and near-silent start/end for tracks analysed before those were measured.
    jobs::submit("levels", "Loudness and silence check", "cpu", None, |h| Ok(Some(analysis::backfill_levels(h)?)));
    // Mood embeddings for tracks analysed before the model was installed (a no-op when there is nothing to do).
    jobs::submit("mood", "Mood model", "cpu", None, |h| Ok(Some(crate::embed::backfill(h)?)));
}

/// Re-analyse tracks whose analysis predates the current analysis::VERSION.
fn queue_outdated_analysis() {
    let ids: Vec<i64> = db::query("SELECT id FROM tracks WHERE deleted=0 AND analysed_at IS NOT NULL", &[]).unwrap_or_default()
        .iter().filter_map(|r| r["id"].as_i64())
        .filter(|&id| analysis::load_analysis(id).and_then(|a| a["version"].as_i64()).unwrap_or(0) < analysis::VERSION)
        .collect();
    queue_analysis(&ids);
}

// Stems in the order the intent asks for: a section first (small), then vocal layers, remixes, and
// plain mixes last; a backup track only when asked for by hand (it is in the list, so it goes last).
const INTENT_ORDER: &[&str] = &["section", "vocal", "remix", "mix", "backup"];
fn intent_rank(tid: i64) -> usize {
    let intent = db::track(tid).ok().flatten().and_then(|t| t.get("intent").and_then(|v| v.as_str()).map(str::to_string)).unwrap_or("mix".into());
    INTENT_ORDER.iter().position(|i| *i == intent).unwrap_or(3)
}
fn queue_stems(track_ids: &[i64]) -> anyhow::Result<Vec<i64>> {
    queue_stems_range(track_ids, None, false)
}
/// `range_ms`: separate only that part of each track (a section), with context; the file stays whole.
/// `force` separates again even when stems exist (a poor grade; a new run can come out differently).
fn queue_stems_range(track_ids: &[i64], range_ms: Option<(f64, f64)>, force: bool) -> anyhow::Result<Vec<i64>> {
    if !models::stems_ready() {
        anyhow::bail!("The stems model isn't downloaded yet (Settings → System)");
    }
    if let Some((a, b)) = range_ms { if !(b > a && a >= 0.0) { anyhow::bail!("The range is empty"); } }
    let mut ordered: Vec<i64> = track_ids.to_vec();
    ordered.sort_by_key(|&t| intent_rank(t));
    let mut out = vec![];
    for &tid in &ordered {
        if jobs::pending_for(tid, "stems") {
            continue;
        }
        // Already made: the whole track, or exactly this range. Nothing is separated twice.
        if let Some(t) = db::track(tid)? {
            let have_range: Option<(f64, f64)> = t.get("stems_range").and_then(Value::as_str).and_then(|s| serde_json::from_str::<Vec<f64>>(s).ok())
                .filter(|v| v.len() == 2).map(|v| (v[0], v[1]));
            let done = t.get("stems").and_then(Value::as_str) == Some("done") && stems::has_stems(tid);
            let same = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1.0 && (a.1 - b.1).abs() < 1.0;
            if !force && done && (have_range.is_none() || range_ms.is_some_and(|r| same(r, have_range.unwrap()))) {
                continue;
            }
        }
        db::tx(|c| Ok(c.execute("UPDATE tracks SET stems='queued' WHERE id=?", [tid])?))?;
        let label = match range_ms { Some((a, b)) => format!("Stems {}–{}: {}", clock_ms(a), clock_ms(b), title_of(tid)), None => format!("Stems: {}", title_of(tid)) };
        // A section goes to the front of the gpu lane in effect: it is small, and the intent order already puts it first.
        out.push(jobs::submit("stems", &label, "gpu", Some(tid), move |h| {
            stems::separate(tid, range_ms, &|f, m| h.progress(f, m), &|| h.cancelled())?;
            Ok(None)
        }));
    }
    Ok(out)
}
fn clock_ms(ms: f64) -> String { let s = (ms / 1000.0).round() as i64; format!("{}:{:02}", s / 60, s % 60) }

// ── Tracks ───────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TrackQuery {
    #[serde(default)]
    q: String,
    #[serde(default)]
    playlist: i64,
}

async fn list_tracks(Query(tq): Query<TrackQuery>) -> R {
    blocking(move || {
        let mut tracks = if tq.playlist != 0 {
            let all = db::query(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0"), &[])?;
            library::playlist_track_ids(tq.playlist)?.iter().filter_map(|i| all.iter().find(|r| r["id"] == json!(i)).cloned()).collect()
        } else {
            db::query(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0 ORDER BY added_at DESC, id DESC"), &[])?
        };
        tracks.iter_mut().for_each(analysis::add_shifted_key);
        for t in tracks.iter_mut() {
            let (id, path) = (t["id"].as_i64().unwrap_or(0), PathBuf::from(t["path"].as_str().unwrap_or("")));
            if let Some(m) = convert::state(id, &path).as_object() {
                m.iter().for_each(|(k, v)| { t.insert(k.clone(), v.clone()); });
            }
            t.insert("file_missing".into(), json!(!path.exists()));
        }
        let words: Vec<String> = tq.q.to_lowercase().split_whitespace().map(str::to_string).collect();
        if !words.is_empty() {
            tracks.retain(|t| {
                let text = ["title", "artist", "album", "genre", "camelot", "musical_key", "comment"].iter()
                    .map(|k| t[*k].as_str().unwrap_or("")).collect::<Vec<_>>().join(" ").to_lowercase();
                words.iter().all(|w| text.contains(w))
            });
        }
        Ok(json!(tracks))
    }).await
}

async fn get_track(P(tid): P<i64>) -> R {
    blocking(move || {
        let mut t = track_or_404(tid)?;
        let a = analysis::load_analysis(tid);
        t.insert("cues".into(), json!(library::cues(tid)?));
        t.insert("mood_why".into(), db::tx(|c| mood::explain(c, tid))?);
        t.insert("phrases".into(), a.as_ref().map(|a| a["phrases"].clone()).unwrap_or(Value::Null));
        // Mix points: the DJ's where set, else the analysed ones (mix_auto says which).
        for k in ["mix_in_ms", "mix_out_ms"] {
            let own = t.get(k).and_then(Value::as_f64);
            let auto = a.as_ref().and_then(|a| a[k].as_f64());
            t.insert(k.into(), json!(own.or(auto)));
            t.insert(format!("{k}_auto"), json!(own.is_none()));
            t.insert(format!("{k}_analysed"), json!(auto));
        }
        // Phrase sections (intro, build, drop...) for the phrase strip under the waveform.
        t.insert("sections".into(), a.as_ref().map(|a| a["sections"].clone()).unwrap_or(json!([])));
        // What the stems said about the singing (vocals.rs): first vocal and the sung phrases, or null.
        t.insert("vocals".into(), vocals::load(tid).unwrap_or(Value::Null));
        // The grid as the deck will get it: the analysis grid plus the user's nudge.
        let a = a.map(|a| analysis::user_grid(&a, &t));
        t.insert("grid".into(), a.as_ref().map(|a| json!({
            "bpm": a["bpm"], "first_beat_ms": a["first_beat_ms"], "period_ms": a["period_ms"],
            "downbeat_index": a["downbeat_index"], "beats_per_bar": a["beats_per_bar"], "duration_ms": a["duration_ms"]})).unwrap_or(Value::Null));
        let mut names: Vec<&str> = if stems::stem_file(tid).exists() { stems::STEM_NAMES.to_vec() } else { vec![] };
        names.sort();
        t.insert("stem_files".into(), json!(names));
        let pls: Vec<Value> = db::query("SELECT DISTINCT playlist_id FROM playlist_tracks WHERE track_id=?", &[json!(tid)])?
            .into_iter().map(|r| r["playlist_id"].clone()).collect();
        t.insert("playlists".into(), json!(pls));
        if let Some(m) = convert::state(tid, &PathBuf::from(t["path"].as_str().unwrap_or(""))).as_object() {
            m.iter().for_each(|(k, v)| { t.insert(k.clone(), v.clone()); });
        }
        t.insert("deck_copy_bytes".into(), json!(t["path"].as_str().and_then(|p| convert::existing(tid, std::path::Path::new(p))).and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len())));
        Ok(json!(t))
    }).await
}

async fn get_waveform(P(tid): P<i64>) -> R {
    blocking(move || {
        let a = analysis::load_analysis(tid).ok_or_else(|| not_found("Not analysed yet"))?;
        Ok(json!({"detail": a["waveforms"]["pwv5"], "duration_ms": a["duration_ms"]}))
    }).await
}

/// Serve a file with Range support, so the player can seek.
async fn serve_file(path: PathBuf, req: Request) -> Response {
    if !path.exists() {
        return not_found("File missing").into_response();
    }
    match tower_http::services::ServeFile::new(path).oneshot(req).await {
        Ok(r) => r.into_response(),
        Err(e) => ApiError::from(anyhow::anyhow!(e.to_string())).into_response(),
    }
}

async fn get_audio(P(tid): P<i64>, req: Request) -> Response {
    let path = db::track(tid).ok().flatten().and_then(|t| t["path"].as_str().map(PathBuf::from)).unwrap_or_default();
    serve_file(path, req).await
}

async fn get_artwork(P(tid): P<i64>, req: Request) -> Response {
    let path = db::track(tid).ok().flatten().and_then(|t| t["artwork"].as_str().map(PathBuf::from)).unwrap_or_default();
    serve_file(path, req).await
}

/// Stem waveforms for the track panel: 150 columns per second (like the PWV5 detail waveform), base64.
async fn get_stem_peaks(P(tid): P<i64>) -> R {
    blocking(move || {
        if !stems::has_stems(tid) {
            return Err(not_found("No stems for this track"));
        }
        use base64::Engine;
        let p = stems::peaks(tid, 150.0)?;
        let b64 = |v: &Vec<u8>| base64::engine::general_purpose::STANDARD.encode(v);
        Ok(json!({"cols_per_s": 150, "names": stems::STEM_NAMES,
                  "peaks": stems::STEM_NAMES.iter().zip(&p).map(|(n, v)| (n.to_string(), json!(b64(v)))).collect::<serde_json::Map<_, _>>()}))
    }).await
}

async fn get_stem(P((tid, name)): P<(i64, String)>, req: Request) -> Response {
    match tokio::task::spawn_blocking(move || stems::stem_preview(tid, &name)).await {
        Ok(Ok(path)) => serve_file(path, req).await,
        _ => not_found("No such stem").into_response(),
    }
}

async fn patch_track(P(tid): P<i64>, Json(fields): Json<Row>) -> R {
    blocking(move || {
        track_or_404(tid)?;
        let mut fields = fields;
        analysis::apply_user_edits(tid, &mut fields)?;
        db::update_track(tid, &fields)?;
        if fields.contains_key("mood") || fields.contains_key("mood_manual") {
            db::tx(mood::refresh)?;       // a hand-picked mood also teaches the others
        }
        let last = db::one("SELECT id FROM history ORDER BY id DESC LIMIT 1", &[])?.map(|r| r["id"].clone());
        Ok(json!({"track": track_or_404(tid)?, "undo": last}))
    }).await
}

async fn put_cues(P(tid): P<i64>, Json(cues): Json<Vec<Value>>) -> R {
    blocking(move || {
        track_or_404(tid)?;
        let hid = library::set_cues(tid, &cues)?;
        Ok(json!({"cues": library::cues(tid)?, "undo": hid}))
    }).await
}

async fn import_files(Json(b): Json<Value>) -> R {
    blocking(move || {
        let paths: Vec<String> = b["paths"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let new = library::import_paths(&paths)?;
        if !new.is_empty() && config::load_settings()["auto_analyse"].as_bool().unwrap_or(true) {
            queue_analysis(&new);
        }
        Ok(json!({"imported": new.len(), "ids": new}))
    }).await
}

async fn remove_tracks(Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!({"undo": library::remove_tracks(&ids(&b, "ids"))?}))).await
}

/// "Convert for the deck" (src/convert.rs): 16-bit FLAC copies the deck can play. With "remove": true it deletes the
/// copies, so the original is exported again.
async fn convert_tracks(Json(b): Json<Value>) -> R {
    blocking(move || {
        let ids = ids(&b, "ids");
        if b["remove"].as_bool().unwrap_or(false) {
            ids.iter().for_each(|&id| convert::remove(id));
            return Ok(json!({"removed": ids.len()}));
        }
        let jobs: Vec<i64> = ids.iter().filter(|t| !jobs::pending_for(**t, "convert")).map(|&tid| {
            jobs::submit("convert", &format!("Convert for the deck: {}", title_of(tid)), "cpu", Some(tid), move |h| {
                let out = convert::convert(tid, &|f, m| h.progress(f, m), &|| h.cancelled())?;
                Ok(Some(json!({"bytes": std::fs::metadata(out)?.len()})))
            })
        }).collect();
        Ok(json!({"jobs": jobs}))
    }).await
}

async fn analyse_tracks(Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!({"jobs": queue_analysis(&ids(&b, "ids"))}))).await
}

async fn make_stems(Json(b): Json<Value>) -> R {
    blocking(move || {
        let range = b["range_ms"].as_array().and_then(|r| Some((r.first()?.as_f64()?, r.get(1)?.as_f64()?)));
        Ok(json!({"jobs": queue_stems_range(&ids(&b, "ids"), range, b.get("force").and_then(Value::as_bool).unwrap_or(false))?}))
    }).await
}

/// What follows `from` well: every analysed track (or `to`) scored as the next track, best first,
/// with the reasons (transitions.rs). Keyed by track id for the list's "mixes with" filter.
async fn transitions_from(Json(b): Json<Value>) -> R {
    blocking(move || {
        let from_id = b["from"].as_i64().ok_or_else(|| anyhow::anyhow!("from is required"))?;
        let only = ids(&b, "to");
        let rows = db::query(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0"), &[])?;
        let from = rows.iter().find(|r| r["id"] == json!(from_id)).and_then(transitions::side).ok_or_else(|| anyhow::anyhow!("That track isn't analysed yet"))?;
        let mut results: Vec<Value> = rows.iter().filter(|r| r["id"] != json!(from_id) && (only.is_empty() || r["id"].as_i64().is_some_and(|i| only.contains(&i))))
            .filter_map(transitions::side).map(|to| transitions::score(&from, &to)).collect();
        results.sort_by(|a, b| b["score"].as_f64().unwrap_or(0.0).total_cmp(&a["score"].as_f64().unwrap_or(0.0)));
        Ok(json!({"from": from_id, "results": results}))
    }).await
}

// ── The practice player (player.rs) ──────────────────────────────────────────

/// The state of the player: what is loaded, where it is, the tempo, the mixer. Cheap: read from a snapshot.
async fn player_state() -> R {
    Ok(Json(json!(player::player().snapshot())))
}

/// Try to open the output device (the page asks once at start-up): `available` says whether the engine plays.
async fn player_open() -> R {
    blocking(|| { let _ = player::player().rate(); Ok(json!(player::player().snapshot())) }).await
}

async fn player_cmd(c: player::Command) -> R {
    blocking(move || { player::player().command(c)?; Ok(json!(player::player().snapshot())) }).await
}

/// Load a track into the player: its original mix, or its four stems with `stems: true` (when it has them), from
/// `position_ms`, playing when `play` is set. Decoding takes a second or two; the audio thread is not blocked.
async fn player_load(Json(b): Json<Value>) -> R {
    blocking(move || {
        let id = b["track_id"].as_i64().ok_or_else(|| anyhow::anyhow!("track_id is required"))?;
        let want_stems = b["stems"].as_bool().unwrap_or(false);
        let position = b["position_ms"].as_f64().unwrap_or(0.0);
        let p = player::player();
        let rate = p.rate()?;
        let (song, stems_loaded) = player::build_song(id, rate, want_stems)?;
        let beats = player::beats_for(id, song.duration_secs(), rate);
        let same = p.snapshot().track_id == Some(id);
        p.load(id, song, beats, position, stems_loaded)?;
        if !same { p.command(player::Command::Loop(None))?; }
        if b["play"].as_bool().unwrap_or(false) { p.command(player::Command::Play)?; }
        Ok(json!(p.snapshot()))
    }).await
}

async fn player_seek(Json(b): Json<Value>) -> R {
    player_cmd(player::Command::SeekMs(b["ms"].as_f64().unwrap_or(0.0))).await
}

/// Tempo (0.25-2), semitones (-24..24), master gain, loop ({start_ms, end_ms} or null), click, vocal shift.
async fn player_set(Json(b): Json<Value>) -> R {
    blocking(move || {
        let p = player::player();
        if let Some(v) = b["tempo"].as_f64() { p.command(player::Command::Tempo(v as f32))?; }
        if let Some(v) = b["semitones"].as_f64() { p.command(player::Command::Semitones(v as f32))?; }
        if let Some(v) = b["gain"].as_f64() { p.command(player::Command::Gain(v as f32))?; }
        if let Some(l) = b.get("loop") {
            let range = match (l["start_ms"].as_f64(), l["end_ms"].as_f64()) { (Some(a), Some(z)) if z > a => Some([a, z]), _ => None };
            p.command(player::Command::Loop(range))?;
        }
        if let Some(c) = b.get("click") {
            p.command(player::Command::Click { on: c["on"].as_bool(), count_in: c["count_in"].as_bool(), gain: c["gain"].as_f64().map(|g| g as f32),
                                               sound: c["sound"].as_str().map(str::to_string), channels: c["channels"].as_str().map(str::to_string) })?;
        }
        if let Some(v) = b.get("vocal") {
            p.command(player::Command::VocalShift { formants: v["formants"].as_bool().unwrap_or(false), tonality_limit: v["tonality_limit"].as_bool().unwrap_or(false) })?;
        }
        Ok(json!(p.snapshot()))
    }).await
}

/// One stem of the mixer: gain (0-4), pan (-1..1), mute, solo, pitch_lock, offset_ms.
async fn player_stem(Json(b): Json<Value>) -> R {
    let name = b["name"].as_str().unwrap_or("").to_string();
    player_cmd(player::Command::Stem { name, gain: b["gain"].as_f64().map(|v| v as f32), pan: b["pan"].as_f64().map(|v| v as f32), mute: b["mute"].as_bool(),
                                      solo: b["solo"].as_bool(), pitch_lock: b["pitch_lock"].as_bool(), offset_ms: b["offset_ms"].as_f64() }).await
}

async fn player_devices() -> R {
    blocking(|| Ok(json!({"devices": player::player().devices(), "current": player::player().snapshot().device}))).await
}

/// Choose the output device (`id` null = the system default). The loaded track is decoded again at the new device's
/// rate and put back where it was.
async fn player_device(Json(b): Json<Value>) -> R {
    blocking(move || {
        let p = player::player();
        let before = p.snapshot();
        let was = p.set_device(b["id"].as_str().map(str::to_string))?;
        if let Some(id) = was {
            let rate = p.rate()?;
            let (song, stems_loaded) = player::build_song(id, rate, before.stems_loaded)?;
            let beats = player::beats_for(id, song.duration_secs(), rate);
            p.load(id, song, beats, before.position_ms, stems_loaded)?;
        }
        Ok(json!(p.snapshot()))
    }).await
}

/// A playlist as a set: the order, the transition between each pair, energy curve, time and load (sets.rs).
async fn set_report(P(pid): P<i64>) -> R {
    blocking(move || Ok(sets::report(pid)?)).await
}

/// Reorder a playlist with the planner: `by` = auto | key | tempo | energy. Undoable.
async fn set_order(P(pid): P<i64>, Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!({"undo": sets::order(pid, b["by"].as_str().unwrap_or("auto"))?}))).await
}

/// Prepare everything the intents ask for, for the given tracks (or the whole library): analysis where
/// missing, stems where the intent wants them, a deck copy where the file won't play. Nothing is redone.
async fn prepare(Json(b): Json<Value>) -> R {
    blocking(move || {
        let wanted = ids(&b, "ids");
        let rows = db::query(&format!("SELECT {TRACK_COLUMNS} FROM tracks WHERE deleted=0"), &[])?;
        let rows: Vec<&Row> = rows.iter().filter(|r| wanted.is_empty() || r["id"].as_i64().is_some_and(|i| wanted.contains(&i))).collect();
        let id = |r: &Row| r["id"].as_i64().unwrap_or(0);
        let analyse: Vec<i64> = rows.iter().filter(|r| r["analysed_at"].is_null()).map(|r| id(r)).collect();
        let stems: Vec<i64> = rows.iter().filter(|r| matches!(r["intent"].as_str(), Some("vocal" | "remix" | "section")) && !matches!(r["stems"].as_str(), Some("done" | "queued" | "running")))
            .map(|r| id(r)).collect();
        let convert: Vec<i64> = rows.iter().filter(|r| r["intent"] != "backup" && r["path"].as_str()
            .is_some_and(|p| crate::convert::state(id(r), std::path::Path::new(p))["deck_problem"].is_string())).map(|r| id(r)).collect();
        let a = queue_analysis(&analyse);
        let s = if stems.is_empty() { vec![] } else { queue_stems(&stems)? };
        let c: Vec<i64> = convert.iter().filter(|t| !jobs::pending_for(**t, "convert")).map(|&tid| {
            jobs::submit("convert", &format!("Convert for the deck: {}", title_of(tid)), "cpu", Some(tid), move |h| {
                let out = crate::convert::convert(tid, &|f, m| h.progress(f, m), &|| h.cancelled())?;
                Ok(Some(json!({"bytes": std::fs::metadata(out)?.len()})))
            })
        }).collect();
        Ok(json!({"analyse": a.len(), "stems": s.len(), "convert": c.len()}))
    }).await
}

async fn delete_stems(P(tid): P<i64>) -> R {
    blocking(move || { stems::remove(tid)?; Ok(json!({"ok": true})) }).await
}

// ── Playlists ────────────────────────────────────────────────────────────────

async fn get_playlists() -> R {
    blocking(|| Ok(json!(library::playlist_tree()?))).await
}

async fn new_playlist(Json(b): Json<Value>) -> R {
    blocking(move || {
        let p = library::create_playlist(b["name"].as_str().unwrap_or(""), b["parent_id"].as_i64().unwrap_or(0), b["is_folder"].as_bool().unwrap_or(false))?;
        if b["rule"].is_object() {
            library::set_playlist_rule(p["id"].as_i64().unwrap_or(0), Some(&b["rule"]))?;
        }
        Ok(json!(p))
    }).await
}

/// Rename a playlist and / or set its rule (`rule: {...}` makes it smart, `rule: null` makes it ordinary again).
async fn rename_playlist(P(pid): P<i64>, Json(b): Json<Value>) -> R {
    blocking(move || {
        if let Some(name) = b["name"].as_str() { library::rename_playlist(pid, name)?; }
        if b.get("rule").is_some() { library::set_playlist_rule(pid, b["rule"].as_object().map(|_| &b["rule"]))?; }
        Ok(json!({"ok": true}))
    }).await
}

async fn remove_playlist(P(pid): P<i64>) -> R {
    blocking(move || { library::delete_playlist(pid)?; Ok(json!({"ok": true})) }).await
}

async fn order_playlist(P(pid): P<i64>, Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!({"undo": library::set_playlist_tracks(pid, &ids(&b, "track_ids"), "edit_playlist")?}))).await
}

async fn add_tracks(P(pid): P<i64>, Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!({"undo": library::add_to_playlist(pid, &ids(&b, "track_ids"))?}))).await
}

// ── Undo / history ───────────────────────────────────────────────────────────

async fn undo(P(hid): P<i64>) -> R {
    blocking(move || {
        if !db::undo(hid)? {
            return Err(ApiError(StatusCode::CONFLICT, "That change can't be undone".into()));
        }
        db::tx(mood::refresh)?;
        Ok(json!({"ok": true}))
    }).await
}

async fn history() -> R {
    blocking(|| Ok(json!(db::query("SELECT id, at, action, detail, undo IS NOT NULL AS can_undo FROM history ORDER BY id DESC LIMIT 200", &[])?))).await
}

// ── Jobs ─────────────────────────────────────────────────────────────────────

async fn get_jobs() -> Json<Value> {
    Json(json!(jobs::listing()))
}

async fn cancel_job(P(id): P<i64>) -> Json<Value> {
    Json(json!({"ok": jobs::cancel(id)}))
}

async fn clear_jobs() -> Json<Value> {
    jobs::clear_finished();
    Json(json!({"ok": true}))
}

// ── Export ───────────────────────────────────────────────────────────────────

fn export_args(b: &Value) -> (String, Option<Vec<i64>>, bool) {
    let pls = b["playlist_ids"].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect());
    (b["root"].as_str().unwrap_or("").to_string(), pls, b["include_stems"].as_bool().unwrap_or(true))
}

#[derive(Deserialize)]
struct DriveQuery {
    #[serde(default)]
    all: bool,
}

async fn drives(Query(q): Query<DriveQuery>) -> R {
    blocking(move || Ok(json!(usb::list_drives(q.all)))).await
}

async fn export_plan(Json(b): Json<Value>) -> R {
    blocking(move || {
        let (root, pls, st) = export_args(&b);
        Ok(usb::plan(&root, pls.as_deref(), st)?)
    }).await
}

async fn export(Json(b): Json<Value>) -> R {
    let (root, pls, st) = export_args(&b);
    if root.is_empty() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "Choose where to export".into()));
    }
    let id = jobs::submit("export", &format!("Export to {root}"), "io", None, move |h| {
        Ok(Some(usb::export(&root, pls.as_deref(), st, &|f, m| h.progress(f, m), &|| h.cancelled())?))
    });
    Ok(Json(json!({"job": id})))
}

// ── Deck ─────────────────────────────────────────────────────────────────────

async fn deck_status() -> R {
    blocking(|| Ok(deck::status())).await
}

async fn deck_ports() -> R {
    blocking(|| Ok(json!(deck::serial_ports()))).await
}

async fn deck_ota(mut form: Multipart) -> R {
    while let Some(field) = form.next_field().await.map_err(anyhow::Error::from)? {
        if field.name() != Some("file") {
            continue;
        }
        // Keep only the base name of the upload, never a path from the request.
        let name = std::path::Path::new(field.file_name().unwrap_or("update.ddjota")).file_name()
            .map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "update.ddjota".into());
        let dir = std::env::temp_dir().join(format!("pocketprep_ota_{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(&name);
        std::fs::write(&tmp, field.bytes().await.map_err(anyhow::Error::from)?)?;
        let id = jobs::submit("ota", &format!("Update deck: {name}"), "io", None, move |h| {
            let report = |f: f64, m: &str| h.progress(f, m);
            Ok(Some(deck::ota_upload(&tmp, &report)?))
        });
        return Ok(Json(json!({"job": id})));
    }
    Err(ApiError(StatusCode::BAD_REQUEST, "No file sent".into()))
}

async fn deck_flash(Json(b): Json<Value>) -> R {
    let (dir, port) = (PathBuf::from(b["build_dir"].as_str().unwrap_or("")), b["port"].as_str().unwrap_or("").to_string());
    let id = jobs::submit("flash", &format!("Flash over USB ({port})"), "io", None, move |h| {
        Ok(Some(deck::wired_flash(&dir, &port, &|f, m| h.progress(f, m))?))
    });
    Ok(Json(json!({"job": id})))
}

// ── Settings, system, file pickers ───────────────────────────────────────────

async fn get_settings() -> Json<Value> {
    Json(json!(config::load_settings()))
}

async fn put_settings(Json(v): Json<Row>) -> R {
    blocking(move || Ok(json!(config::save_settings(&v)?))).await
}

async fn system() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION"), "data_dir": config::data_dir(), "window": APP.get().is_some(),
                "os": std::env::consts::OS}))
}

/// Native file dialogs from the desktop window.
async fn pick(Json(b): Json<Value>) -> R {
    let Some(app) = APP.get().cloned() else { return Ok(Json(json!({"paths": [], "unavailable": true}))) };
    let kind = b["kind"].as_str().unwrap_or("folder").to_string();
    blocking(move || {
        let d = app.dialog().file();
        let paths: Vec<String> = match kind.as_str() {
            "folder" => d.blocking_pick_folder().into_iter().collect(),
            "bundle" => d.add_filter("Signed update", &["ddjota"]).blocking_pick_file().into_iter().collect(),
            _ => d.add_filter("Audio", config::AUDIO_EXTENSIONS).blocking_pick_files().unwrap_or_default(),
        }.into_iter().filter_map(|p| p.into_path().ok()).map(|p| p.to_string_lossy().to_string()).collect();
        Ok(json!({"paths": paths}))
    }).await
}

// ── Static UI ────────────────────────────────────────────────────────────────

async fn static_file(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match Ui::get(path) {
        Some(f) => ([(header::CONTENT_TYPE, mime_guess::from_path(path).first_or_octet_stream().to_string())], Body::from(f.data)).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn local_only(headers: HeaderMap, req: Request, next: Next) -> Response {
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    if !(host.starts_with("127.0.0.1:") || host.starts_with("localhost:")) {
        return StatusCode::FORBIDDEN.into_response();
    }
    next.run(req).await
}

pub fn router() -> Router {
    Router::new()
        .route("/api/jobs", get(get_jobs))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/clear", post(clear_jobs))
        .route("/api/tracks", get(list_tracks))
        .route("/api/moods", get(|| async { Json(mood::list()) }))
        .route("/api/tracks/{tid}", get(get_track).patch(patch_track))
        .route("/api/tracks/{tid}/waveform", get(get_waveform))
        .route("/api/tracks/{tid}/audio", get(get_audio))
        .route("/api/tracks/{tid}/stem/{name}", get(get_stem))
        .route("/api/tracks/{tid}/stems/peaks", get(get_stem_peaks))
        .route("/api/tracks/{tid}/artwork", get(get_artwork))
        .route("/api/tracks/{tid}/cues", put(put_cues))
        .route("/api/tracks/{tid}/stems", delete(delete_stems))
        .route("/api/import", post(import_files))
        .route("/api/tracks/remove", post(remove_tracks))
        .route("/api/analyse", post(analyse_tracks))
        .route("/api/convert", post(convert_tracks))
        .route("/api/stems", post(make_stems))
        .route("/api/prepare", post(prepare))
        .route("/api/transitions", post(transitions_from))
        .route("/api/player", get(player_state))
        .route("/api/player/open", post(player_open))
        .route("/api/player/load", post(player_load))
        .route("/api/player/play", post(|| async { player_cmd(player::Command::Play).await }))
        .route("/api/player/pause", post(|| async { player_cmd(player::Command::Pause).await }))
        .route("/api/player/toggle", post(|| async { player_cmd(player::Command::Toggle).await }))
        .route("/api/player/stop", post(|| async { player_cmd(player::Command::Stop).await }))
        .route("/api/player/seek", post(player_seek))
        .route("/api/player/set", post(player_set))
        .route("/api/player/stem", post(player_stem))
        .route("/api/player/devices", get(player_devices))
        .route("/api/player/device", post(player_device))
        .route("/api/sets/{pid}", get(set_report))
        .route("/api/sets/{pid}/order", post(set_order))
        .route("/api/playlists", get(get_playlists).post(new_playlist))
        .route("/api/playlists/{pid}", axum::routing::patch(rename_playlist).delete(remove_playlist))
        .route("/api/playlists/{pid}/tracks", put(order_playlist))
        .route("/api/playlists/{pid}/add", post(add_tracks))
        .route("/api/undo/{hid}", post(undo))
        .route("/api/history", get(history))
        .route("/api/drives", get(drives))
        .route("/api/export/plan", post(export_plan))
        .route("/api/export", post(export))
        .route("/api/deck/status", get(deck_status))
        .route("/api/deck/ota", post(deck_ota))
        .route("/api/deck/ports", get(deck_ports))
        .route("/api/deck/flash", post(deck_flash))
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/system", get(system))
        .route("/api/models", get(|| async { let mut s = models::status(); s["roformer"] = roformer::status(); Json(s) }).post(|| async { blocking(|| Ok(json!({"job": models::fetch()?}))).await }))
        .route("/api/models/roformer", post(|| async { blocking(|| Ok(json!({"job": roformer::fetch()?}))).await }))
        .route("/api/pick", post(pick))
        .fallback(static_file)
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))     // firmware bundles
        .layer(middleware::from_fn(local_only))
}
