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
use crate::{analysis, config, convert, deck, jobs, library, mood, stems};

pub static APP: OnceLock<tauri::AppHandle> = OnceLock::new();

#[derive(rust_embed::Embed)]
#[folder = "ui/"]
struct Ui;

const TRACK_COLUMNS: &str = "id, path, title, artist, album, genre, year, duration, bpm, musical_key, camelot, energy, \
    rating, color, comment, stems, analysed_at, added_at, artwork IS NOT NULL AS has_artwork, grid_offset_ms, downbeat_shift, key_shift, \
    mood, mood_manual, mood_conf, edited";

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

fn queue_stems(track_ids: &[i64]) -> anyhow::Result<Vec<i64>> {
    let mut out = vec![];
    for &tid in track_ids {
        if jobs::pending_for(tid, "stems") {
            continue;
        }
        db::tx(|c| Ok(c.execute("UPDATE tracks SET stems='queued' WHERE id=?", [tid])?))?;
        out.push(jobs::submit("stems", &format!("Stems: {}", title_of(tid)), "gpu", Some(tid), move |h| {
            stems::separate(tid, &|f, m| h.progress(f, m), &|| h.cancelled())?;
            Ok(None)
        }));
    }
    Ok(out)
}

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
        // Phrase sections (intro, build, drop...) for the phrase strip under the waveform.
        t.insert("sections".into(), a.as_ref().map(|a| a["sections"].clone()).unwrap_or(json!([])));
        // The grid as the deck will get it: the analysis grid plus the user's nudge.
        let a = a.map(|a| analysis::user_grid(&a, &t));
        t.insert("grid".into(), a.as_ref().map(|a| json!({
            "bpm": a["bpm"], "first_beat_ms": a["first_beat_ms"], "period_ms": a["period_ms"],
            "downbeat_index": a["downbeat_index"], "duration_ms": a["duration_ms"]})).unwrap_or(Value::Null));
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
    blocking(move || Ok(json!({"jobs": queue_stems(&ids(&b, "ids"))?}))).await
}

async fn delete_stems(P(tid): P<i64>) -> R {
    blocking(move || { stems::remove(tid)?; Ok(json!({"ok": true})) }).await
}

// ── Playlists ────────────────────────────────────────────────────────────────

async fn get_playlists() -> R {
    blocking(|| Ok(json!(library::playlist_tree()?))).await
}

async fn new_playlist(Json(b): Json<Value>) -> R {
    blocking(move || Ok(json!(library::create_playlist(b["name"].as_str().unwrap_or(""), b["parent_id"].as_i64().unwrap_or(0),
                                                       b["is_folder"].as_bool().unwrap_or(false))?))).await
}

async fn rename_playlist(P(pid): P<i64>, Json(b): Json<Value>) -> R {
    blocking(move || { library::rename_playlist(pid, b["name"].as_str().unwrap_or(""))?; Ok(json!({"ok": true})) }).await
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
        .route("/api/pick", post(pick))
        .fallback(static_file)
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))     // firmware bundles
        .layer(middleware::from_fn(local_only))
}
