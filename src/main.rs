//! Pocket Prep: companion prep app for the Pocket Deck.
//!
//! Starts the local API server in the background, then shows the UI in a
//! native window (Tauri: WebView2 on Windows, WKWebView on macOS).
//!
//!     pocket-prep             desktop window
//!     pocket-prep --browser   open in your web browser instead
//!     pocket-prep --headless  just the server (for tests/e2e.sh)

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod config;
mod convert;
mod db;
mod deck;
mod embed;
mod export;
mod jobs;
mod library;
mod meter;
mod models;
mod player;
mod mood;
mod quality;
mod server;
mod sets;
mod stems;
mod transitions;
mod vocals;
#[cfg(test)]
mod testaudio;


/// Bind 127.0.0.1:8765, or any free port if that's taken.
fn listener() -> std::net::TcpListener {
    std::net::TcpListener::bind(("127.0.0.1", 8765))
        .or_else(|_| std::net::TcpListener::bind(("127.0.0.1", 0)))
        .expect("can't open a local port")
}

fn start_server() -> String {
    let l = listener();
    l.set_nonblocking(true).unwrap();
    let url = format!("http://127.0.0.1:{}/", l.local_addr().unwrap().port());
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async move {
            let l = tokio::net::TcpListener::from_std(l).unwrap();
            axum::serve(l, server::router()).await.unwrap();
        });
    });
    url
}

/// Point ClubCraft at our folders: its scratch files (e.g. the DirectML self-check
/// result) go under our data dir, and models live in <data>/models (filled from the app
/// bundle, or fetched: see models.rs). Set before any thread starts.
fn clubcraft_paths() {
    std::env::set_var("CLUBCRAFT_HOME", config::data_dir().join("clubcraft"));
    if std::env::var_os("CLUBCRAFT_MODELS").is_none() {
        if let Err(e) = models::prepare() { eprintln!("models: {e:#}"); }
    }
}

fn main() {
    config::ensure_dirs().expect("can't create the data folder");
    clubcraft_paths();
    drop(db::conn());                                       // create or upgrade the library now
    let url = start_server();
    server::startup_jobs();
    println!("DKPRP running at {url}  (data: {})", config::data_dir().display());

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--browser" || a == "--headless") {
        if args.iter().any(|a| a == "--browser") {
            let opener = if cfg!(windows) { ("cmd", vec!["/C", "start", ""]) } else if cfg!(target_os = "macos") { ("open", vec![]) } else { ("xdg-open", vec![]) };
            let _ = std::process::Command::new(opener.0).args(opener.1).arg(&url).spawn();
        }
        loop {
            std::thread::park();
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            let _ = server::APP.set(app.handle().clone());
            tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::External(url.parse().unwrap()))
                .title("DKPRP")
                .inner_size(1400.0, 880.0)
                .min_inner_size(980.0, 640.0)
                .disable_drag_drop_handler()                // keeps HTML drag and drop (tracks onto playlists)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running DKPRP");
}
