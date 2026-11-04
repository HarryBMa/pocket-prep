//! A small background job queue, so analysis and stems never block the UI.
//!
//! Three worker threads ("lanes"): cpu (analysis), gpu (stems) and io (export,
//! deck updates), so a long stem run doesn't hold up analysing new imports.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Serialize)]
pub struct Job {
    pub id: i64,
    pub kind: String,           // analyse | stems | export | flash | ota
    pub title: String,
    pub status: String,         // queued | running | done | failed | cancelled
    pub progress: f64,
    pub message: String,
    pub created: f64,
    pub finished: Option<f64>,
    pub track_id: Option<i64>,
    pub result: Option<Value>,
    pub cancel_requested: bool,
}

type Work = Box<dyn FnOnce(&Handle) -> anyhow::Result<Option<Value>> + Send>;

/// What a running job uses to report progress and check for cancel.
pub struct Handle {
    id: i64,
    cancel: Arc<AtomicBool>,
}

impl Handle {
    pub fn progress(&self, fraction: f64, message: &str) {
        update(self.id, |j| {
            j.progress = fraction.clamp(0.0, 1.0);
            if !message.is_empty() {
                j.message = message.to_string();
            }
        });
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

struct State {
    jobs: HashMap<i64, (Job, Arc<AtomicBool>)>,
    lanes: HashMap<&'static str, Sender<(i64, Work)>>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static NEXT: AtomicI64 = AtomicI64::new(1);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.get_or_init(|| {
        let mut lanes = HashMap::new();
        for lane in ["cpu", "gpu", "io"] {
            let (tx, rx) = channel::<(i64, Work)>();
            lanes.insert(lane, tx);
            std::thread::Builder::new().name(format!("jobs-{lane}")).spawn(move || {
                for (id, work) in rx {
                    run(id, work);
                }
            }).unwrap();
        }
        Mutex::new(State { jobs: HashMap::new(), lanes })
    }).lock().unwrap_or_else(|e| e.into_inner())
}

fn update(id: i64, f: impl FnOnce(&mut Job)) {
    if let Some((j, _)) = state().jobs.get_mut(&id) {
        f(j);
    }
}

fn run(id: i64, work: Work) {
    let cancel = {
        let mut s = state();
        let Some((j, c)) = s.jobs.get_mut(&id) else { return };
        if j.status == "cancelled" {
            return;
        }
        j.status = "running".into();
        c.clone()
    };
    let h = Handle { id, cancel: cancel.clone() };
    // catch_unwind keeps the worker alive whatever happens inside a job.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&h)));
    let stopped = cancel.load(Ordering::Relaxed);
    update(id, |j| {
        match res {
            Ok(Ok(r)) => {
                j.result = r;
                j.status = if stopped { "cancelled" } else { "done" }.into();
                j.progress = 1.0;
            }
            Ok(Err(e)) => {
                j.status = if stopped { "cancelled" } else { "failed" }.into();
                j.message = format!("{e:#}").chars().take(500).collect();
                eprintln!("job {id} failed: {e:#}");
            }
            Err(_) => {
                j.status = "failed".into();
                j.message = "Internal error (panic)".into();
            }
        }
        j.finished = Some(crate::db::now());
    });
}

/// work(handle) does the job; it can call handle.progress(fraction, message).
pub fn submit(kind: &str, title: &str, lane: &'static str, track_id: Option<i64>,
              work: impl FnOnce(&Handle) -> anyhow::Result<Option<Value>> + Send + 'static) -> i64 {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let job = Job { id, kind: kind.into(), title: title.into(), status: "queued".into(), progress: 0.0,
                    message: String::new(), created: crate::db::now(), finished: None, track_id, result: None,
                    cancel_requested: false };
    let mut s = state();
    s.jobs.insert(id, (job, Arc::new(AtomicBool::new(false))));
    s.lanes[lane].send((id, Box::new(work))).expect("job worker stopped");
    id
}

pub fn cancel(id: i64) -> bool {
    let mut s = state();
    let Some((j, c)) = s.jobs.get_mut(&id) else { return false };
    if ["done", "failed", "cancelled"].contains(&j.status.as_str()) {
        return false;
    }
    j.cancel_requested = true;
    c.store(true, Ordering::Relaxed);
    if j.status == "queued" {
        j.status = "cancelled".into();
    }
    true
}

pub fn pending_for(track_id: i64, kind: &str) -> bool {
    state().jobs.values().any(|(j, _)| j.track_id == Some(track_id) && j.kind == kind && (j.status == "queued" || j.status == "running"))
}

pub fn listing() -> Vec<Job> {
    let mut v: Vec<Job> = state().jobs.values().map(|(j, _)| j.clone()).collect();
    v.sort_by_key(|j| -j.id);
    v.truncate(60);
    v
}

pub fn clear_finished() {
    state().jobs.retain(|_, (j, _)| !["done", "failed", "cancelled"].contains(&j.status.as_str()));
}
