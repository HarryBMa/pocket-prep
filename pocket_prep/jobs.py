"""A small background job queue, so analysis and stems never block the UI.

Two worker threads: one for analysis (CPU) and one for stems (GPU), so a
long Demucs run doesn't hold up analysing newly imported tracks.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field

_ids = itertools.count(1)


@dataclass
class Job:
    id: int
    kind: str            # analyse | stems | export | flash | ota
    title: str
    status: str = "queued"   # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    finished: float | None = None
    track_id: int | None = None
    result: dict | None = None
    cancel_requested: bool = False


_jobs: dict[int, Job] = {}
_queues = {"cpu": queue.Queue(), "gpu": queue.Queue(), "io": queue.Queue()}
_lock = threading.Lock()


def submit(kind: str, title: str, fn, lane: str = "cpu", track_id: int | None = None) -> Job:
    """fn(job) does the work; it can call job_progress(job, fraction, message)."""
    job = Job(next(_ids), kind, title, track_id=track_id)
    with _lock:
        _jobs[job.id] = job
    _queues[lane].put((job, fn))
    return job


def job_progress(job: Job, fraction: float, message: str = "") -> None:
    job.progress = max(0.0, min(1.0, fraction))
    if message:
        job.message = message


def cancel(job_id: int) -> bool:
    j = _jobs.get(job_id)
    if not j or j.status in ("done", "failed", "cancelled"):
        return False
    j.cancel_requested = True
    if j.status == "queued":
        j.status = "cancelled"
    return True


def pending_for(track_id: int, kind: str) -> bool:
    return any(j.track_id == track_id and j.kind == kind and j.status in ("queued", "running") for j in _jobs.values())


def listing(limit: int = 60) -> list[dict]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.id, reverse=True)[:limit]
    return [asdict(j) for j in jobs]


def clear_finished() -> None:
    with _lock:
        for k in [k for k, j in _jobs.items() if j.status in ("done", "failed", "cancelled")]:
            del _jobs[k]


def _worker(lane: str) -> None:
    q = _queues[lane]
    while True:
        job, fn = q.get()
        if job.status == "cancelled":
            continue
        job.status = "running"
        try:
            job.result = fn(job) or None
            job.status = "cancelled" if job.cancel_requested else "done"
            job.progress = 1.0
        except Exception as e:  # keep the worker alive whatever happens
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.message = str(e)[:500]
            traceback.print_exc()
        job.finished = time.time()


def start() -> None:
    for lane in _queues:
        threading.Thread(target=_worker, args=(lane,), daemon=True, name=f"jobs-{lane}").start()
