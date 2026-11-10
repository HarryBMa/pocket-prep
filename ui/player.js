// The player the app talks to: one object that looks like an <audio> element (so WaveSurfer and the preview bar use it
// unchanged) and plays through the practice engine in the server (src/player.rs, Watershed's engine) when an output
// device can be opened there: tempo and pitch independent of each other, stems as a mixer, an A-B loop done by the
// engine, a click with a count-in. When no device can be opened, or "Browser player" is chosen in Settings, the same
// object forwards to the page's own <audio> element and the practice controls stay off.
//
// Time: the engine reports its position about every 15 ms; the page polls it every 80 ms and interpolates between polls
// (position + elapsed x tempo) so the playhead moves smoothly. A seek shows at once, before the engine confirms it.

class PracticePlayer extends EventTarget {
  constructor(audio) {
    super();
    this.audio = audio;
    this.mode = "audio";                 // "engine" once the probe finds a device
    this.snap = null;                    // last snapshot from /api/player
    this.t0 = 0;                         // performance.now() when it arrived
    this.trackId = null;
    this.loading = null;                 // promise of a load in progress
    this._pendingPlay = false;
    this._seekTo = null;                 // seconds, shown until the engine confirms a seek
    this._seekSent = 0;
    this._volume = 1;
    this._wasPlaying = false;
    this.error = null;
    this.deviceName = "";
    for (const ev of ["play", "pause", "timeupdate", "loadedmetadata", "ended"]) {
      audio.addEventListener(ev, () => { if (this.mode === "audio") this.dispatchEvent(new Event(ev)); });
    }
    this.ready = this._probe();
  }

  async _probe() {
    let forced = false;
    try { forced = localStorage.getItem("player") === "browser"; } catch {}
    if (forced) { this.error = "Browser player chosen in Settings"; return; }
    try {
      const s = await this._post("/api/player/open");
      this.error = s.error;
      if (s.available) {
        this.mode = "engine"; this.snap = s; this.t0 = performance.now(); this.deviceName = s.device?.name || "";
        this._volume = s.gain;
        this._timer = setInterval(() => this._poll(), 80);
        document.addEventListener("visibilitychange", () => { if (!document.hidden) this._poll(); });
      }
    } catch (e) { this.error = e.message; }
    this.dispatchEvent(new Event("modechange"));
  }

  async _post(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
    return r.json();
  }

  get engine() { return this.mode === "engine"; }

  _take(s) {
    const was = this._wasPlaying;
    this.snap = s; this.t0 = performance.now();
    this._wasPlaying = s.playing;
    if (this._seekTo != null && Math.abs(s.position_ms / 1000 - this._seekTo) < 0.35) this._seekTo = null;
    if (s.playing && !was) this.dispatchEvent(new Event("play"));
    if (!s.playing && was) {
      this.dispatchEvent(new Event("pause"));
      if (s.duration_ms && s.position_ms >= s.duration_ms - 150) this.dispatchEvent(new Event("ended"));
    }
    this.dispatchEvent(new Event("snapshot"));
    if (s.needs_reload && this.trackId != null && !this.loading) this.open(this.trackId, { stems: s.stems_loaded, position: s.position_ms / 1000 });
  }

  async _poll() {
    if (this._polling) return;
    this._polling = true;
    try {
      const r = await fetch("/api/player");
      if (r.ok) { this._take(await r.json()); if (this.snap.playing) this.dispatchEvent(new Event("timeupdate")); }
    } catch {} finally { this._polling = false; }
  }

  // ── the HTMLMediaElement surface ───────────────────────────────────────────
  get paused() { return this.engine ? !(this.snap?.playing || this._pendingPlay) : this.audio.paused; }
  get ended() { return this.engine ? false : this.audio.ended; }
  get seeking() { return false; }
  get duration() { return this.engine ? (this.snap?.duration_ms || 0) / 1000 : this.audio.duration; }
  get currentTime() {
    if (!this.engine) return this.audio.currentTime;
    if (this._seekTo != null) return this._seekTo;
    const s = this.snap;
    if (!s) return 0;
    if (!s.playing || s.counting_in) return s.position_ms / 1000;
    const t = s.position_ms / 1000 + (performance.now() - this.t0) / 1000 * s.tempo;
    return Math.min(t, s.duration_ms / 1000);
  }
  set currentTime(v) {
    if (!this.engine) { this.audio.currentTime = v; return; }
    v = Math.max(0, v);
    this._seekTo = v; this.dispatchEvent(new Event("timeupdate"));
    const now = performance.now();
    clearTimeout(this._seekT);
    const send = () => { this._seekSent = performance.now(); this._post("/api/player/seek", { ms: this._seekTo != null ? this._seekTo * 1000 : v * 1000 }).then((s) => this._take(s)).catch(() => {}); };
    if (now - this._seekSent > 60) send(); else this._seekT = setTimeout(send, 60);       // drags send about 16 seeks a second
  }
  get volume() { return this.engine ? this._volume : this.audio.volume; }
  set volume(v) {
    if (!this.engine) { this.audio.volume = v; return; }
    this._volume = v;
    clearTimeout(this._volT);
    this._volT = setTimeout(() => this._post("/api/player/set", { gain: v }).then((s) => this._take(s)).catch(() => {}), 30);
  }
  get muted() { return this.engine ? false : this.audio.muted; }
  set muted(v) { if (!this.engine) this.audio.muted = v; }
  // The tempo is the engine's own (see setTempo): the page's playbackRate stays 1.
  get playbackRate() { return this.engine ? 1 : this.audio.playbackRate; }
  set playbackRate(v) { if (!this.engine) this.audio.playbackRate = v; }
  get defaultPlaybackRate() { return 1; }
  set defaultPlaybackRate(v) { if (!this.engine) this.audio.defaultPlaybackRate = v; }
  get preservesPitch() { return true; }
  set preservesPitch(v) {}
  get src() { return this.engine ? "" : this.audio.src; }
  set src(v) { if (!this.engine) this.audio.src = v; }
  get currentSrc() { return this.engine ? "" : this.audio.currentSrc; }
  get controls() { return false; }
  set controls(v) {}
  get autoplay() { return false; }
  set autoplay(v) {}
  removeAttribute(name) { if (!this.engine && name === "src") this.audio.removeAttribute("src"); }
  load() { if (!this.engine) this.audio.load(); }
  canPlayType(t) { return this.audio.canPlayType(t); }
  setSinkId(id) { return this.engine ? Promise.resolve() : this.audio.setSinkId?.(id); }
  remove() {}

  async play() {
    await this.ready;
    if (!this.engine) return this.audio.play();
    if (this.loading) { this._pendingPlay = true; await this.loading; this._pendingPlay = false; }
    if (this.trackId == null) return;
    this._wasPlaying = false;
    this._take(await this._post("/api/player/play"));
  }
  pause() {
    if (!this.engine) { this.audio.pause(); return; }
    this._pendingPlay = false;
    this._post("/api/player/pause").then((s) => this._take(s)).catch(() => {});
    if (this.snap) { this.snap = { ...this.snap, position_ms: this.currentTime * 1000, playing: false }; this.t0 = performance.now(); }
    this.dispatchEvent(new Event("pause"));
  }

  // ── what the app adds ──────────────────────────────────────────────────────
  /** Make a track the one that plays: its original mix, or its stems with { stems: true }. */
  open(id, { stems = false, position = 0 } = {}) {
    this.trackId = id;
    if (!this.engine) {
      if (this.mode === "audio") this.audio.src = `/api/tracks/${id}/audio`;
      return Promise.resolve();
    }
    this._seekTo = null;
    const keep = this.snap?.track_id === id && this.snap?.playing;
    const job = this._post("/api/player/load", { track_id: id, stems, position_ms: position * 1000, play: keep }).then((s) => {
      this._wasPlaying = false; this._take(s);
      this.dispatchEvent(new Event("loadedmetadata"));
    }).catch((e) => { this.error = e.message; this.dispatchEvent(new CustomEvent("playererror", { detail: e.message })); });
    this.loading = job.finally(() => { if (this.loading === job2) this.loading = null; });
    const job2 = this.loading;
    return this.loading;
  }
  /** Hear one stem alone (the engine's solo), or the full mix with null. Loads the stems first if only the mix is loaded. */
  async solo(id, name) {
    if (!this.engine) {
      const pos = this.audio.currentTime, playing = !this.audio.paused;
      this.audio.src = name ? `/api/tracks/${id}/stem/${name}` : `/api/tracks/${id}/audio`;
      this.audio.addEventListener("loadedmetadata", () => { this.audio.currentTime = pos; if (playing) this.audio.play(); }, { once: true });
      return;
    }
    if (!name && !this.snap?.stems_loaded) return;                    // the mix is already what is playing
    if (!this.snap?.stems_loaded) await this.open(id, { stems: true, position: this.currentTime });
    if (this.loading) await this.loading;
    await Promise.all((this.snap?.stems || []).map((s) => this.stem(s.name, { solo: s.name === name })));
  }
  async set(patch) {
    if (!this.engine) return null;
    const s = await this._post("/api/player/set", patch);
    this._take(s);
    return s;
  }
  async stem(name, patch) {
    if (!this.engine) return null;
    const s = await this._post("/api/player/stem", { name, ...patch });
    this._take(s);
    return s;
  }
  /** Loop between two times in ms, or null. The engine does the jump, so it is seamless and follows the tempo. */
  setLoop(range) { return this.set({ loop: range ? { start_ms: range.start_ms, end_ms: range.end_ms } : null }); }
  /** Output devices and the current one. */
  async devices() { return fetch("/api/player/devices").then((r) => r.json()); }
  async chooseDevice(id) { const s = await this._post("/api/player/device", { id }); this._take(s); this.deviceName = s.device?.name || ""; return s; }
}
