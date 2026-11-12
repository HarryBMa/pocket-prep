// C ABI around PulseBeat::analyze, so Rust can call it without C++ types crossing the boundary.
// Returns 1 when Pulse found a grid that is steady (>= 60 % of beat peaks on one tempo line), else 0.
#include "pulse_beat.h"
#include <cstddef>

extern "C" {

struct PulseGridC {
    int ok, steady, integer_tempo, beat_peaks;
    double bpm, raw_bpm, first_downbeat_s, bar_share, on_grid, analysed_s;
};

int pulse_analyze(const float* interleaved, std::size_t frames, int channels, int rate, double snap_bpm,
                  int threads, PulseGridC* out) {
    if (!out) return 0;
    *out = PulseGridC{};
    try {
        PulseBeat::Options opt;
        opt.threads = threads;
        PulseBeat::Result r = PulseBeat::analyze(interleaved, frames, channels, rate, snap_bpm, opt);
        out->ok = r.ok ? 1 : 0;
        out->steady = r.steady ? 1 : 0;
        out->integer_tempo = r.integerTempo ? 1 : 0;
        out->beat_peaks = r.beatPeaks;
        out->bpm = r.bpm;
        out->raw_bpm = r.rawBpm;
        out->first_downbeat_s = r.firstBeatSeconds;
        out->bar_share = r.barPhaseShare;
        out->on_grid = r.onGridShare;
        out->analysed_s = r.analysedSeconds;
        return (r.ok && r.steady) ? 1 : 0;
    } catch (...) {
        *out = PulseGridC{};
        return 0;
    }
}

}  // extern "C"