// Pulse C++ vs the PyTorch reference (fixtures written by tools pulse_cexport.py), a grid check, and the phase
// stage vs its Python reference (pulse_phase.py).
//   c++ -O2 -std=c++17 -I. test_pulse_beat.cpp pulse_beat.cpp -o test_pulse_beat && ./test_pulse_beat <fixture dir>
#include "pulse_beat.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> testSignal(double seconds = 20.0) {   // same formula as pulse_cexport.test_signal
    const int sr = 22050;
    const double pi = 3.14159265358979323846, beat = 60.0 / 128.0;
    std::vector<float> x(std::size_t(seconds * sr));
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double t = double(n) / sr, tb = std::fmod(t, beat);
        const bool one = (long long)std::floor(t / beat) % 4 == 0;
        const double kick = std::sin(2 * pi * 55.0 * tb) * std::exp(-tb / 0.12) * (one ? 0.8 : 0.55);
        const double off = std::fmod(t + beat / 2, beat);
        const double hat = 0.15 * std::sin(2 * pi * 8000.0 * off) * std::exp(-off / 0.015);
        const long long bar = (long long)std::floor(t / (4 * beat));
        const double bass = 0.25 * std::sin(2 * pi * 55.0 * std::pow(2.0, (bar % 4) * 3 / 12.0) * t);
        x[n] = float(kick + hat + bass);
    }
    return x;
}

// The phase stage on the test signal: grid tempo, a grid beat (s), and what pulse_phase.py's refine() gives for
// that grid: offset (before the guards, s), spread (s), beats that counted. `pulse_phase.py fixture` prints the rows.
struct PhaseRow { double bpm, first, offset, spread; int beats; };
static const PhaseRow kPhaseFixture[] = {
    {128.0, 0.0, -0.002278940305538737, 0.00017586754622200829, 42},
    {128.0, 0.011, -0.01327894030553797, 0.00017586754622200829, 42},
    {128.0, 0.45505, 0.011421059694461309, 0.00017586754622200829, 42},
    {128.0, 0.238375, -0.004188075593100273, 0.0004496978239019178, 43},
    {64.0, 0.006, -0.008278940305538964, 0.0002947240035964249, 21},
    {127.9, 0.002, -0.011919007324891564, 0.0034845295716667835, 42},
};

static std::vector<float> readF32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    int fails = 0;
    const auto x = testSignal();
    int T = 0;
    auto t0 = std::chrono::steady_clock::now();
    const auto mel = PulseBeat::logMel(x, T);
    const auto refMel = readF32(dir + "/fixture_mel.f32");
    double melErr = 0;
    if (refMel.size() != mel.size()) { std::printf("FAIL mel size %zu vs %zu\n", mel.size(), refMel.size()); return 1; }
    for (std::size_t i = 0; i < mel.size(); ++i) melErr = std::max(melErr, double(std::abs(mel[i] - refMel[i])));
    std::printf("log-mel: %d frames, max |C++ - torch| = %.2e\n", T, melErr);
    if (melErr > 2e-3) { std::printf("FAIL mel\n"); ++fails; }

    std::vector<float> b, d;
    PulseBeat::network(refMel, T, b, d);                       // the reference mel: isolates the network
    const auto refY = readF32(dir + "/fixture_logits.f32");
    double yErr = 0;
    for (int t = 0; t < T; ++t)
        yErr = std::max({yErr, double(std::abs(b[t] - refY[t])), double(std::abs(d[t] - refY[T + t]))});
    std::printf("network: max |C++ - torch| logit = %.2e\n", yErr);
    if (yErr > 1e-2) { std::printf("FAIL logits\n"); ++fails; }

    std::vector<float> st(x.size() * 2);
    for (std::size_t i = 0; i < x.size(); ++i) st[2 * i] = st[2 * i + 1] = x[i];
    // the same signal at 44.1 kHz stereo, through the resampler (as the deck feeds it)
    std::vector<float> up(x.size() * 2 * 2);
    for (std::size_t i = 0; i + 1 < x.size(); ++i)
        for (int k = 0; k < 2; ++k) {
            const float v = k == 0 ? x[i] : 0.5f * (x[i] + x[i + 1]);
            up[(2 * i + k) * 2] = up[(2 * i + k) * 2 + 1] = v;
        }
    t0 = std::chrono::steady_clock::now();
    const auto r = PulseBeat::analyze(up.data(), up.size() / 2, 2, 44100);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("analyze (20 s, 44.1 kHz stereo): ok=%d steady=%d bpm=%.4f firstBeat=%.4f s barShare=%.2f "
                "onGrid=%.2f peaks=%d, %.0f ms\n", r.ok, r.steady, r.bpm, r.firstBeatSeconds, r.barPhaseShare,
                r.onGridShare, r.beatPeaks, ms);
    {   // stage timings on the same 20 s
        using clk = std::chrono::steady_clock;
        std::vector<float> mono(up.size() / 2);
        for (std::size_t i = 0; i < mono.size(); ++i) mono[i] = up[2 * i];
        auto a0 = clk::now();
        const auto rs = PulseBeat::resampleTo22050(mono, 44100);
        auto a1 = clk::now();
        int TT = 0;
        const auto mm = PulseBeat::logMel(rs, TT);
        auto a2 = clk::now();
        std::vector<float> bb, dd;
        PulseBeat::network(mm, TT, bb, dd);
        auto a3 = clk::now();
        auto ms_ = [](auto u, auto v) { return std::chrono::duration<double, std::milli>(v - u).count(); };
        std::printf("stages (20 s of audio): resample %.0f ms, log-mel %.0f ms, network %.0f ms\n",
                    ms_(a0, a1), ms_(a1, a2), ms_(a2, a3));
    }
    const double beat = 60.0 / 128.0;
    const double phaseErr = std::abs(std::remainder(r.firstBeatSeconds, beat));
    if (!r.ok || std::abs(r.bpm - 128.0) > 0.05 || phaseErr > 0.02) { std::printf("FAIL grid\n"); ++fails; }

    {   // phase stage: C++ vs pulse_phase.py on the same samples
        double worst = 0;
        bool counts = true;
        for (const PhaseRow& w : kPhaseFixture) {
            const auto f = PulseBeat::phaseFit(x, w.bpm, w.first);
            worst = std::max({worst, std::abs(f.offset - w.offset), std::abs(f.spread - w.spread)});
            counts = counts && f.beats == w.beats;
        }
        std::printf("phase: max |C++ - python| = %.3f us over %zu grids, beat counts %s\n", worst * 1e6,
                    sizeof(kPhaseFixture) / sizeof(kPhaseFixture[0]), counts ? "equal" : "DIFFERENT");
        if (worst > 5e-6 || !counts) { std::printf("FAIL phase\n"); ++fails; }
    }
    {   // the switch: analyze() with it moves the grid it finds without it by what phaseFit measures for that grid
        PulseBeat::Options on;
        on.refinePhase = true;
        const auto p = PulseBeat::analyze(up.data(), up.size() / 2, 2, 44100, 0.0, on);
        const auto fit = PulseBeat::phaseFit(x, r.bpm, r.firstBeatSeconds);       // x: the signal before the 2x upsampling
        const double bar = 4 * 60.0 / r.bpm;
        const double moved = std::remainder(p.firstBeatSeconds - r.firstBeatSeconds, bar);
        std::printf("analyze with refinePhase: grid moved %+.3f ms (offset %+.3f ms, spread %.3f ms, %d beats); "
                    "phaseFit on the 22.05 kHz signal: %+.3f ms\n", moved * 1e3, p.phaseOffset * 1e3, p.phaseSpread * 1e3,
                    p.phaseBeats, fit.offset * 1e3);
        if (!p.ok || p.bpm != r.bpm || p.phaseBeats < 8 || std::abs(moved - p.phaseShift) > 1e-9 ||
            p.phaseShift != p.phaseOffset || std::abs(p.phaseOffset - fit.offset) > 0.0005 || r.phaseShift != 0) {
            std::printf("FAIL phase switch\n");
            ++fails;
        }
    }
    {   // the guards: over 150 BPM, a spread over 6 ms and too few beats leave the grid where it is
        auto grid = [](double bpm, double first) {
            PulseBeat::Result g;
            g.ok = g.steady = true;
            g.bpm = bpm;
            g.firstBeatSeconds = first;
            return g;
        };
        auto fast = grid(174.0, 0.5), loose = grid(128.0, 0.5), few = grid(128.0, 0.5), good = grid(128.0, 0.001);
        const bool f = PulseBeat::applyPhase(fast, {0.004, 0.001, 200}), l = PulseBeat::applyPhase(loose, {0.004, 0.0061, 200});
        const bool w = PulseBeat::applyPhase(few, {0.004, 0.001, 7}), g = PulseBeat::applyPhase(good, {-0.004, 0.001, 200});
        const double bar = 4 * 60.0 / 128.0;
        std::printf("phase guards: 174 BPM moved=%d, spread 6.1 ms moved=%d, 7 beats moved=%d; 128 BPM -4 ms from 0.001 s "
                    "-> %.4f s (moved=%d)\n", f, l, w, good.firstBeatSeconds, g);
        if (f || l || w || fast.firstBeatSeconds != 0.5 || fast.phaseShift != 0 || fast.phaseOffset != 0.004 || !g ||
            std::abs(good.firstBeatSeconds - (bar - 0.003)) > 1e-12 || good.phaseShift != -0.004) {
            std::printf("FAIL phase guards\n");
            ++fails;
        }
    }
    std::printf(fails ? "%d FAILED\n" : "ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
