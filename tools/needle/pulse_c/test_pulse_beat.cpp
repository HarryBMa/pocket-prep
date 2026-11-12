// Pulse C++ vs the PyTorch reference (fixtures written by tools pulse_cexport.py), plus a grid check.
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
    std::printf(fails ? "%d FAILED\n" : "ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
