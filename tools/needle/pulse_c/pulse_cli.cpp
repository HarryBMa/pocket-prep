// pulse_cli <interleaved stereo float32 file> <sample rate> [logits out .f32]
// prints: ok steady bpm firstBeatSeconds barPhaseShare onGridShare beatPeaks milliseconds
#include "pulse_beat.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: pulse_cli in.f32 rate [logits.f32]\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<float> x(b.size() / 4);
    std::memcpy(x.data(), b.data(), x.size() * 4);
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = PulseBeat::analyze(x.data(), x.size() / 2, 2, std::atoi(argv[2]));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d %d %.6f %.6f %.4f %.4f %d %.1f\n", r.ok, r.steady, r.bpm, r.firstBeatSeconds,
                r.barPhaseShare, r.onGridShare, r.beatPeaks, ms);
    if (argc > 3) {
        std::ofstream o(argv[3], std::ios::binary);
        o.write(reinterpret_cast<const char*>(r.beatLogit.data()), r.beatLogit.size() * 4);
        o.write(reinterpret_cast<const char*>(r.downLogit.data()), r.downLogit.size() * 4);
    }
    return 0;
}
