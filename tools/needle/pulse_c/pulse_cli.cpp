// pulse_cli <interleaved stereo float32 file> <sample rate> [threads (0 = auto)] [maxSeconds (0 = whole track)]
//           [doubleTimeRule (1 = on, the default)]
// prints: ok steady bpm firstBeatSeconds barPhaseShare onGridShare beatPeaks milliseconds analysedSeconds
//         beatParity halved gridActivation barVoteMargin rawBpm
#include "pulse_beat.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: pulse_cli in.f32 rate [threads] [maxSeconds] [doubleTimeRule]\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<float> x(b.size() / 4);
    std::memcpy(x.data(), b.data(), x.size() * 4);
    PulseBeat::Options o;
    if (argc > 3) o.threads = std::atoi(argv[3]);
    if (argc > 4) o.maxSeconds = std::atof(argv[4]);
    if (argc > 5) o.doubleTimeRule = std::atoi(argv[5]) != 0;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = PulseBeat::analyze(x.data(), x.size() / 2, 2, std::atoi(argv[2]), 0.0, o);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d %d %.6f %.6f %.4f %.4f %d %.1f %.1f %.4f %d %.4f %.4f %.6f\n", r.ok, r.steady, r.bpm, r.firstBeatSeconds,
                r.barPhaseShare, r.onGridShare, r.beatPeaks, ms, r.analysedSeconds, r.beatParity, int(r.halved),
                r.gridActivation, r.barVoteMargin, r.rawBpm);
    return 0;
}
