// Pulse on stored data, for the evaluation tools (pulse_passfail.py):
//   pulse_logits net <mel.f32> <frames> <out.f32> [threads]
//       the network alone on a ready log-mel ([128][frames], row-major float32) -> beat logits, then downbeat logits
//   pulse_logits grid <set.bin> [doubleTimeRule (1 = on, the default)]
//       the grid fit alone on stored logits. set.bin, per track: u32 length of the id, the id, u32 frames,
//       float32 beat[frames], float32 down[frames]. Prints per track:
//       id ok steady bpm rawBpm firstBeatSeconds barPhaseShare onGridShare beatPeaks integerTempo beatParity
//       halved gridActivation barVoteMargin
#include "pulse_beat.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "net" && argc >= 5) {
        std::ifstream f(argv[2], std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
        const int frames = std::atoi(argv[3]);
        if (frames <= 0 || b.size() != std::size_t(frames) * PulseBeat::kMels * 4) {
            std::fprintf(stderr, "mel size does not match the frame count\n");
            return 3;
        }
        std::vector<float> mel(b.size() / 4), beat, down;
        std::memcpy(mel.data(), b.data(), b.size());
        PulseBeat::network(mel, frames, beat, down, argc > 5 ? std::atoi(argv[5]) : 3);
        std::ofstream o(argv[4], std::ios::binary);
        o.write(reinterpret_cast<const char*>(beat.data()), std::streamsize(beat.size() * 4));
        o.write(reinterpret_cast<const char*>(down.data()), std::streamsize(down.size() * 4));
        return o ? 0 : 4;
    }
    if (mode == "grid" && argc >= 3) {
        std::ifstream f(argv[2], std::ios::binary);
        const bool rule = argc > 3 ? std::atoi(argv[3]) != 0 : true;
        for (;;) {
            std::uint32_t len = 0, n = 0;
            if (!f.read(reinterpret_cast<char*>(&len), 4)) break;
            std::string id(len, ' ');
            f.read(&id[0], len);
            f.read(reinterpret_cast<char*>(&n), 4);
            PulseBeat::Result r;
            r.beatLogit.resize(n);
            r.downLogit.resize(n);
            f.read(reinterpret_cast<char*>(r.beatLogit.data()), std::streamsize(n) * 4);
            if (!f.read(reinterpret_cast<char*>(r.downLogit.data()), std::streamsize(n) * 4)) return 3;
            PulseBeat::fitGrid(r, 0.0, rule);
            std::printf("%s %d %d %.9f %.9f %.9f %.6f %.6f %d %d %.6f %d %.6f %.6f\n", id.c_str(), r.ok, r.steady, r.bpm,
                        r.rawBpm, r.firstBeatSeconds, r.barPhaseShare, r.onGridShare, r.beatPeaks, int(r.integerTempo),
                        r.beatParity, int(r.halved), r.gridActivation, r.barVoteMargin);
        }
        return 0;
    }
    std::fprintf(stderr, "usage: pulse_logits net mel.f32 frames out.f32 [threads] | pulse_logits grid set.bin [rule]\n");
    return 2;
}
