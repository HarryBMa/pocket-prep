#pragma once
// Pulse: a small neural beat / downbeat tracker (81 k parameters, distilled from Beat This!, fine-tuned on
// hand-annotated DJ tracks) plus a straight-line grid fit and a bar-line fit. Self-contained: no library,
// float32, runs on the track-analysis thread (never on an audio callback).
//
//   stereo PCM (any rate) -> mono 22.05 kHz -> 128-band log-mel, 50 frames/s (Beat This!'s front end)
//   -> network (Conv2D / ReLU / add / 2x nearest upsample) -> beat and downbeat logits per frame
//   -> grid: tempo + phase line through the beat peaks (no tempo range cap) -> bar line from the downbeats
#include <cstddef>
#include <vector>

namespace PulseBeat {

constexpr int kSampleRate = 22050;
constexpr int kFps = 50;          // 441-sample hop
constexpr int kMels = 128;

struct Result {
    bool ok = false;               // a grid was found
    bool steady = false;           // >= 60 % of the beat peaks sit on one straight tempo line
    double bpm = 0;                // grid tempo (the whole number when integerSnap applied)
    double rawBpm = 0;             // grid tempo as fitted
    bool integerTempo = false;     // snapped to a whole BPM (and the phase refitted for it)
    double firstBeatSeconds = 0;   // a beat on the grid; the bar line (beat 1) when downbeats are clear
    double barPhaseShare = 0;      // share of detected downbeats on the chosen bar position (0..1)
    double onGridShare = 0;        // share of beat peaks within 35 ms of the grid
    int beatPeaks = 0;
    double analysedSeconds = 0;    // audio actually run through the network
    std::vector<float> beatLogit, downLogit;   // per 20-ms frame (diagnostics / tests)
};

struct Options {
    int threads = 0;          // worker threads; 0 = cores - 1, at most 3 (each at nice +10 on Linux)
    double maxSeconds = 0;    // > 0: analyse three windows of maxSeconds / 3 (at 20 / 50 / 80 %) instead of
                              // the whole track; tracks shorter than maxSeconds + 30 s are analysed whole
};

// Whole-track analysis of interleaved float stereo (or mono with channels = 1). integerSnapBpm > 0: a fitted
// tempo within that distance of a whole BPM becomes the whole BPM, with the phase refitted to it.
Result analyze(const float* pcm, std::size_t frames, int channels, int sampleRate, double integerSnapBpm = 0.0,
               Options options = Options());

// Building blocks (exposed for tests).
std::vector<float> resampleTo22050(const std::vector<float>& mono, int sampleRate, int threads = 1);
std::vector<float> logMel(const std::vector<float>& mono22k, int& outFrames, int threads = 1);   // [mel][frame]
void network(const std::vector<float>& mel, int frames, std::vector<float>& beatLogit, std::vector<float>& downLogit,
             int threads = 1);

}  // namespace PulseBeat
