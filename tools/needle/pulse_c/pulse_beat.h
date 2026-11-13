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
    double beatParity = 1;         // activation of the weaker half of the grid beats (every other one) / the stronger
    bool halved = false;           // the grid was at double time (beatParity < 0.75): every other beat dropped
    double gridActivation = 0;     // mean beat activation at the grid beats (0..1)
    double barVoteMargin = 0;      // (votes for the chosen bar position - votes for the runner-up) / all votes
    double phaseOffset = 0;        // Options::refinePhase: where the drum hits sit against the grid, s (+ = later)
    double phaseSpread = 0;        //   how much they scatter around that, s
    int phaseBeats = 0;            //   beats it was measured on
    double phaseShift = 0;         //   what was added to firstBeatSeconds (0 when the track was left alone)
    double analysedSeconds = 0;    // audio actually run through the network
    std::vector<float> beatLogit, downLogit;   // per 20-ms frame (diagnostics / tests)
};

struct Options {
    int threads = 0;          // worker threads; 0 = cores - 1, at most 3 (each at nice +10 on Linux)
    double maxSeconds = 0;    // > 0: analyse three windows of maxSeconds / 3 (at 20 / 50 / 80 %) instead of
                              // the whole track; tracks shorter than maxSeconds + 30 s are analysed whole
    bool doubleTimeRule = true;   // halve a grid whose every other beat is weak (see fitGrid)
    bool refinePhase = false;     // after the grid fit, move the grid onto the drum hits (see phaseFit)
};

// Whole-track analysis of interleaved float stereo (or mono with channels = 1). integerSnapBpm > 0: a fitted
// tempo within that distance of a whole BPM becomes the whole BPM, with the phase refitted to it.
Result analyze(const float* pcm, std::size_t frames, int channels, int sampleRate, double integerSnapBpm = 0.0,
               Options options = Options());

// The grid stage alone: tempo, phase and bar line from r.beatLogit / r.downLogit (analyze() ends with it).
void fitGrid(Result& r, double integerSnapBpm = 0.0, bool doubleTimeRule = true);

// The phase stage alone (analyze() runs it when Options::refinePhase is set): the grid against the onsets of
// the 22.05 kHz signal x that starts at startSeconds of the track. offset includes the convention shift.
struct PhaseFit {
    double offset = 0;   // s to add to firstBeatSeconds
    double spread = 0;   // s, weighted median distance of the onsets from the shifted grid
    int beats = 0;       // beats that counted
};
PhaseFit phaseFit(const std::vector<float>& mono22k, double bpm, double firstBeatSeconds, int threads = 1,
                  double startSeconds = 0.0);
// What analyze() does with it: notes the measurement in r and moves the grid by fit.offset, unless fewer
// than 8 beats were measured, the tempo is over 150 BPM or the spread is over 6 ms. True when the grid moved.
bool applyPhase(Result& r, const PhaseFit& fit);

// Building blocks (exposed for tests).
std::vector<float> resampleTo22050(const std::vector<float>& mono, int sampleRate, int threads = 1);
std::vector<float> logMel(const std::vector<float>& mono22k, int& outFrames, int threads = 1);   // [mel][frame]
void network(const std::vector<float>& mel, int frames, std::vector<float>& beatLogit, std::vector<float>& downLogit,
             int threads = 1);

}  // namespace PulseBeat
