// Pulse beat / downbeat tracker. See pulse_beat.h. Reference implementation: tools/needle pulse.py,
// pulse_grid.py, pulse_rave.py (bar_fit); this file reproduces them step by step.
#include "pulse_beat.h"
#include "pulse_weights.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <exception>
#include <mutex>
#include <numeric>
#include <thread>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace PulseBeat {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kNfft = 1024;
constexpr int kHop = 441;
constexpr int kChunk = 512;       // frames per network pass (10.24 s)
constexpr int kChunkHop = 448;    // 64 frames of crossfade between passes
constexpr double kDoubleTimeParity = 0.75;   // see fitGrid

// Worker threads for the heavy loops. On Linux each worker drops to nice +10 first, so a track being
// analysed never takes CPU from the audio worker or the UI of a track that is playing.
// An exception in a worker (e.g. std::bad_alloc) stops the remaining work and is rethrown on the calling
// thread after the join, so the caller's try / catch sees it (an exception leaving a std::thread would
// terminate the firmware instead).
template <class F>
void parallelFor(int n, int threads, F&& f) {
    if (threads <= 1 || n < 2) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int> next{0};
    std::exception_ptr failure;
    std::mutex failureMutex;
    auto work = [&] {
#if defined(__linux__)
        setpriority(PRIO_PROCESS, pid_t(syscall(SYS_gettid)), 10);
#endif
        try {
            for (int i; (i = next.fetch_add(1)) < n;) f(i);
        } catch (...) {
            next.store(n);
            std::lock_guard<std::mutex> lock(failureMutex);
            if (!failure) failure = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(std::min(threads, n)));
    try {
        for (int t = 0; t < std::min(threads, n); ++t) pool.emplace_back(work);
    } catch (...) {                       // no more threads available: the ones started finish the work,
        if (pool.empty())                 // or this thread does (at its own priority)
            for (int i; (i = next.fetch_add(1)) < n;) f(i);
    }
    for (auto& t : pool) t.join();
    if (failure) std::rethrow_exception(failure);
}

// ── Resampling: windowed-sinc (Blackman), cutoff just below the lower Nyquist ─────────────────────────────
std::vector<float> resample(const std::vector<float>& x, int inRate, int outRate, int threads) {
    if (inRate == outRate) return x;
    const double ratio = double(inRate) / outRate;            // input samples per output sample
    const double cutoff = 0.97 * 0.5 * std::min(1.0, 1.0 / ratio);   // in cycles per input sample
    const int half = int(std::ceil(8.0 / (2.0 * cutoff)));    // 8 zero crossings each side (mels stop at 11 kHz)
    // kernel table: h(d) for d in [-half, half] at 1/kSub of an input sample, linearly interpolated
    constexpr int kSub = 512;
    const int nTab = 2 * half * kSub + 2;
    std::vector<float> tab(nTab);
    for (int i = 0; i < nTab; ++i) {
        const double d = double(i) / kSub - half;
        const double a = 2.0 * cutoff * d;
        const double sinc = std::abs(a) < 1e-12 ? 1.0 : std::sin(kPi * a) / (kPi * a);
        const double u = (d + half) / (2.0 * half);           // Blackman window over [-half, half]
        tab[i] = (u <= 0 || u >= 1) ? 0.f : float(sinc * (0.42 - 0.5 * std::cos(2 * kPi * u) + 0.08 * std::cos(4 * kPi * u)));
    }
    const std::size_t n = std::size_t(std::floor(double(x.size()) / ratio));
    std::vector<float> y(n);
    const long len = long(x.size());
    constexpr std::size_t kBlock = 1 << 14;
    const auto tap = [&](double d) {                              // h(d), d = output time - input sample
        const double pos = (d + half) * kSub;
        const int i0 = int(pos);
        const float f = float(pos - i0);
        return tab[i0] + f * (tab[i0 + 1] - tab[i0]);
    };
    // The deck's rates (44.1 and 48 kHz) are simple fractions of 22.05 kHz (2/1 and 320/147), so the output
    // samples fall on 1 or 147 different positions between input samples: the 2 x half filter taps of each
    // position are worked out once, and every output sample is one dot product.
    const long g = std::gcd(long(inRate), long(outRate)), up = outRate / g, down = inRate / g;
    if (up <= 512) {
        const int taps = 2 * half;
        std::vector<float> h(std::size_t(up) * taps), norm(up);
        for (long ph = 0; ph < up; ++ph) {
            float wsum = 0;
            for (int m = 0; m < taps; ++m) wsum += h[std::size_t(ph) * taps + m] = tap(double(ph) / up + half - 1 - m);
            norm[ph] = wsum > 0 ? 1.f / wsum : 0.f;
        }
        parallelFor(int((n + kBlock - 1) / kBlock), threads, [&](int blk) {
            for (std::size_t j = std::size_t(blk) * kBlock; j < std::min(n, std::size_t(blk + 1) * kBlock); ++j) {
                const long long num = (long long)j * down;
                const long c = long(num / up), first = c - half + 1;
                const float* hp = h.data() + std::size_t(num % up) * taps;
                float acc = 0;
                if (first >= 0 && first + taps <= len) {
                    const float* xp = x.data() + first;
                    for (int m = 0; m < taps; ++m) acc += hp[m] * xp[m];
                } else {
                    for (int m = 0; m < taps; ++m)
                        if (first + m >= 0 && first + m < len) acc += hp[m] * x[std::size_t(first + m)];
                }
                y[j] = acc * norm[num % up];
            }
        });
        return y;
    }
    parallelFor(int((n + kBlock - 1) / kBlock), threads, [&](int blk) {   // any other rate: taps per sample
        for (std::size_t j = std::size_t(blk) * kBlock; j < std::min(n, std::size_t(blk + 1) * kBlock); ++j) {
            const double t = j * ratio;
            const long c = long(std::floor(t));
            float acc = 0, wsum = 0;
            for (long k = c - half + 1; k <= c + half; ++k) {
                const float h = tap(t - k);
                wsum += h;
                if (k >= 0 && k < len) acc += h * x[std::size_t(k)];
            }
            y[j] = wsum > 0 ? acc / wsum : 0.f;
        }
    });
    return y;
}

// ── FFT: iterative radix-2, complex, size 1024 ─────────────────────────────────────────────────────────
struct Fft {
    std::vector<std::complex<double>> tw;
    std::vector<int> rev;
    explicit Fft(int n) : tw(n / 2), rev(n) {
        for (int i = 0; i < n / 2; ++i) tw[i] = std::polar(1.0, -2 * kPi * i / n);
        int bits = 0;
        while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
            rev[i] = r;
        }
    }
    void run(std::vector<std::complex<double>>& a) const {
        const int n = int(a.size());
        for (int i = 0; i < n; ++i) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
        for (int len = 2; len <= n; len <<= 1) {
            const int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; ++k) {
                    const auto u = a[i + k], v = a[i + k + len / 2] * tw[k * step];
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                }
        }
    }
};

// Slaney mel scale (torchaudio mel_scale="slaney"): linear below 1 kHz, logarithmic above
constexpr double kMelLinearHz = 200.0 / 3, kMelLogFromHz = 1000.0, kMelLogFromMel = kMelLogFromHz / kMelLinearHz;
double melLogStep() { return std::log(6.4) / 27.0; }
double hzToMel(double f) {
    return f < kMelLogFromHz ? f / kMelLinearHz : kMelLogFromMel + std::log(f / kMelLogFromHz) / melLogStep();
}
double melToHz(double m) {
    return m < kMelLogFromMel ? m * kMelLinearHz : kMelLogFromHz * std::exp(melLogStep() * (m - kMelLogFromMel));
}

// torchaudio.functional.melscale_fbanks(n_freqs=513, f_min=30, f_max=11000, n_mels=128, sr=22050, norm=None, slaney)
std::vector<float> melBank() {
    const int nf = kNfft / 2 + 1;
    std::vector<double> freqs(nf), pts(kMels + 2);
    for (int i = 0; i < nf; ++i) freqs[i] = double(kSampleRate) / 2 * i / (nf - 1);
    const double m0 = hzToMel(30.0), m1 = hzToMel(11000.0);
    for (int i = 0; i < kMels + 2; ++i) pts[i] = melToHz(m0 + (m1 - m0) * i / (kMels + 1));
    std::vector<float> fb(std::size_t(kMels) * nf, 0.f);
    for (int m = 0; m < kMels; ++m)
        for (int i = 0; i < nf; ++i) {
            const double down = (freqs[i] - pts[m]) / (pts[m + 1] - pts[m]);
            const double up = (pts[m + 2] - freqs[i]) / (pts[m + 2] - pts[m + 1]);
            fb[std::size_t(m) * nf + i] = float(std::max(0.0, std::min(down, up)));
        }
    return fb;
}

// ── Network ──────────────────────────────────────────────────────────────────────────────────────────────
struct Tensor {
    int c = 0, h = 0, w = 0;
    std::vector<float> d;
    Tensor() = default;
    Tensor(int c_, int h_, int w_) : c(c_), h(h_), w(w_), d(std::size_t(c_) * h_ * w_, 0.f) {}
    float* ch(int i) { return d.data() + std::size_t(i) * h * w; }
    const float* ch(int i) const { return d.data() + std::size_t(i) * h * w; }
};

Tensor conv(const Tensor& x, const PulseWeights::Conv& L) {
    const int oh = (x.h + 2 * L.ph - L.kh) / L.sh + 1;
    const int ow = (x.w + 2 * L.pw - L.kw) / L.sw + 1;
    Tensor y(L.cout, oh, ow);
    // valid output columns [c0, c1) for each horizontal tap
    int c0[8], c1[8];
    for (int kx = 0; kx < L.kw && kx < 8; ++kx) {
        c0[kx] = 0;
        c1[kx] = ow;
        while (c0[kx] < ow && c0[kx] * L.sw + kx - L.pw < 0) ++c0[kx];
        while (c1[kx] > c0[kx] && (c1[kx] - 1) * L.sw + kx - L.pw >= x.w) --c1[kx];
    }
    // One output row at a time: the row being summed (ow floats) stays in the L1 cache while every input
    // channel and tap is added to it. (Tap by tap over the whole plane is the same sum in the same order,
    // but streams the plane through the cache cin x kh x kw times.)
    for (int o = 0; o < L.cout; ++o) {
        const float* wo = L.w + std::size_t(o) * L.cin * L.kh * L.kw;
        for (int r = 0; r < oh; ++r) {
            float* orow = y.ch(o) + std::size_t(r) * ow;
            std::fill(orow, orow + ow, L.b[o]);
            for (int i = 0; i < L.cin; ++i)
                for (int ky = 0; ky < L.kh; ++ky) {
                    const int iy = r * L.sh + ky - L.ph;
                    if (iy < 0 || iy >= x.h) continue;
                    const float* row = x.ch(i) + std::size_t(iy) * x.w;
                    for (int kx = 0; kx < L.kw; ++kx) {
                        const float wv = wo[(std::size_t(i) * L.kh + ky) * L.kw + kx];
                        if (wv == 0.f) continue;
                        if (L.sw == 1) {
                            const float* src = row + (c0[kx] + kx - L.pw);
                            for (int c = c0[kx]; c < c1[kx]; ++c) orow[c] += wv * src[c - c0[kx]];
                        } else {
                            for (int c = c0[kx]; c < c1[kx]; ++c) orow[c] += wv * row[c * L.sw + kx - L.pw];
                        }
                    }
                }
            if (L.relu) for (int c = 0; c < ow; ++c) orow[c] = std::max(orow[c], 0.f);
        }
    }
    return y;
}

// One 512-frame pass: mel chunk [128][512] -> logits [2][512]
void pass(const float* mel, int stride, std::vector<float>& out) {
    using namespace PulseWeights;
    Tensor x(1, kMels, kChunk);
    for (int m = 0; m < kMels; ++m) std::copy(mel + std::size_t(m) * stride, mel + std::size_t(m) * stride + kChunk, x.ch(0) + std::size_t(m) * kChunk);
    int li = 0;
    for (int s = 0; s < kStem; ++s) x = conv(x, kConvs[li++]);
    std::vector<Tensor> skips;
    for (int l = 0; l < kLevels; ++l) {
        skips.push_back(x);
        x = conv(x, kConvs[li++]);   // down (stride 2 in time)
        x = conv(x, kConvs[li++]);   // mid
    }
    for (int l = 0; l < kLevels; ++l) {
        const Tensor& s = skips.back();
        Tensor u(x.c, x.h, x.w * 2);                  // nearest 2x in time + skip
        for (int c = 0; c < x.c; ++c)
            for (int t = 0; t < u.w; ++t) u.ch(c)[t] = x.ch(c)[t / 2] + s.ch(c)[t];
        skips.pop_back();
        x = conv(u, kConvs[li++]);
    }
    x = conv(x, kConvs[li++]);                        // head: 2 x 1 x 512 logits
    out.assign(x.d.begin(), x.d.end());
}

// ── Peaks, grid fit, bar fit (pulse.peaks, pulse_grid.fit_grid, pulse_rave.bar_fit) ──────────────────────
std::vector<double> peaks(const std::vector<float>& l, double minGap) {
    std::vector<double> out;
    long last = -1000000;
    for (std::size_t i = 1; i + 1 < l.size(); ++i)
        if (l[i] > 0 && l[i] >= l[i - 1] && l[i] >= l[i + 1] && double(long(i) - last) / kFps >= minGap) {
            out.push_back(double(i) / kFps);
            last = long(i);
        }
    return out;
}

double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

void lineFit(const std::vector<double>& k, const std::vector<double>& t, double& slope, double& icpt) {
    const double n = double(k.size());
    const double mk = std::accumulate(k.begin(), k.end(), 0.0) / n, mt = std::accumulate(t.begin(), t.end(), 0.0) / n;
    double sxy = 0, sxx = 0;
    for (std::size_t i = 0; i < k.size(); ++i) { sxy += (k[i] - mk) * (t[i] - mt); sxx += (k[i] - mk) * (k[i] - mk); }
    slope = sxy / sxx;
    icpt = mt - slope * mk;
}

// ── Phase refinement (Options::refinePhase; tools/needle/pulse_phase.py is the reference) ────────────────────
// The network marks beats on a 20-ms frame lattice and learned where "the beat" is from labels that do not
// agree with each other, so its grid sits a few ms early on some music and late on other. The drum hits are in
// the audio:
//   1. three bands of the 22.05 kHz signal (below 150 Hz, 150-2000 Hz, above 2 kHz; 4th-order Butterworth run
//      forwards then backwards, so nothing is delayed), each as power per block of 10 samples
//   2. per band: power smoothed (26 blocks below 150 Hz, 13 above), log10, its rise over 10 blocks (4.5 ms);
//      the onset function is the three rises added
//   3. around every grid beat: the highest point of the onset function within +-40 ms, its time to a fraction
//      of a block (parabola through three points) and its height
//   4. the beats whose height is at least 0.3 x the 90th percentile count, weighted by height:
//      offset = weighted median of (onset - beat), spread = weighted median of |onset - beat - offset|
//   5. the grid moves by offset + kPhaseShift, unless the track is faster than 150 BPM (drum & bass: the
//      strongest onset near a beat is often a grace note before it) or the spread is over 6 ms
// kPhaseShift is the convention: with it refined grids sit, in the median, where hand-made (rekordbox) grids do.
constexpr std::size_t kPhaseBlock = 10;
constexpr double kPhaseFs = double(kSampleRate) / double(kPhaseBlock);   // 2205 blocks per second
constexpr std::size_t kPhaseLag = 10;
constexpr double kPhaseCentre = double(kPhaseLag) / 2.0 + 0.3;   // onset[n] belongs to the time (n + this) / kPhaseFs
constexpr double kPhaseWindow = 0.040;
constexpr double kPhaseTau = 0.3;
constexpr double kPhaseShift = 0.0023;
constexpr double kPhaseMaxBpm = 150.0;
constexpr double kPhaseMaxSpread = 0.006;
constexpr std::size_t kPhaseChunk = 6 * kSampleRate;             // samples filtered in one go (whole blocks)
constexpr std::size_t kPhaseSpare = kSampleRate / 5;             // ... with this many more on either side

struct Biquad { double b0, b1, b2, a1, a2; };
// scipy.signal.butter(4, ..., fs=22050, output="sos"): low-pass 150 Hz, band-pass 150-2000 Hz, high-pass 2 kHz
const Biquad kBandLow[] = {
    {1.9739622322682168e-07, 3.9479244645364335e-07, 1.9739622322682168e-07, -1.9222869522443091, 0.9240442445437955},
    {1.0, 2.0, 1.0, -1.966024963538341, 0.9678222397072245}};
const Biquad kBandMid[] = {
    {0.002637848635736277, 0.005275697271472554, 0.002637848635736277, -1.2263990502000652, 0.3987476961287795},
    {1.0, 2.0, 1.0, -1.4371016687604707, 0.697151142890563},
    {1.0, -2.0, 1.0, -1.9140479261474708, 0.9163350658907653},
    {1.0, -2.0, 1.0, -1.9694043527121972, 0.9712662382021913}};
const Biquad kBandHigh[] = {
    {0.4691666404516789, -0.9383332809033578, 0.4691666404516789, -1.123742921678533, 0.33468591437747025},
    {1.0, -2.0, 1.0, -1.3957215810419232, 0.6577189485890346}};
struct Band { const Biquad* sos; int sections; bool everyTenth; int smooth; };
// Below 150 Hz the power is every 10th sample squared, smoothed over 26 blocks (as tools/needle beatgrid_lab.py
// finds kicks); above, the mean square of each 10 samples, smoothed over 13.
const Band kBands[3] = {{kBandLow, 2, true, 26}, {kBandMid, 4, false, 13}, {kBandHigh, 2, false, 13}};

// x -> per band, its power per block. Each band's filter runs forwards then backwards over the signal, from
// rest. The signal is taken in chunks of 6 s with 0.2 s more on either side: what a filter still carries of
// an input 0.2 s old is below 1e-27 of it, so the result is that of one pass over the whole signal, and the
// working memory is one chunk per thread instead of the track three times over.
void bandPowers(const std::vector<float>& x, int threads, std::vector<float> power[3]) {
    const std::size_t n = x.size();
    power[0].assign((n + kPhaseBlock - 1) / kPhaseBlock, 0.f);
    power[1].assign(n / kPhaseBlock, 0.f);
    power[2].assign(n / kPhaseBlock, 0.f);
    parallelFor(int((n + kPhaseChunk - 1) / kPhaseChunk), threads, [&](int chunk) {
        const std::size_t c0 = std::size_t(chunk) * kPhaseChunk, c1 = std::min(n, c0 + kPhaseChunk);
        const std::size_t e0 = c0 > kPhaseSpare ? c0 - kPhaseSpare : 0, e1 = std::min(n, c1 + kPhaseSpare);
        std::vector<float> y(e1 - e0);
        for (int b = 0; b < 3; ++b) {
            const Band& band = kBands[b];
            double z1[4], z2[4];
            const auto step = [&](double v) {                     // direct form II transposed, as scipy's sosfilt
                for (int s = 0; s < band.sections; ++s) {
                    const Biquad& q = band.sos[s];
                    const double out = q.b0 * v + z1[s];
                    z1[s] = q.b1 * v - q.a1 * out + z2[s];
                    z2[s] = q.b2 * v - q.a2 * out;
                    v = out;
                }
                return float(v);
            };
            std::fill(z1, z1 + 4, 0.0);
            std::fill(z2, z2 + 4, 0.0);
            for (std::size_t i = e0; i < e1; ++i) y[i - e0] = step(x[i]);
            std::fill(z1, z1 + 4, 0.0);
            std::fill(z2, z2 + 4, 0.0);
            for (std::size_t i = e1 - e0; i-- > 0;) y[i] = step(y[i]);
            std::vector<float>& p = power[b];
            for (std::size_t k = c0 / kPhaseBlock; k < p.size() && k * kPhaseBlock < c1; ++k) {
                const float* s = y.data() + (k * kPhaseBlock - e0);
                double sum = double(s[0]) * s[0];
                if (!band.everyTenth) {
                    for (std::size_t j = 1; j < kPhaseBlock; ++j) sum += double(s[j]) * s[j];
                    sum /= double(kPhaseBlock);
                }
                p[k] = float(sum);
            }
        }
    });
}

// value at the q-th quantile (0..1), linear between neighbours (numpy.percentile's default)
template <class T>
double quantile(std::vector<T> v, double q) {
    if (v.empty()) return 0.0;
    const double pos = q * double(v.size() - 1);
    const std::size_t lo = std::size_t(pos);
    std::nth_element(v.begin(), v.begin() + long(lo), v.end());
    const double a = double(v[lo]);
    if (lo + 1 >= v.size()) return a;
    const double b = double(*std::min_element(v.begin() + long(lo) + 1, v.end()));
    return a + (pos - double(lo)) * (b - a);
}

// A band's power per block -> in its place, how much the log10 of the smoothed power rises over kPhaseLag
// blocks, never below 0. Element i is the rise from block i to block i + kPhaseLag.
void riseOf(std::vector<float>& p, int smooth) {
    const long n = long(p.size()), ahead = (smooth - 1) / 2;      // numpy.convolve(p, ones(smooth) / smooth, "same")
    std::vector<float> e(p.size());
    for (long i = 0; i < n; ++i) {
        double sum = 0;
        for (long j = std::max(0L, i + ahead + 1 - smooth), end = std::min(n, i + ahead + 1); j < end; ++j)
            sum += double(p[std::size_t(j)]);
        e[std::size_t(i)] = float(sum / smooth);
    }
    const double floor = 1e-4 * quantile(e, 0.99) + 1e-20;       // 40 dB under the loud passages
    for (float& v : e) v = float(std::log10(double(v) + floor));
    p.resize(p.size() > kPhaseLag ? p.size() - kPhaseLag : 0);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = std::max(0.f, e[i + kPhaseLag] - e[i]);
}

std::vector<float> onsetFunction(const std::vector<float>& x, int threads) {
    std::vector<float> band[3];
    bandPowers(x, threads, band);
    parallelFor(3, threads, [&](int b) { riseOf(band[b], kBands[b].smooth); });
    std::vector<float> o = std::move(band[0]);
    o.resize(std::min({o.size(), band[1].size(), band[2].size()}));
    for (std::size_t i = 0; i < o.size(); ++i) o[i] += band[1][i] + band[2][i];
    return o;
}

// Per grid beat inside the onset function o (which starts at t0 seconds of the track): onset time minus beat
// time, and the onset's height. skipStart / skipEnd: seconds left out at the ends of a window cut out of the
// track (the filters start from rest there).
void beatOnsets(const std::vector<float>& o, double t0, double bpm, double first, double skipStart, double skipEnd,
                std::vector<double>& d, std::vector<double>& h) {
    const double period = 60.0 / bpm;
    const long half = std::lround(kPhaseWindow * kPhaseFs), len = long(o.size());
    const double local = first - t0;                              // a grid beat, on this window's clock
    const long k0 = long(std::ceil((kPhaseWindow + 2 / kPhaseFs + skipStart - local) / period));
    const long k1 = long(std::floor((double(len - 2) / kPhaseFs - kPhaseWindow - skipEnd - local) / period));
    for (long k = k0; k <= k1; ++k) {
        const double b = local + double(k) * period;
        const long c = long(std::nearbyint(b * kPhaseFs - kPhaseCentre));     // o's index nearest to the beat
        if (c - half < 0 || c + half >= len) continue;
        long j = c - half;
        for (long i = c - half + 1; i <= c + half; ++i)
            if (o[std::size_t(i)] > o[std::size_t(j)]) j = i;
        if (j == c - half || j == c + half) continue;             // a rise that peaks outside the window
        const double y0 = o[std::size_t(j - 1)], y1 = o[std::size_t(j)], y2 = o[std::size_t(j + 1)];
        const double den = y0 - 2 * y1 + y2;
        const double frac = den < 0 ? 0.5 * (y0 - y2) / den : 0.0;
        d.push_back((double(j) + kPhaseCentre + frac) / kPhaseFs - b);
        h.push_back(y1);
    }
}

double weightedMedian(const std::vector<double>& v, const std::vector<double>& w) {
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    const double half = std::accumulate(w.begin(), w.end(), 0.0) / 2;
    double c = 0;
    for (std::size_t i : idx) {
        c += w[i];
        if (c >= half) return v[i];
    }
    return v[idx.back()];
}

PhaseFit phaseSolve(const std::vector<double>& d, const std::vector<double>& h) {
    PhaseFit fit;
    if (d.size() < 8) return fit;
    const double need = kPhaseTau * quantile(h, 0.9);
    std::vector<double> dk, wk;
    for (std::size_t i = 0; i < d.size(); ++i)
        if (h[i] >= need) { dk.push_back(d[i]); wk.push_back(h[i]); }
    if (dk.size() < 8) return fit;
    const double offset = weightedMedian(dk, wk);
    std::vector<double> dev(dk.size());
    for (std::size_t i = 0; i < dk.size(); ++i) dev[i] = std::abs(dk[i] - offset);
    fit.offset = offset + kPhaseShift;
    fit.spread = weightedMedian(dev, wk);
    fit.beats = int(dk.size());
    return fit;
}

}  // namespace

PhaseFit phaseFit(const std::vector<float>& mono22k, double bpm, double firstBeatSeconds, int threads, double startSeconds) {
    std::vector<double> d, h;
    if (bpm > 0) beatOnsets(onsetFunction(mono22k, threads), startSeconds, bpm, firstBeatSeconds, 0.0, 0.0, d, h);
    return phaseSolve(d, h);
}

bool applyPhase(Result& r, const PhaseFit& fit) {
    r.phaseOffset = fit.offset;
    r.phaseSpread = fit.spread;
    r.phaseBeats = fit.beats;
    r.phaseShift = 0;
    if (!r.ok || !(r.bpm > 0) || fit.beats < 8 || r.bpm > kPhaseMaxBpm || fit.spread > kPhaseMaxSpread) return false;
    const double bar = 4 * 60.0 / r.bpm;
    double anchor = r.firstBeatSeconds + fit.offset;
    anchor -= std::floor(anchor / bar) * bar;                  // still the first bar line of the track
    r.firstBeatSeconds = anchor;
    r.phaseShift = fit.offset;
    return true;
}

std::vector<float> resampleTo22050(const std::vector<float>& mono, int sampleRate, int threads) {
    return resample(mono, sampleRate, kSampleRate, threads);
}

std::vector<float> logMel(const std::vector<float>& x, int& frames, int threads) {
    static const std::vector<float> fb = melBank();
    static const Fft fft(kNfft);
    // each mel band is a triangle over a few FFT bins: [first, last) per band instead of all 513
    static const std::vector<int> band = [] {
        const int bins = kNfft / 2 + 1;
        std::vector<int> b(std::size_t(kMels) * 2, 0);
        for (int m = 0; m < kMels; ++m) {
            int first = bins, last = 0;
            for (int i = 0; i < bins; ++i)
                if (fb[std::size_t(m) * bins + i] != 0.f) { first = std::min(first, i); last = i + 1; }
            b[std::size_t(m) * 2] = std::min(first, last);
            b[std::size_t(m) * 2 + 1] = last;
        }
        return b;
    }();
    const int nf = kNfft / 2 + 1, pad = kNfft / 2;
    if (x.empty()) { frames = 0; return {}; }          // nothing to reflect-pad
    frames = int(x.size() / kHop) + 1;                 // torch.stft, center=True
    std::vector<double> win(kNfft);
    for (int i = 0; i < kNfft; ++i) win[i] = 0.5 - 0.5 * std::cos(2 * kPi * i / kNfft);   // periodic Hann
    const long n = long(x.size());
    auto at = [&](long i) -> double {                  // reflect padding
        if (n == 1) return x[0];
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
        return x[std::size_t(i)];
    };
    std::vector<float> mel(std::size_t(kMels) * frames);
    const double norm = 1.0 / std::sqrt(double(kNfft));   // normalized="frame_length"
    constexpr int kBlock = 256;
    const int nFrames = frames;
    parallelFor((nFrames + kBlock - 1) / kBlock, threads, [&](int blk) {
        std::vector<std::complex<double>> buf(kNfft);
        std::vector<double> mag(nf);
        for (int f = blk * kBlock; f < std::min(nFrames, (blk + 1) * kBlock); ++f) {
            const long s = long(f) * kHop - pad;
            for (int i = 0; i < kNfft; ++i) buf[i] = at(s + i) * win[i];
            fft.run(buf);
            for (int i = 0; i < nf; ++i) mag[i] = std::abs(buf[i]) * norm;
            for (int m = 0; m < kMels; ++m) {
                const float* row = fb.data() + std::size_t(m) * nf;
                double acc = 0;
                for (int i = band[std::size_t(m) * 2]; i < band[std::size_t(m) * 2 + 1]; ++i) acc += row[i] * mag[i];
                mel[std::size_t(m) * nFrames + f] = float(std::log1p(1000.0 * acc));
            }
        }
    });
    return mel;
}

void network(const std::vector<float>& mel, int n, std::vector<float>& beat, std::vector<float>& down, int threads) {
    // pulse.predict_track: 32 frames of zero padding in front, chunks of 512 every 448 frames, linear
    // 64-frame crossfades, then the padding removed.
    const int padR = ((64 - n % 64) % 64) + kChunk;
    const int W = 32 + n + padR;
    std::vector<float> x(std::size_t(kMels) * W, 0.f);
    for (int m = 0; m < kMels; ++m) std::copy(mel.begin() + std::size_t(m) * n, mel.begin() + std::size_t(m + 1) * n, x.begin() + std::size_t(m) * W + 32);
    std::vector<float> acc(std::size_t(2) * W, 0.f), wsum(W, 0.f), w(kChunk, 1.f);
    const int ov = kChunk - kChunkHop;
    for (int i = 0; i < ov; ++i) {
        w[i] = float(double(i + 1) / (ov + 1));
        w[kChunk - ov + i] = float(1.0 - double(i + 1) / (ov + 1));
    }
    std::vector<int> starts;
    for (int a = 0; a + kChunk <= W; a += kChunkHop) starts.push_back(a);
    std::vector<std::vector<float>> ys(starts.size());            // the passes are independent: in parallel
    parallelFor(int(starts.size()), threads, [&](int i) { pass(x.data() + starts[std::size_t(i)], W, ys[std::size_t(i)]); });
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const int a = starts[i];
        const auto& yy = ys[i];
        for (int t = 0; t < kChunk; ++t) {
            acc[a + t] += yy[t] * w[t];
            acc[std::size_t(W) + a + t] += yy[kChunk + t] * w[t];
            wsum[a + t] += w[t];
        }
    }
    beat.resize(n);
    down.resize(n);
    for (int t = 0; t < n; ++t) {
        const float s = std::max(wsum[32 + t], 1e-6f);
        beat[t] = acc[32 + t] / s;
        down[t] = acc[std::size_t(W) + 32 + t] / s;
    }
}

Result analyze(const float* pcm, std::size_t frames, int channels, int sampleRate, double integerSnapBpm,
               Options opt) {
    Result r;
    if (!pcm || frames == 0 || channels < 1 || sampleRate <= 0) return r;
    int threads = opt.threads;
    if (threads <= 0) threads = std::clamp(int(std::thread::hardware_concurrency()) - 1, 1, 3);
    const double seconds = double(frames) / sampleRate;
    const int n = int(std::floor(seconds * kSampleRate) / kHop) + 1;   // frames of the whole track at 50 fps
    r.beatLogit.assign(std::size_t(n), -20.f);                         // no beat outside the analysed windows
    r.downLogit.assign(std::size_t(n), -20.f);
    // Windows to analyse: the whole track, or (opt.maxSeconds) three equal windows centred at 20 / 50 / 80 %
    // of the track. A tempo + phase line fitted across windows minutes apart is as precise as one through
    // the whole track; the bar line is voted over the beats in all three.
    std::vector<std::pair<double, double>> win;
    if (opt.maxSeconds > 0 && seconds > opt.maxSeconds + 30) {
        const double len = opt.maxSeconds / 3;
        for (double c : {0.2, 0.5, 0.8}) {
            const double a = std::clamp(c * seconds - len / 2, 0.0, seconds - len);
            win.push_back({a, a + len});
        }
    } else {
        win.push_back({0.0, seconds});
    }
    std::vector<std::vector<float>> onsets;                            // refinePhase: per analysed window
    std::vector<double> onsetStart;
    std::vector<std::pair<double, double>> onsetCut;
    for (const auto& [a, b] : win) {
        // whole 441-sample frames at 22.05 kHz, so the window's frames land exactly on the track's frame grid
        const long f0 = long(std::floor(a * kFps)), f1 = std::min<long>(n, long(std::ceil(b * kFps)));
        if (f1 - f0 < 64) continue;
        const std::size_t s0 = std::size_t(std::llround(double(f0) * kHop / kSampleRate * sampleRate));
        const std::size_t s1 = std::min(frames, std::size_t(std::llround(double(f1) * kHop / kSampleRate * sampleRate)));
        std::vector<float> mono(s1 - s0);
        for (std::size_t i = s0; i < s1; ++i) {
            double v = 0;
            for (int c = 0; c < channels; ++c) v += pcm[i * channels + c];
            mono[i - s0] = float(v / channels);
        }
        const auto x = resampleTo22050(mono, sampleRate, threads);
        if (opt.refinePhase) {
            onsets.push_back(onsetFunction(x, threads));
            onsetStart.push_back(double(f0) / kFps);
            onsetCut.push_back({f0 > 0 ? 0.25 : 0.0, f1 < n ? 0.25 : 0.0});
        }
        int m = 0;
        const auto mel = logMel(x, m, threads);
        std::vector<float> bl, dl;
        network(mel, m, bl, dl, threads);
        // the first / last 2 s of a cut window have no context on one side: keep only its inner part
        // (except at the track's own start and end)
        const int edge = 2 * kFps;
        for (int t = 0; t < m && f0 + t < n; ++t) {
            if ((f0 > 0 && t < edge) || (f0 + m < n - 1 && t >= m - edge)) continue;
            r.beatLogit[std::size_t(f0 + t)] = bl[std::size_t(t)];
            r.downLogit[std::size_t(f0 + t)] = dl[std::size_t(t)];
        }
        r.analysedSeconds += double(m) / kFps;
    }

    fitGrid(r, integerSnapBpm, opt.doubleTimeRule);
    if (opt.refinePhase && r.ok && r.steady) {
        std::vector<double> d, h;
        for (std::size_t i = 0; i < onsets.size(); ++i)
            beatOnsets(onsets[i], onsetStart[i], r.bpm, r.firstBeatSeconds, onsetCut[i].first, onsetCut[i].second, d, h);
        applyPhase(r, phaseSolve(d, h));
    }
    return r;
}

void fitGrid(Result& r, double integerSnapBpm, bool doubleTimeRule) {
    const int n = int(r.beatLogit.size());
    // grid: tempo hint from the peaks, fine phasor search +-4 % in 0.005-BPM steps, line fit on the peaks
    const auto pk = peaks(r.beatLogit, 0.07);
    r.beatPeaks = int(pk.size());
    if (pk.size() < 16) return;
    std::vector<double> iv(pk.size() - 1);
    for (std::size_t i = 1; i < pk.size(); ++i) iv[i - 1] = pk[i] - pk[i - 1];
    std::nth_element(iv.begin(), iv.begin() + iv.size() / 2, iv.end());
    double med = iv[iv.size() / 2];
    if (iv.size() % 2 == 0) {                           // numpy median of an even count: mean of the middle two
        const double lo = *std::max_element(iv.begin(), iv.begin() + iv.size() / 2);
        med = 0.5 * (med + lo);
    }
    const double hint = 60.0 / med;
    std::vector<double> wgt(pk.size());
    for (std::size_t i = 0; i < pk.size(); ++i) {
        const auto f = std::min<std::size_t>(std::size_t(std::lround(pk[i] * kFps)), r.beatLogit.size() - 1);
        const double p = sigmoid(r.beatLogit[f]);
        wgt[i] = p * p;
    }
    double bestMag = -1, bestBpm = hint, bestAng = 0;
    const int nCand = int(std::ceil((hint * 1.04 - hint * 0.96) / 0.005));
    for (int c = 0; c < nCand; ++c) {
        const double bpm = hint * 0.96 + c * 0.005;
        double re = 0, im = 0;
        for (std::size_t i = 0; i < pk.size(); ++i) {
            const double a = 2 * kPi * pk[i] * bpm / 60.0;
            re += wgt[i] * std::cos(a);
            im += wgt[i] * std::sin(a);
        }
        const double mag = std::hypot(re, im);
        if (mag > bestMag) { bestMag = mag; bestBpm = bpm; bestAng = std::atan2(im, re); }
    }
    double period = 60.0 / bestBpm;
    double phase = std::fmod(bestAng / (2 * kPi) * period, period);
    if (phase < 0) phase += period;
    // a peak is "on the grid" when it is within 35 ms of the line; k = its beat number on that line
    const auto onGrid = [&](double t, double& k) {
        k = std::round((t - phase) / period);
        return std::abs(t - (phase + k * period)) < 0.035;
    };
    for (int it = 0; it < 3; ++it) {
        std::vector<double> ks, ts;
        for (double t : pk) {
            double k;
            if (onGrid(t, k)) { ks.push_back(k); ts.push_back(t); }
        }
        if (ks.size() < 16) break;
        lineFit(ks, ts, period, phase);
    }
    std::size_t on = 0;
    for (double t : pk) {
        double k;
        if (onGrid(t, k)) ++on;
    }
    r.onGridShare = double(on) / pk.size();
    r.bpm = r.rawBpm = 60.0 / period;
    r.steady = r.onGridShare >= 0.6;
    if (!r.steady || !(period > 0)) return;
    const double whole = std::round(r.rawBpm);
    if (integerSnapBpm > 0 && whole > 0 && std::abs(whole - r.rawBpm) <= integerSnapBpm) {
        // a whole-BPM grid: keep the period exact and refit only the phase on the peaks near the line
        const double p2 = 60.0 / whole;
        double sum = 0;
        int cnt = 0;
        for (double t : pk) {
            double k;
            if (onGrid(t, k)) { sum += t - k * p2; ++cnt; }
        }
        if (cnt >= 16) {
            period = p2;
            phase = sum / cnt;
            r.bpm = whole;
            r.integerTempo = true;
        }
    }

    // grid beats over the span with beats
    std::vector<double> grid;
    const double first = phase - std::floor(phase / period) * period;
    for (double t = first; t <= double(n) / kFps; t += period)
        if (t >= pk.front() - period / 2 && t <= pk.back() + period / 2) grid.push_back(t);
    if (grid.size() < 8) return;

    // Double time. On slow tracks (below about 105 BPM) the network often marks the off-beats as beats too,
    // and the grid comes out at twice the tempo. Then every other grid beat is weaker: when one half of the
    // grid beats averages under 75 % of the other half's activation, the weaker half is dropped. Measured
    // (PATCH_NOTE): of the 194 gridded DJ test tracks one changes, 240 -> 120 BPM, the human tempo; on 459
    // mixed-genre tracks 40 of 115 double-time grids are corrected and 3 fast tracks are wrongly halved.
    {
        double sum[2] = {0, 0};
        int cnt[2] = {0, 0};
        for (std::size_t i = 0; i < grid.size(); ++i) {
            const long f = std::lround(grid[i] * kFps);
            float top = -100.f;
            for (long k = std::max(0L, f - 1); k <= std::min<long>(n - 1, f + 1); ++k) top = std::max(top, r.beatLogit[std::size_t(k)]);
            sum[i % 2] += sigmoid(top);
            ++cnt[i % 2];
        }
        const double even = sum[0] / std::max(1, cnt[0]), odd = sum[1] / std::max(1, cnt[1]);
        r.beatParity = std::max(even, odd) > 0 ? std::min(even, odd) / std::max(even, odd) : 1.0;
        r.gridActivation = (sum[0] + sum[1]) / std::max(1, cnt[0] + cnt[1]);
        if (doubleTimeRule && r.beatParity < kDoubleTimeParity && grid.size() >= 32) {
            std::vector<double> half;
            for (std::size_t i = even >= odd ? 0 : 1; i < grid.size(); i += 2) half.push_back(grid[i]);
            grid.swap(half);
            period *= 2;
            r.bpm *= 0.5;
            r.rawBpm *= 0.5;
            r.integerTempo = r.integerTempo && r.bpm == std::round(r.bpm);
            r.halved = true;
            r.gridActivation = std::max(even, odd);          // of the beats that stay
        }
    }

    // bar line: downbeat activation at each grid beat, same bar position summed over 32 bars either side
    const int bars = 32;
    std::vector<double> a(grid.size()), s(grid.size(), 0.0);
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const auto f = std::min<std::size_t>(std::size_t(std::max(0L, std::lround(grid[i] * kFps))), r.downLogit.size() - 1);
        a[i] = sigmoid(r.downLogit[f]);
    }
    for (long i = 0; i < long(grid.size()); ++i)
        for (int m = -bars; m <= bars; ++m) {
            const long j = i + 4L * m;
            if (j >= 0 && j < long(grid.size())) s[std::size_t(i)] += a[std::size_t(j)];
        }
    int votes[4] = {0, 0, 0, 0}, total = 0;
    for (long i = 0; i < long(grid.size()); ++i) {
        bool best = true;
        for (int d = -3; d <= 3 && best; ++d) {
            if (d == 0) continue;
            const long j = i + d;
            if (j >= 0 && j < long(grid.size()) && !(s[std::size_t(i)] > s[std::size_t(j)])) best = false;
        }
        if (best) { ++votes[i % 4]; ++total; }
    }
    const int barPos = int(std::max_element(votes, votes + 4) - votes);
    r.barPhaseShare = total ? double(votes[barPos]) / total : 0.0;
    int runnerUp = 0;
    for (int i = 0; i < 4; ++i)
        if (i != barPos) runnerUp = std::max(runnerUp, votes[i]);
    r.barVoteMargin = total ? double(votes[barPos] - runnerUp) / total : 0.0;
    // anchor: the first grid beat on the majority bar position, moved back by whole bars to the track start
    const double bar = 4 * period;
    double anchor = grid[std::size_t(barPos)];
    anchor -= std::floor(anchor / bar) * bar;
    r.firstBeatSeconds = anchor;
    r.ok = true;
}

}  // namespace PulseBeat
