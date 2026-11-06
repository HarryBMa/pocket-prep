/*
 * Needle STFT/iSTFT benchmark for the Renesas RA8P1 (Cortex-M85 @ 1 GHz, Helium).
 *
 * Question it answers: how much of the M85 does Needle's front and back end cost for two decks?
 * The NPU runs the network; the M85 has to do, per deck and per hop of 512 samples:
 *   - window + forward real FFT (2048 points) for left and right             -> 2 RFFT
 *   - turn the NPU's masks into gains: dequantise int8, normalise 4 sources to sum to 1
 *   - mask x complex spectrum for 4 sources x 2 channels
 *   - inverse real FFT for each of the 8 stems/channels + window + overlap-add -> 8 IRFFT
 * Two decks: 20 transforms per hop, 86.13 hops per second (44.1 kHz / 512).
 *
 * Portable C on CMSIS-DSP (float32: the M85's Helium unit has float32 vectors, and float32 keeps
 * the inverse FFTs free of audible fixed-point noise). Build:
 *   - on the RA8P1 with Renesas FSP: add CMSIS-DSP, call needle_stft_bench() from hal_entry(),
 *     -DNEEDLE_TARGET_M85 (cycle counter), -O3 -mcpu=cortex-m85 -mfloat-abi=hard (see README.md);
 *   - on a PC to check correctness: see README.md (no cycle counter there, uses clock()).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_math.h"

#define NFFT     2048
#define HOP      512
#define BINS     1024            /* bins the network sees; bin 1024 (Nyquist) goes to "other" */
#define SOURCES  4
#define CHANNELS 2
#define DECKS    2
#define OTHER    2               /* drums, bass, other, vocals */
#define HOPS_PER_SECOND (44100.0f / HOP)

/* ── cycle counter ──────────────────────────────────────────────────────── */
#ifdef NEEDLE_TARGET_M85
#include "bsp_api.h"             /* FSP: brings in the CMSIS core header for the M85 */
static void     timer_init(void) { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; }
static uint32_t timer_now(void)  { return DWT->CYCCNT; }
static const char *timer_unit = "cycles";
#else
#include <time.h>
static void     timer_init(void) {}
static uint32_t timer_now(void)  { return (uint32_t)(clock() * (1000000.0 / CLOCKS_PER_SEC)); }
static const char *timer_unit = "us (host clock, not M85 cycles)";
#endif

/* ── state: everything static, nothing allocated at run time ─────────────── */
typedef struct {
    float32_t in_hist[CHANNELS][NFFT];                  /* last 2048 input samples per channel */
    float32_t spec[CHANNELS][NFFT];                     /* packed RFFT output (CMSIS layout)   */
    float32_t ola[SOURCES][CHANNELS][NFFT];             /* overlap-add accumulators            */
    int8_t    mask_q[SOURCES][CHANNELS][BINS];          /* the NPU's output for one frame      */
} deck_t;

static deck_t    decks[DECKS];
static float32_t window[NFFT], synth_window[NFFT];
static float32_t tmp[NFFT], tmp2[NFFT], gain[SOURCES][BINS];
static float32_t out[DECKS][SOURCES][CHANNELS][HOP];   /* finished stem samples of this hop   */
static arm_rfft_fast_instance_f32 rfft;

volatile struct {                                      /* readable in a debugger without UART */
    uint32_t hops, avg, worst, total;
    float    m85_percent_at_1ghz;
    float    reconstruction_error;
} needle_bench_result;

static void init(void) {
    arm_rfft_fast_init_f32(&rfft, NFFT);
    /* periodic Hann, like torch.hann_window; synthesis window scaled so analysis x synthesis
       overlap-adds to 1 at hop 512 (4x overlap: sum of hann^2 = 1.5) */
    for (int n = 0; n < NFFT; n++) {
        window[n] = 0.5f - 0.5f * cosf(2.0f * PI * n / NFFT);
        synth_window[n] = window[n] / 1.5f;
    }
    memset(decks, 0, sizeof decks);
}

/* One hop for one deck: new_in[ch][HOP] in, out[deck] filled. Masks are whatever the NPU wrote. */
static void deck_hop(deck_t *d, const float32_t new_in[CHANNELS][HOP], float32_t o[SOURCES][CHANNELS][HOP]) {
    /* 1. slide the input history and analyse both channels */
    for (int c = 0; c < CHANNELS; c++) {
        memmove(d->in_hist[c], d->in_hist[c] + HOP, (NFFT - HOP) * sizeof(float32_t));
        memcpy(d->in_hist[c] + NFFT - HOP, new_in[c], HOP * sizeof(float32_t));
        arm_mult_f32(d->in_hist[c], window, tmp, NFFT);
        arm_rfft_fast_f32(&rfft, tmp, d->spec[c], 0);          /* tmp is overwritten (scratch) */
    }
    /* 2.-4. per channel: masks -> gains (sum to 1 over sources), apply, inverse, overlap-add */
    for (int c = 0; c < CHANNELS; c++) {
        /* dequantise (scale 1/256, as the NPU's sigmoid output) and normalise over 4 sources */
        for (int k = 0; k < BINS; k++) {
            float32_t m[SOURCES], sum = 1e-4f;
            for (int s = 0; s < SOURCES; s++) { m[s] = (d->mask_q[s][c][k] + 128) * (1.0f / 256.0f); sum += m[s]; }
            float32_t inv = 1.0f / sum;
            for (int s = 0; s < SOURCES; s++) gain[s][k] = m[s] * inv;
        }
        for (int s = 0; s < SOURCES; s++) {
            /* CMSIS packed layout: [DC, Nyquist, re1, im1, ...]: pair 0 gets the DC gain on DC
               and 1 (other) or 0 on Nyquist, the rest are complex x real gain */
            arm_cmplx_mult_real_f32(d->spec[c] + 2, gain[s] + 1, tmp + 2, BINS - 1);
            tmp[0] = d->spec[c][0] * gain[s][0];
            tmp[1] = (s == OTHER) ? d->spec[c][1] : 0.0f;
            arm_rfft_fast_f32(&rfft, tmp, tmp2, 1);                 /* inverse, scaled by 1/N */
            arm_mult_f32(tmp2, synth_window, tmp2, NFFT);
            arm_add_f32(d->ola[s][c], tmp2, d->ola[s][c], NFFT);
            memcpy(o[s][c], d->ola[s][c], HOP * sizeof(float32_t));  /* first HOP samples are final */
            memmove(d->ola[s][c], d->ola[s][c] + HOP, (NFFT - HOP) * sizeof(float32_t));
            memset(d->ola[s][c] + NFFT - HOP, 0, HOP * sizeof(float32_t));
        }
    }
}

/* Runs `hops` hops of two decks on a test signal, returns the result struct (also in
   needle_bench_result). With masks all equal, the four stems must add up to the input, delayed
   by NFFT - HOP samples; the reconstruction error checks the arithmetic. */
void needle_stft_bench(uint32_t hops) {
    static float32_t in[DECKS][CHANNELS][HOP];
    static float32_t history[CHANNELS][NFFT * 4];   /* input kept for the reconstruction check */
    init();
    timer_init();
    for (int dk = 0; dk < DECKS; dk++)
        memset(decks[dk].mask_q, 0, sizeof decks[dk].mask_q);  /* equal masks: 0.5 each -> 1/4 each */
    uint32_t total = 0, worst = 0;
    float32_t err = 0.0f, ref_energy = 1e-9f;
    for (uint32_t h = 0; h < hops; h++) {
        for (int dk = 0; dk < DECKS; dk++)
            for (int c = 0; c < CHANNELS; c++)
                for (int n = 0; n < HOP; n++) {
                    float32_t t = (float32_t)(h * HOP + n) / 44100.0f;
                    in[dk][c][n] = 0.3f * sinf(2 * PI * (110.0f + 220.0f * c) * t) + 0.1f * sinf(2 * PI * 3520.0f * t + dk);
                }
        uint32_t t0 = timer_now();
        for (int dk = 0; dk < DECKS; dk++)
            deck_hop(&decks[dk], (const float32_t (*)[HOP])in[dk], out[dk]);
        uint32_t dt = timer_now() - t0;
        total += dt;
        if (dt > worst) worst = dt;
        /* reconstruction check on deck 0, channel 0 (after the pipeline has filled) */
        memmove(history[0], history[0] + HOP, (NFFT * 4 - HOP) * sizeof(float32_t));
        memcpy(history[0] + NFFT * 4 - HOP, in[0][0], HOP * sizeof(float32_t));
        if (h >= 8) {
            for (int n = 0; n < HOP; n++) {
                float32_t sum = 0;
                for (int s = 0; s < SOURCES; s++) sum += out[0][s][0][n];
                float32_t ref = history[0][NFFT * 4 - HOP - (NFFT - HOP) + n];
                err += (sum - ref) * (sum - ref);
                ref_energy += ref * ref;
            }
        }
    }
    needle_bench_result.hops = hops;
    needle_bench_result.total = total;
    needle_bench_result.avg = total / hops;
    needle_bench_result.worst = worst;
    /* on the M85 avg is in cycles: share of a 1 GHz core = cycles per hop x hops per second / 1e9 */
    needle_bench_result.m85_percent_at_1ghz = 100.0f * (float)needle_bench_result.avg * HOPS_PER_SECOND / 1e9f;
    needle_bench_result.reconstruction_error = sqrtf(err / ref_energy);
    printf("needle stft bench: %lu hops x %d decks, avg %lu %s per hop, worst %lu, "
           "-> %.2f %% of a 1 GHz M85 (valid on target only); stems-sum-to-mix error %.2e\n",
           (unsigned long)hops, DECKS, (unsigned long)needle_bench_result.avg, timer_unit,
           (unsigned long)worst, needle_bench_result.m85_percent_at_1ghz, needle_bench_result.reconstruction_error);
}

#ifndef NEEDLE_TARGET_M85
int main(void) { needle_stft_bench(400); return needle_bench_result.reconstruction_error < 1e-4f ? 0 : 1; }
#endif
