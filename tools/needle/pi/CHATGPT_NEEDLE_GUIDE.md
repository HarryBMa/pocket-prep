# Adding Needle section stem separation to DECK001 (instructions for an AI coding assistant)

You are working on the DECK001 firmware: C++17 on a Raspberry Pi, ALSA audio, KMS display, I2C
controls (the `DECK001_v2.2.25a` source tree). Your job is to add one feature: the DJ selects a
section of the playing track (4-16 bars), presses the **stems** button, and the deck separates
that section into four stems (drums, bass, melody, vocals) that can be muted live while it plays.

The separation model is **Needle**, a small neural network (501 k parameters, a 539 KB `.tflite`
file) delivered in this package. A complete, working reference implementation in Python is
included (`reference/needle_pi.py`), plus golden test vectors (`golden/`) for every stage. Your
C++ must reproduce the reference, checked against those vectors.

Read this whole file before changing anything. Work through the steps in order.

## Ground rules

1. **One step per commit.** Commit message: `Needle S<n>: <what changed>`.
2. **Test first.** Each step lists its test. Write it, see it fail (or not compile), then implement.
3. **Do not change existing behaviour.** With no model file present, or the feature never used, the
   deck must behave exactly as before. All existing tests must still pass:
   `sh tests/run_tests.sh` must end with `PASS: all maintained portable suites`.
4. **`verify_release.sh` greps the source for exact strings.** Run `grep -n` for any line it checks
   before you edit near it; never reformat or refactor those lines.
5. **Real-time rules.** The audio worker (`Player::worker()` in `player.cpp`) must not allocate,
   free large buffers, lock a mutex that another thread holds for long, do file I/O, log, or run
   the network. Separation runs on its own background thread. The worker only reads finished,
   immutable buffers handed over the same way `offerSeamless()` / `releaseRetiredSeamless()` hand
   over a `LiveResidentSource` (read those first and copy that ownership pattern).
6. **The Pi has no internet and no Python at runtime.** New dependencies are vendored in
   `third_party/` like the existing ones, and loaded the way existing optional libraries are
   (the Makefile links with `-ldl`: load the TFLite library with `dlopen`, so the deck still
   builds and runs when it is absent; the stems button then shows as unavailable).
7. **No new settings** unless a step asks for one. Don't upload the code or the model anywhere.
8. If the reference and these instructions disagree, the reference code wins; report the
   disagreement.

## What is in this package

| Path | What |
|---|---|
| `model/needle_int8.tflite` | the network, int8 weights and activations (input and output int8) |
| `model/model.json` | model settings (sample rate, FFT size, hop, bins, frames) and measured speeds |
| `reference/needle_pi.py` | the reference pipeline (numpy + TFLite). `split` writes four WAVs |
| `reference/make_golden.py` | regenerates `golden/` |
| `golden/golden.json` + `*.bin` | every stage for a fixed 6-s synthetic test signal (raw little-endian, C order; shapes, dtypes and quantisation parameters in `golden.json`) |
| `icons/audio-lines.svg` | the stems icon (the same icon the DKPRP prep app uses for stems) |

## The algorithm, exactly

Audio: 44.1 kHz stereo float32 in [-1, 1]. `N` samples per channel. Constants: `N_FFT = 2048`,
`HOP = 512`, `BINS = 1024`, `FRAMES = 32` (read it from the model's input shape), `OVERLAP = 8`,
`STEP = FRAMES - OVERLAP = 24`, sources in this order: drums, bass, other, vocals (show "other" as
**MELODY** in the UI, as DKPRP does).

1. **STFT** (identical to `torch.stft(center=True, pad_mode="reflect")`):
   - window `w[n] = 0.5 - 0.5 cos(2 pi n / N_FFT)`, n = 0..2047 (periodic Hann);
   - pad each channel with `N_FFT/2 = 1024` samples of **reflection** on both sides
     (`x[-1] = x[1]`, `x[-2] = x[2]`, ...; the edge sample is not repeated);
   - frame count `T = 1 + N / HOP` (integer division); frame `t` covers padded samples
     `t*HOP .. t*HOP+2047`;
   - `X[c][k][t] = sum_n w[n] x_pad[c][t*HOP+n] e^{-2 pi i k n / N_FFT}`, k = 0..1024 (real FFT,
     no scaling).
   - Check: `golden/stft_re.bin`, `stft_im.bin` (channel, bin 0..1024, frame).
2. **Network input**: `L[c][k][t] = log1p(|X[c][k][t]|)` for k = 0..1023 (bin 1024, Nyquist, is not
   used by the network). Check: `net_input.bin`.
3. **Chunks**: `n = max(1, ceil((T - OVERLAP) / STEP))` chunks starting at frames `0, 24, 48, ...`.
   Pad `L` at the end with zeros (log1p(0) = 0) to `(n-1)*STEP + FRAMES` frames.
4. **Quantise one chunk** into the input tensor, shape NHWC `(1, 1024, 32, 2)` = (batch, bin, frame,
   channel), int8: `q = clamp(round(L / scale_in + zero_in), -128, 127)`. Read `scale_in`,
   `zero_in` from the model (`TfLiteTensorQuantizationParams`), don't hard-code them. Round half
   to even, as numpy does: `std::nearbyint` in the default rounding mode, so the result matches
   the golden file bit for bit. Check: `chunk0_quantised.bin`.
5. **Run** the interpreter. Output tensor NHWC `(1, 1024, 32, 8)` int8, channel index
   `k = source*2 + audio_channel` (drums L, drums R, bass L, ...). Check: `chunk0_output_raw.bin`
   (allow differences of 1 if your TFLite build uses different kernels than XNNPACK).
6. **Dequantise**: `mask = (q_out - zero_out) * scale_out` (values 0..1). Check: `chunk0_masks.bin`
   (layout (k, bin, frame)).
7. **Crossfade** chunks into one mask track `M[k][bin][t]`: chunk frame `j` gets weight
   `r(j) = (j + 0.5) / 8` for `j < 8`, `r(31 - j)` for `j >= 24`, 1 otherwise. Accumulate
   `weight * mask` and `weight` per frame, divide. Crop to `T` frames.
8. **Normalise over sources** per (channel, bin, frame): `S = sum of the 4 masks`; if `S > 1e-3`
   each gain is `mask / S`, **otherwise each gain is 0.25** (the int8 model can output exactly 0
   for all four; this keeps the stems summing to the mix).
9. **Apply and invert**: for each source and channel, `Y[k][t] = gain * X[k][t]` for k = 0..1023;
   for the Nyquist bin k = 1024, `Y = X` for "other" and 0 for the rest.
   **Inverse STFT** (identical to `torch.istft`): `y_t[n] = irfft(Y[:, t])[n] * w[n]`
   (irfft scaled by 1/N_FFT), overlap-add at `t*HOP`, divide every sample by the overlap-added
   `w[n]^2` envelope (it is 1.5 in the middle), then drop the first 1024 samples and keep `N`.
10. Result: stems `(4, 2, N)`. Check: `stems.bin`, SNR against it >= 40 dB per stem, and the four
    stems summed must equal the input within 1e-5 away from the first and last 2048 samples.

**A section**: the selection `[a, b)` in track frames is separated together with up to 1 s
(44 100 frames) of the track before `a` and after `b` (clipped to the track), then cropped back
to `[a, b)`. That context is why the edges of the selection sound like the middle.

**FFT library**: vendor `pocketfft_hdronly.hpp` (BSD-3, single header, github.com/mreineck/pocketfft,
branch `cpp`) in `third_party/pocketfft/`, or use an FFT already in `third_party/` if there is a
real FFT of size 2048 there. float32 throughout is fine.

## Steps

### S1: TFLite runtime, loaded at runtime

- Build the TensorFlow Lite C library for aarch64 once, on a PC (cross) or on the Pi:
  ```sh
  git clone --depth 1 --branch v2.18.0 https://github.com/tensorflow/tensorflow.git tf
  mkdir tflite_c && cd tflite_c
  cmake ../tf/tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DTFLITE_ENABLE_XNNPACK=ON
  cmake --build . -j4          # -> libtensorflowlite_c.so
  ```
  Put the `.so` and the C API headers it needs (`tensorflow/lite/c/c_api.h`, `c_api_types.h`,
  `common.h` and what they include) in `third_party/tflite_c/`, with its Apache-2.0 licence in
  `licenses/`. Add the `.so` to the release/install scripts next to the other shipped files.
- New `needle_tflite.h/.cpp`: `dlopen("libtensorflowlite_c.so")` (search next to the executable
  first), `dlsym` only the functions used: `TfLiteModelCreateFromFile`, `TfLiteModelDelete`,
  `TfLiteInterpreterOptionsCreate`, `TfLiteInterpreterOptionsSetNumThreads`,
  `TfLiteInterpreterOptionsDelete`, `TfLiteInterpreterCreate`, `TfLiteInterpreterDelete`,
  `TfLiteInterpreterAllocateTensors`, `TfLiteInterpreterGetInputTensor`,
  `TfLiteInterpreterGetOutputTensor`, `TfLiteInterpreterInvoke`, `TfLiteTensorData`,
  `TfLiteTensorDim`, `TfLiteTensorNumDims`, `TfLiteTensorQuantizationParams`, `TfLiteTensorType`.
  Missing library or model -> `available() == false`, one log line at startup, nothing else.
- Threads: 3 on a Pi 5 / Pi 4 (leave one core for audio and UI).
- **Test**: a host test that loads `model/needle_int8.tflite`, feeds `golden/chunk0_quantised.bin`
  and compares with `chunk0_output_raw.bin` (max difference <= 1). A second test: with the
  library renamed away, `available()` is false and nothing crashes.

### S2: STFT and inverse STFT

- New `needle_dsp.h/.cpp` with `stft()` and `istft()` exactly as above, pre-allocating its buffers
  from the section length (allocation happens on the background thread, never in the worker).
- **Tests**: against `stft_re/stft_im.bin` (relative error < 1e-4); `istft(stft(x)) == x` within
  1e-5; odd lengths (N not a multiple of 512) and N shorter than 2048.

### S3: the separator

- New `needle_separator.h/.cpp`: `bool separate(const float* interleavedStereo, uint64_t frames,
  std::array<std::vector<float>,4>& stemsInterleaved, std::atomic<float>* progress,
  const std::atomic<bool>* cancel)`. Steps 3-10 above; process chunk by chunk; update `progress`
  (0..1) after each chunk; return false quickly when `cancel` is set.
- Memory for 16 bars (30 s): keep it near 100 MB at most; compute and invert one source at a time
  instead of holding all 8 masked spectra.
- **Tests**: golden `stems.bin` (SNR >= 40 dB per stem); stems sum to input; cancel works;
  1-frame and 1-second inputs don't crash.

### S4: background job and speed

- New `needle_job.h/.cpp`: one long-lived thread, started at startup only if `available()`.
  `request(track PCM view, sectionStart, sectionEnd)` copies the section plus context (the copy is
  made on that thread), runs `separate()`, then publishes an immutable
  `StemSection { uint64_t start, end; std::array<std::vector<float>,4> stems; }` through an atomic
  pointer swap. A new request cancels the running one. Freed sections are released on the UI
  thread (like retired seamless sources), never in the audio worker.
- **Bench** (tools/, like the other benches): time 16 bars at 128 BPM (30 s). Budget: **3 s on the
  deck's Pi, faster preferred**. Expected on a Pi 5: about 1-3 s. Print network time and DSP time
  separately.

### S5: playback with stems

- The Player gets a `StemSection` from the job (handed over the `offerSeamless` way) and four stem
  gains (atomics, 0 or 1, set by the UI).
- Inside `[start, end)` of the section the worker renders `sum_s gain_s * stem_s` instead of the
  track PCM, at the same source frame positions, through the same resampling/keylock path the
  track PCM goes through. Outside the section, and whenever all four gains are 1, it plays the
  normal track (the stems add up to the mix, so the switch is inaudible).
- Gain changes ramp over 5 ms (220 frames) to avoid clicks. Loops inside the section keep working.
- A new track load, eject or seek far outside the section drops the section.
- **Tests**: host test rendering a block across the section start/end with all gains 1 equals the
  plain track within 1e-4; muting drums removes the drum stem; no allocation in the worker path
  (use the existing real-time checks/trace if present).

### S6: UI

- A stems button with the **AudioLines** icon (`assets/icons/audio-lines.svg`, already compiled into
  `icon_masks.h` as `Icons::Name::AudioLines`; the same icon DKPRP uses for stems). It is also the
  LIVE navigation icon in `cues_ui.cpp`, so place it where the two can't be confused, and **ask the
  user** before reusing it in the same view as LIVE.
- What it separates: the active loop if there is one; otherwise 16 bars from the playhead, starting
  on the beat grid's bar at or before the playhead. (4 and 8 bars as later options.)
- States, coloured like DKPRP's stem icon: unavailable (dimmed, model or library missing); idle;
  separating (accent colour, progress); ready (normal ink) -> four toggles **DRUMS BASS MELODY
  VOCALS**; failed (alert colour). Playback never waits: the stems take over when ready.
- Map toggles to controls the user chooses (ask); until then, touch buttons.

### S7: on the Pi

- Run the S4 bench and a listening check (`reference/needle_pi.py split` on the same section gives
  the expected WAVs). Report times, CPU load while playing, and any underruns.

## Things not to do

- Don't change the model's numbers (FFT size, hop, frames, window, scaling): the network was
  trained with exactly these.
- Don't run the network or the FFTs in the audio worker or the UI thread.
- Don't add Python, PyTorch or TensorFlow (full) to the Pi.
- Don't separate the whole track automatically; full-track stems come from the DKPRP prep app.
- Don't distribute the model: it is trained partly on datasets licensed for non-commercial research
  only (MUSDB18-HQ, MoisesDB). This is a private prototype.
