# Needle on a Raspberry Pi

A test bench for the deck's "separate this section" button: pick a track, a start point and a number
of bars, and Needle separates just that section into drums, bass, other and vocals. `play` loops
the section and lets you mute stems live.

It runs the same int8 `.tflite` file that Vela compiles for the RA8P1's NPU, with the same chunking,
so what you hear is what the deck will produce (the deck will be faster: the NPU does the network).

## Setup (Raspberry Pi OS 64-bit, Bookworm)

    sudo apt install python3-venv libportaudio2 libsndfile1
    python3 -m venv ~/needle && source ~/needle/bin/activate
    pip install numpy scipy soundfile sounddevice ai-edge-litert

(`ai-edge-litert` is Google's TFLite runtime; on older Python versions use `tflite-runtime`.)
Copy `needle_pi.py` and `needle_int8.tflite` to the Pi.

## Use

    python needle_pi.py bench --model needle_int8.tflite
    python needle_pi.py split --model needle_int8.tflite --track song.wav --start 1:04 --bars 16 --bpm 126 --out stems
    python needle_pi.py play  --model needle_int8.tflite --track song.wav --start 1:04 --bars 16 --bpm 126

- Tracks must be 44.1 kHz (WAV or FLAC; MP3 works if your libsndfile is 1.1 or newer).
  Convert others with `ffmpeg -i in.mp3 -ar 44100 out.wav`.
- `--start` is seconds or m:ss; the section length is `--bars` x 4 beats at `--bpm`, or `--seconds`.
- `play`: type 1, 2, 3 or 4 + Enter to mute/unmute drums, bass, other, vocals; `q` quits.
  Audio goes to the default output (HDMI or a USB sound card on a Pi 5, which has no headphone jack).

## What to expect

On the desktop PC the whole pipeline takes 0.6 s for 16 bars at 128 BPM (about 48x real time); the
network is only a quarter of that, the STFT and inverse STFT the rest. A Pi 5 should be a few times
slower (estimate: 2-4 s); a Pi Zero 2 W much slower and tight on memory (512 MB). `bench` prints
the real numbers for your Pi.

The model file changes as training improves; the script stays the same.
