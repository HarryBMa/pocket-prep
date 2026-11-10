//! Running a whole song through a model of fixed length, as MSST's `demix` does: the
//! song is cut in overlapping chunks and their outputs are added with fades.

use crate::StereoBuffer;
use crate::stft::pad_reflect;

const CHANNELS: usize = 2;

/// How a song is cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunking {
    /// Frames of a chunk.
    pub size: usize,
    /// Chunks covering each frame: 2 moves half a chunk at a time.
    pub overlap: usize,
}

/// Fades in and out over `fade` frames.
fn fades(size: usize, fade: usize) -> Vec<f32> {
    let mut window = vec![1.0; size];
    let steps = fade.saturating_sub(1).max(1) as f32;
    for i in 0..fade {
        let gain = i as f32 / steps;
        window[i] = gain;
        window[size - 1 - i] = gain;
    }
    window
}

/// Runs `process` over `input` and returns its `stems` outputs, each as long as `input`.
///
/// `process` takes one chunk of interleaved stereo and the fraction of the song done
/// once this chunk is, and returns the same chunk for each stem.
pub fn overlap_add<E>(
    input: &StereoBuffer,
    stems: usize,
    chunking: Chunking,
    mut process: impl FnMut(&[f32], f32) -> Result<Vec<Vec<f32>>, E>,
) -> Result<Vec<StereoBuffer>, E> {
    let Chunking { size, overlap } = chunking;
    let step = size / overlap;
    let border = size - step;
    let fade = size / 10;
    let frames = input.frames();
    let window = fades(size, fade);

    // The song's first and last moments would be heard only once, faded and at the
    // edge of a chunk, where the model does worst.
    let pad = if frames > 2 * border { border } else { 0 };
    let mix = pad_reflect(&input.samples, CHANNELS, pad, pad);
    let total = mix.len() / CHANNELS;

    let mut results = vec![vec![0.0f32; mix.len()]; stems];
    let mut counter = vec![0.0f32; total];
    let mut part = vec![0.0; size * CHANNELS];

    let mut start = 0;
    while start < total {
        let length = size.min(total - start);
        let samples = &mix[start * CHANNELS..][..length * CHANNELS];
        if length == size {
            part.copy_from_slice(samples);
        } else if length > size / 2 {
            part = pad_reflect(samples, CHANNELS, 0, size - length);
        } else {
            part[..samples.len()].copy_from_slice(samples);
            part[samples.len()..].fill(0.0);
        }
        let done = ((start + step) as f32 / total as f32).min(1.0);
        let outputs = process(&part, done)?;

        let first = start == 0;
        let last = start + step >= total;
        let gain = |i: usize| {
            if (first && i < fade) || (!first && last && i >= size - fade) {
                1.0
            } else {
                window[i]
            }
        };
        for (result, output) in results.iter_mut().zip(&outputs) {
            let target = result[start * CHANNELS..].as_chunks_mut::<CHANNELS>().0;
            let frames = output.as_chunks::<CHANNELS>().0.iter().take(length);
            for (i, (target, frame)) in target.iter_mut().zip(frames).enumerate() {
                target[0] += frame[0] * gain(i);
                target[1] += frame[1] * gain(i);
            }
        }
        for (i, count) in counter[start..start + length].iter_mut().enumerate() {
            *count += gain(i);
        }
        start += step;
    }

    Ok(results
        .into_iter()
        .map(|mut result| {
            for (frame, count) in result
                .as_chunks_mut::<CHANNELS>()
                .0
                .iter_mut()
                .zip(&counter)
            {
                for sample in frame {
                    let value = *sample / count;
                    // A frame no chunk reached with any gain: 0 / 0.
                    *sample = if value.is_nan() { 0.0 } else { value };
                }
            }
            result.truncate((pad + frames) * CHANNELS);
            result.drain(..pad * CHANNELS);
            StereoBuffer::new(input.sample_rate, result)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(frames: usize) -> StereoBuffer {
        let samples = (0..frames * 2).map(|i| (i as f32 * 0.37).sin()).collect();
        StereoBuffer::new(44100, samples)
    }

    fn identity(chunk: &[f32], _done: f32) -> Result<Vec<Vec<f32>>, ()> {
        Ok(vec![
            chunk.to_vec(),
            chunk.iter().map(|v| v * 0.5).collect(),
        ])
    }

    #[test]
    fn fades_match_msst() {
        let window = fades(10, 3);
        assert_eq!(window, [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]);
    }

    /// A model that changes nothing gives back the song, whatever the cut.
    #[test]
    fn an_identity_model_restores_the_song() {
        for (frames, overlap) in [(5000, 2), (5000, 4), (1234, 3), (1000, 2), (2100, 2)] {
            let input = ramp(frames);
            let chunking = Chunking { size: 400, overlap };
            let output = overlap_add(&input, 2, chunking, identity).unwrap();
            assert_eq!(output[0].frames(), frames);
            for (a, b) in output[0].samples.iter().zip(&input.samples) {
                assert!((a - b).abs() < 1e-5, "{frames}, {overlap}: {a} vs {b}");
            }
            for (a, b) in output[1].samples.iter().zip(&input.samples) {
                assert!((a - b * 0.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn a_song_shorter_than_a_chunk_works() {
        let input = ramp(100);
        let chunking = Chunking {
            size: 400,
            overlap: 2,
        };
        let output = overlap_add(&input, 2, chunking, identity).unwrap();
        assert_eq!(output[0].samples, input.samples);
    }

    #[test]
    fn an_empty_song_gives_empty_stems() {
        let input = StereoBuffer::new(44100, Vec::new());
        let chunking = Chunking {
            size: 400,
            overlap: 2,
        };
        let output = overlap_add(&input, 2, chunking, identity).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].frames(), 0);
    }

    #[test]
    fn progress_goes_up_to_the_end() {
        let input = ramp(5000);
        let chunking = Chunking {
            size: 400,
            overlap: 2,
        };
        let mut seen = Vec::new();
        let result = overlap_add(&input, 2, chunking, |chunk, done| {
            seen.push(done);
            identity(chunk, done)
        });
        assert!(result.is_ok());
        assert!(seen.is_sorted());
        assert!(seen[0] > 0.0);
        assert_eq!(seen.last(), Some(&1.0));
    }

    #[test]
    fn errors_stop_the_run() {
        let input = ramp(5000);
        let chunking = Chunking {
            size: 400,
            overlap: 2,
        };
        let mut calls = 0;
        let result = overlap_add(&input, 2, chunking, |_, _| {
            calls += 1;
            Err("stop")
        });
        assert_eq!(result.unwrap_err(), "stop");
        assert_eq!(calls, 1);
    }
}
