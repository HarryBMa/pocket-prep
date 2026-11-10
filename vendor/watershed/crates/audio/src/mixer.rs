use std::sync::Arc;

use crate::Song;

/// Largest block the mixer renders at once, in frames. Longer device buffers are
/// processed in several blocks.
pub(crate) const MAX_BLOCK: usize = 2048;

/// The stereo buses after the mixer. Each goes through its own stretcher:
/// pitch-locked tracks get the tempo change but not the transposition, and vocals
/// are transposed with their own settings (formants, tonality limit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bus {
    Pitched,
    Locked,
    Vocal,
}

impl Bus {
    pub const ALL: [Bus; 3] = [Bus::Pitched, Bus::Locked, Bus::Vocal];

    fn of(params: &crate::TrackParams) -> Self {
        if params.pitch_lock() {
            Bus::Locked
        } else if params.vocal() {
            Bus::Vocal
        } else {
            Bus::Pitched
        }
    }
}

/// A song plus the mixer state that belongs to the audio thread. Built on the UI
/// thread so that every allocation happens there.
pub(crate) struct Voice {
    pub song: Arc<Song>,
    /// Per-track left/right gains on each bus, applied at the end of the previous
    /// block. A track sends to one bus; ramping all of them lets it move between them
    /// without a click when its pitch lock changes.
    gains: Vec<[[f32; 2]; 3]>,
}

impl Voice {
    pub fn new(song: Arc<Song>) -> Self {
        let any_solo = song.tracks.iter().any(|t| t.params.solo());
        let gains = song
            .tracks
            .iter()
            .map(|t| Bus::ALL.map(|bus| target(&t.params, bus, any_solo)))
            .collect();
        Self { song, gains }
    }

    /// Jumps to the current gains without a ramp. For a seek, where the output starts
    /// over anyway.
    pub fn snap_gains(&mut self) {
        let any_solo = self.song.tracks.iter().any(|t| t.params.solo());
        for (track, current) in self.song.tracks.iter().zip(self.gains.iter_mut()) {
            *current = Bus::ALL.map(|bus| target(&track.params, bus, any_solo));
        }
    }

    /// Adds `out.len() / 2` stereo frames of `bus` starting at `position` into `out`.
    ///
    /// Gain changes are ramped linearly across the block to avoid clicks. Each track
    /// is read at `position - offset`; frames outside a track are left untouched.
    pub fn mix_into(&mut self, position: usize, bus: Bus, out: &mut [f32]) {
        let frames = out.len() / 2;
        if position >= self.song.frames() {
            return;
        }
        let track_frames = self.song.track_frames() as i64;
        let any_solo = self.song.tracks.iter().any(|t| t.params.solo());

        for (track, current) in self.song.tracks.iter().zip(self.gains.iter_mut()) {
            let current = &mut current[bus as usize];
            let target = target(&track.params, bus, any_solo);
            let start = *current;
            *current = target;
            if start == [0.0, 0.0] && target == [0.0, 0.0] {
                continue;
            }

            let step = [
                (target[0] - start[0]) / frames as f32,
                (target[1] - start[1]) / frames as f32,
            ];
            // The block's frames that fall inside the track, and where they start in it.
            let first = position as i64 - track.params.offset();
            let begin = (-first).clamp(0, frames as i64) as usize;
            let end = (track_frames - first).clamp(0, frames as i64) as usize;
            if begin >= end {
                continue;
            }
            let src_start = (first + begin as i64) as usize;
            let src = track.samples.frames(src_start..src_start + (end - begin));
            let dst = &mut out.as_chunks_mut::<2>().0[begin..end];
            for (i, (o, s)) in dst.iter_mut().zip(src).enumerate() {
                // Ends exactly on the target so the next block continues seamlessly.
                let t = (begin + i + 1) as f32;
                o[0] += s[0] * (start[0] + step[0] * t);
                o[1] += s[1] * (start[1] + step[1] * t);
            }
        }
    }
}

/// Left/right gains of a track on `bus`: zero unless the track sends to it.
fn target(params: &crate::TrackParams, bus: Bus, any_solo: bool) -> [f32; 2] {
    if Bus::of(params) == bus {
        params.channel_gains(any_solo)
    } else {
        [0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A song where track `i` is a constant `values[i]` on both channels.
    fn song(values: &[f32], frames: usize) -> Arc<Song> {
        let tracks = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (format!("t{i}"), vec![v; frames * 2]))
            .collect();
        Arc::new(Song::new(48_000, tracks))
    }

    /// Mixes one block to settle the gain ramps, then returns the next block.
    fn render(voice: &mut Voice, position: usize, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * 2];
        voice.mix_into(position, Bus::Pitched, &mut out);
        out.fill(0.0);
        voice.mix_into(position, Bus::Pitched, &mut out);
        out
    }

    #[test]
    fn sums_tracks_at_unity() {
        let mut voice = Voice::new(song(&[0.25, 0.5], 64));
        let out = render(&mut voice, 0, 16);
        assert!(out.iter().all(|&s| (s - 0.75).abs() < 1e-6));
    }

    #[test]
    fn mute_and_solo() {
        let song = song(&[0.25, 0.5, 0.125], 64);
        let mut voice = Voice::new(song.clone());

        song.tracks[0].params.set_mute(true);
        assert!((render(&mut voice, 0, 16)[0] - 0.625).abs() < 1e-6);

        // Solo silences the unsoloed tracks, and overrides mute.
        song.tracks[1].params.set_solo(true);
        assert!((render(&mut voice, 0, 16)[0] - 0.5).abs() < 1e-6);
        song.tracks[0].params.set_solo(true);
        assert!((render(&mut voice, 0, 16)[0] - 0.75).abs() < 1e-6);
        // Unsoloed again, the mute holds.
        song.tracks[0].params.set_solo(false);
        song.tracks[1].params.set_solo(false);
        assert!((render(&mut voice, 0, 16)[0] - 0.625).abs() < 1e-6);
    }

    #[test]
    fn gain_and_pan() {
        let song = song(&[1.0], 64);
        let mut voice = Voice::new(song.clone());

        song.tracks[0].params.set_gain(0.5);
        song.tracks[0].params.set_pan(1.0);
        let out = render(&mut voice, 0, 16);
        assert_eq!((out[0], out[1]), (0.0, 0.5));

        song.tracks[0].params.set_pan(-0.5);
        let out = render(&mut voice, 0, 16);
        assert_eq!((out[0], out[1]), (0.5, 0.25));
    }

    #[test]
    fn gain_changes_are_ramped() {
        let song = song(&[1.0], 64);
        let mut voice = Voice::new(song.clone());
        song.tracks[0].params.set_gain(0.0);

        let mut out = vec![0.0; 8 * 2];
        voice.mix_into(0, Bus::Pitched, &mut out);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        assert!(left.windows(2).all(|w| w[1] < w[0]), "{left:?}");
        assert!(left[0] > 0.8 && left[7] == 0.0, "{left:?}");
    }

    #[test]
    fn pitch_locked_tracks_go_to_the_locked_bus() {
        let song = song(&[0.25, 0.5], 64);
        song.tracks[1].params.set_pitch_lock(true);
        let mut voice = Voice::new(song.clone());
        let mut locked = vec![0.0; 16 * 2];
        voice.mix_into(0, Bus::Locked, &mut locked);
        assert!((render(&mut voice, 0, 16)[0] - 0.25).abs() < 1e-6);
        assert!((locked[0] - 0.5).abs() < 1e-6);

        // Unlocking ramps the track out of the locked bus instead of cutting it.
        song.tracks[1].params.set_pitch_lock(false);
        let mut out = vec![0.0; 8 * 2];
        voice.mix_into(0, Bus::Locked, &mut out);
        assert!(out[0] > 0.4 && out[14] == 0.0, "{out:?}");
    }

    #[test]
    fn vocals_go_to_the_vocal_bus_unless_locked() {
        let song = song(&[0.25, 0.5], 64);
        song.tracks[1].params.set_vocal(true);
        let mut voice = Voice::new(song.clone());
        let mut vocal = vec![0.0; 16 * 2];
        voice.mix_into(0, Bus::Vocal, &mut vocal);
        assert!((render(&mut voice, 0, 16)[0] - 0.25).abs() < 1e-6);
        assert!((vocal[0] - 0.5).abs() < 1e-6);

        // The pitch lock wins: a locked vocal is not transposed at all.
        song.tracks[1].params.set_pitch_lock(true);
        let mut voice = Voice::new(song.clone());
        let mut locked = vec![0.0; 16 * 2];
        voice.mix_into(0, Bus::Locked, &mut locked);
        assert!((locked[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn offsets_move_tracks_in_time() {
        let song = Song::new(
            48_000,
            vec![("a".into(), vec![1.0; 20]), ("b".into(), vec![0.5; 20])],
        );
        let song = Arc::new(song);
        // b starts 4 frames later, so the song is 4 frames longer.
        song.tracks[1].params.set_offset(4);
        assert_eq!(song.frames(), 14);
        let mut voice = Voice::new(song.clone());
        let left: Vec<f32> = render(&mut voice, 8, 6)
            .iter()
            .step_by(2)
            .copied()
            .collect();
        assert_eq!(left, [1.5, 1.5, 0.5, 0.5, 0.5, 0.5]);

        // A negative offset cuts the start of b.
        song.tracks[1].params.set_offset(-3);
        assert_eq!(song.frames(), 10);
        let left: Vec<f32> = render(&mut voice, 5, 4)
            .iter()
            .step_by(2)
            .copied()
            .collect();
        assert_eq!(left, [1.5, 1.5, 1.0, 1.0]);
    }

    #[test]
    fn stops_at_end_of_song() {
        let mut voice = Voice::new(song(&[1.0], 10));
        let out = render(&mut voice, 6, 8);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        assert_eq!(left, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(render(&mut voice, 20, 8).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn pads_shorter_tracks() {
        let song = Song::new(
            48_000,
            vec![("a".into(), vec![0.0; 20]), ("b".into(), vec![0.0; 8])],
        );
        assert_eq!(song.frames(), 10);
        assert!(song.tracks.iter().all(|t| t.samples.len() == 20));
    }
}
