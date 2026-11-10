//! The song's beats: where each one falls, from bar 1 on, and how they group into
//! bars. A steady grid is one beat every so often; a tempo map follows the beats the
//! song really has.

/// Beat times from bar 1 on, in any unit (seconds in the UI, frames in the engine).
/// Beat 0 is bar 1. Outside the list the beats go on at the pace of its first and
/// last beat, so there are beats before bar 1 (for a count-in) and after the end.
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    /// At least two, rising.
    times: Vec<f64>,
    /// The first starts at beat 0; the others at rising beats.
    meters: Vec<Meter>,
}

/// Bars of `per_bar` beats from beat `from` on, up to the next meter: where the
/// time signature changes. A bar cut short by the next meter is a bar too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meter {
    pub from: usize,
    pub per_bar: u32,
}

impl Beats {
    /// Beats at `times`, `per_bar` to a bar, or `None` with fewer than two, or times
    /// that are not finite and rising.
    pub fn new(times: Vec<f64>, per_bar: u32) -> Option<Self> {
        Self::with_meters(times, vec![Meter { from: 0, per_bar }])
    }

    /// Beats at `times` grouped by `meters`, or `None` as [`Beats::new`], or when
    /// the meters do not start at beat 0 and rise.
    pub fn with_meters(times: Vec<f64>, mut meters: Vec<Meter>) -> Option<Self> {
        let valid = times.len() >= 2
            && times.iter().all(|t| t.is_finite())
            && times.windows(2).all(|w| w[1] > w[0])
            && meters.first().is_some_and(|m| m.from == 0)
            && meters.windows(2).all(|w| w[1].from > w[0].from);
        for meter in &mut meters {
            meter.per_bar = meter.per_bar.max(1);
        }
        valid.then_some(Self { times, meters })
    }

    /// A beat every `beat` from `start` until past `end`.
    pub fn steady(start: f64, beat: f64, per_bar: u32, end: f64) -> Option<Self> {
        if !(beat > 0.0 && start.is_finite() && end.is_finite()) {
            return None;
        }
        let count = ((end - start) / beat).ceil().max(1.0) as usize + 1;
        Self::new(
            (0..count).map(|k| start + k as f64 * beat).collect(),
            per_bar,
        )
    }

    /// The meter beat `index` is in (the first one before bar 1).
    fn meter(&self, index: i64) -> Meter {
        let after = self
            .meters
            .partition_point(|m| m.from as i64 <= index)
            .max(1);
        self.meters[after - 1]
    }

    /// Beats per bar where beat `index` is.
    pub fn per_bar(&self, index: i64) -> u32 {
        self.meter(index).per_bar
    }

    /// The bar beat `index` is in (0 is bar 1, negative before it) and its beat in
    /// that bar (0 is the first).
    pub fn bar_of(&self, index: i64) -> (i64, u32) {
        let meter = self.meter(index);
        let mut bar = 0;
        for w in self
            .meters
            .windows(2)
            .take_while(|w| w[1].from <= meter.from)
        {
            bar += (w[1].from - w[0].from).div_ceil(w[0].per_bar as usize) as i64;
        }
        let per_bar = i64::from(meter.per_bar);
        let beats = index - meter.from as i64;
        (
            bar + beats.div_euclid(per_bar),
            beats.rem_euclid(per_bar) as u32,
        )
    }

    /// The time of beat `index` (0 is bar 1, negative before it).
    pub fn time(&self, index: i64) -> f64 {
        let last = self.times.len() - 1;
        if index < 0 {
            self.times[0] + index as f64 * (self.times[1] - self.times[0])
        } else if index as usize > last {
            let beat = self.times[last] - self.times[last - 1];
            self.times[last] + (index as usize - last) as f64 * beat
        } else {
            self.times[index as usize]
        }
    }

    /// The index of the first beat at or after `time`.
    pub fn first_from(&self, time: f64) -> i64 {
        let last = self.times.len() - 1;
        if time <= self.times[0] {
            let beat = self.times[1] - self.times[0];
            ((time - self.times[0]) / beat).ceil() as i64
        } else if time > self.times[last] {
            let beat = self.times[last] - self.times[last - 1];
            last as i64 + ((time - self.times[last]) / beat).ceil() as i64
        } else {
            self.times.partition_point(|&t| t < time) as i64
        }
    }

    /// The beats from `range.start` up to before `range.end`: index and time.
    pub fn within(&self, range: std::ops::Range<f64>) -> impl Iterator<Item = (i64, f64)> + '_ {
        (self.first_from(range.start)..)
            .map(|index| (index, self.time(index)))
            .take_while(move |(_, time)| *time < range.end)
    }

    /// Whether beat `index` starts a bar.
    pub fn accented(&self, index: i64) -> bool {
        let meter = self.meter(index);
        (index - meter.from as i64).rem_euclid(i64::from(meter.per_bar)) == 0
    }

    /// The length of the bar that ends at the first beat at or after `time`: as
    /// many beats back as the bar before that beat has.
    pub fn bar_before(&self, time: f64) -> f64 {
        let index = self.first_from(time);
        self.time(index) - self.time(index - i64::from(self.per_bar(index - 1)))
    }

    /// The length of the beat that `time` falls in.
    pub fn beat_at(&self, time: f64) -> f64 {
        let index = self.first_from(time).max(1);
        self.time(index) - self.time(index - 1)
    }

    /// The same beats with every time multiplied by `factor`, e.g. seconds to frames.
    pub fn scaled(&self, factor: f64) -> Option<Self> {
        Self::with_meters(
            self.times.iter().map(|t| t * factor).collect(),
            self.meters.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_beats_go_on_both_ways() {
        let beats = Beats::steady(100.0, 50.0, 3, 200.0).unwrap();
        let found: Vec<(i64, f64)> = beats.within(0.0..360.0).collect();
        let expected = [-2, -1, 0, 1, 2, 3, 4, 5].map(|i| (i, 100.0 + i as f64 * 50.0));
        assert_eq!(found, expected);
        assert!(beats.accented(0) && beats.accented(3) && beats.accented(-3));
        assert!(!beats.accented(-1));
        assert_eq!(beats.bar_before(100.0), 150.0);
        assert_eq!(Beats::steady(0.0, 0.0, 4, 10.0), None);
    }

    #[test]
    fn a_tempo_map_keeps_its_own_beats() {
        // Speeding up: beats 10, 8 and 6 apart.
        let beats = Beats::new(vec![0.0, 10.0, 18.0, 24.0], 2).unwrap();
        assert_eq!(beats.first_from(10.0), 1);
        assert_eq!(beats.first_from(10.5), 2);
        assert_eq!(beats.first_from(-15.0), -1);
        assert_eq!(beats.first_from(25.0), 4);
        assert_eq!(beats.time(-2), -20.0);
        assert_eq!(beats.time(5), 36.0);
        let found: Vec<f64> = beats.within(-10.0..30.0).map(|(_, t)| t).collect();
        assert_eq!(found, vec![-10.0, 0.0, 10.0, 18.0, 24.0]);
        assert_eq!(beats.bar_before(24.0), 14.0);
        assert_eq!(beats.beat_at(20.0), 6.0);
        assert_eq!(beats.beat_at(-5.0), 10.0);
        assert_eq!(beats.scaled(2.0).unwrap().time(3), 48.0);
    }

    #[test]
    fn bars_change_with_the_meter() {
        // Bar 1 in 4/4, a bar cut short to 2 beats, then 3 to a bar from beat 6.
        let meters = vec![
            Meter {
                from: 0,
                per_bar: 4,
            },
            Meter {
                from: 6,
                per_bar: 3,
            },
        ];
        let beats = Beats::with_meters((0..12).map(f64::from).collect(), meters).unwrap();
        let accented: Vec<i64> = (-4..12).filter(|&i| beats.accented(i)).collect();
        assert_eq!(accented, [-4, 0, 4, 6, 9]);
        let bars: Vec<(i64, u32)> = [-1, 0, 5, 6, 8, 9, 20].map(|i| beats.bar_of(i)).into();
        assert_eq!(
            bars,
            [(-1, 3), (0, 0), (1, 1), (2, 0), (2, 2), (3, 0), (6, 2)]
        );
        // A count-in into beat 6 counts 4 beats back, the meter before it; into
        // beat 9, 3.
        assert_eq!(beats.bar_before(6.0), 4.0);
        assert_eq!(beats.bar_before(9.0), 3.0);
        assert_eq!(beats.scaled(2.0).unwrap().bar_of(9), (3, 0));
        let unordered = vec![
            Meter {
                from: 0,
                per_bar: 4,
            },
            Meter {
                from: 0,
                per_bar: 3,
            },
        ];
        assert_eq!(Beats::with_meters(vec![0.0, 1.0], unordered), None);
    }

    #[test]
    fn needs_two_rising_beats() {
        assert_eq!(Beats::new(vec![1.0], 4), None);
        assert_eq!(Beats::new(vec![1.0, 1.0], 4), None);
        assert_eq!(Beats::new(vec![0.0, f64::NAN], 4), None);
    }
}
