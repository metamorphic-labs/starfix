//! Time to hash one million float values with the float options off and on (ITL-1252): clean
//! data, and data where 1% or 50% of the values are hashed as another value (`-0.0`, NaNs with a
//! sign or payload).
//!
//! Run with `cargo run --release --example float_bench`.
#![expect(clippy::unwrap_used, reason = "Benchmark binary")]

use std::hint::black_box;
use std::iter::repeat_with;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use starfix::{ArrowDigester, HasherConfig};

const ROWS: usize = 1_000_000;
const ROUNDS: u32 = 9;

/// A deterministic pseudo-random sequence (a 64-bit LCG), so the benchmark needs no dependency.
struct Lcg(u64);

impl Lcg {
    const fn draw(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0.wrapping_shr(33)
    }

    /// True with probability `percent` / 100.
    const fn chance(&mut self, percent: u64) -> bool {
        self.draw().wrapping_rem(100) < percent
    }

    /// An ordinary value in [-500, 500).
    fn value(&mut self) -> f64 {
        f64::from(u32::try_from(self.draw().wrapping_rem(1_000_000)).unwrap()) / 1000.0 - 500.0
    }
}

/// One slot of the data.
#[derive(Clone, Copy)]
enum Slot {
    Null,
    Value(f64),
    /// `-0.0`
    NegativeZero,
    /// A NaN with the sign bit set and a payload
    OddNan,
}

/// The slots: `dirty_percent` of the values `-0.0` or a NaN other than the canonical one,
/// alternately; with `nulls`, 10% of the slots null.
fn slots(dirty_percent: u64, nulls: bool) -> Vec<Slot> {
    let mut rng = Lcg(42);
    let mut negative_zero = true;
    repeat_with(|| {
        if nulls && rng.chance(10) {
            Slot::Null
        } else if rng.chance(dirty_percent) {
            negative_zero = !negative_zero;
            if negative_zero {
                Slot::NegativeZero
            } else {
                Slot::OddNan
            }
        } else {
            Slot::Value(rng.value())
        }
    })
    .take(ROWS)
    .collect()
}

fn column_batch(array: ArrayRef) -> RecordBatch {
    let field = Field::new("x", array.data_type().clone(), array.null_count() > 0);
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).unwrap()
}

fn batch(data_type: &DataType, dirty_percent: u64, nulls: bool) -> RecordBatch {
    let data = slots(dirty_percent, nulls);
    let array: ArrayRef = if *data_type == DataType::Float32 {
        Arc::new(
            data.iter()
                .map(|slot| match *slot {
                    Slot::Null => None,
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "Benchmark data: an ordinary value in [-500, 500)"
                    )]
                    Slot::Value(x) => Some(x as f32),
                    Slot::NegativeZero => Some(-0.0),
                    Slot::OddNan => Some(f32::from_bits(0xffc0_0001)),
                })
                .collect::<Float32Array>(),
        )
    } else {
        Arc::new(
            data.iter()
                .map(|slot| match *slot {
                    Slot::Null => None,
                    Slot::Value(x) => Some(x),
                    Slot::NegativeZero => Some(-0.0),
                    Slot::OddNan => Some(f64::from_bits(0xfff8_0000_0000_0001)),
                })
                .collect::<Float64Array>(),
        )
    };
    column_batch(array)
}

/// Minimum over `ROUNDS` runs of the time of one call, in milliseconds.
fn best_ms(mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let off = HasherConfig::default();
    let both = HasherConfig {
        normalize_negative_zero: true,
        normalize_nan: true,
        ..HasherConfig::default()
    };
    println!("{ROWS} rows per batch, hash_record_batch, best of {ROUNDS} runs");
    println!("type     nulls   dirty     off (ms)      on (ms)");
    for (data_type, nulls) in [
        (DataType::Float64, false),
        (DataType::Float64, true),
        (DataType::Float32, false),
    ] {
        for dirty_percent in [0, 1, 50] {
            let b = batch(&data_type, dirty_percent, nulls);
            let off_ms = best_ms(|| {
                black_box(ArrowDigester::hash_record_batch(black_box(&b), off));
            });
            let on_ms = best_ms(|| {
                black_box(ArrowDigester::hash_record_batch(black_box(&b), both));
            });
            let type_name = data_type.to_string();
            let null_share = if nulls { "10%" } else { "none" };
            println!(
                "{type_name:<8} {null_share:<6} {dirty_percent:>5}% {off_ms:>12.3} {on_ms:>12.3}"
            );
        }
    }
}
