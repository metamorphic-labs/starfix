//! Time per one-row hash: one-shot path vs prepared path (ITL-1177).
//!
//! Run with `cargo run --release --example prepared_bench`.
#![expect(clippy::unwrap_used, reason = "Benchmark binary")]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch, StringArray, StructArray,
};
use arrow::datatypes::Float64Type;
use arrow_schema::{DataType, Field, Fields, Schema};
use starfix::{ArrowDigester, HasherConfig};

const ITERATIONS: u32 = 10_000;
const ROUNDS: u32 = 7;

/// A one-row batch of a small, Orcapod-like schema: an id, a path string, a struct and a list.
fn one_row_batch(with_metadata: bool) -> RecordBatch {
    let point_fields = Fields::from(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]);
    let mut name = Field::new("name", DataType::Utf8, true);
    if with_metadata {
        name = name.with_metadata([("unit".to_owned(), "none".to_owned())].into());
    }
    let fields = vec![
        Field::new("id", DataType::Int64, false),
        name,
        Field::new("point", DataType::Struct(point_fields.clone()), true),
        Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
    ];
    let mut schema = Schema::new(fields);
    if with_metadata {
        schema = schema.with_metadata([("source".to_owned(), "bench".to_owned())].into());
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7])),
        Arc::new(StringArray::from(vec![Some("/data/file.bin")])),
        Arc::new(StructArray::new(
            point_fields,
            vec![
                Arc::new(Float64Array::from(vec![1.5])),
                Arc::new(Float64Array::from(vec![2.5])),
            ],
            None,
        )),
        Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ])),
    ];
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

/// Minimum over `ROUNDS` rounds of the mean time per call, in microseconds.
fn time_per_call(label: &str, mut f: impl FnMut()) -> f64 {
    for _ in 0..1_000 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            f();
        }
        best = best.min(start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERATIONS));
    }
    println!("{label:<62} {best:>8.2} us");
    best
}

fn main() {
    for (with_metadata, config) in [
        (false, HasherConfig::default()),
        (
            true,
            HasherConfig {
                include_metadata: true,
                ..HasherConfig::default()
            },
        ),
    ] {
        let batch = one_row_batch(with_metadata);
        // A batch built separately: an equal schema that is not the same `Arc`.
        let other = one_row_batch(with_metadata);
        println!("include_metadata = {}", config.include_metadata);
        time_per_call("one-shot ArrowDigester::hash_record_batch", || {
            black_box(ArrowDigester::hash_record_batch(black_box(&batch), config));
        });
        time_per_call("streaming ArrowDigester::new + update + finalize", || {
            let mut d = ArrowDigester::new(black_box(&batch.schema()), config);
            d.update(black_box(&other));
            black_box(d.finalize());
        });
        time_per_call("schema analysis only (ArrowDigester::hash_schema)", || {
            black_box(ArrowDigester::hash_schema(
                black_box(&batch.schema()),
                config,
            ));
        });
        time_per_call(
            "same-schema check: Schema == Schema (equal, not same Arc)",
            || {
                black_box(black_box(batch.schema_ref()) == black_box(other.schema_ref()));
            },
        );
        time_per_call("same-schema check: Arc::ptr_eq", || {
            black_box(Arc::ptr_eq(
                black_box(batch.schema_ref()),
                black_box(batch.schema_ref()),
            ));
        });
        let prepared = ArrowDigester::prepare(&batch.schema(), config);
        time_per_call(
            "prepare once: ArrowDigester::prepare (the analysis)",
            || {
                black_box(ArrowDigester::prepare(black_box(&batch.schema()), config));
            },
        );
        time_per_call(
            "prepared.hash_record_batch (equal schema, other Arc)",
            || {
                black_box(prepared.hash_record_batch(black_box(&other)));
            },
        );
        let same_arc =
            RecordBatch::try_new(Arc::clone(prepared.schema()), other.columns().to_vec()).unwrap();
        time_per_call(
            "prepared.hash_record_batch (batch built on prepared.schema())",
            || {
                black_box(prepared.hash_record_batch(black_box(&same_arc)));
            },
        );
        time_per_call("prepared.digester() + update + finalize", || {
            let mut d = prepared.digester();
            d.update(black_box(&other));
            black_box(d.finalize());
        });
    }
}
