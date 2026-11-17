//! A rejected update leaves the digester usable and unchanged (ITL-1178): for every refusal of
//! `update` (schema mismatch of every kind, a null in a non-nullable field found mid-batch) and
//! for a panic of arrow-rs on invalid data mid-batch, a following valid update and `finalize`
//! give the digest of the digester that never saw the rejected batch; through
//! `ArrowDigester::new` and through `PreparedSchema::digester`.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        make_array, ArrayData, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow::buffer::Buffer;
    use arrow_schema::{DataType, Field, Schema};
    use starfix::{ArrowDigester, HasherConfig, PreparedSchema, StarfixError};

    fn config() -> HasherConfig {
        HasherConfig {
            include_metadata: true,
            ordered_columns: true,
            ordered_struct_fields: false,
        }
    }

    fn dictionary_type() -> DataType {
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
    }

    /// `a: utf8?`, `b: int64` (non-nullable), `d: dictionary<int32, utf8>?`; schema metadata.
    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Int64, false),
            Field::new("d", dictionary_type(), true),
        ])
        .with_metadata(HashMap::from([("k".to_owned(), "v".to_owned())]))
    }

    /// Dictionary keys into `["x", "y"]`, unvalidated (a key may be out of range).
    fn keys(values: &[i32]) -> ArrayRef {
        let builder = ArrayData::builder(dictionary_type())
            .len(values.len())
            .add_buffer(Buffer::from_vec(values.to_vec()))
            .add_child_data(StringArray::from(vec!["x", "y"]).into());
        #[expect(unsafe_code, reason = "Building keys arrow-rs would not validate")]
        // SAFETY: the buffers are consistent; only an out-of-range key is invalid, which the
        // hasher must refuse.
        let data = unsafe { builder.build_unchecked() };
        make_array(data)
    }

    fn batch_of(schema: Schema, b: &[Option<i64>], d: &[i32]) -> RecordBatch {
        let rows = b.len();
        let a: StringArray = (0..rows).map(|i| Some(format!("r{i}"))).collect();
        #[expect(unsafe_code, reason = "Building a null in a non-nullable column")]
        // SAFETY: the columns match the schema in type and length; only nullability is not
        // validated, which the hasher must refuse.
        unsafe {
            RecordBatch::new_unchecked(
                Arc::new(schema),
                vec![Arc::new(a), Arc::new(Int64Array::from(b.to_vec())), keys(d)],
                rows,
            )
        }
    }

    /// One valid row of the schema with columns `b`, `a`, `d` (in that order).
    fn reordered_batch(reordered: Schema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(reordered),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(StringArray::from(vec!["r0"])),
                keys(&[0]),
            ],
        )
        .unwrap()
    }

    fn good(first: i64) -> RecordBatch {
        batch_of(schema(), &[Some(first), first.checked_add(1)], &[0, 1])
    }

    /// Every batch `update` must refuse, with the reason, and whether it panics.
    fn rejected() -> Vec<(&'static str, RecordBatch, bool)> {
        let mut swapped = schema().fields().iter().cloned().collect::<Vec<_>>();
        swapped.swap(0, 1);
        let reordered = Schema::new(swapped).with_metadata(schema().metadata().clone());
        let other_column = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Int64, false),
            Field::new("q", dictionary_type(), true),
        ])
        .with_metadata(schema().metadata().clone());
        let duplicate = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("a", DataType::Int64, false),
            Field::new("d", dictionary_type(), true),
        ])
        .with_metadata(schema().metadata().clone());
        vec![
            (
                "other column",
                batch_of(other_column, &[Some(1)], &[0]),
                false,
            ),
            (
                "column order (ordered_columns)",
                reordered_batch(reordered),
                false,
            ),
            (
                "schema metadata (include_metadata)",
                batch_of(schema().with_metadata(HashMap::new()), &[Some(1)], &[0]),
                false,
            ),
            (
                "duplicate column names",
                batch_of(duplicate, &[Some(1)], &[0]),
                false,
            ),
            // Found after column "a" was hashed.
            (
                "null in a non-nullable column",
                batch_of(schema(), &[Some(1), None], &[0, 1]),
                false,
            ),
            // An out-of-range key in the last column: arrow-rs panics while decoding it.
            (
                "invalid dictionary key",
                batch_of(schema(), &[Some(1), Some(2)], &[0, 7]),
                true,
            ),
        ]
    }

    fn check_usable_after_each_rejection(mut fresh: impl FnMut() -> ArrowDigester) {
        let mut reference = fresh();
        reference.update(&good(0));
        reference.update(&good(10));
        let expected = reference.finalize();
        for (reason, batch, panics) in rejected() {
            let mut digester = fresh();
            digester.update(&good(0));
            if panics {
                let mut target = digester.clone();
                let unwound = catch_unwind(AssertUnwindSafe(|| target.update(&batch)));
                assert!(unwound.is_err(), "{reason}: expected a panic");
                digester = target;
            } else {
                let error = digester.try_update(&batch).unwrap_err();
                assert!(
                    matches!(
                        error,
                        StarfixError::SchemaMismatch
                            | StarfixError::AmbiguousPath(_)
                            | StarfixError::NonNullableNull { .. }
                    ),
                    "{reason}: {error:?}"
                );
            }
            assert_eq!(
                digester.clone().finalize(),
                {
                    let mut once = fresh();
                    once.update(&good(0));
                    once.finalize()
                },
                "{reason}: the rejected batch changed the digester"
            );
            digester.update(&good(10));
            assert_eq!(digester.finalize(), expected, "{reason}");
        }
    }

    #[test]
    fn a_rejected_update_leaves_a_digester_usable() {
        check_usable_after_each_rejection(|| ArrowDigester::new(&schema(), config()));
    }

    #[test]
    fn a_rejected_update_leaves_a_prepared_digester_usable() {
        let prepared = PreparedSchema::new(&schema(), config());
        check_usable_after_each_rejection(|| prepared.digester());
        // The prepared one-shot path keeps no state: a refusal changes nothing after it.
        let expected = prepared.hash_record_batch(&good(0));
        for (reason, batch, panics) in rejected() {
            if panics {
                let unwound = catch_unwind(AssertUnwindSafe(|| prepared.hash_record_batch(&batch)));
                assert!(unwound.is_err(), "{reason}");
            } else {
                prepared.try_hash_record_batch(&batch).unwrap_err();
            }
            assert_eq!(prepared.hash_record_batch(&good(0)), expected, "{reason}");
        }
    }

    #[test]
    fn a_rejected_update_without_options_leaves_a_digester_usable() {
        // Without the options, reordered columns and other metadata are not refusals; the
        // remaining refusals still leave the digester unchanged.
        let plain = HasherConfig::default();
        let mut reference = ArrowDigester::new(&schema(), plain);
        reference.update(&good(0));
        reference.update(&good(10));
        let expected = reference.finalize();
        let mut digester = ArrowDigester::new(&schema(), plain);
        digester.update(&good(0));
        let null_mid_batch = batch_of(schema(), &[Some(1), None], &[0, 1]);
        digester.try_update(&null_mid_batch).unwrap_err();
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("z", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert_eq!(
            digester.try_update(&other),
            Err(StarfixError::SchemaMismatch)
        );
        digester.update(&good(10));
        assert_eq!(digester.finalize(), expected);
    }
}
