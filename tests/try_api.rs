//! The `try_*` API (ITL-1178): every refusal is a `StarfixError`, the panicking API panics with
//! the error's message, and a refused `update` leaves the digester as it was.
#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::expect_used, reason = "Okay in test")]

    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        make_array, Array as _, ArrayData, ArrayRef, Int32Array, ListArray, NullArray, RecordBatch,
        RecordBatchOptions, StringArray, StructArray,
    };
    use arrow::buffer::{Buffer, NullBuffer};
    use arrow::datatypes::Int32Type;
    use arrow_schema::{DataType, Field, Fields, Schema, UnionFields, UnionMode};
    use starfix::{ArrowDigester, HasherConfig, PreparedSchema, StarfixError};

    fn config() -> HasherConfig {
        HasherConfig::default()
    }

    fn panic_message(f: impl FnOnce()) -> String {
        let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap()
    }

    /// `a: utf8?`, `n: list<int32>?`, `s: struct<x: int32?>?`, `z: list<int32 (non-nullable)>`:
    /// every kind of entry (validity bits, list lengths, data), the non-nullable items last.
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new(
                "n",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
            Field::new(
                "s",
                DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, true)])),
                true,
            ),
            Field::new("z", z_type(), false),
        ]))
    }

    fn z_type() -> DataType {
        DataType::List(Arc::new(Field::new("item", DataType::Int32, false)))
    }

    /// One one-item list per row; a `None` item is a null arrow-rs would refuse to build.
    fn z_column(items: &[Option<i32>]) -> ArrayRef {
        let offsets: Vec<i32> = (0..=items.len())
            .map(|i| i32::try_from(i).unwrap())
            .collect();
        let builder = ArrayData::builder(z_type())
            .len(items.len())
            .add_buffer(Buffer::from_vec(offsets))
            .add_child_data(Int32Array::from(items.to_vec()).to_data());
        #[expect(unsafe_code, reason = "Building an array arrow-rs would not validate")]
        // SAFETY: offsets and child length are consistent; only the non-nullable item holds a null,
        // which is what the hasher must refuse.
        let data = unsafe { builder.build_unchecked() };
        make_array(data)
    }

    fn batch(start: i64, z_items: &[Option<i32>]) -> RecordBatch {
        let rows = z_items.len();
        let strings: StringArray = (0..rows)
            .map(|i| i.is_multiple_of(2).then(|| format!("a{start}{i}")))
            .collect();
        let lists = ListArray::from_iter_primitive::<Int32Type, _, _>(
            (0..rows).map(|i| (i != 1).then(|| vec![i32::try_from(i).ok(), None])),
        );
        let xs: ArrayRef = Arc::new(Int32Array::from(
            (0..rows).map(|i| i32::try_from(i).ok()).collect::<Vec<_>>(),
        ));
        let structs = StructArray::new(
            Fields::from(vec![Field::new("x", DataType::Int32, true)]),
            vec![xs],
            Some(NullBuffer::from(
                (0..rows).map(|i| i != 1).collect::<Vec<_>>(),
            )),
        );
        RecordBatch::try_new_with_options(
            schema(),
            vec![
                Arc::new(strings),
                Arc::new(lists),
                Arc::new(structs),
                z_column(z_items),
            ],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap()
    }

    #[test]
    fn a_refused_update_leaves_the_digester_as_it_was() {
        let first = batch(0, &[Some(1), Some(2), Some(3)]);
        let second = batch(10, &[Some(4), Some(5)]);
        // The null is in the last column: every other entry was updated before it was found.
        let refused = batch(20, &[Some(6), None, Some(7)]);

        let mut reference = ArrowDigester::new(&schema(), config());
        reference.update(&first);
        reference.update(&second);
        let expected = reference.finalize();

        let mut digester = ArrowDigester::new(&schema(), config());
        digester.update(&first);
        let before = digester.clone().finalize();
        let mut probe = digester.clone();
        probe.try_update(&refused).unwrap_err();
        // The validity bits of the refused batch are cleared too, not only their length.
        assert_eq!(probe.finalize(), before);
        assert_eq!(
            digester.try_update(&refused),
            Err(StarfixError::NonNullableNull {
                path: "z/".to_owned()
            })
        );
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("q", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert_eq!(
            digester.try_update(&other),
            Err(StarfixError::SchemaMismatch)
        );
        // The panicking `update` refuses the same way and also leaves the digester unchanged.
        let message = panic_message(|| digester.clone().update(&refused));
        assert_eq!(
            message,
            StarfixError::NonNullableNull {
                path: "z/".to_owned()
            }
            .to_string()
        );
        let mut after_panic = digester.clone();
        assert!(catch_unwind(AssertUnwindSafe(|| after_panic.update(&refused))).is_err());
        after_panic.update(&second);
        assert_eq!(after_panic.finalize(), expected);

        digester.try_update(&second).unwrap();
        assert_eq!(digester.finalize(), expected);
    }

    #[test]
    fn the_try_api_returns_each_refusal() {
        let duplicate = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("a", DataType::Int32, true),
        ]);
        assert_eq!(
            ArrowDigester::try_hash_schema(&duplicate, config()),
            Err(StarfixError::AmbiguousPath("a".to_owned()))
        );
        let union = Schema::new(vec![Field::new(
            "u",
            DataType::Union(UnionFields::empty(), UnionMode::Sparse),
            true,
        )]);
        assert_eq!(
            ArrowDigester::try_hash_schema(&union, config()),
            Err(StarfixError::UnsupportedType(DataType::Union(
                UnionFields::empty(),
                UnionMode::Sparse
            )))
        );
        let nulls = Schema::new(vec![Field::new("n", DataType::Null, true)]);
        ArrowDigester::try_hash_schema(&nulls, config()).unwrap();
        assert_eq!(
            ArrowDigester::try_new(&nulls, config()).err(),
            Some(StarfixError::UnsupportedData {
                path: "n".to_owned(),
                data_type: DataType::Null
            })
        );
        assert_eq!(
            ArrowDigester::try_hash_array_with_config(&NullArray::new(2), config()),
            Err(StarfixError::UnsupportedData {
                path: String::new(),
                data_type: DataType::Null
            })
        );
        let refused = batch(0, &[None]);
        assert_eq!(
            ArrowDigester::try_hash_record_batch(&refused, config()),
            Err(StarfixError::NonNullableNull {
                path: "z/".to_owned()
            })
        );

        let prepared = PreparedSchema::try_new(&schema(), config()).unwrap();
        assert_eq!(
            prepared.try_hash_record_batch(&refused),
            Err(StarfixError::NonNullableNull {
                path: "z/".to_owned()
            })
        );
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("q", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert_eq!(
            prepared.try_hash_record_batch(&other),
            Err(StarfixError::PreparedSchemaMismatch)
        );
        let prepared_nulls = PreparedSchema::try_new(&nulls, config()).unwrap();
        assert_eq!(
            prepared_nulls.try_digester().err(),
            Some(StarfixError::UnsupportedData {
                path: "n".to_owned(),
                data_type: DataType::Null
            })
        );
        assert_eq!(
            PreparedSchema::try_new(&duplicate, config()).err(),
            Some(StarfixError::AmbiguousPath("a".to_owned()))
        );
    }

    #[test]
    fn the_try_api_hashes_as_the_panicking_api() {
        let b = batch(0, &[Some(1), Some(2), Some(3)]);
        for config in [
            HasherConfig::default(),
            HasherConfig {
                include_metadata: true,
                ordered_columns: true,
                ordered_struct_fields: true,
                ..HasherConfig::default()
            },
        ] {
            assert_eq!(
                ArrowDigester::try_hash_record_batch(&b, config).unwrap(),
                ArrowDigester::hash_record_batch(&b, config)
            );
            assert_eq!(
                ArrowDigester::try_hash_schema(&schema(), config).unwrap(),
                ArrowDigester::hash_schema(&schema(), config)
            );
            assert_eq!(
                ArrowDigester::try_hash_array_with_config(b.column(1).as_ref(), config).unwrap(),
                ArrowDigester::hash_array_with_config(b.column(1).as_ref(), config)
            );
            let prepared = PreparedSchema::try_new(&schema(), config).unwrap();
            assert_eq!(
                prepared.try_hash_record_batch(&b).unwrap(),
                ArrowDigester::hash_record_batch(&b, config)
            );
            let mut d = prepared.try_digester().unwrap();
            d.try_update(&b).unwrap();
            assert_eq!(d.finalize(), ArrowDigester::hash_record_batch(&b, config));
        }
    }

    #[test]
    fn the_panicking_api_panics_with_the_error_message() {
        let duplicate = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("a", DataType::Int32, true),
        ]);
        assert_eq!(
            panic_message(|| {
                ArrowDigester::hash_schema(&duplicate, config());
            }),
            StarfixError::AmbiguousPath("a".to_owned()).to_string()
        );
        let nulls = Schema::new(vec![Field::new("n", DataType::Null, true)]);
        assert_eq!(
            panic_message(|| {
                ArrowDigester::new(&nulls, config());
            }),
            format!(
                "not yet implemented: {}",
                StarfixError::UnsupportedData {
                    path: "n".to_owned(),
                    data_type: DataType::Null
                }
            )
        );
        let prepared = PreparedSchema::new(&schema(), config());
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("q", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert_eq!(
            panic_message(|| {
                prepared.hash_record_batch(&other);
            }),
            "Record batch schema does not match the prepared schema"
        );
    }

    #[test]
    fn chunks_hash_as_their_concatenation() {
        let whole = batch(0, &[Some(1), Some(2), Some(3), Some(4)]);
        for column in whole.columns() {
            let (left, right) = (column.slice(0, 1), column.slice(1, 3));
            let data_type = column.data_type();
            assert_eq!(
                ArrowDigester::hash_array_chunks(
                    data_type,
                    &[left.as_ref(), right.as_ref()],
                    config()
                ),
                ArrowDigester::hash_array_with_config(column.as_ref(), config())
            );
            let empty = column.slice(0, 0);
            assert_eq!(
                ArrowDigester::hash_array_chunks(data_type, &[], config()),
                ArrowDigester::hash_array_with_config(empty.as_ref(), config())
            );
        }
        let ints = Int32Array::from(vec![1]);
        assert_eq!(
            ArrowDigester::try_hash_array_chunks(&DataType::Int64, &[&ints], config()),
            Err(StarfixError::ChunkTypeMismatch(DataType::Int32))
        );
    }
}
