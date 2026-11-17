//! Dictionary keys under a null enclosing struct are never read (byte-layout spec Section 3.2;
//! ITL-1178). pyarrow builds such keys out of range (an all-null struct over a dictionary child
//! gets key 0 into an empty dictionary): the hash is that of any in-range key, as
//! starfix-python computes it, instead of a panic while decoding.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        make_array, Array as _, ArrayData, ArrayRef, Int32Array, RecordBatch, StringArray,
        StructArray,
    };
    use arrow::buffer::{Buffer, NullBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    fn dictionary(value_type: &DataType, keys: &[i32], values: &ArrayRef) -> ArrayRef {
        let data_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(value_type.clone()));
        let builder = ArrayData::builder(data_type)
            .len(keys.len())
            .add_buffer(Buffer::from_vec(keys.to_vec()))
            .add_child_data(values.to_data());
        #[expect(unsafe_code, reason = "Building keys arrow-rs would not validate")]
        // SAFETY: keys and dictionary are consistent buffers; only out-of-range keys, which
        // the test puts under a null struct (or expects to be refused), are invalid.
        let data = unsafe { builder.build_unchecked() };
        make_array(data)
    }

    /// One column `s: struct<x: dictionary<int32, value_type>>`, rows `[null, valid]`.
    fn batch(
        value_type: &DataType,
        keys: &[i32],
        values: &ArrayRef,
        validity: &[bool],
    ) -> RecordBatch {
        let child = dictionary(value_type, keys, values);
        let fields = Fields::from(vec![Field::new("x", child.data_type().clone(), true)]);
        let column = StructArray::new(
            fields.clone(),
            vec![child],
            Some(NullBuffer::from(validity.to_vec())),
        );
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "s",
                DataType::Struct(fields),
                true,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap()
    }

    #[test]
    fn a_hidden_out_of_range_key_hashes_as_any_in_range_key() {
        let values: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let hidden_out_of_range = batch(&DataType::Utf8, &[5, 0], &values, &[false, true]);
        let hidden_in_range = batch(&DataType::Utf8, &[0, 0], &values, &[false, true]);
        for config in [
            HasherConfig::default(),
            HasherConfig {
                ordered_columns: true,
                ..HasherConfig::default()
            },
        ] {
            assert_eq!(
                ArrowDigester::hash_record_batch(&hidden_out_of_range, config),
                ArrowDigester::hash_record_batch(&hidden_in_range, config)
            );
        }
        // A key into an empty dictionary, as pyarrow builds it for an all-null struct.
        let empty: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
        let all_null = batch(&DataType::Utf8, &[0], &empty, &[false]);
        let all_null_valid = batch(&DataType::Utf8, &[0], &values, &[false]);
        assert_eq!(
            ArrowDigester::hash_record_batch(&all_null, HasherConfig::default()),
            ArrowDigester::hash_record_batch(&all_null_valid, HasherConfig::default())
        );
    }

    #[test]
    fn a_hidden_out_of_range_key_of_a_nested_dictionary_is_not_read() {
        let inner = Fields::from(vec![Field::new("v", DataType::Int32, true)]);
        let values: ArrayRef = Arc::new(StructArray::new(
            inner.clone(),
            vec![Arc::new(Int32Array::from(vec![7]))],
            None,
        ));
        let value_type = DataType::Struct(inner);
        assert_eq!(
            ArrowDigester::hash_record_batch(
                &batch(&value_type, &[9, 0], &values, &[false, true]),
                HasherConfig::default()
            ),
            ArrowDigester::hash_record_batch(
                &batch(&value_type, &[0, 0], &values, &[false, true]),
                HasherConfig::default()
            )
        );
    }

    #[test]
    fn a_visible_out_of_range_key_of_a_nested_dictionary_is_refused() {
        // It was clamped to the last value and hashed (CodeRabbit on 3d2f47e).
        let inner = Fields::from(vec![Field::new("v", DataType::Int32, true)]);
        let values: ArrayRef = Arc::new(StructArray::new(
            inner.clone(),
            vec![Arc::new(Int32Array::from(vec![7, 8]))],
            None,
        ));
        let value_type = DataType::Struct(inner);
        for key in [2, 9, -1] {
            let visible = batch(&value_type, &[0, key], &values, &[true, true]);
            catch_unwind(AssertUnwindSafe(|| {
                ArrowDigester::hash_record_batch(&visible, HasherConfig::default())
            }))
            .unwrap_err();
        }
        // In range, and hidden out of range, still hash.
        ArrowDigester::hash_record_batch(
            &batch(&value_type, &[0, 1], &values, &[true, true]),
            HasherConfig::default(),
        );
        ArrowDigester::hash_record_batch(
            &batch(&value_type, &[0, 9], &values, &[true, false]),
            HasherConfig::default(),
        );
    }

    #[test]
    fn a_visible_out_of_range_key_is_still_refused() {
        let values: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let visible = batch(&DataType::Utf8, &[0, 5], &values, &[false, true]);
        catch_unwind(AssertUnwindSafe(|| {
            ArrowDigester::hash_record_batch(&visible, HasherConfig::default())
        }))
        .unwrap_err();
    }
}
