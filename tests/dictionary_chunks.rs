//! Dictionary-encoded data split into chunks whose combined dictionary would not fit the index
//! type (ITL-1178): each chunk is valid on its own, and the hash is that of the decoded values in
//! one chunk, whatever the index width, whether the dictionaries are disjoint or overlap, with
//! nulls, at the top level and inside a struct or a list; through `hash_array_chunks` and through
//! a streaming digester over record batches.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]
    #![expect(
        clippy::arithmetic_side_effects,
        clippy::integer_division_remainder_used,
        reason = "Small counters building the test data"
    )]

    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, DictionaryArray, ListArray, PrimitiveArray, RecordBatch, StringArray,
        StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{ArrowDictionaryKeyType, Int16Type, Int8Type, UInt8Type};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    /// Rows `start..start + count` over the dictionary `{prefix}{first..first + size}`, every
    /// seventh row null.
    fn chunk<K: ArrowDictionaryKeyType>(
        keys: impl Fn(usize) -> K::Native,
        count: usize,
        first: usize,
        size: usize,
    ) -> (ArrayRef, Vec<Option<String>>) {
        let key_values: Vec<Option<K::Native>> = (0..count)
            .map(|i| (i % 7 != 3).then(|| keys(i % size)))
            .collect();
        let decoded = (0..count)
            .map(|i| (i % 7 != 3).then(|| format!("v{}", first + i % size)))
            .collect();
        let values: ArrayRef = Arc::new(StringArray::from(
            (first..first + size)
                .map(|v| format!("v{v}"))
                .collect::<Vec<_>>(),
        ));
        let dictionary = DictionaryArray::<K>::try_new(
            key_values.into_iter().collect::<PrimitiveArray<K>>(),
            values,
        )
        .unwrap();
        (Arc::new(dictionary), decoded)
    }

    /// Two chunks, each with `size` values, together `size + offset` distinct values (beyond
    /// the index capacity when `offset` is large: disjoint; small: overlapping).
    fn two_chunks<K: ArrowDictionaryKeyType>(
        keys: impl Fn(usize) -> K::Native + Copy,
        size: usize,
        offset: usize,
    ) -> (Vec<ArrayRef>, StringArray) {
        let (left, mut decoded) = chunk::<K>(keys, size + 3, 0, size);
        let (right, decoded_right) = chunk::<K>(keys, size + 5, offset, size);
        decoded.extend(decoded_right);
        (vec![left, right], StringArray::from(decoded))
    }

    fn check(chunks: &[ArrayRef], decoded: &StringArray) {
        let config = HasherConfig::default();
        let data_type = chunks[0].data_type().clone();
        let refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
        assert_eq!(
            ArrowDigester::hash_array_chunks(&data_type, &refs, config),
            ArrowDigester::hash_array_with_config(decoded, config),
        );

        // As a column of two record batches, at the top level, in a struct and in a list.
        let wrap = |array: &ArrayRef, kind: &str| -> (Field, ArrayRef) {
            match kind {
                "struct" => {
                    let fields =
                        Fields::from(vec![Field::new("x", array.data_type().clone(), true)]);
                    let len = array.len();
                    let column = StructArray::new(
                        fields.clone(),
                        vec![Arc::clone(array)],
                        Some(NullBuffer::from(
                            (0..len).map(|i| i % 5 != 1).collect::<Vec<_>>(),
                        )),
                    );
                    (
                        Field::new("c", DataType::Struct(fields), true),
                        Arc::new(column),
                    )
                }
                "list" => {
                    let item = Arc::new(Field::new("item", array.data_type().clone(), true));
                    let len = array.len();
                    // Lists of 0, 1 or 2 items over the array, in order.
                    let mut lengths = Vec::new();
                    let mut used = 0;
                    while used < len {
                        let n = [2, 0, 1][lengths.len() % 3].min(len - used);
                        lengths.push(n);
                        used += n;
                    }
                    let column = ListArray::new(
                        Arc::clone(&item),
                        OffsetBuffer::from_lengths(lengths),
                        Arc::clone(array),
                        None,
                    );
                    (
                        Field::new("c", DataType::List(item), true),
                        Arc::new(column),
                    )
                }
                _ => (
                    Field::new("c", array.data_type().clone(), true),
                    Arc::clone(array),
                ),
            }
        };
        let decoded_ref: ArrayRef = Arc::new(decoded.clone());
        for kind in ["top", "struct", "list"] {
            let mut digester = None;
            for chunk in chunks {
                let (field, column) = wrap(chunk, kind);
                let schema = Arc::new(Schema::new(vec![field]));
                digester
                    .get_or_insert_with(|| ArrowDigester::new(&schema, config))
                    .update(&RecordBatch::try_new(schema, vec![column]).unwrap());
            }
            // The decoded values in one batch, with the struct validity and list lengths of the
            // chunks one after the other.
            let streamed = digester.unwrap().finalize();
            let (field, column) = wrap_decoded(&decoded_ref, chunks, kind);
            let one =
                RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap();
            assert_eq!(
                streamed,
                ArrowDigester::hash_record_batch(&one, config),
                "{kind}"
            );
        }
    }

    /// The decoded column of `check`, built with the same struct validity and list lengths as
    /// the chunks streamed one after the other.
    fn wrap_decoded(decoded: &ArrayRef, chunks: &[ArrayRef], kind: &str) -> (Field, ArrayRef) {
        match kind {
            "struct" => {
                let fields = Fields::from(vec![Field::new("x", DataType::Utf8, true)]);
                let validity: Vec<bool> = chunks
                    .iter()
                    .flat_map(|c| (0..c.len()).map(|i| i % 5 != 1))
                    .collect();
                let column = StructArray::new(
                    fields.clone(),
                    vec![Arc::clone(decoded)],
                    Some(NullBuffer::from(validity)),
                );
                (
                    Field::new("c", DataType::Struct(fields), true),
                    Arc::new(column),
                )
            }
            "list" => {
                let item = Arc::new(Field::new("item", DataType::Utf8, true));
                let mut lengths = Vec::new();
                for c in chunks {
                    let (len, mut used, mut k) = (c.len(), 0, 0);
                    while used < len {
                        let n = [2, 0, 1][k % 3].min(len - used);
                        lengths.push(n);
                        used += n;
                        k += 1;
                    }
                }
                let column = ListArray::new(
                    Arc::clone(&item),
                    OffsetBuffer::from_lengths(lengths),
                    Arc::clone(decoded),
                    None,
                );
                (
                    Field::new("c", DataType::List(item), true),
                    Arc::new(column),
                )
            }
            _ => (Field::new("c", DataType::Utf8, true), Arc::clone(decoded)),
        }
    }

    #[test]
    fn int8_keys_beyond_capacity() {
        let keys = |i: usize| i8::try_from(i).unwrap();
        let (disjoint, disjoint_decoded) = two_chunks::<Int8Type>(keys, 100, 100); // disjoint: 200 values
        check(&disjoint, &disjoint_decoded);
        let (overlapping, overlapping_decoded) = two_chunks::<Int8Type>(keys, 120, 60); // overlapping: 180 values
        check(&overlapping, &overlapping_decoded);
    }

    #[test]
    fn uint8_keys_beyond_capacity() {
        let keys = |i: usize| u8::try_from(i).unwrap();
        let (disjoint, disjoint_decoded) = two_chunks::<UInt8Type>(keys, 200, 200);
        check(&disjoint, &disjoint_decoded);
        let (overlapping, overlapping_decoded) = two_chunks::<UInt8Type>(keys, 250, 100);
        check(&overlapping, &overlapping_decoded);
    }

    #[test]
    fn int16_keys_beyond_capacity() {
        let keys = |i: usize| i16::try_from(i).unwrap();
        let (disjoint, disjoint_decoded) = two_chunks::<Int16Type>(keys, 20_000, 20_000);
        check(&disjoint, &disjoint_decoded);
        let (overlapping, overlapping_decoded) = two_chunks::<Int16Type>(keys, 30_000, 10_000);
        check(&overlapping, &overlapping_decoded);
    }
}
