//! The float options `normalize_negative_zero` and `normalize_nan` (ITL-1252; byte-layout spec
//! Sections 3.2, 6 and 11).
//!
//! The invariant: hashing data with a float option on equals hashing, with both options off, a
//! copy of the data whose float values are normalized (`-0.0` to `+0.0`; every NaN to the quiet
//! NaN with a clear sign and no payload). Checked for record batches, streams, the prepared
//! schema and `hash_array`, on explicit inputs (each option alone and both, several values in
//! two orders, top level and nested) and as a property over generated tables: list, large list,
//! fixed-size list (nested too), struct and dictionary nesting, `Float16` / `Float32` /
//! `Float64` leaves, nulls, nulls hidden by enclosing structs, values hidden under null slots,
//! and sliced batches. Map data is not supported (Section 8), so maps are not generated.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::iter::repeat_with;
    use std::sync::Arc;

    use arrow::array::{
        make_array, Array as _, ArrayData, ArrayRef, DictionaryArray, FixedSizeBinaryArray,
        FixedSizeListArray, Int16Array, Int32Array, Int64Array, LargeListArray, ListArray,
        RecordBatch, StructArray, UInt64Array,
    };
    use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow::datatypes::Int32Type;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use starfix::{ArrowDigester, HasherConfig, PreparedSchema};

    // ---------------------------------------------------------------------------------------
    // float widths, values and their normalization (written from the spec, not from the crate)
    // ---------------------------------------------------------------------------------------

    #[derive(Clone, Copy)]
    enum Width {
        F16,
        F32,
        F64,
    }

    impl Width {
        const ALL: [Self; 3] = [Self::F16, Self::F32, Self::F64];

        const fn of(data_type: &DataType) -> Option<Self> {
            match *data_type {
                DataType::Float16 => Some(Self::F16),
                DataType::Float32 => Some(Self::F32),
                DataType::Float64 => Some(Self::F64),
                _ => None,
            }
        }

        const fn data_type(self) -> DataType {
            match self {
                Self::F16 => DataType::Float16,
                Self::F32 => DataType::Float32,
                Self::F64 => DataType::Float64,
            }
        }

        /// Bytes per value.
        const fn bytes(self) -> usize {
            match self {
                Self::F16 => 2,
                Self::F32 => 4,
                Self::F64 => 8,
            }
        }

        /// (sign bit, exponent mask, fraction mask, canonical quiet NaN).
        const fn layout(self) -> (u64, u64, u64, u64) {
            match self {
                Self::F16 => (0x8000, 0x7c00, 0x03ff, 0x7e00),
                Self::F32 => (0x8000_0000, 0x7f80_0000, 0x007f_ffff, 0x7fc0_0000),
                Self::F64 => (
                    0x8000_0000_0000_0000,
                    0x7ff0_0000_0000_0000,
                    0x000f_ffff_ffff_ffff,
                    0x7ff8_0000_0000_0000,
                ),
            }
        }

        /// Ordinary and special values, as bits: `+0.0`, `-0.0`, 1.5, -2.25, `+inf`, `-inf`, the
        /// canonical NaN, `-NaN`, a quiet NaN with a payload, a signaling NaN, a negative
        /// signaling NaN with a payload, and the smallest subnormal.
        const fn values(self) -> [u64; 12] {
            match self {
                Self::F16 => [
                    0x0000, 0x8000, 0x3e00, 0xc080, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7e2a, 0x7c01,
                    0xfd55, 0x0001,
                ],
                Self::F32 => [
                    0x0000_0000,
                    0x8000_0000,
                    0x3fc0_0000,
                    0xc010_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_0000,
                    0xffc0_0000,
                    0x7fc0_002a,
                    0x7f80_0001,
                    0xff95_5555,
                    0x0000_0001,
                ],
                Self::F64 => [
                    0x0000_0000_0000_0000,
                    0x8000_0000_0000_0000,
                    0x3ff8_0000_0000_0000,
                    0xc002_0000_0000_0000,
                    0x7ff0_0000_0000_0000,
                    0xfff0_0000_0000_0000,
                    0x7ff8_0000_0000_0000,
                    0xfff8_0000_0000_0000,
                    0x7ff8_0000_0000_002a,
                    0x7ff0_0000_0000_0001,
                    0xfff5_5555_5555_5555,
                    0x0000_0000_0000_0001,
                ],
            }
        }

        /// The bits hashed for `bits` under `flags` (spec Section 6): a NaN is an all-ones
        /// exponent with a non-zero fraction.
        const fn normalized(self, bits: u64, flags: Flags) -> u64 {
            let (sign, exponent, fraction, canonical) = self.layout();
            let nan = (bits & exponent) == exponent && (bits & fraction) != 0;
            if flags.nan && nan {
                canonical
            } else if flags.negative_zero && bits == sign {
                0
            } else {
                bits
            }
        }

        /// A float array of this width with the values `bits` and validity `nulls`.
        fn array(self, bits: &[u64], nulls: Option<NullBuffer>) -> ArrayRef {
            let buffer = match self {
                Self::F16 => Buffer::from_vec(
                    bits.iter()
                        .map(|b| u16::try_from(*b).unwrap())
                        .collect::<Vec<_>>(),
                ),
                Self::F32 => Buffer::from_vec(
                    bits.iter()
                        .map(|b| u32::try_from(*b).unwrap())
                        .collect::<Vec<_>>(),
                ),
                Self::F64 => Buffer::from_vec(bits.to_vec()),
            };
            make_array(
                ArrayData::builder(self.data_type())
                    .len(bits.len())
                    .add_buffer(buffer)
                    .nulls(nulls)
                    .build()
                    .unwrap(),
            )
        }
    }

    /// One setting of the two float options.
    #[derive(Clone, Copy)]
    struct Flags {
        negative_zero: bool,
        nan: bool,
    }

    const OFF: Flags = Flags {
        negative_zero: false,
        nan: false,
    };
    const ZERO_ONLY: Flags = Flags {
        negative_zero: true,
        nan: false,
    };
    const NAN_ONLY: Flags = Flags {
        negative_zero: false,
        nan: true,
    };
    const BOTH: Flags = Flags {
        negative_zero: true,
        nan: true,
    };
    /// The settings with an option on.
    const ON: [Flags; 3] = [ZERO_ONLY, NAN_ONLY, BOTH];

    impl Flags {
        const fn config(self, ordered_columns: bool, ordered_struct_fields: bool) -> HasherConfig {
            HasherConfig {
                include_metadata: false,
                ordered_columns,
                ordered_struct_fields,
                normalize_negative_zero: self.negative_zero,
                normalize_nan: self.nan,
            }
        }

        const fn plain(self) -> HasherConfig {
            self.config(false, false)
        }
    }

    // ---------------------------------------------------------------------------------------
    // the normalized copy: every float buffer at every depth (dictionary values are child data)
    // ---------------------------------------------------------------------------------------

    fn normalized_buffer(buffer: &Buffer, width: Width, flags: Flags) -> Buffer {
        match width {
            Width::F16 => Buffer::from_vec(
                buffer
                    .typed_data::<u16>()
                    .iter()
                    .map(|b| u16::try_from(width.normalized(u64::from(*b), flags)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Width::F32 => Buffer::from_vec(
                buffer
                    .typed_data::<u32>()
                    .iter()
                    .map(|b| u32::try_from(width.normalized(u64::from(*b), flags)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Width::F64 => Buffer::from_vec(
                buffer
                    .typed_data::<u64>()
                    .iter()
                    .map(|b| width.normalized(*b, flags))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn normalized_data(data: &ArrayData, flags: Flags) -> ArrayData {
        let children: Vec<ArrayData> = data
            .child_data()
            .iter()
            .map(|child| normalized_data(child, flags))
            .collect();
        let buffers = Width::of(data.data_type()).map_or_else(
            || data.buffers().to_vec(),
            |width| {
                vec![normalized_buffer(
                    data.buffers().first().unwrap(),
                    width,
                    flags,
                )]
            },
        );
        data.clone()
            .into_builder()
            .buffers(buffers)
            .child_data(children)
            .build()
            .unwrap()
    }

    fn normalized(array: &ArrayRef, flags: Flags) -> ArrayRef {
        make_array(normalized_data(&array.to_data(), flags))
    }

    fn normalized_batch(batch: &RecordBatch, flags: Flags) -> RecordBatch {
        let columns = batch
            .columns()
            .iter()
            .map(|column| normalized(column, flags))
            .collect();
        RecordBatch::try_new(batch.schema(), columns).unwrap()
    }

    // ---------------------------------------------------------------------------------------
    // the invariant, through every entry point
    // ---------------------------------------------------------------------------------------

    fn streamed(batch: &RecordBatch, split: usize, config: HasherConfig) -> Vec<u8> {
        let mut digester = ArrowDigester::new(&batch.schema(), config);
        digester.update(&batch.slice(0, split));
        digester.update(&batch.slice(split, batch.num_rows().checked_sub(split).unwrap()));
        digester.finalize()
    }

    /// Hash `batch` with `flags` on through every entry point and compare with the normalized
    /// copy with the options off. Returns whether the copy hashes differently from the data with
    /// the options off (the options had something to normalize).
    fn check_invariant(
        batch: &RecordBatch,
        flags: Flags,
        ordered: (bool, bool),
        split: usize,
        why: &str,
    ) -> bool {
        let on = flags.config(ordered.0, ordered.1);
        let off = OFF.config(ordered.0, ordered.1);
        let copy = normalized_batch(batch, flags);
        let want = ArrowDigester::hash_record_batch(&copy, off);
        assert_eq!(
            ArrowDigester::hash_record_batch(batch, on),
            want,
            "{why}: hash_record_batch"
        );
        assert_eq!(streamed(batch, split, on), want, "{why}: stream");
        assert_eq!(
            streamed(&copy, split, off),
            want,
            "{why}: stream of the copy"
        );
        let prepared = PreparedSchema::new(&batch.schema(), on);
        assert_eq!(prepared.hash_record_batch(batch), want, "{why}: prepared");
        let mut digester = prepared.digester();
        digester.update(batch);
        assert_eq!(digester.finalize(), want, "{why}: prepared digester");
        // the copy holds nothing to normalize: with the options on it hashes as with them off
        assert_eq!(
            ArrowDigester::hash_record_batch(&copy, on),
            want,
            "{why}: normalized data with the options on"
        );
        for (column, copied) in batch.columns().iter().zip(copy.columns()) {
            let array_want = ArrowDigester::hash_array_with_config(copied.as_ref(), off);
            assert_eq!(
                ArrowDigester::hash_array_with_config(column.as_ref(), on),
                array_want,
                "{why}: hash_array"
            );
            let head = column.slice(0, split);
            let tail = column.slice(split, column.len().checked_sub(split).unwrap());
            assert_eq!(
                ArrowDigester::hash_array_chunks(
                    column.data_type(),
                    &[head.as_ref(), tail.as_ref()],
                    on
                ),
                array_want,
                "{why}: hash_array_chunks"
            );
        }
        ArrowDigester::hash_record_batch(batch, off) != want
    }

    // ---------------------------------------------------------------------------------------
    // explicit inputs
    // ---------------------------------------------------------------------------------------

    /// Eight values in one order: 1.5, -0.0, payload NaN, +0.0, -NaN, signaling NaN, -2.25,
    /// canonical NaN (indices into `Width::values`).
    const MIXED: [usize; 8] = [2, 1, 8, 0, 7, 9, 3, 6];
    /// The same values in another order.
    const MIXED_REORDERED: [usize; 8] = [6, 9, 0, 3, 1, 7, 8, 2];

    fn bits_of(width: Width, indices: &[usize]) -> Vec<u64> {
        let values = width.values();
        indices
            .iter()
            .map(|i| values.get(*i).copied().unwrap())
            .collect()
    }

    fn item(data_type: DataType) -> Arc<Field> {
        Arc::new(Field::new("item", data_type, true))
    }

    /// `values` (8 values) as a column of each nesting form.
    fn nesting_forms(values: &ArrayRef) -> Vec<(&'static str, ArrayRef)> {
        let data_type = values.data_type().clone();
        let list: ArrayRef = Arc::new(
            ListArray::try_new(
                item(data_type.clone()),
                OffsetBuffer::from_lengths([3, 0, 5]),
                Arc::clone(values),
                None,
            )
            .unwrap(),
        );
        // the middle row is null: its two items are never hashed
        let large_list: ArrayRef = Arc::new(
            LargeListArray::try_new(
                item(data_type.clone()),
                OffsetBuffer::from_lengths([2, 2, 4]),
                Arc::clone(values),
                Some(NullBuffer::from(vec![true, false, true])),
            )
            .unwrap(),
        );
        let pairs: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(item(data_type.clone()), 2, Arc::clone(values), None)
                .unwrap(),
        );
        let pairs_of_pairs: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                item(pairs.data_type().clone()),
                2,
                Arc::clone(&pairs),
                None,
            )
            .unwrap(),
        );
        let structure: ArrayRef = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("v", data_type, true)]),
            vec![Arc::clone(values)],
            None,
        ));
        // keys reach -0.0 and a NaN other than the canonical one in both value orders
        let keys = Int32Array::from(vec![Some(3), None, Some(0), Some(7), Some(1), Some(4)]);
        let dictionary: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::clone(values)).unwrap());
        vec![
            ("top level", Arc::clone(values)),
            ("sliced", values.slice(1, 6)),
            ("list", list),
            ("large list with a null row", large_list),
            ("fixed-size list", pairs),
            ("fixed-size list of fixed-size lists", pairs_of_pairs),
            ("struct", structure),
            ("dictionary values", dictionary),
        ]
    }

    fn one_column(column: &ArrayRef) -> RecordBatch {
        let field = Field::new("x", column.data_type().clone(), true);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::clone(column)]).unwrap()
    }

    #[test]
    fn each_option_alone_and_both_in_every_nesting_form() {
        for width in Width::ALL {
            for order in [MIXED, MIXED_REORDERED] {
                let values = width.array(&bits_of(width, &order), None);
                for (form, column) in nesting_forms(&values) {
                    let batch = one_column(&column);
                    for flags in ON {
                        let why = format!(
                            "{} {form}: zero {} nan {}",
                            width.data_type(),
                            flags.negative_zero,
                            flags.nan
                        );
                        assert!(
                            check_invariant(&batch, flags, (false, false), 1, &why),
                            "{why}: the option must change this hash"
                        );
                    }
                    // the two options are independent: each alone differs from both
                    let hash =
                        |flags: Flags| ArrowDigester::hash_record_batch(&batch, flags.plain());
                    assert_ne!(
                        hash(ZERO_ONLY),
                        hash(BOTH),
                        "{form}: NaNs need normalize_nan"
                    );
                    assert_ne!(hash(NAN_ONLY), hash(BOTH), "{form}: -0.0 needs its option");
                    assert_ne!(hash(OFF), hash(ZERO_ONLY), "{form}: off hashes the bits");
                }
            }
        }
    }

    #[test]
    fn negative_zero_hashes_as_positive_zero_only_with_its_option() {
        for width in Width::ALL {
            // [1.5, -0.0, +0.0, -0.0] and [-0.0, -0.0, 1.5, +0.0], each against its +0.0 twin
            for (order, twin) in [([2, 1, 0, 1], [2, 0, 0, 0]), ([1, 1, 2, 0], [0, 0, 2, 0])] {
                let data = one_column(&width.array(&bits_of(width, &order), None));
                let zeroed = one_column(&width.array(&bits_of(width, &twin), None));
                let hash = |batch: &RecordBatch, flags: Flags| {
                    ArrowDigester::hash_record_batch(batch, flags.plain())
                };
                assert_eq!(hash(&data, ZERO_ONLY), hash(&zeroed, OFF));
                assert_eq!(hash(&data, BOTH), hash(&zeroed, OFF));
                assert_ne!(hash(&data, OFF), hash(&zeroed, OFF));
                assert_ne!(hash(&data, NAN_ONLY), hash(&zeroed, OFF));
            }
        }
    }

    #[test]
    fn every_nan_hashes_as_the_canonical_nan_only_with_its_option() {
        for width in Width::ALL {
            // canonical, -NaN, payload, signaling, negative signaling: in two orders
            for order in [[6, 7, 8, 9, 10], [10, 8, 6, 9, 7]] {
                let data = one_column(&width.array(&bits_of(width, &order), None));
                let canonical = one_column(&width.array(&bits_of(width, &[6; 5]), None));
                let hash = |batch: &RecordBatch, flags: Flags| {
                    ArrowDigester::hash_record_batch(batch, flags.plain())
                };
                assert_eq!(hash(&data, NAN_ONLY), hash(&canonical, OFF));
                assert_eq!(hash(&data, BOTH), hash(&canonical, OFF));
                assert_ne!(hash(&data, OFF), hash(&canonical, OFF));
                assert_ne!(hash(&data, ZERO_ONLY), hash(&canonical, OFF));
            }
        }
    }

    #[test]
    fn values_under_nulls_never_count() {
        for width in Width::ALL {
            let nulls = NullBuffer::from(vec![true, false, true, false, false]);
            // the null slots hold -0.0, a payload NaN, +0.0 / +0.0, +0.0, -NaN
            let first = width.array(&bits_of(width, &[2, 1, 3, 8, 0]), Some(nulls.clone()));
            let second = width.array(&bits_of(width, &[2, 0, 3, 0, 7]), Some(nulls));
            let in_struct = |values: &ArrayRef| -> ArrayRef {
                Arc::new(StructArray::new(
                    Fields::from(vec![Field::new("v", values.data_type().clone(), true)]),
                    vec![Arc::clone(values)],
                    Some(NullBuffer::from(vec![true, true, true, false, true])),
                ))
            };
            for flags in [OFF, ZERO_ONLY, NAN_ONLY, BOTH] {
                for (a, b) in [
                    (Arc::clone(&first), Arc::clone(&second)),
                    (in_struct(&first), in_struct(&second)),
                ] {
                    assert_eq!(
                        ArrowDigester::hash_record_batch(&one_column(&a), flags.plain()),
                        ArrowDigester::hash_record_batch(&one_column(&b), flags.plain())
                    );
                    assert_eq!(
                        ArrowDigester::hash_array_with_config(a.as_ref(), flags.plain()),
                        ArrowDigester::hash_array_with_config(b.as_ref(), flags.plain())
                    );
                }
            }
        }
    }

    #[test]
    fn other_types_are_untouched() {
        let neg_zero_64 = i64::from_le_bytes(0x8000_0000_0000_0000_u64.to_le_bytes());
        let nan_64 = i64::from_le_bytes(0xfff8_0000_0000_002a_u64.to_le_bytes());
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int16Array::from(vec![
                i16::from_le_bytes(0x8000_u16.to_le_bytes()),
                0,
                i16::from_le_bytes(0xfe00_u16.to_le_bytes()),
            ])),
            Arc::new(Int64Array::from(vec![neg_zero_64, nan_64, 0, 1])),
            Arc::new(UInt64Array::from(vec![
                0x7ff0_0000_0000_0001,
                0x8000_0000_0000_0000,
                7,
            ])),
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    [
                        0x8000_0000_0000_0000_u64.to_le_bytes(),
                        0xfff8_0000_0000_002a_u64.to_le_bytes(),
                    ]
                    .into_iter(),
                )
                .unwrap(),
            ),
        ];
        for column in &columns {
            let batch = one_column(column);
            let off = ArrowDigester::hash_record_batch(&batch, OFF.plain());
            for flags in ON {
                assert_eq!(ArrowDigester::hash_record_batch(&batch, flags.plain()), off);
                assert_eq!(
                    ArrowDigester::hash_array_with_config(column.as_ref(), flags.plain()),
                    ArrowDigester::hash_array_with_config(column.as_ref(), OFF.plain())
                );
            }
        }
    }

    #[test]
    fn the_options_are_not_in_the_hash() {
        for width in Width::ALL {
            // clean values (1.5, -2.25, +0.0, +inf, canonical NaN) in two orders and two forms
            for order in [[2, 3, 0, 4, 6], [6, 0, 4, 3, 2]] {
                let values = width.array(&bits_of(width, &order), None);
                for column in [
                    Arc::clone(&values),
                    Arc::new(
                        ListArray::try_new(
                            item(values.data_type().clone()),
                            OffsetBuffer::from_lengths([2, 3]),
                            Arc::clone(&values),
                            None,
                        )
                        .unwrap(),
                    ),
                ] {
                    let batch = one_column(&column);
                    let schema = batch.schema();
                    let off = OFF.plain();
                    for flags in ON {
                        assert_eq!(
                            ArrowDigester::hash_schema(&schema, flags.plain()),
                            ArrowDigester::hash_schema(&schema, off)
                        );
                        assert_eq!(
                            ArrowDigester::hash_record_batch(&batch, flags.plain()),
                            ArrowDigester::hash_record_batch(&batch, off)
                        );
                        assert_eq!(
                            ArrowDigester::hash_array_with_config(column.as_ref(), flags.plain()),
                            ArrowDigester::hash_array_with_config(column.as_ref(), off)
                        );
                        let prepared = PreparedSchema::new(&schema, flags.plain());
                        assert_eq!(prepared.config(), flags.plain());
                        assert_eq!(
                            prepared.schema_digest(),
                            ArrowDigester::hash_schema(&schema, off)
                        );
                    }
                }
            }
        }
    }

    /// Float values around and across the 16 KiB blocks in which the normalized bytes are hashed:
    /// valid-value byte counts just below, at and just above one block, two blocks, and three
    /// blocks with a partial tail; without nulls and with a null after every fourth valid value;
    /// every entry point, the stream split inside a block.
    #[test]
    fn normalized_bytes_across_block_boundaries() {
        const BLOCK: usize = 0x4000;
        for width in Width::ALL {
            let per_block = BLOCK.checked_div(width.bytes()).unwrap();
            let counts = [
                per_block.checked_sub(1).unwrap(),
                per_block,
                per_block.checked_add(1).unwrap(),
                per_block.checked_mul(2).unwrap(),
                per_block.checked_mul(3).unwrap().checked_add(5).unwrap(),
            ];
            let values = width.values();
            for valid_count in counts {
                for with_nulls in [false, true] {
                    let mut bits = Vec::new();
                    let mut valid = Vec::new();
                    for k in 0..valid_count {
                        let index = k.checked_rem(values.len()).unwrap();
                        bits.push(values.get(index).copied().unwrap());
                        valid.push(true);
                        if with_nulls && k.checked_rem(4) == Some(3) {
                            bits.push(values.get(1).copied().unwrap()); // -0.0 in a null slot
                            valid.push(false);
                        }
                    }
                    let split = bits.len().checked_div(3).unwrap();
                    let nulls = with_nulls.then(|| NullBuffer::from(valid));
                    let batch = one_column(&width.array(&bits, nulls));
                    for flags in ON {
                        let why = format!(
                            "{} {valid_count} valid values, nulls {with_nulls}, zero {} nan {}",
                            width.data_type(),
                            flags.negative_zero,
                            flags.nan
                        );
                        assert!(
                            check_invariant(&batch, flags, (false, false), split, &why),
                            "{why}: the option must change this hash"
                        );
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // the property over generated tables
    // ---------------------------------------------------------------------------------------

    /// A deterministic pseudo-random sequence (a 64-bit LCG).
    struct Rng(u64);

    impl Rng {
        const fn draw(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0.wrapping_shr(33)
        }

        /// A number in `0..n` (`n > 0`).
        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.draw().checked_rem(u64::try_from(n).unwrap()).unwrap()).unwrap()
        }

        /// True with probability `percent` / 100.
        const fn chance(&mut self, percent: u64) -> bool {
            self.draw().wrapping_rem(100) < percent
        }

        fn pick<T: Copy>(&mut self, items: &[T]) -> T {
            let index = self.below(items.len());
            items.get(index).copied().unwrap()
        }

        fn bools(&mut self, len: usize, percent: u64) -> Vec<bool> {
            repeat_with(|| self.chance(percent)).take(len).collect()
        }
    }

    enum Shape {
        Float(Width),
        /// `Int64` holding the bits of special float values: never normalized.
        Int,
        List {
            large: bool,
            item: Box<Self>,
            item_nullable: bool,
        },
        FixedSizeList {
            size: i32,
            item: Box<Self>,
            item_nullable: bool,
        },
        Struct(Vec<(&'static str, Self, bool)>),
        /// `Int32` keys over values of the inner shape (a float or a struct of floats).
        Dictionary(Box<Self>),
    }

    fn random_shape(rng: &mut Rng, depth: usize) -> Shape {
        let leaf = |r: &mut Rng| {
            if r.chance(85) {
                Shape::Float(r.pick(&Width::ALL))
            } else {
                Shape::Int
            }
        };
        if depth >= 3 || rng.chance(35) {
            return leaf(rng);
        }
        let next = depth.checked_add(1).unwrap();
        match rng.below(5) {
            0 => Shape::List {
                large: rng.chance(50),
                item: Box::new(random_shape(rng, next)),
                item_nullable: rng.chance(70),
            },
            1 => Shape::FixedSizeList {
                size: rng.pick(&[1, 2, 3]),
                item: Box::new(random_shape(rng, next)),
                item_nullable: rng.chance(70),
            },
            2 => {
                let inner = if rng.chance(50) {
                    Shape::Struct(vec![("f", leaf(rng), true), ("g", leaf(rng), false)])
                } else {
                    Shape::Float(rng.pick(&Width::ALL))
                };
                Shape::Dictionary(Box::new(inner))
            }
            _ => {
                let names = ["z", "a", "m"];
                let count = rng.below(3).checked_add(1).unwrap();
                Shape::Struct(
                    names
                        .into_iter()
                        .take(count)
                        .map(|name| (name, random_shape(rng, next), rng.chance(60)))
                        .collect(),
                )
            }
        }
    }

    fn data_type(shape: &Shape) -> DataType {
        match shape {
            Shape::Float(width) => width.data_type(),
            Shape::Int => DataType::Int64,
            Shape::List {
                large,
                item,
                item_nullable,
            } => {
                let field = Arc::new(Field::new("item", data_type(item), *item_nullable));
                if *large {
                    DataType::LargeList(field)
                } else {
                    DataType::List(field)
                }
            }
            Shape::FixedSizeList {
                size,
                item,
                item_nullable,
            } => DataType::FixedSizeList(
                Arc::new(Field::new("item", data_type(item), *item_nullable)),
                *size,
            ),
            Shape::Struct(children) => DataType::Struct(struct_fields(children)),
            Shape::Dictionary(inner) => {
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(data_type(inner)))
            }
        }
    }

    fn struct_fields(children: &[(&'static str, Shape, bool)]) -> Fields {
        children
            .iter()
            .map(|(name, child, nullable)| Field::new(*name, data_type(child), *nullable))
            .collect()
    }

    /// Validity of a node: random when nullable; else null only where `masked` (the enclosing
    /// struct) is null, sometimes (hidden nulls).
    fn validity(
        rng: &mut Rng,
        len: usize,
        nullable: bool,
        masked: Option<&NullBuffer>,
    ) -> Option<NullBuffer> {
        if nullable {
            return rng.chance(80).then(|| NullBuffer::from(rng.bools(len, 75)));
        }
        let mask = masked.filter(|nulls| nulls.null_count() > 0)?;
        let valid: Vec<bool> = mask.iter().map(|v| v || rng.chance(50)).collect();
        Some(NullBuffer::from(valid))
    }

    fn lengths(rng: &mut Rng, len: usize) -> Vec<usize> {
        repeat_with(|| rng.below(4)).take(len).collect()
    }

    fn build(
        rng: &mut Rng,
        shape: &Shape,
        len: usize,
        nullable: bool,
        masked: Option<&NullBuffer>,
    ) -> ArrayRef {
        let nulls = validity(rng, len, nullable, masked);
        match shape {
            Shape::Float(width) => {
                let values = width.values();
                let bits: Vec<u64> = repeat_with(|| rng.pick(&values)).take(len).collect();
                width.array(&bits, nulls)
            }
            Shape::Int => {
                let values = Width::F64.values();
                let ints: Vec<i64> =
                    repeat_with(|| i64::from_le_bytes(rng.pick(&values).to_le_bytes()))
                        .take(len)
                        .collect();
                Arc::new(Int64Array::new(ScalarBuffer::from(ints), nulls))
            }
            Shape::List {
                large,
                item,
                item_nullable,
            } => {
                let sizes = lengths(rng, len);
                let total = sizes.iter().sum();
                let field = Arc::new(Field::new("item", data_type(item), *item_nullable));
                let child = build(rng, item, total, *item_nullable, None);
                if *large {
                    Arc::new(
                        LargeListArray::try_new(
                            field,
                            OffsetBuffer::from_lengths(sizes),
                            child,
                            nulls,
                        )
                        .unwrap(),
                    )
                } else {
                    Arc::new(
                        ListArray::try_new(field, OffsetBuffer::from_lengths(sizes), child, nulls)
                            .unwrap(),
                    )
                }
            }
            Shape::FixedSizeList {
                size,
                item,
                item_nullable,
            } => {
                let count = len.checked_mul(usize::try_from(*size).unwrap()).unwrap();
                let field = Arc::new(Field::new("item", data_type(item), *item_nullable));
                let child = build(rng, item, count, *item_nullable, None);
                Arc::new(FixedSizeListArray::try_new(field, *size, child, nulls).unwrap())
            }
            Shape::Struct(children) => {
                let columns = children
                    .iter()
                    .map(|(_, child, child_nullable)| {
                        build(rng, child, len, *child_nullable, nulls.as_ref())
                    })
                    .collect();
                Arc::new(StructArray::try_new(struct_fields(children), columns, nulls).unwrap())
            }
            Shape::Dictionary(inner) => {
                let values = build(rng, inner, 4, false, None);
                let keys: Vec<i32> = repeat_with(|| rng.pick(&[0, 1, 2, 3])).take(len).collect();
                let key_array = Int32Array::new(ScalarBuffer::from(keys), nulls);
                Arc::new(DictionaryArray::<Int32Type>::try_new(key_array, values).unwrap())
            }
        }
    }

    fn random_batch(rng: &mut Rng) -> RecordBatch {
        let len = rng.pick(&[0, 1, 2, 5, 8, 13]);
        let count = rng.below(3).checked_add(1).unwrap();
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for name in ["c", "a", "b"].into_iter().take(count) {
            let shape = random_shape(rng, 0);
            let nullable = rng.chance(70);
            columns.push(build(rng, &shape, len, nullable, None));
            fields.push(Field::new(name, data_type(&shape), nullable));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        if len > 2 && rng.chance(40) {
            batch.slice(1, len.checked_sub(2).unwrap())
        } else {
            batch
        }
    }

    #[test]
    fn hashing_with_an_option_on_equals_hashing_the_normalized_copy_with_it_off() {
        let cases = 400_u64;
        let mut normalizing = [0_usize; 3];
        for seed in 0..cases {
            let mut rng = Rng(seed.wrapping_add(0x1252));
            let batch = random_batch(&mut rng);
            let ordered = (rng.chance(30), rng.chance(30));
            let split = rng.below(batch.num_rows().checked_add(1).unwrap());
            for (count, flags) in normalizing.iter_mut().zip(ON) {
                let why = format!(
                    "seed {seed}, zero {}, nan {}, schema {}",
                    flags.negative_zero,
                    flags.nan,
                    batch.schema()
                );
                if check_invariant(&batch, flags, ordered, split, &why) {
                    *count = count.checked_add(1).unwrap();
                }
            }
        }
        // the generated data must exercise the options, not only clean values
        for count in normalizing {
            assert!(
                count >= 100,
                "only {count} of {cases} tables had values to normalize"
            );
        }
    }
}
