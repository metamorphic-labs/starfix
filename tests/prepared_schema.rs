//! The prepared-schema interface (ITL-1177): `ArrowDigester::prepare` / `PreparedSchema`.
//!
//! The prepared path must be byte-identical to the one-shot path for every input and every
//! option combination, and must refuse a batch whose schema is not the prepared one under the
//! prepared options (byte-layout spec Section 5): a batch is accepted iff its schema has the
//! prepared schema digest under the prepared options and it passes Section 7.

#[path = "common/leaf_types.rs"]
mod leaf_types;
#[path = "common/nested_types.rs"]
mod nested_types;

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::expect_used, reason = "Okay in test")]
    #![expect(clippy::panic, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]
    #![expect(clippy::too_many_lines, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::fs;
    use std::io::Cursor;
    use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};
    use std::sync::{Arc, Once};

    use arrow::array::{
        Array as _, ArrayRef, DictionaryArray, FixedSizeListArray, Int32Array, Int64Array,
        Int8Array, LargeListArray, LargeStringArray, ListArray, RecordBatch, StringArray,
        StructArray,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Int32Type, Int8Type};
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    use hex::encode;
    use serde::Deserialize;
    use starfix::{ArrowDigester, HasherConfig, PreparedSchema};

    use crate::leaf_types::{leaf_types, make, wrap, Position};
    use crate::nested_types::{all_types, arrow_type, build, row_sets, vals};

    // ---------------------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------------------

    /// The 8 option combinations.
    fn all_configs() -> Vec<HasherConfig> {
        (0_u8..8)
            .map(|bits| HasherConfig {
                include_metadata: bits & 0x04 != 0,
                ordered_columns: bits & 0x01 != 0,
                ordered_struct_fields: bits & 0x02 != 0,
                ..HasherConfig::default()
            })
            .collect()
    }

    /// Silence the default panic hook once: many cases below expect a panic.
    fn quiet_panics() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let default = take_hook();
            set_hook(Box::new(move |info| {
                let msg = info
                    .payload()
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| info.payload().downcast_ref::<&str>().copied())
                    .unwrap_or("");
                // Show unexpected panics only.
                if !msg.contains("does not match")
                    && !msg.contains("not supported")
                    && !msg.contains("Ambiguous field path")
                    && !msg.contains("cannot be hashed")
                    && !msg.contains("non-nullable")
                {
                    default(info);
                }
            }));
        });
    }

    fn try_hash<F>(f: F) -> Result<Vec<u8>, String>
    where
        F: FnOnce() -> Vec<u8>,
    {
        quiet_panics();
        catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default()
        })
    }

    fn one_shot(batch: &RecordBatch, config: HasherConfig) -> Result<Vec<u8>, String> {
        try_hash(|| ArrowDigester::hash_record_batch(batch, config))
    }

    fn prepared(schema: &Schema, config: HasherConfig) -> Result<PreparedSchema, String> {
        quiet_panics();
        catch_unwind(|| ArrowDigester::prepare(schema, config)).map_err(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default()
        })
    }

    fn assert_refused(p: &PreparedSchema, batch: &RecordBatch, why: &str) {
        let r = try_hash(|| p.hash_record_batch(batch));
        let msg = r.expect_err(&format!("expected a refusal: {why}"));
        assert!(
            msg.contains("does not match the prepared schema")
                || msg.contains("Ambiguous field path"),
            "{why}: unexpected panic message {msg:?}"
        );
    }

    fn assert_accepted_like_one_shot(p: &PreparedSchema, batch: &RecordBatch, why: &str) {
        let got = try_hash(|| p.hash_record_batch(batch))
            .unwrap_or_else(|m| panic!("expected acceptance: {why}: {m}"));
        assert_eq!(
            got,
            one_shot(batch, p.config()).unwrap(),
            "{why}: prepared != one-shot"
        );
    }

    // ---------------------------------------------------------------------------------------
    // golden vectors
    // ---------------------------------------------------------------------------------------

    #[derive(Deserialize)]
    struct GoldenFixture {
        vectors: Vec<GoldenVector>,
    }

    #[derive(Deserialize)]
    #[expect(clippy::struct_excessive_bools, reason = "One field per option")]
    struct GoldenVector {
        id: String,
        method: String,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
        /// The float options: absent (false) in every vector that does not set them.
        #[serde(default)]
        normalize_negative_zero: bool,
        #[serde(default)]
        normalize_nan: bool,
        ipc_b64: String,
        expected_hash: String,
    }

    impl GoldenVector {
        const fn config(&self) -> HasherConfig {
            HasherConfig {
                include_metadata: self.include_metadata,
                ordered_columns: self.ordered_columns,
                ordered_struct_fields: self.ordered_struct_fields,
                normalize_negative_zero: self.normalize_negative_zero,
                normalize_nan: self.normalize_nan,
            }
        }
    }

    fn load_fixture() -> GoldenFixture {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/format2_v0.5.json"
        );
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    /// As in `tests/golden_vectors_format2.rs`: vectors arrow-rs's IPC validation rejects.
    const UNVALIDATED_VECTORS: [&str; 6] = [
        "hidden_null_list_rb_f0",
        "hidden_null_large_list_rb_f0",
        "fsl_fsl_hidden_nulls_rb_f0",
        "fsl_fsl_hidden_nulls_array_f0",
        "fsl_struct_masked_nulls_rb_f0",
        "fsl_struct2_masked_nulls_rb_f0",
    ];

    fn decode_ipc(v: &GoldenVector) -> (SchemaRef, Vec<RecordBatch>) {
        let bytes = STANDARD.decode(&v.ipc_b64).unwrap();
        let skip =
            UNVALIDATED_VECTORS.contains(&v.id.as_str()) || v.id.starts_with("nested_hidden_");
        #[expect(unsafe_code, reason = "Decoding vectors arrow-rs would not validate")]
        // SAFETY: as in tests/golden_vectors_format2.rs.
        let reader = unsafe {
            StreamReader::try_new(Cursor::new(bytes), None)
                .unwrap()
                .with_skip_validation(skip)
        };
        let schema = reader.schema();
        let batches = reader.map(|b| b.expect("IPC batch")).collect();
        (schema, batches)
    }

    /// Every golden vector except `hash_array` (which has no schema), through the prepared
    /// path, equals its committed hash.
    #[test]
    fn prepared_path_matches_every_golden_vector() {
        let fixture = load_fixture();
        let mut checked = 0;
        for v in &fixture.vectors {
            let config = v.config();
            let (schema, batches) = decode_ipc(v);
            let got = match v.method.as_str() {
                // `prepare` succeeds whenever `hash_schema` does (also for schemas whose data
                // cannot be hashed, such as Map)
                "hash_schema" => ArrowDigester::prepare(&schema, config).schema_digest(),
                "hash_record_batch" => {
                    assert_eq!(batches.len(), 1, "{}", v.id);
                    let batch = &batches[0];
                    // The reader's schema is equal to, but maybe not the same `Arc` as, the
                    // batch's; check both the batch's own schema and the reader's.
                    let a = ArrowDigester::prepare(&batch.schema(), config);
                    let b = ArrowDigester::prepare(&schema, config);
                    let ha = a.hash_record_batch(batch);
                    assert_eq!(ha, b.hash_record_batch(batch), "{}", v.id);
                    // A prepared schema is reusable: hashing again gives the same hash.
                    assert_eq!(ha, a.hash_record_batch(batch), "{}", v.id);
                    ha
                }
                "hash_stream" => {
                    let p = ArrowDigester::prepare(&schema, config);
                    let mut d = p.digester();
                    for b in &batches {
                        d.update(b);
                    }
                    // A second digester of the same prepared schema starts fresh.
                    let mut d2 = p.digester();
                    for b in &batches {
                        d2.update(b);
                    }
                    let h = d.finalize();
                    assert_eq!(h, d2.finalize(), "{}", v.id);
                    h
                }
                "hash_array" => continue,
                other => panic!("unknown method {other}"),
            };
            assert_eq!(encode(got), v.expected_hash, "vector {}", v.id);
            checked += 1;
        }
        assert!(checked >= 370, "only {checked} vectors checked");
    }

    /// Every golden input, under every one of the 8 option combinations: the prepared path
    /// equals the one-shot path (also when both reject the input).
    #[test]
    fn prepared_equals_one_shot_for_every_option_combination() {
        let fixture = load_fixture();
        let mut compared = 0;
        for v in fixture.vectors.iter().filter(|v| v.method != "hash_array") {
            let (schema, batches) = decode_ipc(v);
            for config in all_configs() {
                // schema digest
                assert_eq!(
                    try_hash(|| ArrowDigester::hash_schema(&schema, config)),
                    try_hash(|| ArrowDigester::prepare(&schema, config).schema_digest()),
                    "{} {config:?}: schema digest",
                    v.id
                );
                // stream
                let one_shot_stream = try_hash(|| {
                    let mut d = ArrowDigester::new(&schema, config);
                    for b in &batches {
                        d.update(b);
                    }
                    d.finalize()
                });
                let prepared_stream = try_hash(|| {
                    let mut d = ArrowDigester::prepare(&schema, config).digester();
                    for b in &batches {
                        d.update(b);
                    }
                    d.finalize()
                });
                assert_eq!(
                    one_shot_stream, prepared_stream,
                    "{} {config:?}: stream",
                    v.id
                );
                // every batch on its own, one prepared schema reused for all of them
                if let Ok(p) = prepared(&schema, config) {
                    for b in &batches {
                        let prepared_batch = try_hash(|| p.hash_record_batch(b));
                        assert_eq!(
                            one_shot(b, config),
                            prepared_batch,
                            "{} {config:?}: batch",
                            v.id
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000, "only {compared} batch comparisons");
    }

    /// The same-schema rule as a property over every pair of golden record-batch inputs and
    /// every option combination: preparing the schema of input `i` and hashing the batch of
    /// input `j` is accepted iff the two schemas have the same schema digest under the options
    /// (and the batch passes Section 7), and an accepted batch hashes exactly like the one-shot
    /// path. It never hashes with an analysis that is not its own.
    #[test]
    fn accepted_iff_same_schema_digest_and_then_equal_to_one_shot() {
        let fixture = load_fixture();
        // one batch per distinct schema
        let mut inputs: Vec<RecordBatch> = Vec::new();
        for v in fixture
            .vectors
            .iter()
            .filter(|v| v.method == "hash_record_batch")
        {
            let (_, batches) = decode_ipc(v);
            let b = batches.into_iter().next().unwrap();
            if !inputs.iter().any(|x| x.schema() == b.schema()) {
                inputs.push(b);
            }
        }
        assert!(inputs.len() > 50, "only {} distinct schemas", inputs.len());
        let (mut accepted, mut refused) = (0_usize, 0_usize);
        for config in all_configs() {
            let digests: Vec<Option<Vec<u8>>> = inputs
                .iter()
                .map(|b| try_hash(|| ArrowDigester::hash_schema(&b.schema(), config)).ok())
                .collect();
            for (i, bi) in inputs.iter().enumerate() {
                let Ok(p) = prepared(&bi.schema(), config) else {
                    continue;
                };
                for (j, bj) in inputs.iter().enumerate() {
                    let result = try_hash(|| p.hash_record_batch(bj));
                    let same = digests[j].is_some() && digests[j] == digests[i];
                    match result {
                        Ok(h) => {
                            assert!(same, "accepted with a different schema: {i} {j} {config:?}");
                            assert_eq!(Ok(h), one_shot(bj, config), "{i} {j} {config:?}");
                            accepted += 1;
                        }
                        Err(msg) => {
                            // Refused for the schema, or rejected for the data exactly as
                            // the one-shot path rejects it.
                            if same {
                                assert!(
                                    !msg.contains("does not match"),
                                    "refused with the same schema digest: {i} {j} {config:?}"
                                );
                                assert!(one_shot(bj, config).is_err(), "{i} {j} {config:?}");
                            } else {
                                assert!(
                                    msg.contains("does not match the prepared schema"),
                                    "{i} {j} {config:?}: {msg}"
                                );
                                refused += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            accepted > 400 && refused > 10_000,
            "{accepted} accepted, {refused} refused"
        );
    }

    // ---------------------------------------------------------------------------------------
    // sweeps: leaf types and nested types, one prepared schema reused for many tables
    // ---------------------------------------------------------------------------------------

    #[test]
    fn leaf_sweep_prepared_equals_one_shot() {
        let positions = [
            Position::Top,
            Position::InStruct,
            Position::InList,
            Position::InNullableStruct,
            Position::InNullableList,
        ];
        for dt in leaf_types() {
            for nullable in [false, true] {
                for position in positions {
                    let seed_sets: Vec<Vec<Option<i64>>> = if nullable {
                        vec![vec![], vec![Some(1)], vec![Some(1), None, Some(3)]]
                    } else {
                        vec![vec![], vec![Some(1)], vec![Some(1), Some(2), Some(3)]]
                    };
                    let batches: Vec<RecordBatch> = seed_sets
                        .iter()
                        .map(|s| wrap(position, &make(&dt, s), nullable, s))
                        .collect();
                    for config in all_configs() {
                        let p = ArrowDigester::prepare(&batches[0].schema(), config);
                        for b in &batches {
                            assert_eq!(
                                p.hash_record_batch(b),
                                ArrowDigester::hash_record_batch(b, config),
                                "{dt} nullable={nullable} {position:?} {config:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nested_sweep_prepared_equals_one_shot() {
        let types = all_types();
        assert_eq!(types.len(), 420);
        for node in &types {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "c",
                arrow_type(node),
                node.nullable,
            )]));
            let batches: Vec<RecordBatch> = row_sets(&vals(node))
                .iter()
                .map(|rows| RecordBatch::try_new(Arc::clone(&schema), vec![build(node, rows)]))
                .map(Result::unwrap)
                .collect();
            for config in all_configs() {
                let p = ArrowDigester::prepare(&schema, config);
                for b in &batches {
                    assert_eq!(
                        p.hash_record_batch(b),
                        ArrowDigester::hash_record_batch(b, config),
                        "{node:?} {config:?}"
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // the same-schema rule, case by case
    // ---------------------------------------------------------------------------------------

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    /// The parts of the base schema a variant may change.
    #[derive(Clone)]
    struct Spec {
        id_name: &'static str,
        id_type: DataType,
        id_nullable: bool,
        id_meta: HashMap<String, String>,
        struct_children: Vec<(&'static str, bool)>, // (name, nullable), both Int32
        struct_child_meta: HashMap<String, String>,
        list_item_name: &'static str,
        list_item_meta: HashMap<String, String>,
        columns_reversed: bool,
        schema_meta: HashMap<String, String>,
        string_type: DataType, // Utf8 / LargeUtf8 / Dictionary(Int32, Utf8)
        large_list: bool,
    }

    fn base_spec() -> Spec {
        Spec {
            id_name: "id",
            id_type: DataType::Int64,
            id_nullable: false,
            id_meta: meta(&[
                ("ARROW:extension:name", "example.id"),
                ("ARROW:extension:metadata", "{\"v\":1}"),
            ]),
            struct_children: vec![("x", false), ("y", true)],
            struct_child_meta: meta(&[("unit", "m")]),
            list_item_name: "item",
            list_item_meta: meta(&[("k", "v")]),
            columns_reversed: false,
            schema_meta: meta(&[("source", "test")]),
            string_type: DataType::Utf8,
            large_list: false,
        }
    }

    /// A two-row batch of the schema described by `s` (the values do not depend on `s`).
    fn batch_of(s: &Spec) -> RecordBatch {
        let id: ArrayRef = match &s.id_type {
            DataType::Int64 => Arc::new(Int64Array::from(vec![1, 2])),
            DataType::Int32 => Arc::new(Int32Array::from(vec![1, 2])),
            other => panic!("unexpected type {other}"),
        };
        let id_field = Field::new(s.id_name, s.id_type.clone(), s.id_nullable)
            .with_metadata(s.id_meta.clone());

        let children: Vec<Field> = s
            .struct_children
            .iter()
            .map(|(n, nullable)| {
                let f = Field::new(*n, DataType::Int32, *nullable);
                if *n == "x" {
                    f.with_metadata(s.struct_child_meta.clone())
                } else {
                    f
                }
            })
            .collect();
        let child_arrays: Vec<ArrayRef> = s
            .struct_children
            .iter()
            .map(|(n, _)| -> ArrayRef {
                if *n == "x" {
                    Arc::new(Int32Array::from(vec![10, 20]))
                } else {
                    Arc::new(Int32Array::from(vec![Some(30), None]))
                }
            })
            .collect();
        let point = StructArray::new(Fields::from(children), child_arrays, None);
        let point_field = Field::new("point", point.data_type().clone(), true);

        let item = Arc::new(
            Field::new(s.list_item_name, DataType::Int32, true)
                .with_metadata(s.list_item_meta.clone()),
        );
        let values = Int32Array::from(vec![Some(1), None, Some(3)]);
        let offsets = [0_i32, 2, 3];
        let list: ArrayRef = if s.large_list {
            Arc::new(LargeListArray::new(
                item,
                OffsetBuffer::new(
                    offsets
                        .iter()
                        .map(|o| i64::from(*o))
                        .collect::<Vec<_>>()
                        .into(),
                ),
                Arc::new(values),
                None,
            ))
        } else {
            Arc::new(ListArray::new(
                item,
                OffsetBuffer::new(offsets.to_vec().into()),
                Arc::new(values),
                None,
            ))
        };
        let list_field = Field::new("values", list.data_type().clone(), true);

        let name: ArrayRef = match &s.string_type {
            DataType::Utf8 => Arc::new(StringArray::from(vec![Some("a"), None])),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![Some("a"), None])),
            DataType::Dictionary(_, _) => Arc::new(
                vec![Some("a"), None]
                    .into_iter()
                    .collect::<DictionaryArray<Int32Type>>(),
            ),
            other => panic!("unexpected type {other}"),
        };
        let name_field = Field::new("name", name.data_type().clone(), true);

        let mut fields = vec![id_field, point_field, list_field, name_field];
        let mut columns = vec![id, Arc::new(point) as ArrayRef, list, name];
        if s.columns_reversed {
            fields.reverse();
            columns.reverse();
        }
        let schema = Schema::new(fields).with_metadata(s.schema_meta.clone());
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    /// Each variant of the base schema, with the condition under which it must be refused.
    #[test]
    fn same_schema_rule_case_by_case() {
        type Refused = fn(HasherConfig) -> bool;
        let always: Refused = |_| true;
        let never: Refused = |_| false;
        let with_metadata: Refused = |c| c.include_metadata;
        let ordered_columns: Refused = |c| c.ordered_columns;
        let ordered_struct_fields: Refused = |c| c.ordered_struct_fields;

        let b = base_spec();
        let variants: Vec<(&str, Spec, Refused)> = vec![
            ("identical", b.clone(), never),
            (
                "column renamed",
                Spec {
                    id_name: "ID",
                    ..b.clone()
                },
                always,
            ),
            (
                "column type",
                Spec {
                    id_type: DataType::Int32,
                    ..b.clone()
                },
                always,
            ),
            (
                "column nullability",
                Spec {
                    id_nullable: true,
                    ..b.clone()
                },
                always,
            ),
            (
                "struct field renamed",
                Spec {
                    struct_children: vec![("x", false), ("z", true)],
                    ..b.clone()
                },
                always,
            ),
            (
                "struct field nullability",
                Spec {
                    struct_children: vec![("x", true), ("y", true)],
                    ..b.clone()
                },
                always,
            ),
            (
                "struct field reordered",
                Spec {
                    struct_children: vec![("y", true), ("x", false)],
                    ..b.clone()
                },
                ordered_struct_fields,
            ),
            (
                "columns reordered",
                Spec {
                    columns_reversed: true,
                    ..b.clone()
                },
                ordered_columns,
            ),
            (
                "column metadata value",
                Spec {
                    id_meta: meta(&[
                        ("ARROW:extension:name", "example.id"),
                        ("ARROW:extension:metadata", "{\"v\":2}"),
                    ]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "extension name",
                Spec {
                    id_meta: meta(&[
                        ("ARROW:extension:name", "example.other"),
                        ("ARROW:extension:metadata", "{\"v\":1}"),
                    ]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "column metadata removed",
                Spec {
                    id_meta: meta(&[]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "nested struct field metadata",
                Spec {
                    struct_child_meta: meta(&[("unit", "cm")]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "list element metadata",
                Spec {
                    list_item_meta: meta(&[("k", "w")]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "schema metadata",
                Spec {
                    schema_meta: meta(&[("source", "x")]),
                    ..b.clone()
                },
                with_metadata,
            ),
            (
                "schema metadata removed",
                Spec {
                    schema_meta: meta(&[]),
                    ..b.clone()
                },
                with_metadata,
            ),
            // canonically equal, physically different: accepted (spec Section 5)
            (
                "Utf8 -> LargeUtf8",
                Spec {
                    string_type: DataType::LargeUtf8,
                    ..b.clone()
                },
                never,
            ),
            (
                "Utf8 -> Dictionary",
                Spec {
                    string_type: DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::Utf8),
                    ),
                    ..b.clone()
                },
                never,
            ),
            (
                "List -> LargeList",
                Spec {
                    large_list: true,
                    ..b.clone()
                },
                never,
            ),
            (
                "list element name (not hashed)",
                Spec {
                    list_item_name: "element",
                    ..b.clone()
                },
                never,
            ),
        ];
        let base_batch = batch_of(&b);
        for config in all_configs() {
            // Prepared from an equal schema built separately (not the batch's `Arc`) ...
            let p = ArrowDigester::prepare(&batch_of(&b).schema(), config);
            for (case, spec, refused) in &variants {
                let batch = batch_of(spec);
                let why = format!("{case} {config:?}");
                if refused(config) {
                    assert_refused(&p, &batch, &why);
                    // ... and the streaming digester refuses it too
                    let r = try_hash(|| {
                        let mut d = p.digester();
                        d.update(&batch);
                        d.finalize()
                    });
                    assert!(r.unwrap_err().contains("does not match"), "{why}");
                } else {
                    assert_accepted_like_one_shot(&p, &batch, &why);
                }
            }
            // The base batch through the prepared path equals the one-shot path.
            assert_accepted_like_one_shot(&p, &base_batch, "base");
        }
    }

    /// An unordered top level is keyed by name, so a batch with a duplicated column has the
    /// prepared Phase 1 JSON; it is still refused (Section 7), as by the streaming digester.
    #[test]
    fn duplicate_columns_are_refused() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        let dup = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            dup,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        for config in all_configs() {
            let p = ArrowDigester::prepare(&schema, config);
            assert_refused(&p, &batch, &format!("{config:?}"));
        }
    }

    /// A dictionary key type does not enter the hash: a batch with other key types is accepted.
    #[test]
    fn dictionary_key_type_is_not_part_of_the_rule() {
        let make = |key: DataType| -> RecordBatch {
            let a: ArrayRef = match key {
                DataType::Int8 => Arc::new(
                    vec!["a", "b"]
                        .into_iter()
                        .collect::<DictionaryArray<Int8Type>>(),
                ),
                _ => Arc::new(
                    vec!["a", "b"]
                        .into_iter()
                        .collect::<DictionaryArray<Int32Type>>(),
                ),
            };
            let f = Field::new("d", a.data_type().clone(), true);
            RecordBatch::try_new(Arc::new(Schema::new(vec![f])), vec![a]).unwrap()
        };
        let b8 = make(DataType::Int8);
        let b32 = make(DataType::Int32);
        for config in all_configs() {
            let p = ArrowDigester::prepare(&b8.schema(), config);
            assert_accepted_like_one_shot(&p, &b32, "Int8 keys -> Int32 keys");
        }
    }

    // ---------------------------------------------------------------------------------------
    // the object itself
    // ---------------------------------------------------------------------------------------

    #[test]
    fn prepared_schema_accessors() {
        let b = batch_of(&base_spec());
        for config in all_configs() {
            let p = PreparedSchema::new(&b.schema(), config);
            assert_eq!(p.config(), config);
            assert_eq!(p.schema().as_ref(), b.schema().as_ref());
            assert_eq!(
                p.schema_digest(),
                ArrowDigester::hash_schema(&b.schema(), config)
            );
            assert_eq!(p.schema_digest().len(), 35);
            // a clone is the same prepared schema
            let q = p.clone();
            assert_eq!(q.hash_record_batch(&b), p.hash_record_batch(&b));
        }
    }

    #[test]
    fn prepared_schema_is_send_and_sync() {
        const fn assert_send_sync<T>()
        where
            T: Send + Sync,
        {
        }
        assert_send_sync::<PreparedSchema>();
    }

    /// `prepare` rejects exactly what `hash_schema` rejects (Section 7).
    #[test]
    fn prepare_rejects_like_hash_schema() {
        let schemas = [
            Schema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("a", DataType::Int32, true),
            ]),
            Schema::new(vec![
                Field::new("a/b", DataType::Int32, true),
                Field::new(
                    "a",
                    DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int32, true)])),
                    true,
                ),
            ]),
        ];
        for s in &schemas {
            for config in all_configs() {
                let msg = prepared(s, config).unwrap_err();
                assert!(msg.contains("Ambiguous field path"), "{s:?}: {msg}");
                try_hash(|| ArrowDigester::hash_schema(s, config)).unwrap_err();
            }
        }
    }

    /// A schema whose data cannot be hashed is prepared (its schema digest exists), but
    /// `hash_record_batch` and `digester()` reject it with the rejection of
    /// `ArrowDigester::new`.
    #[test]
    fn data_unsupported_schema_is_prepared_but_hashes_no_data() {
        let schemas = [
            Schema::new(vec![Field::new("n", DataType::Null, true)]),
            Schema::new(vec![Field::new("u", DataType::Utf8View, true)]),
            Schema::new(vec![
                Field::new("ok", DataType::Int32, true),
                Field::new(
                    "s",
                    DataType::Struct(Fields::from(vec![Field::new(
                        "b",
                        DataType::BinaryView,
                        true,
                    )])),
                    true,
                ),
            ]),
        ];
        for s in &schemas {
            for config in all_configs() {
                let p = ArrowDigester::prepare(s, config);
                assert_eq!(p.schema_digest(), ArrowDigester::hash_schema(s, config));
                let expected = try_hash(|| ArrowDigester::new(s, config).finalize()).unwrap_err();
                assert!(expected.contains("is not supported"), "{expected}");
                let empty = RecordBatch::new_empty(Arc::clone(p.schema()));
                assert_eq!(
                    try_hash(|| p.hash_record_batch(&empty)).unwrap_err(),
                    expected,
                    "{s:?}"
                );
                assert_eq!(
                    try_hash(|| p.digester().finalize()).unwrap_err(),
                    expected,
                    "{s:?}"
                );
            }
        }
    }

    /// Completeness of the structural fast path (tier 2) where pyarrow's
    /// `Schema.equals(check_metadata=True)` is incomplete: (a) field metadata nested in a
    /// dictionary's value type, (b) an extension type's parameters (in Arrow, the
    /// `ARROW:extension:metadata` field metadata). With `include_metadata` the batch is refused;
    /// without it the batch is accepted and hashes like the one-shot path.
    #[test]
    fn fast_path_is_complete_for_dictionary_values_and_extension_parameters() {
        // (a) dictionary<struct<x>> whose x carries metadata
        let dict_of_struct = |unit: &str| -> RecordBatch {
            let x = Field::new("x", DataType::Int32, true).with_metadata(meta(&[("unit", unit)]));
            let values = StructArray::new(
                Fields::from(vec![x]),
                vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
                None,
            );
            let keys = Int8Array::from(vec![0, 1, 0]);
            let dict: ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap());
            let f = Field::new("d", dict.data_type().clone(), true);
            RecordBatch::try_new(Arc::new(Schema::new(vec![f])), vec![dict]).unwrap()
        };
        // (b) an extension type (fixed-shape tensor) whose serialized parameters differ
        let tensor = |shape: &str| -> RecordBatch {
            let values = Int32Array::from(vec![1, 2, 3, 4]);
            let fsl: ArrayRef = Arc::new(
                FixedSizeListArray::try_new(
                    Arc::new(Field::new("item", DataType::Int32, false)),
                    4,
                    Arc::new(values),
                    None,
                )
                .unwrap(),
            );
            let f = Field::new("t", fsl.data_type().clone(), true).with_metadata(meta(&[
                ("ARROW:extension:name", "arrow.fixed_shape_tensor"),
                ("ARROW:extension:metadata", shape),
            ]));
            RecordBatch::try_new(Arc::new(Schema::new(vec![f])), vec![fsl]).unwrap()
        };
        let pairs = [
            (
                dict_of_struct("m"),
                dict_of_struct("cm"),
                "dictionary value metadata",
            ),
            (
                tensor("{\"shape\":[2,2]}"),
                tensor("{\"shape\":[4,1]}"),
                "extension parameters",
            ),
        ];
        for (prepared_from, batch, why) in &pairs {
            // the schemas differ only there, so arrow-rs equality must see the difference
            assert_ne!(prepared_from.schema(), batch.schema(), "{why}");
            for config in all_configs() {
                let p = ArrowDigester::prepare(&prepared_from.schema(), config);
                assert_accepted_like_one_shot(&p, prepared_from, why);
                if config.include_metadata {
                    assert_refused(&p, batch, &format!("{why} {config:?}"));
                } else {
                    assert_accepted_like_one_shot(&p, batch, &format!("{why} {config:?}"));
                }
            }
        }
    }

    /// A batch is hashed with the prepared analysis even when it is built from a different
    /// but equal schema object: hashing does not depend on which equal schema object is used.
    #[test]
    fn equal_schema_objects_are_interchangeable() {
        let a = batch_of(&base_spec());
        let b = batch_of(&base_spec());
        assert!(!Arc::ptr_eq(a.schema_ref(), b.schema_ref()));
        for config in all_configs() {
            let p = ArrowDigester::prepare(&a.schema(), config);
            assert_eq!(p.hash_record_batch(&a), p.hash_record_batch(&b));
            assert_eq!(
                p.hash_record_batch(&b),
                ArrowDigester::hash_record_batch(&b, config)
            );
        }
    }
}
