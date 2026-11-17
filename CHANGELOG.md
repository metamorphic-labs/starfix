# Changelog

## Unreleased

### Added — prepared schema (ITL-1177)

- `ArrowDigester::prepare(&schema, config)` / `PreparedSchema::new`: analyse a schema once and
  hash many record batches of it (`hash_record_batch`), take its schema digest
  (`schema_digest`) or start streaming digesters (`digester`) without analysing it again. The
  object is immutable and owned by the caller; starfix keeps no cache. Every hash is
  byte-identical to the one-shot calls (tested over every golden vector under all 8 option
  combinations, the leaf and nested-type sweeps, and every pair of golden schemas).
- A batch whose schema is not the prepared one under the prepared options is refused (panic
  "Record batch schema does not match the prepared schema"), by the streaming digester's rule
  (byte-layout spec Section 5). The check is a pointer or structural comparison for the same or
  an equal schema (about 0.05 µs) and analyses only a schema that differs.
- `prepare` succeeds whenever `hash_schema` does; for a schema whose data cannot be hashed,
  `hash_record_batch` and `digester` panic as `ArrowDigester::new` does.
- Python binding (`python/starfix`): `prepare(...)`, `ArrowDigester.prepare(...)` and
  `PreparedSchema` (`schema_digest`, `hash_record_batch`, `hash_table`, `digester`), with the
  three options as keywords.
- `examples/prepared_bench.rs`: time per one-row hash, one-shot and prepared.

### Added — a `Result` API (ITL-1178)

- `StarfixError`, and `try_*` variants that return it instead of panicking:
  `ArrowDigester::try_new`, `try_update`, `try_hash_schema`, `try_hash_record_batch`,
  `try_hash_array_with_config`, and `PreparedSchema::try_new`, `try_hash_record_batch`,
  `try_digester`. Every refusal of the spec (Sections 5, 7 and 8) is a variant. The panicking
  methods are unchanged and panic with the error's message.
- `update` is atomic: a refused batch (for example a null in a non-nullable field, found while
  the batch is traversed) leaves the digester as it was, so it can be used further. Before, its
  state was unspecified. The state is saved per node, not per row.

### Fixed — the Python binding of this crate (`python/starfix`, ITL-1178)

**Hashes produced through this binding change with this release for multi-chunk tables, and
with schema-level metadata, because they were wrong.** The hash format does not change (still
2): the binding now gives the digester the data the Rust crate and starfix-python hash. This is
the binding built from this repository with maturin, not starfix-python.

- Every chunk of a table is hashed. `hash_record_batch(table)` and `PyArrowDigester.update(table)`
  hashed the first chunk only: a multi-chunk table hashed differently from the same rows in one
  chunk, and two tables that differ after their first chunk had the same hash. A table now
  hashes like the same rows in one record batch, as in the Rust crate and starfix-python.
- An empty table (no record batches) hashes, to the hash of a zero-row batch of its schema. It
  raised `IndexError`.
- The schema-level metadata of a record batch reaches the digester. The import dropped it, so
  the metadata a schema carries was missing from what the digester received (in 0.5.0 the
  options could not be set, so it showed when hashing with `include_metadata`).
- The options `include_metadata`, `ordered_columns` and `ordered_struct_fields` (keyword-only,
  default `False`) are available on `hash_schema`, `hash_record_batch` and `PyArrowDigester`;
  only the defaults were exposed.
- `ArrowDigester` has starfix-python's interface: `ArrowDigester(schema, *, ...)` with `update`
  and `finalize`, and `prepare`, `hash_schema`, `hash_record_batch`, `hash_table` and
  `hash_array` (a `ChunkedArray` hashes as its chunks combined), with the same option names and
  defaults. Rejected input raises `ValueError` (it raised the binding's internal error) and a
  wrong container type `TypeError`. The differences that remain are listed in the package
  docstring (after a rejected `update` the digester cannot be used; starfix-python restores it).
- Two inputs pyarrow exports failed to import: a struct inside a sliced struct or fixed-size
  list (arrow-rs applied the slice offset twice), and buffers less aligned than arrow-rs requires
  (such as `decimal128` values at an 8-byte boundary, as read from IPC), which are now copied.
- CI builds the binding and runs its tests, including a differential test against starfix-python
  0.5.0 (golden vectors and seeded random multi-chunk tables, all 8 option combinations).

### Changed

- The schema is analysed once per one-shot call: `hash_record_batch` and `ArrowDigester::new`
  built the canonical schema JSON three times and the metadata document twice. One-row
  `hash_record_batch` takes about half the time (about 25 µs to about 12 µs in the benchmark).
  No hash changes; the hash format is still 2.

## 0.5.0 (2026-11-11)

### Changed — hash format 2 (every hash changes)

The hash format version prefix is now `00 00 02` (format 2); format 1 (`00 00 01`) was produced
by starfix 0.4 and earlier. **Every hash changes** with the default (main) API, and a format-1
hash can always be told from a format-2 hash by its first three bytes. Every format-2 digest
starts with a header that records the format, the entry point (schema, record batch / stream,
array) and the order options, so outputs of different entry points or options never coincide. The 0.4 behaviour stays
available as `starfix::legacy::v0_4`. `tests/golden/format1_to_format2.json` maps every
format-1 golden value of this repository to its format-2 value. Specification:
`docs/byte-layout-spec.md` (format history in Section 10).

### Fixed — information losses of format 1 (ITL-898)

These inputs collided, or hashed differently depending on batch boundaries, in format 1:

- Booleans: bit-packed per call without a count (`[true]` == `[true, false, false]` in a
  one-column non-nullable table; batch splits changed the hash). Now one byte per value.
- Nullable structs: a null struct equalled a struct whose fields are all null; null struct rows
  vanished under non-nullable fields. Now every nullable node records its validity.
- Lists whose element is a nullable list lost their own lengths; lists of nullable structs did
  not distinguish a null element. Now every list node records its lengths and every nullable
  element its validity.
- The row count of zero-column tables and empty structs was not hashed. Now the row count is
  hashed.

### Added — `FixedSizeList` data hashing (ITL-927)

- Record batches, streams and `hash_array` now accept `FixedSizeList` data (any size, also 0;
  at any depth, also dictionary-encoded and as extension storage such as
  `arrow.fixed_shape_tensor`). starfix 0.4 panicked on `FixedSizeList` data.
- `FixedSizeList<T, n>` is a different type from `List<T>` / `LargeList<T>` and from
  `FixedSizeList<T, m>` (`m != n`): equal values hash differently (schema and data).
- Layout (byte-layout spec Section 3.2, Example 3b): a fixed-size list node's entry is its
  validity if nullable and nothing else (no lengths); its element node's elements are the `n`
  items of every effectively valid value. Items under null values (and child positions past the
  last value) never affect the hash; a null of a non-nullable element there is accepted, a
  visible one is rejected.
- A `FixedSizeList` whose element type contains a type whose data is not supported (`Null`,
  `Map`, `Interval`, view types) is rejected.
- `FixedSizeList` support moved no other format-2 hash: every input containing `FixedSizeList`
  data was rejected before any data was hashed. `tests/golden/format2_v0.5.json` gained 73
  `fsl_*` vectors and 34 relations for it, appended after the 423 earlier vectors and 114
  relations, which are unchanged.

### Added

- `starfix::legacy::v0_4`: the frozen **0.4 legacy mode**, reproducing starfix 0.4.0 (hash
  format 1) byte for byte with the 0.4.0 API, only to read or match data hashed by 0.4.x. Never
  use it for new data. Tested against the restored v0.4.0 tests, the format-1 golden values and
  the published 0.4.0 crate.

- `HasherConfig::ordered_columns` and `HasherConfig::ordered_struct_fields`: make column order
  and struct field order (at every depth) part of the hash. Default off.
- `ArrowDigester::hash_array_with_config`. `hash_array` always treats the array as nullable:
  an array hashes the same with or without an all-valid validity buffer.
- Dictionaries resolve recursively (a dictionary of lists or of dictionaries hashes as the
  decoded array); `FixedSizeBinary(0)` is supported; nulls in list child slots that are not
  elements (under a null list value) are ignored for `List` and `LargeList` alike.
- Golden vectors `tests/golden/format2_v0.5.json`, generated by
  `cargo run --bin emit_golden_format2` (append-only).

### Changed (breaking)

- `HasherConfig` has new fields: exhaustive struct literals must add `..HasherConfig::default()`.
- In every mode, schemas with duplicate or `/`-aliased field paths, and `ListView`,
  `LargeListView`, `Union` and `RunEndEncoded` types, are rejected (they were accepted with
  ill-defined hashes).
- Map types hash without their entries/key/value field names and without the entries field's
  metadata; field metadata is keyed by node path.
- Data hashing of a schema containing a type whose data is not supported (`Null`, `Map`,
  `Interval`, view types; also inside a `FixedSizeList`) is rejected when the digester is
  created, also with zero rows; a non-nullable field with unmasked nulls is rejected.
- With `include_metadata`, an extension field without `ARROW:extension:metadata` hashes like one
  with an empty value.
- `emit_golden_metadata` is removed; `tests/golden/include_metadata_v0.3.json` is kept as the
  historical format-1 record.

### Fixed

- A panic when hashing a Boolean under a nullable struct at a byte-aligned array offset ≥ 8
  (present since 0.4).

### Known limitations

- Performance: dictionary-encoded columns whose values are nested (struct or list) are hashed
  per row; plain nested columns and dictionaries of scalars and strings are unaffected.
