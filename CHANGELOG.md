# Changelog

## 0.6.0 (2026-11-19)

### Added — float options (ITL-1252)

- `HasherConfig::normalize_negative_zero`: a `-0.0` value of a `Float16`, `Float32` or `Float64`
  leaf, at every depth and in `hash_array`, is hashed as `+0.0`.
- `HasherConfig::normalize_nan`: every NaN value of a float leaf (any sign or payload, quiet or
  signaling) is hashed as the quiet NaN with a clear sign and no payload (`0x7e00`,
  `0x7fc00000`, `0x7ff8000000000000`).
- Both default to `false`: every value is hashed by its bits as before, and with both off every
  hash is unchanged (format 2, the same header; every golden vector and the differential
  harness against starfix-python 0.5.0 pass unchanged). With an option on, data hashes exactly
  as the same data with those values normalized hashes with it off (byte-layout spec Section 6,
  invariant I6), tested as a property over generated tables in Rust and through the harness's
  seeded tables in Python. Null slots, every other type and the stored data are never changed.
- **The float options are not recorded in the hash** (unlike the order options, they are not in
  the header): do not mix hashes taken with different float options where bit-exact identity
  matters (invariant I7).
- Cost: with both options off the hashing path is the one before. With an option on, each float
  value buffer is scanned once; a buffer with nothing to normalize is then hashed as with the
  options off, otherwise its normalized bytes are hashed in 16 KiB blocks
  (`examples/float_bench.rs`).
- Python package: the keywords `normalize_negative_zero` and `normalize_nan` on every entry
  point (`ArrowDigester(...)`, `prepare`, `hash_schema`, `hash_record_batch`, `hash_table`,
  `hash_array`) and the `PreparedSchema` properties of the same names.
- Golden vectors (`float_*`, appended): `+0.0`, `-0.0`, NaN, `-NaN`, payload and signaling NaNs at
  each float width, alone and mixed, at the top level, in a list, in a struct and through
  `hash_array`, under each float option setting. A vector sets the keys `normalize_negative_zero`
  / `normalize_nan` only when `true`.
- `HasherConfig` has two more fields: a struct literal that lists every field must add them
  (`..HasherConfig::default()` keeps working).

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
- Python package (`python/starfix`): `ArrowDigester.prepare(...)` and `PreparedSchema`
  (`schema_digest`, `hash_record_batch`, `hash_table`, `digester`), with the three options as
  keywords (starfix-python's interface; see the ITL-1178 section below).
- `examples/prepared_bench.rs`: time per one-row hash, one-shot and prepared.

### Added — a `Result` API (ITL-1178)

- `StarfixError`, and `try_*` variants that return it instead of panicking:
  `ArrowDigester::try_new`, `try_update`, `try_hash_schema`, `try_hash_record_batch`,
  `try_hash_array_with_config`, and `PreparedSchema::try_new`, `try_hash_record_batch`,
  `try_digester`. Every refusal of the spec (Sections 5, 7 and 8) is a variant, with the node
  path where there is one. The panicking methods are unchanged and panic with the error's
  message.
- `update` is atomic: a refused batch (for example a null in a non-nullable field, found while
  the batch is traversed) leaves the digester as it was, so it can be used further. Before, its
  state was unspecified. The state is saved per node, not per row.
- `ArrowDigester::hash_array_chunks` / `try_hash_array_chunks`: the hash of the concatenation
  of several arrays of one type (the chunks of a column), without concatenating them.

### Fixed — dictionary keys under a null struct (ITL-1178)

- A dictionary key under a null enclosing struct is never hashed (spec Section 3.2), but it was
  decoded, and an out-of-range one panicked. pyarrow builds such keys (an all-null struct over a
  dictionary child gets key 0 into an empty dictionary). They are no longer read: such data
  hashes as starfix-python hashes it. No hash of valid data changes.

### Changed — the Python package (`python/starfix`, ITL-1178)

The Python package built from this crate is rebuilt on PyO3 (it used uniffi) and takes
starfix-python's interface, so that it can replace starfix-python. **Hashes produced through
the 0.5.0 binding change for multi-chunk tables, and with schema-level metadata, because they
were wrong.** The hash format does not change (still 2).

- Fixed: every chunk of a table is hashed. The 0.5.0 binding's `hash_record_batch(table)` and
  `PyArrowDigester.update(table)` hashed the first chunk only, so a multi-chunk table hashed
  differently from the same rows in one chunk, and two tables that differ after their first
  chunk had the same hash. An empty table hashes as a zero-row batch (it raised `IndexError`).
  Schema-level metadata of a record batch reaches the digester (the import dropped it; in 0.5.0
  the options could not be set, so it showed only through the prepared schema).
- Interface: `ArrowDigester(schema, *, include_metadata=False, ordered_columns=False,
  ordered_struct_fields=False)` with `update` and `finalize`; `ArrowDigester.prepare`,
  `hash_schema`, `hash_record_batch`, `hash_table`, `hash_array` (an Array or ChunkedArray);
  `PreparedSchema` (`schema`, the three option properties, `schema_digest`,
  `hash_record_batch`, `hash_table`, `digester`; immutable, picklable); `__version__`. Results,
  exception types and messages are starfix-python's: refusals raise `ValueError` (also for
  invalid Arrow data, never `PanicException`), a wrong container type `TypeError`; repeated
  metadata keys are refused with `include_metadata`. **Breaking** for users of the 0.5.0
  binding: `PyArrowDigester` and the module-level `hash_schema`, `hash_record_batch` and
  `prepare` are removed (use `ArrowDigester`).
- `starfix.legacy.v0_4` (format 1) is starfix-python's frozen pure-Python module, shipped
  unchanged.
- Arrow data crosses through the Arrow PyCapsule interface (pyarrow >= 14; arrow-rs's `pyarrow`
  feature for schemas). Inputs pyarrow exports that arrow-rs's import refused or mishandled are
  handled: a struct nested in a sliced struct or fixed-size list (arrow-rs applies the offset
  twice), buffers less aligned than arrow-rs requires (copied), and empty arrays without an
  offsets buffer at any depth, such as the empty child of a list whose rows are all empty
  (starfix-python 0.5.0 hashes them as the allocated equivalent; so does the package now).
- With `include_metadata`, as starfix-python: explicit `ARROW:extension:*` field metadata that
  conflicts with the field's registered extension type is refused (pyarrow's export hides the
  conflict), and repeated metadata keys are refused below dictionaries of dictionaries too.
- `PreparedSchema.hash_table` of a table without record batches checks the table's schema also
  when pyarrow cannot build an empty batch of it (a year-month interval column, for example).
- The GIL is released while batches and arrays are hashed. Wheels are abi3 (Python >= 3.10).
- uniffi, its bindgen binary and `uniffi.toml` are removed (Python was their only consumer).
  PyO3 is an optional dependency (feature `python`): the Rust crate does not depend on it.
- Runtime dependencies: `pyarrow>=14` only (`ipykernel` and `maturin` were listed as runtime
  dependencies).
- A seeded differential harness (`python/tests/test_harness.py`, `_harness.py`) compares the
  package with starfix-python 0.5.0 on generated tables: every leaf type, nesting, dictionaries
  and extension types, names and time zones with NUL, metadata that is not UTF-8, offsets at
  every depth, bufferless empty arrays, many and empty batches, and invalid data. Both sides
  build each table in process. Each call must give the same digest, or both sides must raise
  in the same class. CI runs 2,000 seeds on every pull request and 100,000 nightly and before a
  release (`differential-harness.yml`), on the newest pyarrow and on Python 3.10 with pyarrow
  19 (starfix-python 0.5.0's floor).
- CI builds the package and runs its tests, the differential test against starfix-python 0.5.0,
  and starfix-python's own test suite against it, on the newest pyarrow and on the floor
  (Python 3.10, pyarrow 14). `maturin-release.yml` builds and tests the wheels of every
  platform; a release tag `v*` publishes them to TestPyPI, then PyPI, by trusted publishing
  (ITL-1193).

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
