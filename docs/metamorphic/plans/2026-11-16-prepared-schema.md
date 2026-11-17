# Prepared schema (ITL-1177) — implementation plan

## Goal

Let a caller analyse a schema once (`ArrowDigester::prepare(&schema, config)` /
`PreparedSchema::new`) and hash many record batches of that schema, or start streaming
digesters, without the analysis being redone. No cache, no hash change.

## Design

- `PreparedSchema` (public, immutable, `Clone` via `Arc`) holds the schema (`SchemaRef`), the
  `HasherConfig`, the raw schema digest, the canonical schema key (Phase 1 JSON, `|` Phase 2
  document when metadata is included and present) and the empty per-node entry template.
  Data-type support and node paths are validated once, in `new`.
- `ArrowDigesterCore` keeps an `Arc` of the prepared analysis plus its own entries and row count;
  `new_with_config` = prepare + `digester`.
- Same-schema rule: unchanged (byte-layout spec Section 5). Checked in tiers: `Arc::ptr_eq`,
  then arrow-rs structural equality (sufficient), then the canonical key of the batch's schema
  (today's check) plus Section 7 path validation.
- One-shot `hash_record_batch` prepares from `batch.schema()` (same `Arc`), so the check hits
  tier 1 and the analysis runs once. `hash_schema` builds the Phase 1 JSON and Phase 2 document
  once, shared with the key.
- Python binding: `InternalPreparedSchema` uniffi object; `prepare`, `ArrowDigester.prepare`,
  `PreparedSchema` in `python/starfix/__init__.py`.

## Steps (TDD)

1. `tests/prepared_schema.rs`: prepared vs one-shot over every golden vector (all methods,
   all option combinations in the fixture) and over every option combination (8) for each
   golden input; `schema_digest == hash_schema`; `digester()` == `ArrowDigester::new` streams;
   refusal cases (name, type, nullability, order when ordered, field / nested / extension /
   schema metadata when included); acceptance cases (Utf8 vs LargeUtf8, dictionary, reordered
   columns when unordered, metadata when not included) equal to one-shot; the property
   "accepted ⇒ equals one-shot of that batch, else refused" over pairs of golden schemas.
   Sweeps from `lossless_properties` / `ordered_hashing` re-run through the prepared path.
2. Watch them fail (no `PreparedSchema`).
3. Implement in `src/arrow_digester_core.rs` and `src/lib.rs`.
4. Benchmark example `examples/prepared_bench.rs`, before/after numbers.
5. Binding, README, spec Section 5 note, CHANGELOG `Unreleased`.
6. `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
