# Starfix

**Deterministic, logical hashing for Apache Arrow data.**

Starfix computes stable, content-based hashes of Apache Arrow tables, record batches, arrays, and schemas—**independent of how the data happens to be represented in memory**.

Two Arrow tables containing the same logical data should produce the same hash, even if they differ in chunk boundaries, buffer offsets, dictionary encoding, or other physical representation details.

This makes Starfix useful for content-addressed storage, data deduplication, caching, reproducible computational pipelines, and any system that needs a reliable way to identify the contents of Arrow data.

Starfix is implemented in Rust, with Python bindings for PyArrow.

## Why logical hashing?

Apache Arrow provides an efficient, columnar representation of data. However, the same logical data can be represented in many physically different ways.

For example:

- A column might consist of one contiguous array or multiple chunks.
- An array might be a slice of a larger buffer with a nonzero offset.
- Strings might use 32-bit or 64-bit offsets.
- Values might be stored directly or through dictionary encoding.
- Columns and struct fields might appear in different orders without changing the data they represent.

These differences can arise naturally from how data is constructed, processed, serialized, or transferred between systems.

A conventional hash of serialized Arrow data or its underlying buffers may change when any of these details change—even when the logical content remains identical.

**Starfix hashes the logical content of Arrow data rather than its incidental physical representation.**

The goal is to establish a stable identity for data that survives representation changes while remaining sensitive to meaningful differences in its contents.

### A simple example

```python
import pyarrow as pa
from starfix import ArrowDigester

table = pa.table({
    "id": [1, 2, 3],
    "name": ["alice", "bob", "carol"],
})

# Construct the same data with different chunk boundaries
# and a different column order.
repacked = pa.concat_tables([
    table.slice(0, 1),
    table.slice(1),
]).select(["name", "id"])

assert (
    ArrowDigester.hash_table(table)
    == ArrowDigester.hash_table(repacked)
)
```

Despite the different physical layout and column order, both tables contain the same logical data and produce the same hash.

But changing the order of rows changes the hash:

```python
reordered_rows = table.take(pa.array([2, 1, 0]))

assert (
    ArrowDigester.hash_table(table)
    != ArrowDigester.hash_table(reordered_rows)
)
```

This reflects an important principle: **Starfix ignores differences in representation, not differences that are meaningful to the data model.**

## What determines the hash?

Starfix computes a versioned SHA-256 hash from a canonical representation of the Arrow schema and data.

Its behavior is guided by the following rules.

### Representation-independent properties

The hash remains unchanged under the following transformations, provided the logical data remains the same:

| Transformation | Hash changes? |
|---|---|
| Split or combine chunks | No |
| Split or combine record batches | No |
| Change array offsets through slicing while retaining the same visible values | No |
| Use `Utf8` instead of `LargeUtf8` | No |
| Use `Binary` instead of `LargeBinary` | No |
| Use `List` instead of `LargeList` | No |
| Dictionary-encode or decode the same values | No |
| Reorder top-level columns | No, by default |
| Reorder struct fields | No, by default |
| Change memory allocation or buffer layout without changing visible data | No |

These properties make hashes stable across many transformations that affect Arrow's physical representation without affecting its logical contents.

### Content-sensitive properties

Starfix also preserves distinctions that are meaningful to the data.

| Transformation | Hash changes? |
|---|---|
| Change a non-null value | Yes, subject to hash collisions |
| Change a value's null status | Yes |
| Change the number of rows | Yes |
| Reorder rows | Generally yes |
| Reorder list elements | Generally yes |
| Rename a column or struct field | Yes |
| Change a field's logical type or nullability | Yes, unless the types are canonically equivalent |
| Change schema or field metadata | Only with `include_metadata=True` |

By default, floating-point values are distinguished by their bit representations. In particular, `-0.0` and `+0.0` hash differently, as do NaNs with different bit patterns. Optional normalization is available.

As with any cryptographic hash, different inputs are not mathematically guaranteed to have different digests because collisions are theoretically possible.

### What does "same logical data" mean?

Starfix does not attempt to determine whether two datasets are semantically equivalent in an application-specific sense.

For example, two columns containing identical numerical values but having different integer types are not necessarily considered identical.

Similarly, two tables containing the same rows in different orders are not considered equivalent by default.

Instead, Starfix implements a precise definition of logical identity based on Arrow's types, values, validity, and structure, together with a small number of configurable options.

This definition is documented in the [byte-layout specification](https://github.com/metamorphic-labs/starfix/blob/main/docs/byte-layout-spec.md).

---

## Installation

### Rust

Add the `starfix` crate to your Rust project:

```bash
cargo add starfix
```

Starfix 0.5.0 and later implement hash format 2.

### Python

Starfix provides a Python package backed by its Rust implementation, using PyO3 and the Arrow PyCapsule interface.

The binding supports Python 3.10+ and PyArrow 14+.

Install it with uv (preferred):

```bash
uv add starfix
```

Or with pip: `python -m pip install starfix`.

Wheels are published for Linux (x86_64 and aarch64, glibc and musl), macOS (universal2), and Windows (x64). On other platforms, the source distribution builds with a Rust toolchain.

To build the Python binding from source in a development environment:

```bash
python -m pip install maturin "pyarrow>=14"
maturin develop
```

Run this inside an activated Python virtual environment with a Rust toolchain available.

**Release compatibility:** From 0.6.0, the PyPI `starfix` package is backed by the Rust implementation. `starfix` 0.5.0 on PyPI is the pure-Python implementation of hash format 2; 0.4.0 and earlier use hash format 1. See [Hash format and compatibility](#hash-format-and-compatibility) below.

---

## Usage

### Python

The Python API provides one-shot hashing for Arrow tables, record batches, schemas, and arrays.

```python
import pyarrow as pa
from starfix import ArrowDigester

table = pa.table({
    "id": [1, 2, 3],
    "score": [0.9, 0.7, 0.95],
})

# Hash an entire table, regardless of chunking.
table_hash = ArrowDigester.hash_table(table)

# Hash a record batch.
batch_hash = ArrowDigester.hash_record_batch(
    table.to_batches()[0]
)

# Hash a schema.
schema_hash = ArrowDigester.hash_schema(table.schema)

# Hash an array or chunked array.
array_hash = ArrowDigester.hash_array(
    table.column("score")
)
```

Table and record-batch hashing are based on the same logical table representation. The resulting digest is independent of how rows are divided among batches.

### Streaming record batches

For large datasets, Starfix can incrementally hash record batches without first combining them into a single table.

```python
digester = ArrowDigester(table.schema)

for batch in table.to_batches():
    digester.update(batch)

hash_value = digester.finalize()
```

The final result is independent of the number and sizes of the input batches, provided they collectively represent the same rows in the same order and have compatible schemas.

This supports efficient hashing of data produced incrementally or processed in streams.

### Preparing a schema for repeated hashing

When hashing many batches or tables with the same schema, use `PreparedSchema` to avoid repeatedly analyzing that schema.

```python
prepared = ArrowDigester.prepare(table.schema)

# Hash multiple tables with the same schema.
hash_value = prepared.hash_table(table)

# Reuse the prepared schema for streaming.
digester = prepared.digester()
for batch in table.to_batches():
    digester.update(batch)

hash_value = digester.finalize()
```

Preparation precomputes the canonical schema representation and associated hashing information.

It produces the same hashes as the corresponding one-shot operations.

The prepared object is immutable and reusable. Starfix does not maintain a hidden global schema cache.

### Rust

The Rust API provides equivalent one-shot and streaming operations.

```rust
use starfix::{ArrowDigester, HasherConfig};

// Hash a record batch.
let hash = ArrowDigester::hash_record_batch(
    &batch,
    HasherConfig::default(),
);

// Hash multiple batches as one logical table.
let mut digester = ArrowDigester::new(
    &schema,
    HasherConfig::default(),
);

digester.update(&batch1);
digester.update(&batch2);

let hash = digester.finalize();

// Hash a schema or individual array.
let schema_hash = ArrowDigester::hash_schema(
    &schema,
    HasherConfig::default(),
);

let array_hash = ArrowDigester::hash_array(&array);
```

For repeated hashing under a shared schema:

```rust
use starfix::{ArrowDigester, HasherConfig};

let prepared = ArrowDigester::prepare(
    &schema,
    HasherConfig::default(),
);

for batch in &batches {
    let hash = prepared.hash_record_batch(batch);
}
```

`PreparedSchema` can substantially reduce the overhead of hashing many small batches sharing the same schema.

See `examples/prepared_bench.rs` for a benchmark.

---

## Configuration

Starfix provides several options for specifying which differences should matter to a logical hash.

All options default to `false`.

| Option | Behavior |
|---|---|
| `include_metadata` | Include schema-level and field-level metadata |
| `ordered_columns` | Treat top-level column order as significant |
| `ordered_struct_fields` | Treat struct-field order as significant at every depth |
| `normalize_negative_zero` | Treat negative floating-point zero as positive zero |
| `normalize_nan` | Normalize all NaN bit patterns to a canonical quiet NaN |

### Column and struct-field order

By default, Starfix treats top-level columns and struct fields as name-identified collections.

Their order does not affect the hash.

Applications where those orders are semantically significant can enable order-sensitive hashing.

```python
ordered_hash = ArrowDigester.hash_table(
    table,
    ordered_columns=True,
    ordered_struct_fields=True,
)
```

In Rust:

```rust
let config = HasherConfig {
    ordered_columns: true,
    ordered_struct_fields: true,
    ..HasherConfig::default()
};
```

These options affect the identity of the resulting hash.

Row order and list-element order remain significant regardless of these options.

### Schema and field metadata

Schema and field metadata are excluded by default.

Use `include_metadata=True` when metadata should be considered part of the data's identity.

```python
hash_value = ArrowDigester.hash_table(
    table,
    include_metadata=True,
)
```

Metadata keys are handled canonically, so the ordering of metadata entries does not affect the hash.

### Floating-point normalization

By default, Starfix preserves floating-point bit distinctions:

- `-0.0` differs from `+0.0`.
- NaNs with different payloads or representations may differ.

Applications that do not wish to distinguish these values can enable normalization.

```python
hash_value = ArrowDigester.hash_table(
    table,
    normalize_negative_zero=True,
    normalize_nan=True,
)
```

These options do not modify the original Arrow data.

**Important:** Unlike the ordering options, floating-point normalization settings are not encoded in the hash header. Callers should use a consistent normalization policy when comparing hashes, particularly when bit-exact identity matters.

---

## Supported Arrow types

Starfix supports hashing data with the following Arrow types:

- Boolean
- Signed and unsigned integers
- Float16, Float32, and Float64
- Date, Time, Timestamp, and Duration
- Decimal32, Decimal64, Decimal128, and Decimal256
- FixedSizeBinary
- Binary and LargeBinary
- Utf8 and LargeUtf8
- List and LargeList
- FixedSizeList
- Struct
- Dictionary encoding of supported value types, including nested values

Supported nested combinations are handled recursively.

Starfix also supports extension types through their storage types. When metadata hashing is enabled, extension-related metadata participates in the hash.

Some Arrow types have more limited support.

Schemas containing `Null`, `Map`, `Interval`, `Utf8View`, or `BinaryView` can be hashed, but data hashing for these types is not currently supported.

`ListView`, `LargeListView`, `Union`, and `RunEndEncoded` are rejected.

For the precise type rules and unsupported cases, see the [byte-layout specification](https://github.com/metamorphic-labs/starfix/blob/main/docs/byte-layout-spec.md).

---

## How Starfix works

Starfix constructs a canonical representation of the logical Arrow data and hashes that representation using SHA-256.

At a high level, the process involves three steps.

### 1. Canonicalize the schema

The schema representation includes column names, canonical data types, field nullability, and optionally metadata.

Equivalent physical representations of the same logical type are normalized. For example, `Utf8` and `LargeUtf8` share a canonical type representation.

### 2. Traverse the logical data

Starfix hashes the contents of each node in the Arrow type tree, including nested structures and list elements.

It accounts for:

- The number of rows
- Validity and null information
- List lengths and boundaries
- Values of supported primitive and nested types

Physical details that do not change the visible logical data are excluded.

For example, values stored beneath null parent structures do not contribute as independently visible values.

### 3. Produce a versioned digest

Starfix uses SHA-256 to compute the final digest from the canonical schema and data representation.

The hash format also distinguishes the type of hashing operation, so schema hashes, array hashes, and table/record-batch hashes belong to separate domains.

The precise byte-level algorithm is specified in [docs/byte-layout-spec.md](https://github.com/metamorphic-labs/starfix/blob/main/docs/byte-layout-spec.md), enabling independent implementations to reproduce identical hashes.

---

## Hash format and compatibility

Every Starfix hash consists of 35 bytes:

```text
[ 3-byte format version ][ 32-byte SHA-256 digest ]
```

Starfix 0.5.0 and later use hash format 2:

```text
00 00 02
```

The format version is independent of the library's package version.

Hash format 2 is documented byte-for-byte and validated against a set of golden test vectors.

### Cross-language compatibility

The Rust implementation and the corresponding format-2 `starfix-python` implementation produce identical hashes for the defined format, with one known exception: `starfix-python` 0.5.0 hashes a dictionary of sliced nested fixed-size lists incorrectly (a pyarrow `take()` defect).

From 0.6.0, the `starfix` Python package uses the Rust hashing implementation and exposes a compatible Python interface.

### Legacy compatibility

Starfix 0.4 and earlier use hash format 1:

```text
00 00 01
```

Format 2 corrects several information-loss issues in format 1, including edge cases involving Boolean arrays, nullable structs, nested lists, and zero-column tables.

**Format-1 and format-2 hashes are not interchangeable.**

The Rust crate retains a frozen compatibility implementation under:

```rust
use starfix::legacy::v0_4::{
    ArrowDigester,
    HasherConfig,
};
```

The Python package also provides a legacy module at `starfix.legacy.v0_4`.

These implementations exist to support previously stored hashes. New applications should use format 2.

See the [changelog](https://github.com/metamorphic-labs/starfix/blob/main/CHANGELOG.md) and [byte-layout specification](https://github.com/metamorphic-labs/starfix/blob/main/docs/byte-layout-spec.md) for migration details.

---

## Validation and testing

Starfix includes tests for its logical-hashing invariants, including:

- Independence from batch splitting and array offsets
- Column-order and struct-field-order options
- Type canonicalization
- Nested arrays and nullable values
- Dictionary encoding
- Floating-point normalization
- Hash-format conformance
- Cross-implementation compatibility through golden vectors

The test suite includes exhaustive small-input testing across supported type families, along with targeted regression tests for historically problematic cases.

Run the Rust test suite with:

```bash
cargo test
```

The Python tests are available under `python/tests`.

Golden vectors for hash format 2 are stored in `tests/golden/format2_v0.5.json`.

For the normative hashing rules, see [Starfix Byte Layout Specification — Hash Format 2](https://github.com/metamorphic-labs/starfix/blob/main/docs/byte-layout-spec.md).

## License

Starfix is dual-licensed under [MIT](https://github.com/metamorphic-labs/starfix/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/metamorphic-labs/starfix/blob/main/LICENSE-APACHE), at your option.
