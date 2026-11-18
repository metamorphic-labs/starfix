# StarFix: Arrow Data Deterministic Hashing

## Overview

StarFix computes deterministic, content-based SHA-256 hashes of Apache Arrow schemas, record
batches, streams of record batches, and arrays. Inputs that hold the same logical data under the
same schema get the same hash, regardless of how the data is laid out in memory (chunking,
offsets, small vs large offset types, dictionary encoding); inputs that differ get different
hashes, up to SHA-256 collisions (tested exhaustively on small inputs of every supported type).

Every hash is 35 bytes: a 3-byte **hash format version** followed by a SHA-256 digest. This crate
produces **hash format 2** (`00 00 02`). The exact bytes are specified in
[`docs/byte-layout-spec.md`](docs/byte-layout-spec.md), precisely enough for an independent
implementation. [starfix-python](https://github.com/metamorphic-labs/starfix-python) is such an
implementation in pure Python (format 2 since its 0.5.0); it matches the Rust-generated golden
vectors in `tests/golden/format2_v0.5.json` exactly. This repository also builds a Python
package from the crate (see [Python](#python)).

## Usage

```rust
use starfix::{ArrowDigester, HasherConfig};

// One batch
let hash = ArrowDigester::hash_record_batch(&batch, HasherConfig::default());

// A stream of batches (the result does not depend on how rows are split into batches)
let mut digester = ArrowDigester::new(&schema, HasherConfig::default());
digester.update(&batch1);
digester.update(&batch2);
let hash = digester.finalize();

// A schema, or a single array
let schema_hash = ArrowDigester::hash_schema(&schema, HasherConfig::default());
let array_hash = ArrowDigester::hash_array(&array);
let array_hash_ordered = ArrowDigester::hash_array_with_config(
    &array,
    HasherConfig { ordered_struct_fields: true, ..HasherConfig::default() },
);
```

### Many batches of one schema: `prepare`

Analysing the schema (validation, canonical schema JSON, schema digest, node layout) is a large
part of hashing a small batch. To hash many batches of one schema, prepare the schema once and
hash with the prepared object; the hashes are byte-identical to the one-shot calls:

```rust
use starfix::{ArrowDigester, HasherConfig, PreparedSchema};

let prepared: PreparedSchema = ArrowDigester::prepare(&schema, HasherConfig::default());
let schema_hash = prepared.schema_digest();       // == ArrowDigester::hash_schema(&schema, config)
for batch in &batches {
    let hash = prepared.hash_record_batch(batch); // == ArrowDigester::hash_record_batch(batch, config)
}
let mut digester = prepared.digester();           // == ArrowDigester::new(&schema, config)
```

The prepared object is immutable, cheap to clone and owned by the caller: starfix keeps no
cache. A batch whose schema is not the prepared one under the prepared options is refused
(panic "Record batch schema does not match the prepared schema"), by the rule of the streaming
digester (byte-layout spec Section 5): the batch's schema must have the prepared schema digest
under the options (names, canonical types, nullability, column and struct-field order when
ordered, all field, extension and schema metadata with `include_metadata`) and pass the checks
of Section 7. The check is a pointer comparison when the batch is built on `prepared.schema()`
and an allocation-free structural comparison for an equal schema; only a schema that differs is
analysed to decide (canonically equal types such as `Utf8` and `LargeUtf8` are accepted).
`prepare` succeeds whenever `hash_schema` does; for a schema whose data cannot be hashed (see
below) only `schema_digest` is available.

Time per hash of a one-row, four-column batch (`cargo run --release --example prepared_bench`):
one-shot `hash_record_batch` about 12 µs (about 25 µs in 0.5.0, which analysed the schema three
times per call); `prepared.hash_record_batch` about 3 µs.

## Python

`python/starfix` is a Python package built from this crate with maturin and PyO3 (feature
`python`): the hashing runs in Rust, with starfix-python's interface and results, so it can
replace starfix-python.

```python
import pyarrow as pa
from starfix import ArrowDigester

table = pa.table({"id": [1, 2], "name": ["a", None]})
ArrowDigester.hash_table(table)                        # every chunk of the table
ArrowDigester.hash_record_batch(table.to_batches()[0])
ArrowDigester.hash_schema(table.schema, include_metadata=True)
ArrowDigester.hash_array(table.column("id"))           # an Array or a ChunkedArray

prepared = ArrowDigester.prepare(table.schema)         # analyse the schema once
prepared.hash_table(table)                             # == ArrowDigester.hash_table(table)

digester = ArrowDigester(table.schema, ordered_columns=True)
for batch in table.to_batches():
    digester.update(batch)
digester.finalize()
```

Same names, keyword-only options, defaults, results and exception types as starfix-python:
refusals raise `ValueError` (with starfix-python's messages), a wrong container type `TypeError`.
A refused `update` leaves the digester unchanged. Hashing releases the GIL. Arrow data crosses
through the Arrow PyCapsule interface (pyarrow >= 14); wheels are abi3 (Python >= 3.10).
`starfix.legacy.v0_4` (format 1, starfix-python 0.4) is starfix-python's frozen pure-Python
module, shipped unchanged: it is not backed by the crate.

Differences from starfix-python:

- Invalid Arrow data that must be read (for example a dictionary key out of range in a visible
  row) raises `ValueError` ("Invalid Arrow data: ..."); starfix-python raises whatever pyarrow
  or Python raises there. A key under a null struct is never read, by either.
- `PreparedSchema` accepts a batch whose schema equals the prepared one by a structural
  comparison (arrow-rs schema equality), starfix-python by comparing IPC bytes; both then fall
  back to the same schema-key comparison, so they accept and refuse the same batches.
- Its private helpers (`starfix._format2`) do not exist.
- Two more keyword options, `normalize_negative_zero` and `normalize_nan` (see Options), off by
  default; with both off every hash is starfix-python's.

## What is hashed

- **The schema**: column names, canonical types (`Utf8`≡`LargeUtf8`, `Binary`≡`LargeBinary`,
  `List`≡`LargeList`, a dictionary is its value type), nullability; with `include_metadata`,
  schema and field metadata (including extension-type keys).
- **The row count.**
- **Every node of the type tree** (each column, struct field and list element): its validity if
  it is nullable, its lengths if it is a list, its values if it is a leaf. Booleans are hashed one
  byte per value. Floats are hashed by their bits as they are (`-0.0` differs from `+0.0`, and
  NaNs with different bits differ) unless a float option is on.

**Ordering principle.** By default starfix ignores exactly two orders: the order of the
top-level columns and the order of struct fields (at every depth). It never sorts or reorders
anything else: row order, list element order (also for `list<struct<key, value>>`) and the
logical values of dictionary-encoded arrays are hashed as given.

Not hashed (they are not part of the logical data, or pyarrow cannot observe them): the list
element field name, the map entries/key/value field names, the map entries field's metadata, the
dictionary index type and `ordered` flag, and metadata unless `include_metadata` is set.

## Options

All options are per call (or per streaming digester), never global, and default to `false`:

| `HasherConfig` field | Effect |
|---|---|
| `include_metadata` | schema-level and field-level metadata are part of the hash |
| `ordered_columns` | the order of the top-level columns is part of the hash |
| `ordered_struct_fields` | the order of struct fields is part of the hash, at every depth (struct in struct, list, large list, fixed-size list, map, dictionary value type) |
| `normalize_negative_zero` | a `-0.0` value of a `Float16`, `Float32` or `Float64` leaf, at every depth, is hashed as `+0.0` |
| `normalize_nan` | every NaN value of a float leaf (any sign, payload, quiet or signaling), at every depth, is hashed as the quiet NaN with a clear sign and no payload (`0x7e00`, `0x7fc00000`, `0x7ff8000000000000`) |

```rust
// Identity of a table value whose column and struct-field order matter:
let config = HasherConfig {
    ordered_columns: true,
    ordered_struct_fields: true,
    ..HasherConfig::default()
};
```

Hashes taken with different order options never coincide: the options are recorded in the
header that starts every digest. The header also names the entry point, so a schema hash, a
record batch or stream hash and an array hash never coincide either. `hash_array` always treats
the array as nullable (its validity is always hashed). Construct `HasherConfig` with `..HasherConfig::default()` to
stay compatible with future options.

**The float options are not recorded in the hash.** By default starfix hashes every value's bits
as they are and normalizes only the structural arrangement and encoding of Arrow data, which is
also the fastest. With `normalize_negative_zero` or `normalize_nan` on, data hashes exactly as the
same data with those float values normalized hashes with the options off, so values that compare
equal in Polars (`-0.0 == 0.0`, NaN == NaN) can hash equal. Unlike the order options they are not
in the header: whether they were used is the caller's to know, and hashes taken with different
float options must not be mixed where bit-exact identity matters. Null slots, every other type
and the stored data are never changed. Cost with an option on: one scan of each float value
buffer; a buffer with nothing to normalize is then hashed as with the options off
(`examples/float_bench.rs`).

```rust
// -0.0 hashes as +0.0, and every NaN as the canonical NaN:
let config = HasherConfig {
    normalize_negative_zero: true,
    normalize_nan: true,
    ..HasherConfig::default()
};
```

## Rejected inputs

In every mode, the following panic (starfix-python raises `ValueError`):

- schemas with two fields at the same path: duplicate column names, duplicate field names in a
  struct, or a name containing `/` that aliases a nested field (a column `"a/b"` next to a struct
  column `a` with child `b`);
- `ListView`, `LargeListView`, `Union` and `RunEndEncoded` types;
- data hashing of a schema that contains a type whose data is not supported (see below), also
  with zero rows;
- a non-nullable field whose array contains nulls not masked by an enclosing null struct (nulls
  in child slots that are not elements, such as under a null list value, are ignored);
- a streamed batch whose schema does not match the digester's (with the order options, this
  includes a different column or struct-field order).

These checks run before any data is hashed, except the null check, which runs while a batch is
traversed; after a panic, discard the digester.

Metadata that repeats a key (possible in stored Arrow metadata) is not portable: arrow-rs keeps
the last value when it reads IPC or imports over the C data interface, before starfix sees it,
while starfix-python rejects such inputs with `include_metadata` (spec Section 7).

## Supported data types

Data hashing supports Boolean, all integer and float types, Date32/64, Time32/64, Timestamp,
Duration, Decimal32/64/128/256, FixedSizeBinary, Binary/LargeBinary, Utf8/LargeUtf8,
List/LargeList, FixedSizeList, Struct, and Dictionary of any of these (a dictionary of
dictionaries resolves recursively; `FixedSizeBinary(0)` and `FixedSizeList<_, 0>` are supported).
`FixedSizeList<T, n>` is its own type: it never hashes like `List<T>` / `LargeList<T>` with the
same values, nor like a fixed-size list of another size or `FixedSizeBinary(n)`. Schemas of
`Null`, `Map`, `Interval`, `Utf8View` and `BinaryView` can be hashed, but their data cannot yet. Extension types
hash as their storage type; with `include_metadata`, their `ARROW:extension:*` metadata counts
too (an absent `ARROW:extension:metadata` equals an empty one).

Performance: dictionary-encoded columns whose values are nested (struct, list or fixed-size list)
are hashed per row; plain nested columns and dictionaries of scalars and strings are unaffected.

## Legacy 0.4 mode (format 1)

`starfix::legacy::v0_4` reproduces the output of starfix **0.4.0** (Rust) byte for byte, with
the same API as 0.4.0 (`ArrowDigester`, `HasherConfig { include_metadata }`). It exists **only to
read or match data hashed by starfix 0.4.x** — for example to find stored data under its old
hash. **Never hash new data with it.** It is frozen: never fixed, never extended (no order
options), and it keeps every format-1 defect listed below. Migrating code that must keep
producing 0.4 hashes changes one import:

```rust
use starfix::legacy::v0_4::{ArrowDigester, HasherConfig}; // hash format 1, as starfix 0.4.0
```

Exactness is tested against the v0.4.0 test suite and golden files (restored as
`tests/legacy_v0_4_*.rs`), the format-1 side of `tests/golden/format1_to_format2.json`, and a
differential test against the published starfix 0.4.0 crate; its source files are pinned by
`tests/legacy_v0_4_freeze.rs`. It reproduces **Rust** starfix 0.4.0; starfix-python's
`starfix.legacy.v0_4` reproduces **starfix-python** 0.4.0, which differed from Rust 0.4.0 on a
few inputs (see the divergence table in starfix-python's documentation).

## Hash format history

- **Format 2** (`00 00 02`, starfix 0.5): the current format.
- **Format 1** (`00 00 01`, starfix 0.4 and earlier). Every hash changed in format 2. Format 1
  lost information for some inputs, which format 2 fixes: Booleans were bit-packed without a
  count (a one-column non-nullable `[true]` equalled `[true, false, false]`, and the hash
  depended on batch boundaries); a null struct equalled a struct with all-null fields and null
  struct rows vanished under non-nullable fields; a list whose element is a nullable list lost
  its own lengths; a list of nullable structs did not distinguish a null element; and the row
  count of zero-column tables and empty structs was not hashed. Format-1 hashes are computed by
  the legacy 0.4 mode (`starfix::legacy::v0_4`, above). `tests/golden/format1_to_format2.json` maps every format-1 golden
  value of this repository to its format-2 value.

See `docs/byte-layout-spec.md`, Section 10, and the [CHANGELOG](CHANGELOG.md).

## Testing

- `tests/format2_bytes.rs`, `tests/digest_bytes.rs`: byte-level conformance with the spec.
- `tests/golden_vectors_format2.rs`: the golden vectors and their relations.
- `tests/nested_family.rs`: 420 nested column types, exhaustive small inputs, no collisions.
- `tests/lossless_properties.rs`: every supported leaf type × nullability × position × row
  count, batch split, slicing and empty-vs-absent.
- `tests/ordered_hashing.rs`, `tests/arrow_digester.rs`: options, rejections, invariants.
- `tests/float_normalization.rs`: the float options, and the property that hashing with one on
  equals hashing the normalized data with it off, over generated nested tables.

Regenerate the golden file (append-only) with `cargo run --bin emit_golden_format2`.
`tests/golden_vectors_format2.rs` also requires a minimum vector and relation count, so a
regeneration that drops vectors fails even if the generator was edited too.

### The Python package (`python/starfix`)

`python/tests` holds the package's own tests (the golden vectors through every call, tables in
several chunks, slices and empty tables, metadata, errors, the float options in
`test_float_options.py`) and `test_differential.py`, which
compares every result with starfix-python 0.5.0 under all 8 option combinations. Both packages
import as `starfix`, so starfix-python runs in its own environment, named by `STARFIX_PYTHON`
(the test is skipped without it). `python/tests/parity` runs starfix-python's own test suite
against this package (tests of starfix-python's private helpers are skipped, with the reason):

```sh
maturin develop --skip-install        # builds python/starfix/_internal*.so in place
python -m venv /tmp/sfp && /tmp/sfp/bin/pip install pyarrow \
  && /tmp/sfp/bin/pip install --no-deps --require-hashes -r python/tests/starfix-python-requirements.txt
STARFIX_PYTHON=/tmp/sfp/bin/python PYTHONPATH=python python -m pytest python/tests
# starfix-python's suite, from a checkout of metamorphic-labs/starfix-python:
PYTHONPATH=$OLDPWD/python:$OLDPWD/python/tests/parity python -m pytest -p starfix_python_private_stub tests
```

CI runs these against a built wheel (`.github/workflows/ci.yml`, job `python-binding`, also on
Python 3.10 with pyarrow 14), and `.github/workflows/maturin-release.yml` builds and tests the
wheels of every platform; only a release tag `v*` publishes them, to TestPyPI and then PyPI
(see [RELEASING.md](RELEASING.md)).
