/// Crate for starfix.
extern crate uniffi as uniffi_external;
uniffi_external::setup_scaffolding!();

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow_schema::{Schema, SchemaRef};
use sha2::Sha256;

use crate::arrow_digester_core::{ArrowDigesterCore, SchemaAnalysis};

/// Hash format version: the first 3 bytes of every hash, a 24-bit big-endian format number,
/// independent of the crate version. Format 1 (`00 00 01`): starfix 0.4 and earlier. Format 2
/// (`00 00 02`): this crate (byte-layout spec Section 1; format history in Section 10).
pub(crate) const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 2_u8];

/// Configuration for the Arrow hasher.
///
/// Every option defaults to `false`. Options are per call (or per streaming digester), never
/// global, so ordered and unordered hashing can be used side by side. Construct with struct-update syntax to stay forward compatible:
///
/// ```
/// use starfix::HasherConfig;
/// let config = HasherConfig { ordered_columns: true, ..HasherConfig::default() };
/// ```
///
/// The order options are recorded in the header of every digest, so hashes taken with different
/// order options never coincide; the header also names the entry point (schema, record batch or
/// stream, array), so outputs of different entry points never coincide either (see
/// `docs/byte-layout-spec.md`, Sections 1 and 6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HasherConfig {
    /// When `true`, schema-level and per-field Arrow metadata are included in the hash.
    /// Default is `false`.
    pub include_metadata: bool,
    /// When `true`, the order of the top-level columns is part of the hash.
    /// Default is `false`: reordering columns does not change the hash.
    pub ordered_columns: bool,
    /// When `true`, the order of struct fields is part of the hash, at every nesting depth
    /// (struct in struct, list, large list, fixed-size list, map, dictionary value type).
    /// Default is `false`: reordering struct fields does not change the hash.
    pub ordered_struct_fields: bool,
}

/// SHA-256 hasher of Arrow data, producing hash format 2 (`00 00 02 || SHA-256`, 35 bytes).
///
/// # Panics
///
/// The rejections are the same in every mode (`docs/byte-layout-spec.md`, Section 7); each method
/// lists its own under `# Panics`. After a panic, discard the digester.
#[derive(Clone)]
pub struct ArrowDigester {
    digester: ArrowDigesterCore<Sha256>,
}

impl ArrowDigester {
    /// Create a new instance of `ArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the schema has two fields at the same path (duplicate column names, duplicate struct field
    ///   names, or a name containing `/` that aliases a nested field);
    /// - the schema contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type;
    /// - the schema contains a type whose data is not supported (`Null`, `Map`, `Interval`,
    ///   `Utf8View`, `BinaryView`).
    pub fn new(schema: &Schema, config: HasherConfig) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new_with_config(schema, config),
        }
    }

    /// Analyse `schema` once under `config`, for hashing many record batches of that schema, or
    /// starting many streaming digesters, without analysing it again. Same as
    /// [`PreparedSchema::new`].
    ///
    /// # Panics
    ///
    /// As [`Self::hash_schema`].
    pub fn prepare(schema: &Schema, config: HasherConfig) -> PreparedSchema {
        PreparedSchema::new(schema, config)
    }

    /// Update the digester with a new `RecordBatch`.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the batch's schema does not match the digester's under its options (for example a
    ///   different column order with `ordered_columns`), or has two fields at the same path, or
    ///   contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type;
    /// - a non-nullable field holds a null at an element that no enclosing null struct masks;
    /// - the total row count overflows `u64`.
    ///
    /// After a panic the digester's state is unspecified: discard it.
    pub fn update(&mut self, record_batch: &RecordBatch) {
        self.digester.update(record_batch);
    }

    /// Consume the digester and finalize the hash computation.
    pub fn finalize(self) -> Vec<u8> {
        Self::prepend_version_bytes(self.digester.finalize())
    }

    /// Hash an array in one go with the default configuration.
    ///
    /// Equivalent to `hash_array_with_config(array, HasherConfig::default())`.
    ///
    /// # Panics
    ///
    /// As [`Self::hash_array_with_config`].
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        Self::hash_array_with_config(array, HasherConfig::default())
    }

    /// Hash an array in one go with the given configuration.
    ///
    /// `include_metadata` has no effect here (there is no schema). `ordered_columns` has no
    /// effect on the layout, but like every order option it is recorded in the header. The array
    /// is always treated as nullable: its validity is always hashed.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the array's type has two fields at the same path, or contains a `ListView`,
    ///   `LargeListView`, `Union` or `RunEndEncoded` type;
    /// - the array's type contains a type whose data is not supported (`Null`, `Map`,
    ///   `Interval`, `Utf8View`, `BinaryView`);
    /// - a non-nullable field inside the array holds a null at an element that no enclosing null
    ///   struct masks.
    pub fn hash_array_with_config(array: &dyn Array, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_array_with_config(
            array, config,
        ))
    }

    /// Hash a complete `RecordBatch` in one go.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the schema has two fields at the same path (duplicate column names, duplicate struct field
    ///   names, or a name containing `/` that aliases a nested field);
    /// - the schema contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type;
    /// - the schema contains a type whose data is not supported (`Null`, `Map`, `Interval`,
    ///   `Utf8View`, `BinaryView`), whatever the number of rows;
    /// - a non-nullable field holds a null at an element that no enclosing null struct masks.
    pub fn hash_record_batch(record_batch: &RecordBatch, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_record_batch(
            record_batch,
            config,
        ))
    }

    /// Hash a schema only.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the schema has two fields at the same path (duplicate column names, duplicate struct field
    ///   names, or a name containing `/` that aliases a nested field);
    /// - the schema contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type.
    pub fn hash_schema(schema: &Schema, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_schema_with_config(
            schema, config,
        ))
    }

    fn prepend_version_bytes(digest: Vec<u8>) -> Vec<u8> {
        let mut complete_hash = VERSION_BYTES.clone().to_vec();
        complete_hash.extend(digest);
        complete_hash
    }
}

/// A schema analysed once under one [`HasherConfig`] (ITL-1177): hash many record batches of it,
/// or start many streaming digesters, without the schema analysis being redone.
///
/// The analysis (validation, the schema digest, the canonical schema key, the per-node entry
/// layout) runs once, in [`Self::new`]. The object is immutable and owned by the caller; starfix
/// keeps no cache. Cloning is cheap (the analysis is shared, never modified).
///
/// Every hash is byte-identical to the one-shot path:
/// `prepared.hash_record_batch(&batch) == ArrowDigester::hash_record_batch(&batch, config)`,
/// `prepared.schema_digest() == ArrowDigester::hash_schema(&schema, config)`, and
/// `prepared.digester()` hashes like `ArrowDigester::new(&schema, config)`.
///
/// # Same schema
///
/// A batch is hashed only if its schema is the prepared one under the prepared options: the
/// per-batch schema check of the streaming digester (`docs/byte-layout-spec.md`, Section 5).
/// Its schema must have the same schema digest under the options (equal Phase 1 JSON: names,
/// canonical types, nullability, column order with `ordered_columns`, struct field order with
/// `ordered_struct_fields`; and with `include_metadata` equal field metadata at every depth,
/// extension type names and metadata included, and equal schema metadata), and it must pass
/// Section 7. Canonically equal but physically different types (`Utf8` and `LargeUtf8`,
/// dictionary and plain, `List` and `LargeList`) are accepted, as by the streaming digester.
///
/// The check costs a pointer comparison when the batch carries the prepared schema object
/// ([`Self::schema`]), and a structural comparison without allocation when it carries an equal
/// one; only a schema that differs from the prepared one is analysed to decide.
#[derive(Clone)]
pub struct PreparedSchema {
    analysis: Arc<SchemaAnalysis<Sha256>>,
}

impl PreparedSchema {
    /// Analyse `schema` under `config`.
    ///
    /// Succeeds whenever [`ArrowDigester::hash_schema`] does, so [`Self::schema_digest`] is
    /// available for every schema that has one. A schema whose data cannot be hashed (it
    /// contains `Null`, `Map`, `Interval`, `Utf8View` or `BinaryView`) is prepared, but
    /// [`Self::hash_record_batch`] and [`Self::digester`] reject it, as [`ArrowDigester::new`]
    /// does.
    ///
    /// # Panics
    ///
    /// As [`ArrowDigester::hash_schema`].
    pub fn new(schema: &Schema, config: HasherConfig) -> Self {
        Self {
            analysis: Arc::new(SchemaAnalysis::new(Arc::new(schema.clone()), config)),
        }
    }

    /// The prepared schema. A record batch built with this `SchemaRef` passes the same-schema
    /// check by pointer comparison.
    pub fn schema(&self) -> &SchemaRef {
        self.analysis.schema()
    }

    /// The options the schema was prepared with.
    pub fn config(&self) -> HasherConfig {
        self.analysis.config()
    }

    /// The schema digest: equal to `ArrowDigester::hash_schema(schema, config)`.
    pub fn schema_digest(&self) -> Vec<u8> {
        ArrowDigester::prepend_version_bytes(self.analysis.schema_digest().to_vec())
    }

    /// Hash a complete `RecordBatch` of the prepared schema: equal to
    /// `ArrowDigester::hash_record_batch(record_batch, config)`.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the prepared schema contains a type whose data is not supported (`Null`, `Map`,
    ///   `Interval`, `Utf8View`, `BinaryView`), whatever the batch;
    /// - the batch's schema is not the prepared schema under the prepared options (see
    ///   [`PreparedSchema`], "Same schema"), with the message "Record batch schema does not
    ///   match the prepared schema";
    /// - a non-nullable field holds a null at an element that no enclosing null struct masks.
    pub fn hash_record_batch(&self, record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = ArrowDigesterCore::from_analysis(Arc::clone(&self.analysis));
        assert!(
            self.analysis.matches(record_batch.schema_ref()),
            "Record batch schema does not match the prepared schema"
        );
        digester.update_matched(record_batch);
        ArrowDigester::prepend_version_bytes(digester.finalize())
    }

    /// A fresh streaming digester of the prepared schema, without analysing it again: hashes
    /// like `ArrowDigester::new(schema, config)`. Its `update` applies the same schema check.
    ///
    /// # Panics
    ///
    /// When the prepared schema contains a type whose data is not supported, as
    /// [`ArrowDigester::new`].
    pub fn digester(&self) -> ArrowDigester {
        ArrowDigester {
            digester: ArrowDigesterCore::from_analysis(Arc::clone(&self.analysis)),
        }
    }
}

impl fmt::Debug for PreparedSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSchema")
            .field("schema", self.schema())
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

pub(crate) mod arrow_digester_core;
pub mod legacy;
pub mod pyarrow;

// Write a test to check that int32 digest is consistent
