//! Crate for starfix.

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema, SchemaRef};
use sha2::Sha256;

use crate::arrow_digester_core::{ArrowDigesterCore, SchemaAnalysis};
use crate::error::or_panic;

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
///
/// The float options `normalize_negative_zero` and `normalize_nan` are **not** recorded in the
/// hash: with a float option on, data is hashed exactly as the same data with its float values
/// normalized is hashed with the option off. Whether they were used is the caller's to know; do
/// not mix hashes taken with different float options where bit-exact identity matters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent on/off options, each a keyword argument of the Python binding"
)]
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
    /// When `true`, a `-0.0` value of a `Float16`, `Float32` or `Float64` leaf, at every nesting
    /// depth, is hashed as `+0.0` (all bits zero). Not recorded in the hash.
    /// Default is `false`: every float is hashed by its bits as is.
    pub normalize_negative_zero: bool,
    /// When `true`, every NaN value of a `Float16`, `Float32` or `Float64` leaf, at every nesting
    /// depth (any sign, quiet or signaling, any payload), is hashed as the quiet NaN with a clear
    /// sign bit and no payload (`0x7e00`, `0x7fc00000`, `0x7ff8000000000000`). Not recorded in
    /// the hash. Default is `false`: every float is hashed by its bits as is.
    pub normalize_nan: bool,
}

/// SHA-256 hasher of Arrow data, producing hash format 2 (`00 00 02 || SHA-256`, 35 bytes).
///
/// # Panics
///
/// The rejections are the same in every mode (`docs/byte-layout-spec.md`, Section 7); each method
/// lists its own under `# Panics`; each has a `try_*` variant returning [`StarfixError`] instead. A
/// refused `update` leaves the digester as it was.
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
        or_panic(Self::try_new(schema, config))
    }

    /// [`Self::new`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::AmbiguousPath`], [`StarfixError::UnsupportedType`] or
    /// [`StarfixError::UnsupportedData`], in the cases listed under [`Self::new`].
    pub fn try_new(schema: &Schema, config: HasherConfig) -> Result<Self, StarfixError> {
        Ok(Self {
            digester: ArrowDigesterCore::<Sha256>::try_new_with_config(schema, config)?,
        })
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
    /// A refused batch leaves the digester as it was (the panic can be caught and the digester
    /// used further).
    pub fn update(&mut self, record_batch: &RecordBatch) {
        or_panic(self.try_update(record_batch));
    }

    /// [`Self::update`], returning the refusal instead of panicking. A refused batch leaves the
    /// digester as it was: the per-node state is saved before the batch is hashed and restored
    /// on a refusal (a cost per node, not per row).
    ///
    /// # Errors
    ///
    /// [`StarfixError::SchemaMismatch`], [`StarfixError::AmbiguousPath`],
    /// [`StarfixError::UnsupportedType`], [`StarfixError::NonNullableNull`] or
    /// [`StarfixError::RowCountOverflow`], in the cases listed under [`Self::update`].
    pub fn try_update(&mut self, record_batch: &RecordBatch) -> Result<(), StarfixError> {
        self.digester.try_update(record_batch)
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
        or_panic(Self::try_hash_array_with_config(array, config))
    }

    /// [`Self::hash_array_with_config`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::AmbiguousPath`], [`StarfixError::UnsupportedType`],
    /// [`StarfixError::UnsupportedData`] or [`StarfixError::NonNullableNull`], in the cases
    /// listed under [`Self::hash_array_with_config`].
    pub fn try_hash_array_with_config(
        array: &dyn Array,
        config: HasherConfig,
    ) -> Result<Vec<u8>, StarfixError> {
        ArrowDigesterCore::<Sha256>::try_hash_array_with_config(array, config)
            .map(Self::prepend_version_bytes)
    }

    /// Hash the concatenation of `chunks`, every one of type `data_type` (for example the chunks
    /// of a chunked column), without concatenating them: equal to
    /// [`Self::hash_array_with_config`] of their concatenation. No chunks hash as an empty array
    /// of the type.
    ///
    /// # Panics
    ///
    /// As [`Self::hash_array_with_config`], and when a chunk's type is not `data_type`.
    pub fn hash_array_chunks(
        data_type: &DataType,
        chunks: &[&dyn Array],
        config: HasherConfig,
    ) -> Vec<u8> {
        or_panic(Self::try_hash_array_chunks(data_type, chunks, config))
    }

    /// [`Self::hash_array_chunks`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// As [`Self::try_hash_array_with_config`], and [`StarfixError::ChunkTypeMismatch`] when a
    /// chunk's type is not `data_type`.
    pub fn try_hash_array_chunks(
        data_type: &DataType,
        chunks: &[&dyn Array],
        config: HasherConfig,
    ) -> Result<Vec<u8>, StarfixError> {
        ArrowDigesterCore::<Sha256>::try_hash_array_chunks(data_type, chunks, config)
            .map(Self::prepend_version_bytes)
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
        or_panic(Self::try_hash_record_batch(record_batch, config))
    }

    /// [`Self::hash_record_batch`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::AmbiguousPath`], [`StarfixError::UnsupportedType`],
    /// [`StarfixError::UnsupportedData`] or [`StarfixError::NonNullableNull`], in the cases
    /// listed under [`Self::hash_record_batch`].
    pub fn try_hash_record_batch(
        record_batch: &RecordBatch,
        config: HasherConfig,
    ) -> Result<Vec<u8>, StarfixError> {
        ArrowDigesterCore::<Sha256>::try_hash_record_batch(record_batch, config)
            .map(Self::prepend_version_bytes)
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
        or_panic(Self::try_hash_schema(schema, config))
    }

    /// [`Self::hash_schema`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::AmbiguousPath`] or [`StarfixError::UnsupportedType`], in the cases
    /// listed under [`Self::hash_schema`].
    pub fn try_hash_schema(schema: &Schema, config: HasherConfig) -> Result<Vec<u8>, StarfixError> {
        ArrowDigesterCore::<Sha256>::try_hash_schema_with_config(schema, config)
            .map(Self::prepend_version_bytes)
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
        or_panic(Self::try_new(schema, config))
    }

    /// [`Self::new`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// As [`ArrowDigester::try_hash_schema`].
    pub fn try_new(schema: &Schema, config: HasherConfig) -> Result<Self, StarfixError> {
        Ok(Self {
            analysis: Arc::new(SchemaAnalysis::try_new(Arc::new(schema.clone()), config)?),
        })
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
        or_panic(self.try_hash_record_batch(record_batch))
    }

    /// [`Self::hash_record_batch`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::UnsupportedData`], [`StarfixError::PreparedSchemaMismatch`],
    /// [`StarfixError::AmbiguousPath`], [`StarfixError::UnsupportedType`] or
    /// [`StarfixError::NonNullableNull`], in the cases listed under [`Self::hash_record_batch`].
    pub fn try_hash_record_batch(
        &self,
        record_batch: &RecordBatch,
    ) -> Result<Vec<u8>, StarfixError> {
        let mut digester = ArrowDigesterCore::try_from_analysis(Arc::clone(&self.analysis))?;
        if !self.analysis.try_matches(record_batch.schema_ref())? {
            return Err(StarfixError::PreparedSchemaMismatch);
        }
        digester.ingest(record_batch)?;
        Ok(ArrowDigester::prepend_version_bytes(digester.finalize()))
    }

    /// A fresh streaming digester of the prepared schema, without analysing it again: hashes
    /// like `ArrowDigester::new(schema, config)`. Its `update` applies the same schema check.
    ///
    /// # Panics
    ///
    /// When the prepared schema contains a type whose data is not supported, as
    /// [`ArrowDigester::new`].
    pub fn digester(&self) -> ArrowDigester {
        or_panic(self.try_digester())
    }

    /// [`Self::digester`], returning the refusal instead of panicking.
    ///
    /// # Errors
    ///
    /// [`StarfixError::UnsupportedData`] when the prepared schema contains a type whose data is
    /// not supported.
    pub fn try_digester(&self) -> Result<ArrowDigester, StarfixError> {
        Ok(ArrowDigester {
            digester: ArrowDigesterCore::try_from_analysis(Arc::clone(&self.analysis))?,
        })
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
mod error;
pub use error::StarfixError;
pub mod legacy;
#[cfg(feature = "python")]
mod python;

// Write a test to check that int32 digest is consistent
