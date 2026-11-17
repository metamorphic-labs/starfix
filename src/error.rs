//! Why starfix refuses an input: the error of the `try_*` API (ITL-1178).
//!
//! Every refusal of the byte-layout spec (Sections 5, 7 and 8) is a [`StarfixError`]. The
//! panicking API (`ArrowDigester::new`, `update`, `hash_*`, `PreparedSchema`) panics with the
//! error's message, so both report the same thing.
use std::error::Error;
use std::fmt;

use arrow_schema::DataType;

/// Why an input cannot be hashed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StarfixError {
    /// Two nodes of the schema share a node path: duplicate column or struct field names, or a
    /// name containing `/` that aliases a nested field (byte-layout spec Section 7). Holds the
    /// path.
    AmbiguousPath(String),
    /// The schema contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type
    /// (byte-layout spec Section 7). Holds the type.
    UnsupportedType(DataType),
    /// Data of this type cannot be hashed (`Null`, `Map`, `Interval`, `Utf8View`,
    /// `BinaryView`; byte-layout spec Section 8). Its schema can.
    UnsupportedData {
        /// The node path of the first such node (byte-layout spec Section 3.1).
        path: String,
        /// Its logical type (a dictionary's value type), as in the schema.
        data_type: DataType,
    },
    /// A record batch whose schema is not the streaming digester's under its options
    /// (byte-layout spec Section 5).
    SchemaMismatch,
    /// A record batch whose schema is not the prepared schema under the prepared options.
    PreparedSchemaMismatch,
    /// A non-nullable field holds a null that no enclosing null struct masks (byte-layout spec
    /// Section 7).
    NonNullableNull {
        /// The node path of the field (byte-layout spec Section 3.1).
        path: String,
    },
    /// The total row count of a stream does not fit in a `u64`.
    RowCountOverflow,
}

impl fmt::Display for StarfixError {
    #[expect(
        clippy::use_debug,
        reason = "The path is quoted and escaped, as in the panic message before the error type"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AmbiguousPath(path) => write!(
                f,
                "Ambiguous field path {path:?}: duplicate field names, or a field name containing \
                 \"/\" that collides with a nested field, cannot be hashed"
            ),
            Self::UnsupportedType(_) => f.write_str(
                "ListView, LargeListView, Union and RunEndEncoded types cannot be hashed",
            ),
            Self::UnsupportedData { data_type, .. } => {
                write!(f, "hashing data of type {data_type} is not supported")
            }
            Self::SchemaMismatch => {
                f.write_str("Record batch schema does not match ArrowDigester schema")
            }
            Self::PreparedSchemaMismatch => {
                f.write_str("Record batch schema does not match the prepared schema")
            }
            Self::NonNullableNull { .. } => {
                f.write_str("A non-nullable field contains null values")
            }
            Self::RowCountOverflow => f.write_str("row count overflow"),
        }
    }
}

impl Error for StarfixError {}

/// The value, or a panic with the error's message: the panicking API on top of the `try_*` API.
/// Unsupported data panics as `todo!` did before the `try_*` API existed.
#[expect(
    clippy::panic,
    clippy::todo,
    reason = "The panicking API reports refusals by panicking"
)]
pub fn or_panic<T>(result: Result<T, StarfixError>) -> T {
    match result {
        Ok(value) => value,
        Err(error @ StarfixError::UnsupportedData { .. }) => todo!("{error}"),
        Err(error) => panic!("{error}"),
    }
}
