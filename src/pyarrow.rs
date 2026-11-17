//! The Python binding (`python/starfix`), over the Arrow C Data Interface.
//!
//! Python passes the raw pointers of the `PyCapsule`s that pyarrow's `__arrow_c_schema__` /
//! `__arrow_c_array__` return; each import moves the structures out of them. The Python layer
//! feeds every record batch of a table (ITL-1178); this module hashes what it is given.
#![expect(
    unsafe_code,
    clippy::expect_used,
    reason = "Converting raw pointers to Arrow structures"
)]
use std::sync::{Arc, Mutex};

use crate::{ArrowDigester, HasherConfig, PreparedSchema};
use arrow::array::{
    make_array, ArrayData, ArrayDataBuilder, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_schema::{DataType, Schema};

const fn config(
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> HasherConfig {
    HasherConfig {
        include_metadata,
        ordered_columns,
        ordered_struct_fields,
    }
}

/// Import a schema through the C Data Interface, keeping its schema-level metadata.
///
/// # Panics
/// The pointer must be a valid Arrow C Data Interface schema from Python's pyarrow.
fn import_schema(schema_ptr: u64) -> Schema {
    // SAFETY:
    // The pointer is a valid `FFI_ArrowSchema` exported by pyarrow; `from_raw` moves it out.
    let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema) };
    Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
}

/// Import an array through the C Data Interface.
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow.
fn import_array(array_ptr: u64, schema_ptr: u64) -> (ArrayData, FFI_ArrowSchema) {
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "Okay since we are doing the same operation of dereferencing pointers"
    )]
    // SAFETY:
    // The pointers are a valid `FFI_ArrowArray` and `FFI_ArrowSchema` exported together by
    // pyarrow; `from_raw` moves them out and `from_ffi` imports the array they describe.
    let (mut array_data, ffi_schema) = unsafe {
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        let array_data =
            from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data");
        (array_data, ffi_schema)
    };
    // pyarrow's buffers can be less aligned than arrow-rs requires (for example 8-byte aligned
    // decimals read from IPC): copy those, as arrow-rs's IPC reader does.
    array_data.align_buffers();
    (without_struct_offsets(array_data), ffi_schema)
}

/// The same logical array, with the offset of every struct (and fixed-size list) moved down
/// into its children.
///
/// The C Data Interface gives a struct an offset that applies to its unsliced children (pyarrow
/// exports a sliced struct, or a struct column of a sliced batch, this way). arrow-rs's
/// `ArrayData::slice` of a struct both keeps the offset and slices the children, and
/// `StructArray::from` then applies the offset to the children again: a struct nested in a
/// sliced struct or fixed-size list fails to import. With the offsets moved down, no struct
/// has both.
fn without_struct_offsets(data: ArrayData) -> ArrayData {
    let (offset, len) = (data.offset(), data.len());
    let children: Vec<ArrayData> = match data.data_type() {
        DataType::Struct(_) => data
            .child_data()
            .iter()
            .map(|child| without_struct_offsets(rows(child, offset, len)))
            .collect(),
        DataType::FixedSizeList(_, value_length) => {
            let size =
                usize::try_from(*value_length).expect("A fixed-size list's size is not negative");
            let (child_offset, child_len) = (
                offset.checked_mul(size).expect("Array offset overflow"),
                len.checked_mul(size).expect("Array length overflow"),
            );
            data.child_data()
                .iter()
                .map(|child| without_struct_offsets(rows(child, child_offset, child_len)))
                .collect()
        }
        _ if data.child_data().is_empty() => return data,
        // Lists, maps and dictionaries index their child by its logical position.
        _ => {
            let children = data
                .child_data()
                .iter()
                .cloned()
                .map(without_struct_offsets)
                .collect();
            return rebuilt(data.into_builder().child_data(children));
        }
    };
    let nulls = data.nulls().cloned();
    rebuilt(
        data.into_builder()
            .offset(0)
            .nulls(nulls)
            .child_data(children),
    )
}

/// Build an imported array re-expressed by [`without_struct_offsets`] or [`rows`], unchecked.
///
/// Full validation is not applied, as `from_ffi` does not apply it to the imported array: it
/// would refuse arrays the format hashes, such as a null in a non-nullable child under a null
/// parent (Section 7 decides those).
fn rebuilt(builder: ArrayDataBuilder) -> ArrayData {
    // SAFETY:
    // The builder holds the buffers, children and validity of an array imported from the C Data
    // Interface, with the same logical rows: only where a struct's offset is applied changed
    // (moved from the struct to its children, within their bounds; `rows` slices bounds-checked).
    unsafe { builder.build_unchecked() }
}

/// Rows `offset .. offset + len` of `data`, without slicing a struct's children.
fn rows(data: &ArrayData, offset: usize, len: usize) -> ArrayData {
    let end = offset.checked_add(len).expect("Array offset overflow");
    assert!(
        end <= data.len(),
        "Rows out of bounds of the imported array"
    );
    match data.data_type() {
        DataType::Struct(_) => rebuilt(
            data.clone()
                .into_builder()
                .offset(
                    data.offset()
                        .checked_add(offset)
                        .expect("Array offset overflow"),
                )
                .len(len)
                .nulls(data.nulls().map(|nulls| nulls.slice(offset, len))),
        ),
        _ => data.slice(offset, len),
    }
}

/// Import a record batch through the C Data Interface, keeping the schema-level metadata of the
/// exported schema (a plain `StructArray` import would drop it).
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow.
fn import_record_batch(array_ptr: u64, schema_ptr: u64) -> RecordBatch {
    let (array_data, ffi_schema) = import_array(array_ptr, schema_ptr);
    let schema =
        Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema");
    let num_rows = array_data.len();
    // The columns (any struct offset was moved into the children on import).
    let (_, columns, _) = StructArray::from(array_data).into_parts();
    // The row count is explicit: a batch may have no columns.
    RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .expect("Failed to build the record batch")
}

/// Hash one record batch (`ArrowDigester::hash_record_batch`).
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow. Panics as
/// `ArrowDigester::hash_record_batch` otherwise.
#[uniffi::export]
pub fn hash_record_batch(
    array_ptr: u64,
    schema_ptr: u64,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> Vec<u8> {
    ArrowDigester::hash_record_batch(
        &import_record_batch(array_ptr, schema_ptr),
        config(include_metadata, ordered_columns, ordered_struct_fields),
    )
}

/// Hash a schema (`ArrowDigester::hash_schema`).
///
/// # Panics
/// The pointer must be a valid Arrow schema from Python's pyarrow. Panics as
/// `ArrowDigester::hash_schema` otherwise.
#[uniffi::export]
pub fn hash_schema(
    schema_ptr: u64,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> Vec<u8> {
    ArrowDigester::hash_schema(
        &import_schema(schema_ptr),
        config(include_metadata, ordered_columns, ordered_struct_fields),
    )
}

/// Hash one array (`ArrowDigester::hash_array_with_config`; an array has no metadata to include).
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow. Panics as
/// `ArrowDigester::hash_array_with_config` otherwise.
#[uniffi::export]
pub fn hash_array(
    array_ptr: u64,
    schema_ptr: u64,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> Vec<u8> {
    let (array_data, _) = import_array(array_ptr, schema_ptr);
    ArrowDigester::hash_array_with_config(
        make_array(array_data).as_ref(),
        config(false, ordered_columns, ordered_struct_fields),
    )
}

/// A streaming digester. A rejected update (a panic) poisons it: every later call panics.
#[derive(uniffi::Object)]
pub struct InternalPyArrowDigester {
    digester: Mutex<ArrowDigester>,
}

#[uniffi::export]
impl InternalPyArrowDigester {
    /// A streaming digester of `schema` under the options (`ArrowDigester::new`).
    ///
    /// # Panics
    /// The pointer must be a valid Arrow schema from Python's pyarrow. Panics as
    /// `ArrowDigester::new` otherwise.
    #[uniffi::constructor]
    pub fn new(
        schema_ptr: u64,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
    ) -> Self {
        Self {
            digester: Mutex::new(ArrowDigester::new(
                &import_schema(schema_ptr),
                config(include_metadata, ordered_columns, ordered_struct_fields),
            )),
        }
    }

    /// Update the digester with a record batch, keeping its schema-level metadata.
    ///
    /// # Panics
    /// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow. Panics as
    /// `ArrowDigester::update` otherwise, after which the digester cannot be used.
    pub fn update(&self, array_ptr: u64, schema_ptr: u64) {
        let batch = import_record_batch(array_ptr, schema_ptr);
        self.digester
            .lock()
            .expect("The digester cannot be used after a rejected update")
            .update(&batch);
    }

    /// The hash of everything added so far; the digester can still be updated.
    ///
    /// # Panics
    /// When an earlier update was rejected.
    pub fn finalize(&self) -> Vec<u8> {
        self.digester
            .lock()
            .expect("The digester cannot be used after a rejected update")
            .clone()
            .finalize()
    }
}

/// A schema analysed once under the hashing options (see `PreparedSchema`).
#[derive(uniffi::Object)]
pub struct InternalPreparedSchema {
    prepared: PreparedSchema,
}

#[uniffi::export]
impl InternalPreparedSchema {
    /// Analyse the schema once.
    ///
    /// # Panics
    /// The pointer must be a valid Arrow schema from Python's pyarrow. Panics as
    /// `ArrowDigester::hash_schema` otherwise.
    #[uniffi::constructor]
    pub fn new(
        schema_ptr: u64,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
    ) -> Self {
        Self {
            prepared: PreparedSchema::new(
                &import_schema(schema_ptr),
                config(include_metadata, ordered_columns, ordered_struct_fields),
            ),
        }
    }

    /// The schema digest (equal to `hash_schema` with the same options).
    pub fn schema_digest(&self) -> Vec<u8> {
        self.prepared.schema_digest()
    }

    /// Hash a record batch of the prepared schema.
    ///
    /// # Panics
    /// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow. Panics as
    /// `PreparedSchema::hash_record_batch` otherwise (for example when the batch's schema is not
    /// the prepared one).
    pub fn hash_record_batch(&self, array_ptr: u64, schema_ptr: u64) -> Vec<u8> {
        self.prepared
            .hash_record_batch(&import_record_batch(array_ptr, schema_ptr))
    }

    /// A fresh streaming digester of the prepared schema.
    ///
    /// # Panics
    /// As `PreparedSchema::digester`.
    pub fn digester(&self) -> Arc<InternalPyArrowDigester> {
        Arc::new(InternalPyArrowDigester {
            digester: Mutex::new(self.prepared.digester()),
        })
    }
}
