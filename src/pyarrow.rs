#![expect(
    unsafe_code,
    clippy::expect_used,
    reason = "Converting raw pointers to Arrow structures"
)]
use std::sync::{Arc, Mutex};

use crate::{ArrowDigester, HasherConfig, PreparedSchema};
use arrow::array::{RecordBatch, RecordBatchOptions, StructArray};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_schema::Schema;

/// Process an Arrow table via C Data Interface
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow

#[uniffi::export]
pub fn hash_record_batch(array_ptr: u64, schema_ptr: u64) -> Vec<u8> {
    #[expect(
        unsafe_code,
        reason = "Need to convert raw pointers to Arrow data structures"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        clippy::expect_used,
        reason = "Okay since we are doing the same operation of dereferencing pointers, Will add proper errors later"
    )]
    // SAFETY:
    // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
    let array_data = unsafe {
        // Construct ArrayData from FFI structures
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data")
    };

    // Hash the table
    ArrowDigester::hash_record_batch(
        &RecordBatch::from(StructArray::from(array_data)),
        HasherConfig::default(),
    )
}

/// Process an Arrow schema via C Data Interface
///
/// # Panics
/// The pointer must be a valid Arrow schema from Python's pyarrow
#[uniffi::export]
pub fn hash_schema(schema_ptr: u64) -> Vec<u8> {
    #[expect(
        unsafe_code,
        reason = "Need to convert raw pointers to Arrow data structures"
    )]
    // SAFETY:
    // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
    let schema = unsafe {
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
    };

    // Hash the schema
    ArrowDigester::hash_schema(&schema, HasherConfig::default())
}

#[derive(uniffi::Object)]
pub struct InternalPyArrowDigester {
    digester: Arc<Mutex<ArrowDigester>>,
}

#[uniffi::export]
impl InternalPyArrowDigester {
    /// Create a new instance of `PyArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    ///
    /// # Panics
    /// The pointer must be a valid Arrow schema from Python's pyarrow. Panics if conversion fails.

    #[uniffi::constructor]
    pub fn new(schema_ptr: u64) -> Self {
        // SAFETY:
        // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
        let schema = unsafe {
            let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
        };
        Self {
            digester: Arc::new(Mutex::new(ArrowDigester::new(
                &schema,
                HasherConfig::default(),
            ))),
        }
    }

    /// Update the digester with a new `RecordBatch`
    ///
    /// # Panics
    /// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow
    pub fn update(&self, array_ptr: u64, schema_ptr: u64) {
        #[expect(
            unsafe_code,
            reason = "Need to convert raw pointers to Arrow data structures"
        )]
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            clippy::expect_used,
            reason = "Okay since we are doing the same operation of dereferencing pointers, Will add proper errors later"
        )]
        // SAFETY:
        // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
        let array_data = unsafe {
            // Construct ArrayData from FFI structures
            let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
            let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data")
        };

        self.digester
            .lock()
            .expect("Failed to acquire lock on digester")
            .update(&RecordBatch::from(StructArray::from(array_data)));
    }

    /// Consume the digester and finalize the hash computation
    ///
    /// # Panics
    /// Panics if it fails to acquire the lock on the digester.
    pub fn finalize(&self) -> Vec<u8> {
        self.digester
            .lock()
            .expect("Failed to acquire lock on digester")
            .clone()
            .finalize()
    }
}

/// Import a record batch through the C Data Interface, keeping the schema-level metadata of the
/// exported schema (a plain `StructArray` import would drop it).
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow
fn import_record_batch(array_ptr: u64, schema_ptr: u64) -> RecordBatch {
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "Okay since we are doing the same operation of dereferencing pointers"
    )]
    // SAFETY:
    // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
    let (array_data, schema) = unsafe {
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        let schema =
            Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema");
        let array_data =
            from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data");
        (array_data, schema)
    };
    let num_rows = array_data.len();
    let (_, columns, _) = StructArray::from(array_data).into_parts();
    // The row count is explicit: a batch may have no columns.
    RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .expect("Failed to build the record batch")
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
        // SAFETY:
        // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
        let schema = unsafe {
            let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
        };
        Self {
            prepared: PreparedSchema::new(
                &schema,
                HasherConfig {
                    include_metadata,
                    ordered_columns,
                    ordered_struct_fields,
                },
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
            digester: Arc::new(Mutex::new(self.prepared.digester())),
        })
    }
}

#[uniffi::export]
impl InternalPyArrowDigester {
    /// Update the digester with a record batch, keeping its schema-level metadata.
    ///
    /// # Panics
    /// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow. Panics as
    /// `ArrowDigester::update` otherwise.
    pub fn update_record_batch(&self, array_ptr: u64, schema_ptr: u64) {
        let batch = import_record_batch(array_ptr, schema_ptr);
        self.digester
            .lock()
            .expect("Failed to acquire lock on digester")
            .update(&batch);
    }
}
