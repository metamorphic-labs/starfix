//! The Python extension module `starfix._internal` (feature `python`, built by maturin).
//!
//! Arrow data crosses through the Arrow `PyCapsule` interface (`__arrow_c_schema__`,
//! `__arrow_c_array__`): schemas and arrays through arrow-rs's `pyarrow` feature, record
//! batches through their capsules directly (arrow-rs's `RecordBatch` import applies a struct's
//! offset twice for a struct nested in a sliced struct, see [`without_struct_offsets`]). The
//! Python package (`python/starfix`) wraps these classes in starfix-python's interface; every
//! table is fed one record batch at a time.
//!
//! Refusals ([`StarfixError`]) raise `ValueError`. A panic of arrow-rs on invalid Arrow data is
//! caught at the boundary and raised as `ValueError` too, never as `PanicException`. The GIL is
//! released while a batch or array is hashed (not while it is imported: that reads Python
//! objects).
#![expect(
    unsafe_code,
    reason = "Reading Arrow C Data Interface structures from PyCapsules"
)]
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::{ptr, slice};

use arrow::array::{
    make_array, Array, ArrayData, ArrayDataBuilder, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::pyarrow::{FromPyArrow as _, ToPyArrow as _};
use arrow_schema::{DataType, Schema};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyTuple};

use crate::{ArrowDigester, HasherConfig, PreparedSchema, StarfixError};

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

/// A refusal as `ValueError`, with starfix-python's message: node paths as Python writes
/// them, types by pyarrow's names.
fn value_error(py: Python<'_>, error: &StarfixError) -> PyErr {
    let message = match error {
        StarfixError::AmbiguousPath(path) => format!(
            "Ambiguous node path {}: two schema nodes share one path (duplicate names, or a name \
             containing '/' that aliases another node)",
            py_repr(path)
        ),
        StarfixError::UnsupportedType(data_type) => format!(
            "Unsupported Arrow type for hashing: {}",
            pyarrow_name(py, data_type)
        ),
        StarfixError::UnsupportedData { path, data_type } => format!(
            "Hashing data of type {} (node {}) is not supported",
            pyarrow_name(py, data_type),
            py_repr(path)
        ),
        StarfixError::NonNullableNull { path } => {
            format!("Non-nullable field {} contains nulls", py_repr(path))
        }
        other => other.to_string(),
    };
    PyValueError::new_err(message)
}

/// pyarrow's name of a type (`str(pa_type)`), or arrow-rs's when pyarrow cannot take it.
fn pyarrow_name(py: Python<'_>, data_type: &DataType) -> String {
    data_type
        .to_pyarrow(py)
        .and_then(|pa_type| pa_type.str().map(|name| name.to_string()))
        .unwrap_or_else(|_| data_type.to_string())
}

/// Run `f` (hashing; no Python objects) without the GIL. A panic (arrow-rs on invalid Arrow
/// data) becomes `ValueError`.
fn hashing<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Result<T, StarfixError> + Send,
) -> PyResult<T> {
    py.detach(|| catch_unwind(AssertUnwindSafe(f)))
        .map_err(|payload| {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown error");
            PyValueError::new_err(format!("Invalid Arrow data: {message}"))
        })?
        .map_err(|error| value_error(py, &error))
}

/// The capsule `name` of `obj.method()` (a capsule or a pair of capsules, by position).
fn capsules<'py>(
    obj: &Bound<'py, PyAny>,
    method: &str,
) -> PyResult<(Bound<'py, PyCapsule>, Option<Bound<'py, PyCapsule>>)> {
    let missing = || {
        PyTypeError::new_err(format!(
            "expected an Arrow object with {method} (pyarrow >= 14), got {}",
            obj.get_type()
                .name()
                .map_or_else(|_| "an object".to_owned(), |name| name.to_string())
        ))
    };
    if !obj.hasattr(method)? {
        return Err(missing());
    }
    let result = obj.call_method0(method)?;
    if let Ok(tuple) = result.cast::<PyTuple>() {
        let schema = tuple.get_item(0)?.cast_into::<PyCapsule>()?;
        let array = tuple.get_item(1)?.cast_into::<PyCapsule>()?;
        return Ok((schema, Some(array)));
    }
    Ok((result.cast_into::<PyCapsule>()?, None))
}

fn check_capsule_name(capsule: &Bound<'_, PyCapsule>, expected: &CStr) -> PyResult<()> {
    if capsule.name()? == Some(expected) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "expected a PyCapsule named {expected:?}"
        )))
    }
}

/// A record batch from its `__arrow_c_array__` capsules, with its schema's metadata.
///
/// With `include_metadata`, metadata repeating a key is refused first, as starfix-python
/// refuses it (arrow-rs keeps one value of a repeated key on import).
fn import_record_batch(batch: &Bound<'_, PyAny>, include_metadata: bool) -> PyResult<RecordBatch> {
    let (schema_capsule, maybe_array_capsule) = capsules(batch, "__arrow_c_array__")?;
    let array_capsule = maybe_array_capsule
        .ok_or_else(|| PyTypeError::new_err("__arrow_c_array__ must return two capsules"))?;
    check_capsule_name(&schema_capsule, c"arrow_schema")?;
    check_capsule_name(&array_capsule, c"arrow_array")?;
    // SAFETY:
    // The capsule holds a valid `ArrowSchema` exported by its producer and stays alive (and
    // owns it) until the end of this function.
    let ffi_schema = unsafe { schema_capsule.reference::<FFI_ArrowSchema>() };
    if include_metadata {
        check_unique_metadata(ffi_schema)?;
    }
    let schema = Schema::try_from(ffi_schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
    // SAFETY:
    // The capsule holds a valid `ArrowArray` exported with this schema; `from_raw` moves it out,
    // leaving a released struct for the capsule's destructor.
    let ffi_array =
        unsafe { FFI_ArrowArray::from_raw(array_capsule.pointer().cast::<FFI_ArrowArray>()) };
    // SAFETY: `ffi_array` was exported with `ffi_schema` (the capsule pair of one export).
    let raw_data = unsafe { from_ffi(ffi_array, ffi_schema) }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !matches!(raw_data.data_type(), DataType::Struct(_)) {
        return Err(PyTypeError::new_err(
            "expected a record batch (a struct array from __arrow_c_array__)",
        ));
    }
    let array_data = imported(raw_data);
    let num_rows = array_data.len();
    let (_, columns, _) = StructArray::from(array_data).into_parts();
    RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// An array (`pa.Array`) through arrow-rs's `pyarrow` feature.
fn import_array(array: &Bound<'_, PyAny>) -> PyResult<ArrayData> {
    Ok(imported(ArrayData::from_pyarrow_bound(array)?))
}

/// A schema through arrow-rs's `pyarrow` feature, metadata included. With `include_metadata`,
/// metadata repeating a key is refused first.
fn import_schema(schema: &Bound<'_, PyAny>, include_metadata: bool) -> PyResult<Schema> {
    if include_metadata {
        let (capsule, _) = capsules(schema, "__arrow_c_schema__")?;
        check_capsule_name(&capsule, c"arrow_schema")?;
        // SAFETY: as in `import_record_batch`; the capsule outlives the reference.
        check_unique_metadata(unsafe { capsule.reference::<FFI_ArrowSchema>() })?;
    }
    Schema::from_pyarrow_bound(schema)
}

/// Buffers copied where less aligned than arrow-rs requires (pyarrow allows, for example,
/// `decimal128` values at an 8-byte boundary, as read from IPC), and struct offsets moved into
/// the children.
fn imported(mut data: ArrayData) -> ArrayData {
    data.align_buffers();
    without_struct_offsets(data)
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
            let size = usize::try_from(*value_length).unwrap_or_default();
            let (child_offset, child_len) = (offset.saturating_mul(size), len.saturating_mul(size));
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
    // (moved from the struct to its children, within their bounds, checked by `rows`).
    unsafe { builder.build_unchecked() }
}

/// Rows `offset .. offset + len` of `data`, without slicing a struct's children.
fn rows(data: &ArrayData, offset: usize, len: usize) -> ArrayData {
    // Out of bounds only for a malformed export: an empty array of the type, which the hasher
    // then refuses or hashes as empty, instead of a panic.
    if offset.checked_add(len).is_none_or(|end| end > data.len()) {
        return ArrayData::new_empty(data.data_type());
    }
    match data.data_type() {
        DataType::Struct(_) => rebuilt(
            data.clone()
                .into_builder()
                .offset(data.offset().saturating_add(offset))
                .len(len)
                .nulls(data.nulls().map(|nulls| nulls.slice(offset, len))),
        ),
        _ => data.slice(offset, len),
    }
}

/// The C ABI of an exported schema node (`ArrowSchema`, Arrow C Data Interface), to read the
/// metadata as exported: arrow-rs's `FFI_ArrowSchema` keeps its fields private and returns the
/// metadata as a map, which drops repeated keys.
#[repr(C)]
struct RawSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut RawSchema,
    dictionary: *mut RawSchema,
    release: Option<unsafe extern "C" fn(*mut RawSchema)>,
    private_data: *mut c_void,
}

/// Refuse metadata that repeats a key, on the schema or on any node (byte-layout spec Section
/// 7, as starfix-python), with starfix-python's message.
fn check_unique_metadata(schema: &FFI_ArrowSchema) -> PyResult<()> {
    let raw = ptr::from_ref(schema).cast::<RawSchema>();
    // SAFETY:
    // `FFI_ArrowSchema` is `#[repr(C)]` with exactly the `ArrowSchema` layout of the C Data
    // Interface, and `schema` is a valid exported schema, alive for this call.
    let root = unsafe { &*raw };
    if let Some(key) = repeated_key(root) {
        return Err(PyValueError::new_err(format!(
            "Duplicate metadata key {} on the schema: ambiguous, rejected with include_metadata",
            py_repr(&key)
        )));
    }
    for child in children(root) {
        check_node(child, &node_name(child))?;
    }
    Ok(())
}

/// A string as Python's `repr` writes it (the quotes of starfix-python's messages).
fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
    out
}

fn node_name(node: &RawSchema) -> String {
    if node.name.is_null() {
        return String::new();
    }
    // SAFETY: a non-null `name` is a NUL-terminated UTF-8 string (C Data Interface).
    unsafe { CStr::from_ptr(node.name) }
        .to_string_lossy()
        .into_owned()
}

fn format_of(node: &RawSchema) -> &str {
    if node.format.is_null() {
        return "";
    }
    // SAFETY: `format` is a NUL-terminated string (C Data Interface).
    unsafe { CStr::from_ptr(node.format) }
        .to_str()
        .unwrap_or_default()
}

fn children(node: &RawSchema) -> impl Iterator<Item = &RawSchema> {
    let count = usize::try_from(node.n_children).unwrap_or_default();
    (0..count).filter_map(move |i| {
        if node.children.is_null() {
            return None;
        }
        // SAFETY: `children` points to `n_children` valid child pointers (C Data Interface).
        let slot = unsafe { node.children.add(i) };
        // SAFETY: as above, `slot` is in bounds.
        let child = unsafe { *slot };
        // SAFETY: each child pointer is a valid `ArrowSchema` owned by the parent.
        (!child.is_null()).then(|| unsafe { &*child })
    })
}

/// The field node at `path` and its descendants (node paths of byte-layout spec Section 3.1;
/// a dictionary is its value type, a map's entries struct is not a node).
fn check_node(node: &RawSchema, path: &str) -> PyResult<()> {
    if let Some(key) = repeated_key(node) {
        return Err(PyValueError::new_err(format!(
            "Field {}: Duplicate metadata key {} on the field: ambiguous, rejected with \
             include_metadata",
            py_repr(path),
            py_repr(&key)
        )));
    }
    let logical = if node.dictionary.is_null() {
        node
    } else {
        // SAFETY: a non-null `dictionary` is the valid value-type schema (C Data Interface).
        unsafe { &*node.dictionary }
    };
    match format_of(logical) {
        "+s" => {
            for child in children(logical) {
                check_node(child, &format!("{path}/{}", node_name(child)))?;
            }
        }
        "+m" => {
            for entries in children(logical) {
                for (child, name) in children(entries).zip(["key", "value"]) {
                    check_node(child, &format!("{path}//{name}"))?;
                }
            }
        }
        format
            if format.starts_with("+l") || format.starts_with("+L") || format.starts_with("+w") =>
        {
            for child in children(logical) {
                check_node(child, &format!("{path}/"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The first key that a node's exported metadata repeats: `int32` count, then per pair `int32`
/// length + key bytes and `int32` length + value bytes (C Data Interface).
fn repeated_key(node: &RawSchema) -> Option<String> {
    if node.metadata.is_null() {
        return None;
    }
    let mut pos = node.metadata.cast::<u8>();
    let count = read_length(&mut pos);
    let mut seen = HashSet::new();
    for _ in 0..count {
        let key_len = read_length(&mut pos);
        // SAFETY: `key_len` bytes of key follow (C Data Interface).
        let key = unsafe { slice::from_raw_parts(pos, key_len) }.to_vec();
        // SAFETY: as above.
        pos = unsafe { pos.add(key_len) };
        let value_len = read_length(&mut pos);
        // SAFETY: `value_len` bytes of value follow (C Data Interface).
        pos = unsafe { pos.add(value_len) };
        if !seen.insert(key.clone()) {
            return Some(String::from_utf8_lossy(&key).into_owned());
        }
    }
    None
}

/// Read an `int32` of the metadata encoding at `pos` and advance past it.
fn read_length(pos: &mut *const u8) -> usize {
    // SAFETY: the metadata encoding is well formed (C Data Interface); reads stay in it.
    let value = unsafe { pos.cast::<i32>().read_unaligned() };
    // SAFETY: as above.
    *pos = unsafe { pos.add(4) };
    usize::try_from(value).unwrap_or_default()
}

fn bytes<'py>(py: Python<'py>, digest: &[u8]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, digest)
}

/// `ArrowDigester.hash_schema`.
#[pyfunction]
fn hash_schema<'py>(
    py: Python<'py>,
    py_schema: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let schema = import_schema(py_schema, include_metadata)?;
    let config = config(include_metadata, ordered_columns, ordered_struct_fields);
    let digest = hashing(py, || ArrowDigester::try_hash_schema(&schema, config))?;
    Ok(bytes(py, &digest))
}

/// `ArrowDigester.hash_record_batch`.
#[pyfunction]
fn hash_record_batch<'py>(
    py: Python<'py>,
    py_batch: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = import_record_batch(py_batch, include_metadata)?;
    let config = config(include_metadata, ordered_columns, ordered_struct_fields);
    let digest = hashing(py, || ArrowDigester::try_hash_record_batch(&batch, config))?;
    Ok(bytes(py, &digest))
}

/// `ArrowDigester.hash_table`: a digester over `batches`, the table's record batches (at least
/// one: the Python layer passes an empty batch for a table without any).
#[pyfunction]
fn hash_table<'py>(
    py: Python<'py>,
    py_schema: &Bound<'py, PyAny>,
    batches: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut digester = Digester::new(
        py,
        py_schema,
        include_metadata,
        ordered_columns,
        ordered_struct_fields,
    )?;
    for py_batch in batches.try_iter()? {
        digester.update(py, &py_batch?)?;
    }
    Ok(digester.finalize(py))
}

/// `ArrowDigester.hash_array`: the chunks of a `ChunkedArray` (or one array), all of type
/// `py_type`, hashed as their concatenation without concatenating them.
#[pyfunction]
fn hash_array<'py>(
    py: Python<'py>,
    py_type: &Bound<'py, PyAny>,
    py_chunks: &Bound<'py, PyAny>,
    ordered_columns: bool,
    ordered_struct_fields: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let data_type = DataType::from_pyarrow_bound(py_type)?;
    let chunks = py_chunks
        .try_iter()?
        .map(|chunk| Ok(make_array(import_array(&chunk?)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let config = config(false, ordered_columns, ordered_struct_fields);
    let digest = hashing(py, || {
        let chunk_refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
        ArrowDigester::try_hash_array_chunks(&data_type, &chunk_refs, config)
    })?;
    Ok(bytes(py, &digest))
}

/// A streaming digester (`ArrowDigester`); `update` takes one record batch.
#[pyclass(module = "starfix._internal", name = "Digester")]
struct Digester {
    digester: ArrowDigester,
    include_metadata: bool,
}

#[pymethods]
impl Digester {
    #[new]
    fn new(
        py: Python<'_>,
        py_schema: &Bound<'_, PyAny>,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
    ) -> PyResult<Self> {
        let schema = import_schema(py_schema, include_metadata)?;
        let config = config(include_metadata, ordered_columns, ordered_struct_fields);
        Ok(Self {
            digester: hashing(py, || ArrowDigester::try_new(&schema, config))?,
            include_metadata,
        })
    }

    /// Add a record batch; a refused batch leaves the digester as it was.
    fn update(&mut self, py: Python<'_>, py_batch: &Bound<'_, PyAny>) -> PyResult<()> {
        let batch = import_record_batch(py_batch, self.include_metadata)?;
        let digester = &mut self.digester;
        hashing(py, || digester.try_update(&batch))
    }

    /// The hash of everything added so far (the digester stays usable).
    fn finalize<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        bytes(py, &self.digester.clone().finalize())
    }
}

/// A prepared schema (`PreparedSchema`).
#[pyclass(module = "starfix._internal", name = "Prepared", frozen)]
struct Prepared {
    prepared: PreparedSchema,
}

#[pymethods]
impl Prepared {
    #[new]
    fn new(
        py: Python<'_>,
        py_schema: &Bound<'_, PyAny>,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
    ) -> PyResult<Self> {
        let schema = import_schema(py_schema, include_metadata)?;
        let config = config(include_metadata, ordered_columns, ordered_struct_fields);
        Ok(Self {
            prepared: hashing(py, || PreparedSchema::try_new(&schema, config))?,
        })
    }

    /// The schema digest (`hash_schema` with the same options).
    fn schema_digest<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        bytes(py, &self.prepared.schema_digest())
    }

    /// Whether data of the schema can be hashed: `None`, or the refusal's message.
    fn data_error(&self) -> Option<String> {
        self.prepared.try_digester().err().map(|e| e.to_string())
    }

    /// Hash one record batch of the prepared schema.
    fn hash_record_batch<'py>(
        &self,
        py: Python<'py>,
        py_batch: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let batch = import_record_batch(py_batch, self.prepared.config().include_metadata)?;
        let prepared = &self.prepared;
        let digest = hashing(py, || prepared.try_hash_record_batch(&batch))?;
        Ok(bytes(py, &digest))
    }

    /// Hash a table of the prepared schema, given its record batches (at least one).
    fn hash_table<'py>(
        &self,
        py: Python<'py>,
        batches: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut digester = self.digester(py)?;
        for py_batch in batches.try_iter()? {
            let batch = import_record_batch(&py_batch?, digester.include_metadata)?;
            let inner = &mut digester.digester;
            hashing(py, || {
                inner.try_update(&batch).map_err(|error| match error {
                    StarfixError::SchemaMismatch => StarfixError::PreparedSchemaMismatch,
                    other => other,
                })
            })
            .map_err(|error| {
                if error
                    .to_string()
                    .contains("does not match the prepared schema")
                {
                    PyValueError::new_err("Table schema does not match the prepared schema")
                } else {
                    error
                }
            })?;
        }
        Ok(digester.finalize(py))
    }

    /// A fresh streaming digester of the prepared schema.
    fn digester(&self, py: Python<'_>) -> PyResult<Digester> {
        let prepared = &self.prepared;
        Ok(Digester {
            digester: hashing(py, || prepared.try_digester())?,
            include_metadata: prepared.config().include_metadata,
        })
    }
}

/// The extension module.
#[pymodule]
fn _internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(hash_schema, module)?)?;
    module.add_function(wrap_pyfunction!(hash_record_batch, module)?)?;
    module.add_function(wrap_pyfunction!(hash_table, module)?)?;
    module.add_function(wrap_pyfunction!(hash_array, module)?)?;
    module.add_class::<Digester>()?;
    module.add_class::<Prepared>()?;
    Ok(())
}
