//! The Python extension module `starfix._internal` (feature `python`, built by maturin).
//!
//! Arrow data crosses through the Arrow `PyCapsule` interface (`__arrow_c_schema__`,
//! `__arrow_c_array__`), read with arrow-rs's C Data Interface import after these repairs: an
//! empty node may omit the offsets buffer that arrow-rs requires ([`fill_empty_offsets`]);
//! metadata that is not hashed is cleared, as arrow-rs refuses metadata that is not UTF-8
//! ([`exported_schema`]); names cut at a NUL are taken from pyarrow ([`with_full_names`]);
//! arrow-rs's `RecordBatch` import applies a struct's offset twice for a struct nested in a
//! sliced struct ([`without_struct_offsets`]). The Python package (`python/starfix`) wraps these
//! classes in starfix-python's interface; every table is fed one record batch at a time.
//!
//! Refusals ([`StarfixError`]) raise `ValueError`. A panic of arrow-rs on invalid Arrow data,
//! while it is imported or hashed, is caught and raised as `ValueError` too, never as
//! `PanicException`. The GIL is released while a batch or array is hashed (not while it is
//! imported: that reads Python objects).
#![expect(
    unsafe_code,
    reason = "Reading Arrow C Data Interface structures from PyCapsules"
)]
use std::any::Any;
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::{ptr, slice, str};

use arrow::array::{
    make_array, Array, ArrayData, ArrayDataBuilder, ArrayRef, RecordBatch, RecordBatchOptions,
    StructArray,
};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::pyarrow::{PyArrowException, ToPyArrow as _};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, Schema};
use pyo3::exceptions::{PyTypeError, PyUnicodeDecodeError, PyValueError};
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

/// pyarrow's name of a type (`str(pa_type)`), or arrow-rs's when pyarrow cannot take it or a
/// field name in it contains NUL (arrow-rs's export of such a name panics).
fn pyarrow_name(py: Python<'_>, data_type: &DataType) -> String {
    if has_nul_name(data_type) {
        return data_type.to_string();
    }
    data_type
        .to_pyarrow(py)
        .and_then(|pa_type| pa_type.str().map(|name| name.to_string()))
        .unwrap_or_else(|_| data_type.to_string())
}

/// Whether a field name in `data_type`, at any depth, contains NUL.
fn has_nul_name(data_type: &DataType) -> bool {
    let nul_in = |field: &Field| field.name().contains('\0') || has_nul_name(field.data_type());
    match data_type {
        DataType::Struct(fields) => fields.iter().any(|field| nul_in(field)),
        DataType::Union(fields, _) => fields.iter().any(|(_, field)| nul_in(field)),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => nul_in(field),
        DataType::RunEndEncoded(run_ends, values) => nul_in(run_ends) || nul_in(values),
        DataType::Dictionary(_, value) => has_nul_name(value),
        _ => false,
    }
}

/// Run `f` (hashing; no Python objects) without the GIL. A panic (arrow-rs on invalid Arrow
/// data) becomes `ValueError`.
fn hashing<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Result<T, StarfixError> + Send,
) -> PyResult<T> {
    py.detach(|| catch_unwind(AssertUnwindSafe(f)))
        .map_err(|payload| invalid_data(payload.as_ref()))?
        .map_err(|error| value_error(py, &error))
}

/// Run an import step of Arrow data: a panic of arrow-rs on malformed data (a struct child
/// shorter than the struct, say) becomes `ValueError`, as in [`hashing`].
fn importing<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| invalid_data(payload.as_ref()))?
}

/// The `ValueError` of a panic of arrow-rs on invalid Arrow data.
fn invalid_data(payload: &(dyn Any + Send)) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown error");
    PyValueError::new_err(format!("Invalid Arrow data: {message}"))
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

/// The schema and array capsules of `obj.__arrow_c_array__()`, checked by name.
fn array_capsules<'py>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let (schema_capsule, maybe_array_capsule) = capsules(obj, "__arrow_c_array__")?;
    let array_capsule = maybe_array_capsule
        .ok_or_else(|| PyTypeError::new_err("__arrow_c_array__ must return two capsules"))?;
    check_capsule_name(&schema_capsule, c"arrow_schema")?;
    check_capsule_name(&array_capsule, c"arrow_array")?;
    Ok((schema_capsule, array_capsule))
}

/// The array of an `arrow_array` capsule exported with `ffi_schema` (of type `data_type`),
/// moved out of the capsule, its empty nodes given offsets first ([`fill_empty_offsets`]).
fn take_array(
    array_capsule: &Bound<'_, PyCapsule>,
    ffi_schema: &FFI_ArrowSchema,
    data_type: &DataType,
) -> Result<ArrayData, ArrowError> {
    let raw = array_capsule.pointer().cast::<RawArray>();
    // SAFETY:
    // The capsule holds a valid `ArrowArray` exported with `ffi_schema`, alive (and owned by the
    // capsule) until it is moved out below.
    fill_empty_offsets(unsafe { &*raw }, data_type);
    // SAFETY:
    // As above; `from_raw` moves the `ArrowArray` out, leaving a released struct for the
    // capsule's destructor.
    let ffi_array = unsafe { FFI_ArrowArray::from_raw(raw.cast::<FFI_ArrowArray>()) };
    // SAFETY: `ffi_array` was exported with `ffi_schema` (the capsule pair of one export).
    unsafe { from_ffi(ffi_array, ffi_schema) }
}

/// A record batch from its `__arrow_c_array__` capsules, with its schema's metadata.
///
/// With `include_metadata`, metadata that starfix-python refuses is refused first: a repeated
/// key (arrow-rs keeps one value of a repeated key on import) and explicit extension metadata
/// that conflicts with a registered extension type ([`check_extension_metadata`]). With
/// `full_names` (the batch's `pyarrow.Schema`, when a name contains NUL), the field names are
/// taken from it ([`with_full_names`]).
fn import_record_batch(
    batch: &Bound<'_, PyAny>,
    include_metadata: bool,
    full_names: Option<&Bound<'_, PyAny>>,
) -> PyResult<RecordBatch> {
    let (schema_capsule, array_capsule) = array_capsules(batch)?;
    let ffi_schema = exported_schema(&schema_capsule, include_metadata);
    if include_metadata && check_metadata(batch.py(), ffi_schema)? {
        check_extension_metadata(&batch.getattr("schema")?)?;
    }
    let exported =
        Schema::try_from(ffi_schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let schema = match full_names {
        Some(py_schema) => with_full_names(&exported, py_schema)?,
        None => exported.clone(),
    };
    importing(|| {
        let struct_type = DataType::Struct(exported.fields().clone());
        let raw_data = take_array(&array_capsule, ffi_schema, &struct_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if !matches!(raw_data.data_type(), DataType::Struct(_)) {
            return Err(PyTypeError::new_err(
                "expected a record batch (a struct array from __arrow_c_array__)",
            ));
        }
        let array_data = match full_names {
            Some(_) => retyped(
                imported(raw_data),
                &DataType::Struct(schema.fields().clone()),
            ),
            None => imported(raw_data),
        };
        let num_rows = array_data.len();
        let (_, columns, _) = StructArray::from(array_data).into_parts();
        RecordBatch::try_new_with_options(
            Arc::new(schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// An array (`pa.Array`) from its `__arrow_c_array__` capsules, as arrow-rs's `pyarrow` feature
/// imports it (an import error raises `pyarrow.ArrowException`), its empty nodes given offsets
/// and its metadata (never hashed for an array) cleared. With `full_type` (its
/// `pyarrow.DataType`, when a name contains NUL), the field names are taken from it.
fn import_array(
    array: &Bound<'_, PyAny>,
    full_type: Option<&Bound<'_, PyAny>>,
) -> PyResult<ArrayRef> {
    let (schema_capsule, array_capsule) = array_capsules(array)?;
    let ffi_schema = exported_schema(&schema_capsule, false);
    let arrow_error = |error: ArrowError| PyArrowException::new_err(error.to_string());
    let data_type = DataType::try_from(ffi_schema).map_err(arrow_error)?;
    let named = full_type
        .map(|py_type| with_full_type_names(&data_type, py_type))
        .transpose()?;
    importing(|| {
        let data =
            imported(take_array(&array_capsule, ffi_schema, &data_type).map_err(arrow_error)?);
        Ok(make_array(match &named {
            Some(named_type) => retyped(data, named_type),
            None => data,
        }))
    })
}

/// A type (`pa.DataType`) from its `__arrow_c_schema__` capsule, its metadata cleared.
fn import_type(py_type: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let (capsule, _) = capsules(py_type, "__arrow_c_schema__")?;
    check_capsule_name(&capsule, c"arrow_schema")?;
    DataType::try_from(exported_schema(&capsule, false))
        .map_err(|error| PyArrowException::new_err(error.to_string()))
}

/// The `ArrowSchema` of a schema capsule; with `keep_metadata` false, its metadata (at every
/// node) is cleared first: arrow-rs decodes metadata as UTF-8 on import and refuses other bytes,
/// which starfix-python accepts when metadata is not hashed. Only the export's pointers are
/// cleared: the producer's release callback frees what the producer allocated.
fn exported_schema<'py>(
    capsule: &Bound<'py, PyCapsule>,
    keep_metadata: bool,
) -> &'py FFI_ArrowSchema {
    if !keep_metadata {
        // SAFETY:
        // The capsule holds a valid `ArrowSchema` exported by its producer, alive as long as the
        // capsule; nothing refers to it yet.
        clear_metadata(unsafe { &mut *capsule.pointer().cast::<RawSchema>() });
    }
    // SAFETY: as above; the capsule outlives the reference.
    unsafe { capsule.reference::<FFI_ArrowSchema>() }
}

/// Clear the metadata of `node` and of every node below it (children and dictionaries).
fn clear_metadata(node: &mut RawSchema) {
    node.metadata = ptr::null();
    let count = usize::try_from(node.n_children).unwrap_or_default();
    if !node.children.is_null() {
        for i in 0..count {
            // SAFETY: `children` points to `n_children` valid child pointers (C Data Interface).
            let slot = unsafe { node.children.add(i) };
            // SAFETY: as above, `slot` is in bounds.
            let child = unsafe { *slot };
            if !child.is_null() {
                // SAFETY: each child pointer is a valid `ArrowSchema` owned by the parent.
                clear_metadata(unsafe { &mut *child });
            }
        }
    }
    if !node.dictionary.is_null() {
        // SAFETY: a non-null `dictionary` is the valid value-type schema (C Data Interface).
        clear_metadata(unsafe { &mut *node.dictionary });
    }
}

/// A schema from its `__arrow_c_schema__` capsule, as arrow-rs's `pyarrow` feature imports it,
/// metadata included only with `include_metadata`. Then metadata that starfix-python refuses
/// is refused first, as in [`import_record_batch`]. With `full_names` (the same schema, when a
/// name contains NUL), the names are taken from pyarrow.
fn import_schema(
    schema: &Bound<'_, PyAny>,
    include_metadata: bool,
    full_names: Option<&Bound<'_, PyAny>>,
) -> PyResult<Schema> {
    let (capsule, _) = capsules(schema, "__arrow_c_schema__")?;
    check_capsule_name(&capsule, c"arrow_schema")?;
    let ffi_schema = exported_schema(&capsule, include_metadata);
    if include_metadata && check_metadata(schema.py(), ffi_schema)? {
        check_extension_metadata(schema)?;
    }
    let exported = Schema::try_from(ffi_schema)
        .map_err(|error| PyArrowException::new_err(error.to_string()))?;
    match full_names {
        Some(py_schema) => with_full_names(&exported, py_schema),
        None => Ok(exported),
    }
}

/// `schema` with the field names, at every depth, of `py_schema`: the `pyarrow.Schema` it was
/// exported from. The Arrow C Data Interface carries a name as a C string, which ends at the
/// first NUL; starfix-python hashes the whole name (the Python layer passes `py_schema` only
/// when a name contains NUL).
fn with_full_names(schema: &Schema, py_schema: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let fields = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| with_full_field_names(field, &py_schema.call_method1("field", (i,))?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// `field` with the name of `py_field` (its `pyarrow.Field`) and the names below it.
fn with_full_field_names(field: &Field, py_field: &Bound<'_, PyAny>) -> PyResult<Field> {
    let name: String = py_field.getattr("name")?.extract()?;
    let data_type = with_full_type_names(field.data_type(), &py_field.getattr("type")?)?;
    Ok(field.clone().with_name(name).with_data_type(data_type))
}

/// `data_type` with the field names of `py_type` (its `pyarrow.DataType`) at every depth. An
/// extension type is its storage type here, as in arrow-rs.
fn with_full_type_names(data_type: &DataType, py_type: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let mut storage = py_type.clone();
    while storage.hasattr("storage_type")? {
        storage = storage.getattr("storage_type")?;
    }
    let child = |field: &FieldRef, attribute: &str| -> PyResult<FieldRef> {
        Ok(Arc::new(with_full_field_names(
            field,
            &storage.getattr(attribute)?,
        )?))
    };
    Ok(match data_type {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    with_full_field_names(field, &storage.call_method1("field", (i,))?)
                })
                .collect::<PyResult<Fields>>()?,
        ),
        DataType::List(field) => DataType::List(child(field, "value_field")?),
        DataType::LargeList(field) => DataType::LargeList(child(field, "value_field")?),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(child(field, "value_field")?, *size)
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(key_value) = entries.data_type() else {
                return Ok(data_type.clone());
            };
            let names = key_value
                .iter()
                .zip(["key_field", "item_field"])
                .map(|(field, attribute)| child(field, attribute))
                .collect::<PyResult<Fields>>()?;
            let named_entries = entries
                .as_ref()
                .clone()
                .with_data_type(DataType::Struct(names));
            DataType::Map(Arc::new(named_entries), *sorted)
        }
        DataType::Dictionary(key, value) => DataType::Dictionary(
            key.clone(),
            Box::new(with_full_type_names(
                value,
                &storage.getattr("value_type")?,
            )?),
        ),
        _ => data_type.clone(),
    })
}

/// `data` as an array of `data_type`, which differs from its type in field names only.
fn retyped(data: ArrayData, data_type: &DataType) -> ArrayData {
    let children: Vec<ArrayData> = match data_type {
        DataType::Struct(fields) => data
            .child_data()
            .iter()
            .zip(fields.iter())
            .map(|(child, field)| retyped(child.clone(), field.data_type()))
            .collect(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => data
            .child_data()
            .iter()
            .map(|child| retyped(child.clone(), field.data_type()))
            .collect(),
        DataType::Dictionary(_, value) => data
            .child_data()
            .iter()
            .map(|child| retyped(child.clone(), value))
            .collect(),
        _ => data.child_data().to_vec(),
    };
    rebuilt(
        data.into_builder()
            .data_type(data_type.clone())
            .child_data(children),
    )
}

/// Refuse explicit `ARROW:extension:*` field metadata that conflicts with the field's
/// registered extension type (byte-layout spec Section 7, as starfix-python). pyarrow's export
/// lets an explicit value replace the type's own, so the conflict shows only in the pyarrow
/// fields; `starfix.arrow_digester._check_extension_metadata` reads them. Called when the export
/// carries an extension name.
fn check_extension_metadata(schema: &Bound<'_, PyAny>) -> PyResult<()> {
    schema
        .py()
        .import("starfix.arrow_digester")?
        .getattr("_check_extension_metadata")?
        .call1((schema,))?;
    Ok(())
}

/// The C ABI of an exported array (`ArrowArray`, Arrow C Data Interface), to give empty arrays
/// the offsets buffer arrow-rs's import requires ([`fill_empty_offsets`]).
#[repr(C)]
struct RawArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut Self,
    dictionary: *mut Self,
    release: Option<unsafe extern "C" fn(*mut Self)>,
    private_data: *mut c_void,
}

/// The offsets buffer of an empty array: one offset `0`, of either width.
static ZERO_OFFSET: u64 = 0;

/// Whether arrays of the type have an offsets buffer (buffer 1).
const fn has_offsets(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Map(_, _)
    )
}

/// Give every empty node of `array` (any depth, of type `data_type`) that has an offsets type but
/// no offsets buffer the single offset `0`.
///
/// Arrow lets an empty array omit its offsets buffer, and pyarrow exports such arrays (a list
/// whose rows are all empty, over an empty child without buffers). arrow-rs's import requires
/// `offset + length + 1` offsets and refuses a null offsets buffer at any depth, though the
/// array's content does not depend on it. Only the pointer in the export's `buffers` list is
/// set: the producer's release callback frees what the producer allocated, not what the list
/// points to, and the imported buffer borrows the static zero.
fn fill_empty_offsets(array: &RawArray, data_type: &DataType) {
    if array.length == 0
        && array.offset == 0
        && array.n_buffers >= 2
        && !array.buffers.is_null()
        && has_offsets(data_type)
    {
        // SAFETY: `buffers` points to `n_buffers` (at least 2) buffer pointers (C Data Interface).
        let slot = unsafe { array.buffers.add(1) };
        // SAFETY: as above, `slot` is in bounds.
        if unsafe { slot.read() }.is_null() {
            // SAFETY: as above; the export belongs to this import until it is moved out.
            unsafe {
                slot.write(ptr::from_ref(&ZERO_OFFSET).cast::<c_void>());
            }
        }
    }
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => {
            if let Some(child) = array_children(array).next() {
                fill_empty_offsets(child, field.data_type());
            }
        }
        DataType::Struct(fields) => {
            for (child, field) in array_children(array).zip(fields.iter()) {
                fill_empty_offsets(child, field.data_type());
            }
        }
        DataType::Dictionary(_, value_type) if !array.dictionary.is_null() => {
            // SAFETY: a non-null `dictionary` is the valid array of the values (C Data Interface).
            fill_empty_offsets(unsafe { &*array.dictionary }, value_type);
        }
        _ => {}
    }
}

fn array_children(array: &RawArray) -> impl Iterator<Item = &RawArray> {
    let count = usize::try_from(array.n_children).unwrap_or_default();
    (0..count).filter_map(move |i| {
        if array.children.is_null() {
            return None;
        }
        // SAFETY: `children` points to `n_children` valid child pointers (C Data Interface).
        let slot = unsafe { array.children.add(i) };
        // SAFETY: as above, `slot` is in bounds.
        let child = unsafe { *slot };
        // SAFETY: each child pointer is a valid `ArrowArray` owned by the parent.
        (!child.is_null()).then(|| unsafe { &*child })
    })
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

/// Build an imported array re-expressed by [`without_struct_offsets`], [`rows`] or [`retyped`],
/// unchecked.
///
/// Full validation is not applied, as `from_ffi` does not apply it to the imported array: it
/// would refuse arrays the format hashes, such as a null in a non-nullable child under a null
/// parent (Section 7 decides those).
fn rebuilt(builder: ArrayDataBuilder) -> ArrayData {
    // SAFETY:
    // The builder holds the buffers, children and validity of an array imported from the C Data
    // Interface, with the same logical rows: only where a struct's offset is applied changed
    // (moved from the struct to its children, within their bounds, checked by `rows`), or the
    // field names in its type (`retyped`).
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

const EXTENSION_NAME: &[u8] = b"ARROW:extension:name";
const EXTENSION_METADATA: &[u8] = b"ARROW:extension:metadata";

/// `schema` as the C ABI it is: `FFI_ArrowSchema` is `#[repr(C)]` with exactly the
/// `ArrowSchema` layout of the C Data Interface.
const fn raw_schema(schema: &FFI_ArrowSchema) -> &RawSchema {
    let raw = ptr::from_ref(schema).cast::<RawSchema>();
    // SAFETY: same layout (above); `schema` is a valid exported schema, alive as long as `&`.
    unsafe { &*raw }
}

/// Refuse metadata that repeats a key, on the schema or on any node (byte-layout spec Section
/// 7, as starfix-python), with starfix-python's message. Returns whether a field node or a
/// dictionary below one carries `ARROW:extension:name` (an extension type, or its metadata).
fn check_metadata(py: Python<'_>, schema: &FFI_ArrowSchema) -> PyResult<bool> {
    let root = raw_schema(schema);
    let pairs = metadata_pairs(root);
    check_utf8(py, &pairs)?;
    if let Some(key) = repeated_key(&pairs) {
        return Err(PyValueError::new_err(format!(
            "Duplicate metadata key {} on the schema: ambiguous, rejected with include_metadata",
            py_repr(&key)
        )));
    }
    let mut extension = false;
    for child in children(root) {
        extension |= check_node(py, child, &node_name(child))?;
    }
    Ok(extension)
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
/// a dictionary, at any depth, is its value type; a map's entries struct is not a node).
/// Returns whether any of them carries `ARROW:extension:name`.
fn check_node(py: Python<'_>, node: &RawSchema, path: &str) -> PyResult<bool> {
    let pairs = metadata_pairs(node);
    check_utf8(py, &pairs)?;
    if let Some(key) = repeated_key(&pairs) {
        return Err(PyValueError::new_err(format!(
            "Field {}: Duplicate metadata key {} on the field: ambiguous, rejected with \
             include_metadata",
            py_repr(path),
            py_repr(&key)
        )));
    }
    let mut extension = has_key(&pairs, EXTENSION_NAME);
    let mut logical = node;
    while !logical.dictionary.is_null() {
        // SAFETY: a non-null `dictionary` is the valid value-type schema (C Data Interface).
        logical = unsafe { &*logical.dictionary };
        extension |= has_key(&metadata_pairs(logical), EXTENSION_NAME);
    }
    match format_of(logical) {
        "+s" => {
            for child in children(logical) {
                extension |= check_node(py, child, &format!("{path}/{}", node_name(child)))?;
            }
        }
        "+m" => {
            for entries in children(logical) {
                for (child, name) in children(entries).zip(["key", "value"]) {
                    extension |= check_node(py, child, &format!("{path}//{name}"))?;
                }
            }
        }
        format
            if format.starts_with("+l") || format.starts_with("+L") || format.starts_with("+w") =>
        {
            for child in children(logical) {
                extension |= check_node(py, child, &format!("{path}/"))?;
            }
        }
        _ => {}
    }
    Ok(extension)
}

/// A node's exported metadata, every pair in order (repeated keys kept): `int32` count, then per
/// pair `int32` length + key bytes and `int32` length + value bytes (C Data Interface).
fn metadata_pairs(node: &RawSchema) -> Vec<(Vec<u8>, Vec<u8>)> {
    if node.metadata.is_null() {
        return Vec::new();
    }
    let mut pos = node.metadata.cast::<u8>();
    let count = read_length(&mut pos);
    let mut pairs = Vec::new();
    for _ in 0..count {
        let key_len = read_length(&mut pos);
        // SAFETY: `key_len` bytes of key follow (C Data Interface).
        let key = unsafe { slice::from_raw_parts(pos, key_len) }.to_vec();
        // SAFETY: as above.
        pos = unsafe { pos.add(key_len) };
        let value_len = read_length(&mut pos);
        // SAFETY: `value_len` bytes of value follow (C Data Interface).
        let value = unsafe { slice::from_raw_parts(pos, value_len) }.to_vec();
        // SAFETY: as above.
        pos = unsafe { pos.add(value_len) };
        pairs.push((key, value));
    }
    pairs
}

/// Refuse metadata that is not UTF-8 with the `UnicodeDecodeError` starfix-python raises when it
/// decodes the metadata it hashes.
fn check_utf8(py: Python<'_>, pairs: &[(Vec<u8>, Vec<u8>)]) -> PyResult<()> {
    for text in pairs.iter().flat_map(|(key, value)| [key, value]) {
        if let Err(error) = str::from_utf8(text) {
            return Err(PyErr::from_value(
                PyUnicodeDecodeError::new_utf8(py, text, error)?.into_any(),
            ));
        }
    }
    Ok(())
}

/// The first key that the pairs repeat.
fn repeated_key(pairs: &[(Vec<u8>, Vec<u8>)]) -> Option<String> {
    let mut seen = HashSet::new();
    pairs
        .iter()
        .find(|(key, _)| !seen.insert(key.as_slice()))
        .map(|(key, _)| String::from_utf8_lossy(key).into_owned())
}

fn has_key(pairs: &[(Vec<u8>, Vec<u8>)], key: &[u8]) -> bool {
    pairs.iter().any(|(k, _)| k.as_slice() == key)
}

/// The value of `key` in the pairs (the first), or empty.
fn value_of<'a>(pairs: &'a [(Vec<u8>, Vec<u8>)], key: &[u8]) -> &'a [u8] {
    pairs
        .iter()
        .find(|(k, _)| k.as_slice() == key)
        .map_or(&[], |(_, value)| value.as_slice())
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

/// The `ARROW:extension:name` and `ARROW:extension:metadata` values pyarrow exports for
/// `field` (empty when absent): for a field of an extension type without metadata, the type's
/// own name and serialization.
#[pyfunction]
fn extension_metadata<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let (capsule, _) = capsules(field, "__arrow_c_schema__")?;
    check_capsule_name(&capsule, c"arrow_schema")?;
    // SAFETY: as in `import_record_batch`; the capsule outlives the reference.
    let pairs = metadata_pairs(raw_schema(unsafe {
        capsule.reference::<FFI_ArrowSchema>()
    }));
    Ok((
        bytes(py, value_of(&pairs, EXTENSION_NAME)),
        bytes(py, value_of(&pairs, EXTENSION_METADATA)),
    ))
}

/// `ArrowDigester.hash_schema`. Each entry point takes `full_names`: the `pyarrow.Schema` (or
/// type) to take field names from when one contains NUL, else `None` (see [`with_full_names`]).
#[pyfunction]
#[pyo3(signature = (py_schema, include_metadata, ordered_columns, ordered_struct_fields, full_names))]
fn hash_schema<'py>(
    py: Python<'py>,
    py_schema: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    full_names: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let schema = import_schema(py_schema, include_metadata, full_names)?;
    let config = config(include_metadata, ordered_columns, ordered_struct_fields);
    let digest = hashing(py, || ArrowDigester::try_hash_schema(&schema, config))?;
    Ok(bytes(py, &digest))
}

/// `ArrowDigester.hash_record_batch`.
#[pyfunction]
#[pyo3(signature = (py_batch, include_metadata, ordered_columns, ordered_struct_fields, full_names))]
fn hash_record_batch<'py>(
    py: Python<'py>,
    py_batch: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    full_names: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = import_record_batch(py_batch, include_metadata, full_names)?;
    let config = config(include_metadata, ordered_columns, ordered_struct_fields);
    let digest = hashing(py, || ArrowDigester::try_hash_record_batch(&batch, config))?;
    Ok(bytes(py, &digest))
}

/// `ArrowDigester.hash_table`: a digester over `batches`, the table's record batches (a table
/// without any hashes as a digester without updates, equal to one empty batch).
#[pyfunction]
#[pyo3(signature = (
    py_schema, batches, include_metadata, ordered_columns, ordered_struct_fields, full_names
))]
fn hash_table<'py>(
    py: Python<'py>,
    py_schema: &Bound<'py, PyAny>,
    batches: &Bound<'py, PyAny>,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    full_names: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut digester = Digester::new(
        py,
        py_schema,
        include_metadata,
        ordered_columns,
        ordered_struct_fields,
        full_names,
    )?;
    for py_batch in batches.try_iter()? {
        digester.update(py, &py_batch?, full_names)?;
    }
    Ok(digester.finalize(py))
}

/// `ArrowDigester.hash_array`: the chunks of a `ChunkedArray` (or one array), all of type
/// `py_type`, hashed as their concatenation without concatenating them.
#[pyfunction]
#[pyo3(signature = (py_type, py_chunks, ordered_columns, ordered_struct_fields, full_names))]
fn hash_array<'py>(
    py: Python<'py>,
    py_type: &Bound<'py, PyAny>,
    py_chunks: &Bound<'py, PyAny>,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    full_names: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let exported_type = import_type(py_type)?;
    let data_type = match full_names {
        Some(full_type) => with_full_type_names(&exported_type, full_type)?,
        None => exported_type,
    };
    let chunks = py_chunks
        .try_iter()?
        .map(|chunk| import_array(&chunk?, full_names))
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
    #[pyo3(signature = (
        py_schema, include_metadata, ordered_columns, ordered_struct_fields, full_names
    ))]
    fn new(
        py: Python<'_>,
        py_schema: &Bound<'_, PyAny>,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
        full_names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema = import_schema(py_schema, include_metadata, full_names)?;
        let config = config(include_metadata, ordered_columns, ordered_struct_fields);
        Ok(Self {
            digester: hashing(py, || ArrowDigester::try_new(&schema, config))?,
            include_metadata,
        })
    }

    /// Add a record batch; a refused batch leaves the digester as it was.
    #[pyo3(signature = (py_batch, full_names))]
    fn update(
        &mut self,
        py: Python<'_>,
        py_batch: &Bound<'_, PyAny>,
        full_names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let batch = import_record_batch(py_batch, self.include_metadata, full_names)?;
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
    #[pyo3(signature = (
        py_schema, include_metadata, ordered_columns, ordered_struct_fields, full_names
    ))]
    fn new(
        py: Python<'_>,
        py_schema: &Bound<'_, PyAny>,
        include_metadata: bool,
        ordered_columns: bool,
        ordered_struct_fields: bool,
        full_names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema = import_schema(py_schema, include_metadata, full_names)?;
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
    #[pyo3(signature = (py_batch, full_names))]
    fn hash_record_batch<'py>(
        &self,
        py: Python<'py>,
        py_batch: &Bound<'py, PyAny>,
        full_names: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let include_metadata = self.prepared.config().include_metadata;
        let batch = import_record_batch(py_batch, include_metadata, full_names)?;
        let prepared = &self.prepared;
        let digest = hashing(py, || prepared.try_hash_record_batch(&batch))?;
        Ok(bytes(py, &digest))
    }

    /// Hash a table of the prepared schema, given its schema and record batches. A table
    /// without batches hashes as one empty batch of its schema, so its schema is checked too.
    #[pyo3(signature = (py_schema, batches, full_names))]
    fn hash_table<'py>(
        &self,
        py: Python<'py>,
        py_schema: &Bound<'py, PyAny>,
        batches: &Bound<'py, PyAny>,
        full_names: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut digester = self.digester(py)?;
        let include_metadata = digester.include_metadata;
        let mut empty = true;
        for py_batch in batches.try_iter()? {
            empty = false;
            let batch = import_record_batch(&py_batch?, include_metadata, full_names)?;
            table_update(py, &mut digester.digester, &batch)?;
        }
        if empty {
            let schema = import_schema(py_schema, include_metadata, full_names)?;
            table_update(
                py,
                &mut digester.digester,
                &RecordBatch::new_empty(Arc::new(schema)),
            )?;
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

/// Add a batch of a table to a prepared schema's digester; a schema mismatch is the table's.
fn table_update(py: Python<'_>, digester: &mut ArrowDigester, batch: &RecordBatch) -> PyResult<()> {
    hashing(py, || {
        digester.try_update(batch).map_err(|error| match error {
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
    })
}

/// The extension module.
#[pymodule]
fn _internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(hash_schema, module)?)?;
    module.add_function(wrap_pyfunction!(hash_record_batch, module)?)?;
    module.add_function(wrap_pyfunction!(hash_table, module)?)?;
    module.add_function(wrap_pyfunction!(hash_array, module)?)?;
    module.add_function(wrap_pyfunction!(extension_metadata, module)?)?;
    module.add_class::<Digester>()?;
    module.add_class::<Prepared>()?;
    Ok(())
}
