"""starfix: logical hashing of Arrow data, through the Rust crate (hash format 2).

``ArrowDigester`` has starfix-python's interface: ``ArrowDigester(schema, *, ...)`` with
``update`` and ``finalize``, and ``ArrowDigester.prepare``, ``hash_schema``,
``hash_record_batch``, ``hash_table`` and ``hash_array``, with the same keyword-only options
and defaults (``include_metadata``, ``ordered_columns``, ``ordered_struct_fields``, all
``False``) and the same hashes. Rejected input raises ``ValueError``; a wrong container type
raises ``TypeError``.

The binding's own calls remain: ``hash_schema``, ``hash_record_batch`` (a table or a record
batch), ``PyArrowDigester`` (whose ``update`` also takes a table) and ``prepare``, with the same
options. Every chunk of a table is hashed, so a table hashes like the same rows in one batch.

Differences from starfix-python:

- After a rejected ``update`` the digester cannot be used (every later call raises
  ``ValueError``); starfix-python restores the digester to its state before the update. The
  crate leaves a digester's state unspecified after a rejection, and restoring it here would
  mean copying the digester's buffered state on every update.
- ``hash_array`` hashes a ``ChunkedArray`` by combining its chunks into one array first (one
  copy), as the crate hashes a single array; the hash is the same.
- ``PreparedSchema`` can also be constructed directly, has no option properties, and is not
  picklable.
"""

from __future__ import annotations

import ctypes

import pyarrow as pa

from . import _internal as sfr

__all__ = [
    "ArrowDigester",
    "PreparedSchema",
    "PyArrowDigester",
    "hash_record_batch",
    "hash_schema",
    "prepare",
]

# A private prototype, so the shared ``ctypes.pythonapi`` function is left untouched.
_PyCapsule_GetPointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


def _pointer(capsule: object, name: bytes) -> int:
    return _PyCapsule_GetPointer(capsule, name)


def _rust(call, *args):
    """Call into the crate; a rejection (a Rust panic) raises ``ValueError``."""
    try:
        return call(*args)
    except sfr.InternalError as e:
        raise ValueError(str(e)) from e


def _require(obj: object, cls, entry_point: str, expected: str) -> None:
    if not isinstance(obj, cls):
        raise TypeError(f"{entry_point} expects {expected}, got {type(obj).__name__}")


def _options(include_metadata: bool, ordered_columns: bool, ordered_struct_fields: bool) -> tuple:
    return bool(include_metadata), bool(ordered_columns), bool(ordered_struct_fields)


def _with_schema(call, schema: pa.Schema, *args):
    # The capsule stays referenced until the call returns: its pointer is valid only while the
    # capsule is alive. The crate moves the structure out of it.
    schema_capsule = schema.__arrow_c_schema__()
    return _rust(call, _pointer(schema_capsule, b"arrow_schema"), *args)


def _with_array(call, data, *args):
    """``call(array_ptr, schema_ptr, *args)`` for a record batch or an array."""
    schema_capsule, array_capsule = data.__arrow_c_array__()
    return _rust(
        call,
        _pointer(array_capsule, b"arrow_array"),
        _pointer(schema_capsule, b"arrow_schema"),
        *args,
    )


def _batches_of(table: pa.Table) -> list[pa.RecordBatch]:
    """Every record batch of a table; one empty batch of its schema when it has none (so an
    empty table hashes as a zero-row batch, and its schema is still checked)."""
    batches = table.to_batches()
    if not batches:
        schema = table.schema
        batches = [pa.RecordBatch.from_arrays([pa.nulls(0, f.type) for f in schema], schema=schema)]
    return batches


def _combined(chunked: pa.ChunkedArray) -> pa.Array:
    """A ChunkedArray as one array (its chunks hash as their concatenation)."""
    if chunked.num_chunks == 1:
        return chunked.chunk(0)
    try:
        return chunked.combine_chunks()
    except pa.ArrowNotImplementedError as e:
        # pyarrow cannot concatenate some types the format does not hash the data of.
        raise ValueError(f"Cannot hash a ChunkedArray of type {chunked.type}: {e}") from e


_BATCH_TYPE = "a pyarrow.RecordBatch (use hash_table for a Table)"
_TABLE_TYPE = "a pyarrow.Table (use hash_record_batch for a RecordBatch)"


class ArrowDigester:
    """Hash Arrow schemas, record batches, tables and arrays (hash format 2), with the Rust crate.

    Hashes ignore the order of the top-level columns and of struct fields (at every depth)
    unless ``ordered_columns`` / ``ordered_struct_fields`` is set. Arrow schema- and
    field-level metadata are hashed only with ``include_metadata=True``.

    An instance is a streaming digester: ``update`` with record batches of its schema, then
    ``finalize``. Splitting rows into batches does not change the hash. Not thread-safe: feed one
    digester from one thread at a time. After a rejected ``update`` the digester cannot be used.
    """

    __slots__ = ("_internal",)

    def __init__(
        self,
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> None:
        _require(schema, pa.Schema, "ArrowDigester", "a pyarrow.Schema")
        self._internal = _with_schema(
            sfr.InternalPyArrowDigester,
            schema,
            *_options(include_metadata, ordered_columns, ordered_struct_fields),
        )

    @classmethod
    def _from_internal(cls, internal) -> ArrowDigester:
        digester = cls.__new__(cls)
        digester._internal = internal
        return digester

    def _update(self, batch: pa.RecordBatch) -> None:
        _with_array(self._internal.update, batch)

    def update(self, record_batch: pa.RecordBatch) -> None:
        """Add a record batch of the digester's schema."""
        _require(record_batch, pa.RecordBatch, "update", "a pyarrow.RecordBatch (use hash_table for a Table)")
        self._update(record_batch)

    def finalize(self) -> bytes:
        """Return the 35-byte hash of everything added so far (the digester stays usable)."""
        return _rust(self._internal.finalize)

    @staticmethod
    def prepare(
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> PreparedSchema:
        """Analyse a schema once, for hashing many tables or record batches of it."""
        return PreparedSchema(
            schema,
            include_metadata=include_metadata,
            ordered_columns=ordered_columns,
            ordered_struct_fields=ordered_struct_fields,
        )

    @staticmethod
    def hash_schema(
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash an Arrow schema."""
        _require(schema, pa.Schema, "hash_schema", "a pyarrow.Schema")
        return _with_schema(
            sfr.hash_schema, schema, *_options(include_metadata, ordered_columns, ordered_struct_fields)
        )

    @staticmethod
    def hash_record_batch(
        record_batch: pa.RecordBatch,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash one record batch (equal to streaming it through a digester)."""
        _require(record_batch, pa.RecordBatch, "hash_record_batch", _BATCH_TYPE)
        return _with_array(
            sfr.hash_record_batch,
            record_batch,
            *_options(include_metadata, ordered_columns, ordered_struct_fields),
        )

    @staticmethod
    def hash_table(
        table: pa.Table,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash a table: every one of its record batches, through one digester."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        digester = ArrowDigester(
            table.schema,
            include_metadata=include_metadata,
            ordered_columns=ordered_columns,
            ordered_struct_fields=ordered_struct_fields,
        )
        for batch in _batches_of(table):
            digester._update(batch)
        return digester.finalize()

    @staticmethod
    def hash_array(
        array: pa.Array,
        *,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash a single array or ChunkedArray (equal to its chunks combined).

        There is no ``include_metadata``: an array has no schema or field metadata.
        """
        _require(array, (pa.Array, pa.ChunkedArray), "hash_array", "a pyarrow.Array or pyarrow.ChunkedArray")
        if isinstance(array, pa.ChunkedArray):
            array = _combined(array)
        return _with_array(sfr.hash_array, array, bool(ordered_columns), bool(ordered_struct_fields))


class PyArrowDigester(ArrowDigester):
    """The binding's streaming digester: an ``ArrowDigester`` whose ``update`` also takes a
    table (every one of its record batches)."""

    __slots__ = ()

    def update(self, table: pa.Table | pa.RecordBatch) -> None:
        """Add a table (every record batch of it) or a record batch of the digester's schema."""
        if isinstance(table, pa.Table):
            for batch in _batches_of(table):
                self._update(batch)
        else:
            _require(table, pa.RecordBatch, "update", "a pyarrow.Table or pyarrow.RecordBatch")
            self._update(table)


def hash_schema(
    schema: pa.Schema,
    *,
    include_metadata: bool = False,
    ordered_columns: bool = False,
    ordered_struct_fields: bool = False,
) -> bytes:
    """Hash an Arrow schema (``ArrowDigester.hash_schema``)."""
    return ArrowDigester.hash_schema(
        schema,
        include_metadata=include_metadata,
        ordered_columns=ordered_columns,
        ordered_struct_fields=ordered_struct_fields,
    )


def hash_record_batch(
    table: pa.Table | pa.RecordBatch,
    *,
    include_metadata: bool = False,
    ordered_columns: bool = False,
    ordered_struct_fields: bool = False,
) -> bytes:
    """Hash a table (every one of its record batches) or a record batch."""
    options = {
        "include_metadata": include_metadata,
        "ordered_columns": ordered_columns,
        "ordered_struct_fields": ordered_struct_fields,
    }
    if isinstance(table, pa.Table):
        return ArrowDigester.hash_table(table, **options)
    _require(table, pa.RecordBatch, "hash_record_batch", "a pyarrow.Table or pyarrow.RecordBatch")
    return ArrowDigester.hash_record_batch(table, **options)


class PreparedSchema:
    """A schema analysed once under the hashing options.

    Hash many record batches or tables of that schema, or start streaming digesters, without
    the schema being analysed again. Immutable; starfix keeps no cache. Every hash equals the
    one-shot path's. A batch or table whose schema is not the prepared one under the prepared
    options (byte-layout spec Section 5) is refused with ``ValueError``.
    """

    def __init__(
        self,
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> None:
        _require(schema, pa.Schema, "prepare", "a pyarrow.Schema")
        self._schema = schema
        self._internal = _with_schema(
            sfr.InternalPreparedSchema,
            schema,
            *_options(include_metadata, ordered_columns, ordered_struct_fields),
        )
        self._schema_digest = bytes(self._internal.schema_digest())

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def schema_digest(self) -> bytes:
        """The schema digest: equal to ``hash_schema`` with the same options."""
        return self._schema_digest

    def hash_record_batch(self, record_batch: pa.RecordBatch) -> bytes:
        """Hash one record batch of this schema."""
        _require(record_batch, pa.RecordBatch, "hash_record_batch", _BATCH_TYPE)
        return _with_array(self._internal.hash_record_batch, record_batch)

    def hash_table(self, table: pa.Table) -> bytes:
        """Hash a table of this schema (every one of its record batches)."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        digester = self.digester()
        for batch in _batches_of(table):
            digester._update(batch)
        return digester.finalize()

    def digester(self) -> PyArrowDigester:
        """A fresh streaming digester of the prepared schema (an ``ArrowDigester``)."""
        return PyArrowDigester._from_internal(_rust(self._internal.digester))


def prepare(
    schema: pa.Schema,
    *,
    include_metadata: bool = False,
    ordered_columns: bool = False,
    ordered_struct_fields: bool = False,
) -> PreparedSchema:
    """Analyse ``schema`` once under the hashing options (``ArrowDigester.prepare``)."""
    return PreparedSchema(
        schema,
        include_metadata=include_metadata,
        ordered_columns=ordered_columns,
        ordered_struct_fields=ordered_struct_fields,
    )
