from . import _internal as sfr  # import rust binding
from typing import TYPE_CHECKING
import ctypes

if TYPE_CHECKING:
    import pyarrow as pa


def hash_record_batch(table: "pa.Table") -> bytes:
    # Covert table to record batch first (so we can extract the pointers), since the default behavior is 1 batch, we can just get the first element
    # After that we can extract the PyCapsules
    schema_capsule, array_capsule = table.to_batches()[0].__arrow_c_array__()

    # Extract raw pointers from capsules due to uniffi limitations
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_GetPointer.restype = ctypes.c_void_p

    return sfr.hash_record_batch(
        PyCapsule_GetPointer(array_capsule, b"arrow_array"),
        PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
    )


def hash_schema(schema: "pa.Schema") -> bytes:
    schema_capsule = schema.__arrow_c_schema__()

    # Extract raw pointers from capsules due to uniffi limitations
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_GetPointer.restype = ctypes.c_void_p

    return sfr.hash_schema(
        PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
    )


class PyArrowDigester:
    def __init__(self, schema: "pa.Schema") -> None:

        schema_capsule = schema.__arrow_c_schema__()

        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p

        schema_ptr = PyCapsule_GetPointer(schema_capsule, b"arrow_schema")

        self._internal = sfr.InternalPyArrowDigester(schema_ptr)

    def update(self, table: "pa.Table") -> None:
        # Covert table to record batch first (so we can extract the pointers), since the default behavior is 1 batch, we can just get the first element
        # After that we can extract the PyCapsules
        schema_capsule, array_capsule = table.to_batches()[0].__arrow_c_array__()

        # Extract raw pointers from capsules due to uniffi limitations
        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p

        self._internal.update(
            PyCapsule_GetPointer(array_capsule, b"arrow_array"),
            PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
        )

    def finalize(self) -> bytes:
        return self._internal.finalize()


def _pointer(capsule: object, name: bytes) -> int:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    return get_pointer(capsule, name)


def _batches_of(table: "pa.Table") -> "list[pa.RecordBatch]":
    """The record batches of a table; one empty batch of its schema when it has none."""
    import pyarrow as pa

    batches = table.to_batches()
    if not batches:
        schema = table.schema
        batches = [
            pa.RecordBatch.from_arrays(
                [pa.array([], type=f.type) for f in schema], schema=schema
            )
        ]
    return batches


class PreparedSchema:
    """A schema analysed once under the hashing options.

    Hash many record batches or tables of that schema, or start streaming digesters, without
    the schema being analysed again. Immutable; starfix keeps no cache. Every hash equals the
    one-shot path's. A batch or table whose schema is not the prepared one under the prepared
    options (byte-layout spec Section 5) is refused with an exception.
    """

    def __init__(
        self,
        schema: "pa.Schema",
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> None:
        self._schema = schema
        # Keep the capsule referenced until the call returns: its pointer is only valid while
        # the capsule is alive.
        schema_capsule = schema.__arrow_c_schema__()
        self._internal = sfr.InternalPreparedSchema(
            _pointer(schema_capsule, b"arrow_schema"),
            include_metadata,
            ordered_columns,
            ordered_struct_fields,
        )
        self._schema_digest = bytes(self._internal.schema_digest())

    @property
    def schema(self) -> "pa.Schema":
        return self._schema

    @property
    def schema_digest(self) -> bytes:
        """The schema digest: equal to ``hash_schema`` with the same options."""
        return self._schema_digest

    def hash_record_batch(self, batch: "pa.RecordBatch") -> bytes:
        schema_capsule, array_capsule = batch.__arrow_c_array__()
        return self._internal.hash_record_batch(
            _pointer(array_capsule, b"arrow_array"),
            _pointer(schema_capsule, b"arrow_schema"),
        )

    def hash_table(self, table: "pa.Table") -> bytes:
        digester = self._internal.digester()
        for batch in _batches_of(table):
            schema_capsule, array_capsule = batch.__arrow_c_array__()
            digester.update_record_batch(
                _pointer(array_capsule, b"arrow_array"),
                _pointer(schema_capsule, b"arrow_schema"),
            )
        return digester.finalize()

    def digester(self) -> "PyArrowDigester":
        """A fresh streaming digester of the prepared schema."""
        digester = PyArrowDigester.__new__(PyArrowDigester)
        digester._internal = self._internal.digester()
        return digester


def prepare(
    schema: "pa.Schema",
    *,
    include_metadata: bool = False,
    ordered_columns: bool = False,
    ordered_struct_fields: bool = False,
) -> PreparedSchema:
    """Analyse ``schema`` once under the hashing options."""
    return PreparedSchema(
        schema,
        include_metadata=include_metadata,
        ordered_columns=ordered_columns,
        ordered_struct_fields=ordered_struct_fields,
    )


class ArrowDigester:
    """Namespace matching the Rust ``ArrowDigester`` and starfix-python's interface."""

    prepare = staticmethod(prepare)
