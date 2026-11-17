"""The starfix Arrow logical hasher, backed by the Rust crate (hash format 2).

The interface is starfix-python's (``ArrowDigester``, ``PreparedSchema``): same names,
signatures, defaults, results and exception types. Every hash is 35 bytes: the 3-byte format
number ``00 00 02`` followed by a SHA-256 digest (``docs/byte-layout-spec.md`` in
nauticalab/starfix).
"""

from __future__ import annotations

import sys

import pyarrow as pa

from starfix import _internal

__all__ = ["ArrowDigester", "PreparedSchema"]

_BATCH_TYPE = "a pyarrow.RecordBatch (use hash_table for a Table)"
_TABLE_TYPE = "a pyarrow.Table (use hash_record_batch for a RecordBatch)"


def _require(obj: object, cls, entry_point: str, expected: str) -> None:
    if not isinstance(obj, cls):
        raise TypeError(f"{entry_point} expects {expected}, got {type(obj).__name__}")


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise NotImplementedError(
            "starfix hash format 2 is defined over little-endian buffers; big-endian hosts are not supported"
        )


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch | None:
    """A zero-row batch of `schema` with every buffer present (None if pyarrow cannot build one).

    A zero-length array may omit its buffers (an empty list without an offsets buffer), which
    the Arrow C Data Interface import of arrow-rs refuses; its hash depends only on its type.
    """
    try:
        return pa.RecordBatch.from_arrays([pa.nulls(0, f.type) for f in schema], schema=schema)
    except (KeyError, NotImplementedError, pa.ArrowException):
        # pyarrow cannot build an empty array of some types (such as year-month intervals)
        # whose data the format does not hash: the digester refuses the schema itself.
        return None


def _batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """The batch, or for zero rows an equal batch with every buffer present."""
    if batch.num_rows:
        return batch
    empty = _empty_batch(batch.schema)  # a zero-row batch is falsy: no `or`
    return batch if empty is None else empty


def _array(array: pa.Array) -> pa.Array:
    """The array, or for length zero an equal array with every buffer present."""
    if len(array):
        return array
    try:
        return pa.nulls(0, array.type)
    except (KeyError, NotImplementedError, pa.ArrowException):
        return array


def _batches_of(table: pa.Table) -> list[pa.RecordBatch]:
    """Every record batch of a table; one empty batch of its schema when it has none (so an
    empty table hashes as a zero-row batch, and its schema is still checked)."""
    batches = [_batch(batch) for batch in table.to_batches()]
    if not batches:
        empty = _empty_batch(table.schema)
        batches = [] if empty is None else [empty]
    return batches


class ArrowDigester:
    """Hash Arrow schemas, record batches, tables and arrays (hash format 2).

    Hashes ignore the order of the top-level columns and of struct fields (at every
    depth) unless ``ordered_columns`` / ``ordered_struct_fields`` is set; nothing else is
    reordered. Arrow schema- and field-level metadata are hashed only with
    ``include_metadata=True``; a schema without metadata hashes the same either way.

    Raises ``ValueError`` (in every mode) for: schemas in which two nodes share a path
    (duplicate names, or a name containing ``/`` that aliases another node); ListView,
    LargeListView, Union and RunEndEncoded types; hashing data of a schema containing
    Null, Map, Interval or view types (their schemas hash; also inside a FixedSizeList),
    whatever the number of rows; a non-nullable field holding a null that no enclosing
    struct null explains; duplicate metadata keys when ``include_metadata`` is on; a streamed
    batch whose schema differs from the digester's; and invalid Arrow data. Raises
    ``TypeError`` for a wrong container type (for example a Table passed to
    ``hash_record_batch``).

    A digester is not thread-safe: feed one digester from one thread at a time. A batch
    rejected by ``update`` leaves the digester unchanged. Hashing releases the GIL.

    To hash many tables or batches of one schema, analyse the schema once with
    ``ArrowDigester.prepare`` and hash through the returned ``PreparedSchema``.
    """

    __slots__ = ("_digester",)

    def __init__(
        self,
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> None:
        """Start a streaming hash of record batches with this schema.

        Args:
            schema: The Arrow schema every record batch must have.
            include_metadata: Hash schema- and field-level metadata.
            ordered_columns: Make the order of the top-level columns part of the hash.
            ordered_struct_fields: Make the order of struct fields, at every depth,
                part of the hash.

        Hashes taken with different order options never coincide.
        """
        _require_little_endian()
        _require(schema, pa.Schema, "ArrowDigester", "a pyarrow.Schema")
        self._digester = _internal.Digester(
            schema, bool(include_metadata), bool(ordered_columns), bool(ordered_struct_fields)
        )

    @classmethod
    def _from_internal(cls, digester: _internal.Digester) -> ArrowDigester:
        instance = cls.__new__(cls)
        instance._digester = digester
        return instance

    @staticmethod
    def prepare(
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> PreparedSchema:
        """Analyse a schema once, for hashing many tables or record batches of it.

        Returns an immutable ``PreparedSchema`` owned by the caller; its hashes equal the
        one-shot calls with the same options. Raises what ``hash_schema`` raises. A schema
        whose data cannot be hashed (Null, Map, Interval, view types) still prepares, for its
        ``schema_digest``; hashing data with it raises ``ValueError``.

        Args:
            schema: The Arrow schema of the data to hash.
            include_metadata: Hash schema- and field-level metadata.
            ordered_columns: Make the order of the top-level columns part of the hash.
            ordered_struct_fields: Make the order of struct fields, at every depth,
                part of the hash.
        """
        _require_little_endian()
        _require(schema, pa.Schema, "prepare", "a pyarrow.Schema")
        options = (bool(include_metadata), bool(ordered_columns), bool(ordered_struct_fields))
        return PreparedSchema._create(schema, _internal.Prepared(schema, *options), options)

    def update(self, record_batch: pa.RecordBatch) -> None:
        """Add a record batch. Splitting rows into batches does not change the hash."""
        _require(record_batch, pa.RecordBatch, "update", _BATCH_TYPE)
        self._digester.update(_batch(record_batch))

    def finalize(self) -> bytes:
        """Return the 35-byte hash of everything added so far."""
        return self._digester.finalize()

    @staticmethod
    def hash_schema(
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash an Arrow schema."""
        _require_little_endian()
        _require(schema, pa.Schema, "hash_schema", "a pyarrow.Schema")
        return _internal.hash_schema(
            schema, bool(include_metadata), bool(ordered_columns), bool(ordered_struct_fields)
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
        _require_little_endian()
        return _internal.hash_record_batch(
            _batch(record_batch), bool(include_metadata), bool(ordered_columns), bool(ordered_struct_fields)
        )

    @staticmethod
    def hash_table(
        table: pa.Table,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash a table: a digester over ``table.to_batches()``."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        _require_little_endian()
        return _internal.hash_table(
            table.schema,
            _batches_of(table),
            bool(include_metadata),
            bool(ordered_columns),
            bool(ordered_struct_fields),
        )

    @staticmethod
    def hash_array(
        array: pa.Array,
        *,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
    ) -> bytes:
        """Hash a single array or ChunkedArray (equal to its chunks combined).

        There is deliberately no ``include_metadata``: an array has no schema or field
        metadata. ``ordered_columns`` changes only the header of an array's hash.
        """
        _require_little_endian()
        _require(array, (pa.Array, pa.ChunkedArray), "hash_array", "a pyarrow.Array or pyarrow.ChunkedArray")
        chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
        return _internal.hash_array(
            array.type, [_array(chunk) for chunk in chunks], bool(ordered_columns), bool(ordered_struct_fields)
        )


class PreparedSchema:
    """A schema analysed once, with its hashing options (create with ``ArrowDigester.prepare``).

    Hashes many tables or record batches of the schema without analysing it again; every
    hash is byte-identical to the one-shot ``ArrowDigester`` call with the same options.
    Immutable and owned by the caller: starfix keeps no cache or other state of its own. Safe
    to share between threads (each hash or ``digester()`` has its own state).

    **Same-schema rule.** A table or record batch is accepted exactly when its schema has the
    prepared schema's canonical schema key under the prepared options, i.e. when
    ``ArrowDigester.hash_schema(its_schema, **options) == prepared.schema_digest``: the same
    column names, logical types and nullability at every depth (column order when
    ``ordered_columns``, struct field order when ``ordered_struct_fields``) and, with
    ``include_metadata``, the same field metadata at every node, schema metadata and extension
    type names and serializations. This is the rule ``ArrowDigester.update`` applies to
    streamed batches. Anything else raises ``ValueError``; nothing is hashed with another
    schema's analysis. An input whose schema equals the prepared one is accepted by a
    structural comparison; other inputs, such as a ``string`` column for a prepared
    ``large_string`` one, are checked by computing their schema key.
    """

    __slots__ = ("_options", "_prepared", "_schema")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("PreparedSchema is created by ArrowDigester.prepare(schema, ...)")

    @classmethod
    def _create(cls, schema: pa.Schema, prepared: _internal.Prepared, options: tuple) -> PreparedSchema:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_schema", schema)
        object.__setattr__(instance, "_prepared", prepared)
        object.__setattr__(instance, "_options", options)
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"PreparedSchema is immutable (cannot set {name!r})")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"PreparedSchema is immutable (cannot delete {name!r})")

    def __copy__(self) -> PreparedSchema:
        return self

    def __deepcopy__(self, memo: dict) -> PreparedSchema:
        return self

    def __reduce__(self) -> tuple:
        # Pickled as (schema, options) and prepared again when loaded.
        return (_prepare, (self._schema, *self._options))

    def __repr__(self) -> str:
        return (
            f"PreparedSchema(include_metadata={self.include_metadata}, ordered_columns={self.ordered_columns}, "
            f"ordered_struct_fields={self.ordered_struct_fields}, schema_digest={self.schema_digest.hex()})"
        )

    @property
    def schema(self) -> pa.Schema:
        """The prepared Arrow schema."""
        return self._schema

    @property
    def include_metadata(self) -> bool:
        return self._options[0]

    @property
    def ordered_columns(self) -> bool:
        return self._options[1]

    @property
    def ordered_struct_fields(self) -> bool:
        return self._options[2]

    @property
    def schema_digest(self) -> bytes:
        """The 35-byte schema hash, equal to ``ArrowDigester.hash_schema`` with the same options."""
        return self._prepared.schema_digest()

    def hash_record_batch(self, record_batch: pa.RecordBatch) -> bytes:
        """Hash one record batch of this schema (see the same-schema rule)."""
        _require(record_batch, pa.RecordBatch, "hash_record_batch", _BATCH_TYPE)
        return self._prepared.hash_record_batch(_batch(record_batch))

    def hash_table(self, table: pa.Table) -> bytes:
        """Hash a table of this schema (see the same-schema rule)."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        return self._prepared.hash_table(_batches_of(table))

    def digester(self) -> ArrowDigester:
        """A fresh streaming ``ArrowDigester`` for this schema and options, without analysis."""
        return ArrowDigester._from_internal(self._prepared.digester())


def _prepare(schema: pa.Schema, include_metadata: bool, ordered_columns: bool, ordered_struct_fields: bool) -> PreparedSchema:
    return ArrowDigester.prepare(
        schema,
        include_metadata=include_metadata,
        ordered_columns=ordered_columns,
        ordered_struct_fields=ordered_struct_fields,
    )
