"""The starfix Arrow logical hasher, backed by the Rust crate (hash format 2).

The interface is starfix-python's (``ArrowDigester``, ``PreparedSchema``): same names,
signatures, defaults, results and exception types. Every hash is 35 bytes: the 3-byte format
number ``00 00 02`` followed by a SHA-256 digest (``docs/byte-layout-spec.md`` in
metamorphic-labs/starfix).
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
    """Every record batch of a table (none for an empty table: the native module hashes it as
    one empty batch of the table's schema, which a prepared schema checks)."""
    return [_batch(batch) for batch in table.to_batches()]


_EXTENSION_KEYS = (b"ARROW:extension:name", b"ARROW:extension:metadata")


def _logical(t: pa.DataType) -> pa.DataType:
    """Extension -> storage type, dictionary -> value type, repeatedly."""
    while True:
        if isinstance(t, pa.BaseExtensionType):
            t = t.storage_type
        elif pa.types.is_dictionary(t):
            t = t.value_type
        else:
            return t


def _child_nodes(t: pa.DataType, path: str) -> list[tuple[str, pa.Field]]:
    """The child nodes of a node of type `t`, as (node path, field) (spec Section 3.1)."""
    t = _logical(t)
    if pa.types.is_struct(t):
        return [(f"{path}/{t.field(i).name}", t.field(i)) for i in range(t.num_fields)]
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return [(path + "/", t.value_field)]
    if pa.types.is_map(t):
        return [(path + "//key", t.key_field), (path + "//value", t.item_field)]
    return []


# Types whose own strings or children may hold a NUL: wrappers, and timestamps (time zone).
_WRAPPERS = (pa.DictionaryType, pa.BaseExtensionType, pa.TimestampType)


def _type_has_nul_name(t: pa.DataType) -> bool:
    """Whether a field name below type `t`, or a time zone at or below it, contains NUL."""
    t = _logical(t)
    if pa.types.is_timestamp(t):
        return t.tz is not None and "\x00" in t.tz
    if pa.types.is_struct(t):
        fields = [t.field(i) for i in range(t.num_fields)]
    elif pa.types.is_map(t):
        fields = [t.key_field, t.item_field]
    else:
        fields = [t.value_field] if hasattr(t, "value_field") else []
    for field in fields:
        if "\x00" in field.name:
            return True
        child = field.type
        if (child.num_fields or isinstance(child, _WRAPPERS)) and _type_has_nul_name(child):
            return True
    return False


def _has_nul_name(schema: pa.Schema) -> bool:
    """Whether a field name of `schema` or a time zone, at any depth, contains NUL.

    The Arrow C Data Interface carries names and type formats (a timestamp's time zone) as C
    strings, which end at the first NUL; for such a schema the native module takes the names and
    time zones from pyarrow (they are hashed in full).
    """
    if any("\x00" in name for name in schema.names):
        return True
    return any(_type_has_nul_name(t) for t in schema.types if t.num_fields or isinstance(t, _WRAPPERS))


def _full_names(schema: pa.Schema, analysed: pa.Schema, analysed_has_nul: bool) -> pa.Schema | None:
    """`schema` when a name in it contains NUL (see `_has_nul_name`), else None; decided without
    a walk when `schema` equals the `analysed` one (names included)."""
    has_nul = analysed_has_nul if schema.equals(analysed) else _has_nul_name(schema)
    return schema if has_nul else None


def _check_extension_metadata(schema: pa.Schema) -> None:
    """Refuse explicit ``ARROW:extension:*`` field metadata that differs from what the field's
    registered extension type exports (spec Section 7 with ``include_metadata``, as
    starfix-python). pyarrow's export lets the explicit value replace the type's, so only the
    pyarrow fields show the conflict; the native module calls this when the exported schema
    carries an extension name."""
    stack = [(schema.field(i).name, schema.field(i)) for i in reversed(range(len(schema)))]
    while stack:
        path, field = stack.pop()
        t = field.type
        while pa.types.is_dictionary(t):  # a dictionary of an extension type keeps its keys
            t = t.value_type
        explicit = field.metadata
        if explicit and isinstance(t, pa.BaseExtensionType):
            own = zip(_EXTENSION_KEYS, _internal.extension_metadata(pa.field("x", t)))
            for key, value in own:
                if key in explicit and explicit[key] != value:
                    raise ValueError(
                        f"Field {path!r}: {key.decode()}={explicit[key].decode()!r} is set, conflicting with its "
                        f"extension type ({value.decode()!r}): ambiguous, rejected with include_metadata"
                    )
        stack.extend(reversed(_child_nodes(field.type, path)))


class ArrowDigester:
    """Hash Arrow schemas, record batches, tables and arrays (hash format 2).

    Hashes ignore the order of the top-level columns and of struct fields (at every
    depth) unless ``ordered_columns`` / ``ordered_struct_fields`` is set; nothing else is
    reordered. Arrow schema- and field-level metadata are hashed only with
    ``include_metadata=True``; a schema without metadata hashes the same either way.

    Float values are hashed by their bits as they are, unless ``normalize_negative_zero``
    (-0.0 hashed as +0.0) or ``normalize_nan`` (every NaN hashed as the quiet NaN with a clear
    sign and no payload) is set. Unlike the order options these two are not recorded in the
    hash: with one on, data hashes exactly as the same data with its float values normalized
    hashes with it off. Do not mix hashes taken with different float options where bit-exact
    identity matters.

    Raises ``ValueError`` (in every mode) for: schemas in which two nodes share a path
    (duplicate names, or a name containing ``/`` that aliases another node); ListView,
    LargeListView, Union and RunEndEncoded types; hashing data of a schema containing
    Null, Map, Interval or view types (their schemas hash; also inside a FixedSizeList),
    whatever the number of rows; a non-nullable field holding a null that no enclosing
    struct null explains; duplicate metadata keys when ``include_metadata`` is on; a streamed
    batch whose schema differs from the digester's; explicit ``ARROW:extension:*`` field
    metadata conflicting with the field's registered extension type when ``include_metadata`` is
    on; and invalid Arrow data. Raises ``TypeError`` for a wrong container type (for example a Table passed to
    ``hash_record_batch``).

    A digester is not thread-safe: feed one digester from one thread at a time. A batch
    rejected by ``update`` leaves the digester unchanged. Hashing releases the GIL.

    To hash many tables or batches of one schema, analyse the schema once with
    ``ArrowDigester.prepare`` and hash through the returned ``PreparedSchema``.
    """

    __slots__ = ("_digester", "_has_nul", "_schema")

    def __init__(
        self,
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
    ) -> None:
        """Start a streaming hash of record batches with this schema.

        Args:
            schema: The Arrow schema every record batch must have.
            include_metadata: Hash schema- and field-level metadata.
            ordered_columns: Make the order of the top-level columns part of the hash.
            ordered_struct_fields: Make the order of struct fields, at every depth,
                part of the hash.
            normalize_negative_zero: Hash every -0.0 float value (float16, float32,
                float64, at every depth) as +0.0. Not recorded in the hash.
            normalize_nan: Hash every NaN float value (any sign or payload) as the
                quiet NaN with a clear sign and no payload. Not recorded in the hash.

        Hashes taken with different order options never coincide.
        """
        _require_little_endian()
        _require(schema, pa.Schema, "ArrowDigester", "a pyarrow.Schema")
        has_nul = _has_nul_name(schema)
        self._digester = _internal.Digester(
            schema,
            _options(include_metadata, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan),
            schema if has_nul else None,
        )
        self._schema = schema
        self._has_nul = has_nul

    @classmethod
    def _from_internal(cls, digester: _internal.Digester, schema: pa.Schema, has_nul: bool) -> ArrowDigester:
        instance = cls.__new__(cls)
        instance._digester = digester
        instance._schema = schema
        instance._has_nul = has_nul
        return instance

    @staticmethod
    def prepare(
        schema: pa.Schema,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
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
            normalize_negative_zero: Hash every -0.0 float value (float16, float32,
                float64, at every depth) as +0.0. Not recorded in the hash.
            normalize_nan: Hash every NaN float value (any sign or payload) as the
                quiet NaN with a clear sign and no payload. Not recorded in the hash.
        """
        _require_little_endian()
        _require(schema, pa.Schema, "prepare", "a pyarrow.Schema")
        options = _options(include_metadata, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan)
        has_nul = _has_nul_name(schema)
        prepared = _internal.Prepared(schema, options, schema if has_nul else None)
        return PreparedSchema._create(schema, prepared, options, has_nul)

    def update(self, record_batch: pa.RecordBatch) -> None:
        """Add a record batch. Splitting rows into batches does not change the hash."""
        _require(record_batch, pa.RecordBatch, "update", _BATCH_TYPE)
        batch = _batch(record_batch)
        self._digester.update(batch, _full_names(batch.schema, self._schema, self._has_nul))

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
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
    ) -> bytes:
        """Hash an Arrow schema. The float options have no effect on a schema's hash; they are
        accepted as by every entry point."""
        _require_little_endian()
        _require(schema, pa.Schema, "hash_schema", "a pyarrow.Schema")
        return _internal.hash_schema(
            schema,
            _options(include_metadata, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan),
            schema if _has_nul_name(schema) else None,
        )

    @staticmethod
    def hash_record_batch(
        record_batch: pa.RecordBatch,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
    ) -> bytes:
        """Hash one record batch (equal to streaming it through a digester)."""
        _require(record_batch, pa.RecordBatch, "hash_record_batch", _BATCH_TYPE)
        _require_little_endian()
        batch = _batch(record_batch)
        return _internal.hash_record_batch(
            batch,
            _options(include_metadata, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan),
            batch.schema if _has_nul_name(batch.schema) else None,
        )

    @staticmethod
    def hash_table(
        table: pa.Table,
        *,
        include_metadata: bool = False,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
    ) -> bytes:
        """Hash a table: a digester over ``table.to_batches()``."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        _require_little_endian()
        return _internal.hash_table(
            table.schema,
            _batches_of(table),
            _options(include_metadata, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan),
            table.schema if _has_nul_name(table.schema) else None,
        )

    @staticmethod
    def hash_array(
        array: pa.Array,
        *,
        ordered_columns: bool = False,
        ordered_struct_fields: bool = False,
        normalize_negative_zero: bool = False,
        normalize_nan: bool = False,
    ) -> bytes:
        """Hash a single array or ChunkedArray (equal to its chunks combined).

        There is deliberately no ``include_metadata``: an array has no schema or field
        metadata. ``ordered_columns`` changes only the header of an array's hash.
        ``normalize_negative_zero`` and ``normalize_nan`` apply to the array's float values at
        every depth, as for a table.
        """
        _require_little_endian()
        _require(array, (pa.Array, pa.ChunkedArray), "hash_array", "a pyarrow.Array or pyarrow.ChunkedArray")
        chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
        t = array.type
        return _internal.hash_array(
            t,
            [_array(chunk) for chunk in chunks],
            _options(False, ordered_columns, ordered_struct_fields, normalize_negative_zero, normalize_nan),
            t if _type_has_nul_name(t) else None,
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

    __slots__ = ("_has_nul", "_options", "_prepared", "_schema")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("PreparedSchema is created by ArrowDigester.prepare(schema, ...)")

    @classmethod
    def _create(cls, schema: pa.Schema, prepared: _internal.Prepared, options: tuple, has_nul: bool) -> PreparedSchema:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_has_nul", has_nul)
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
            f"ordered_struct_fields={self.ordered_struct_fields}, "
            f"normalize_negative_zero={self.normalize_negative_zero}, normalize_nan={self.normalize_nan}, "
            f"schema_digest={self.schema_digest.hex()})"
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
    def normalize_negative_zero(self) -> bool:
        return self._options[3]

    @property
    def normalize_nan(self) -> bool:
        return self._options[4]

    @property
    def schema_digest(self) -> bytes:
        """The 35-byte schema hash, equal to ``ArrowDigester.hash_schema`` with the same options."""
        return self._prepared.schema_digest()

    def hash_record_batch(self, record_batch: pa.RecordBatch) -> bytes:
        """Hash one record batch of this schema (see the same-schema rule)."""
        _require(record_batch, pa.RecordBatch, "hash_record_batch", _BATCH_TYPE)
        batch = _batch(record_batch)
        return self._prepared.hash_record_batch(batch, _full_names(batch.schema, self._schema, self._has_nul))

    def hash_table(self, table: pa.Table) -> bytes:
        """Hash a table of this schema (see the same-schema rule)."""
        _require(table, pa.Table, "hash_table", _TABLE_TYPE)
        full_names = _full_names(table.schema, self._schema, self._has_nul)
        return self._prepared.hash_table(table.schema, _batches_of(table), full_names)

    def digester(self) -> ArrowDigester:
        """A fresh streaming ``ArrowDigester`` for this schema and options, without analysis."""
        return ArrowDigester._from_internal(self._prepared.digester(), self._schema, self._has_nul)


def _options(
    include_metadata: object,
    ordered_columns: object,
    ordered_struct_fields: object,
    normalize_negative_zero: object,
    normalize_nan: object,
) -> tuple[bool, bool, bool, bool, bool]:
    """The options of a call as the extension module takes them."""
    return (
        bool(include_metadata),
        bool(ordered_columns),
        bool(ordered_struct_fields),
        bool(normalize_negative_zero),
        bool(normalize_nan),
    )


def _prepare(
    schema: pa.Schema,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    normalize_negative_zero: bool = False,
    normalize_nan: bool = False,
) -> PreparedSchema:
    return ArrowDigester.prepare(
        schema,
        include_metadata=include_metadata,
        ordered_columns=ordered_columns,
        ordered_struct_fields=ordered_struct_fields,
        normalize_negative_zero=normalize_negative_zero,
        normalize_nan=normalize_nan,
    )
