"""Inputs built in process, shared by the binding's tests and the starfix-python worker.

These inputs cannot be written to an IPC stream for ``test_differential.py``: an IPC round trip
allocates the buffers an empty array may omit, and a registered extension type does not
survive it in a process that has not registered it. Both sides of the differential build them
with these functions instead. Imports pyarrow only (the worker's ``starfix`` is
starfix-python's).
"""

import pyarrow as pa

# --- empty descendants without buffers (Arrow allows an empty array to omit them) ---------


def bufferless_empty(t: pa.DataType) -> pa.Array:
    """An empty array of `t` without validity or offsets buffers (children likewise)."""
    if pa.types.is_struct(t):
        return pa.StructArray.from_arrays([bufferless_empty(f.type) for f in t], fields=list(t))
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return pa.Array.from_buffers(t, 0, [None, None], children=[bufferless_empty(t.value_type)])
    # pyarrow requires a data buffer (it may be empty); an offsets buffer it does not.
    return pa.Array.from_buffers(t, 0, [None] * (t.num_buffers - 1) + [pa.py_buffer(b"")])


def one_row_over(child: pa.Array, t: pa.DataType) -> pa.Array:
    """One row of list type `t` (offsets [0, 0]) over the empty `child`."""
    offsets = pa.array([0, 0], pa.int64() if pa.types.is_large_list(t) else pa.int32())
    return pa.Array.from_buffers(t, 1, [None, offsets.buffers()[1]], children=[child])


INT_LISTS = pa.list_(pa.int64())
NESTED_LISTS = [
    pa.list_(INT_LISTS),
    pa.list_(pa.string()),
    pa.large_list(pa.large_binary()),
    pa.list_(pa.struct([("s", pa.string()), ("l", INT_LISTS)])),
    pa.list_(pa.list_(pa.list_(pa.utf8()))),
]


def bufferless_nested(t: pa.DataType) -> pa.Array:
    """One empty list of type `t` whose empty descendants have no offsets buffers."""
    return one_row_over(bufferless_empty(t.value_type), t)


# --- a registered extension type ----------------------------------------------------------

POINT_NAME = "starfix.test.parity-point"
POINT_SERIALIZED = "v1"


class Point(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.struct([("x", pa.int64())]), POINT_NAME)

    def __arrow_ext_serialize__(self):
        return POINT_SERIALIZED.encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


try:
    pa.register_extension_type(Point())
except pa.ArrowKeyError:
    pass  # registered by an earlier import


def point_batch(schema: pa.Schema) -> pa.RecordBatch:
    """Two rows of the schema's one column: a Point, a struct of a Point or a list of Points."""
    storage = pa.array([{"x": 1}, {"x": 2}], Point().storage_type)
    column = pa.ExtensionArray.from_storage(Point(), storage)
    t = schema.field(0).type
    if pa.types.is_struct(t):
        column = pa.StructArray.from_arrays([column], fields=list(t))
    elif pa.types.is_list(t):
        column = pa.ListArray.from_arrays(pa.array([0, 1, 2], pa.int32()), column, type=t)
    return pa.RecordBatch.from_arrays([column], schema=schema)


def extension_conflicts():
    """(id, schema, node path, key, explicit value, the type's own value): a Point field whose
    explicit ARROW:extension:* metadata differs from what the type itself exports."""
    for key, explicit, own in [
        ("ARROW:extension:name", "other", POINT_NAME),
        ("ARROW:extension:metadata", "v2", POINT_SERIALIZED),
    ]:
        field = pa.field("p", Point(), metadata={key: explicit})
        for where, schema, path in [
            ("top", pa.schema([field]), "p"),
            ("struct", pa.schema([pa.field("s", pa.struct([field]))]), "s/p"),
            ("list", pa.schema([pa.field("l", pa.list_(field))]), "l/"),
        ]:
            yield f"{key.rsplit(':', 1)[1]}-{where}", schema, path, key, explicit, own


def without_metadata(field: pa.Field) -> pa.Field:
    """The field without metadata at any depth of structs and lists."""
    t = field.type
    if pa.types.is_struct(t):
        t = pa.struct([without_metadata(f) for f in t])
    elif pa.types.is_list(t):
        t = pa.list_(without_metadata(t.value_field))
    return field.remove_metadata().with_type(t)


# --- repeated metadata keys below nested dictionaries -------------------------------------

REPEATED = pa.KeyValueMetadata([(b"k", b"1"), (b"k", b"2")])


def nested_dictionary_schema(depth: int) -> pa.Schema:
    """`depth` dictionaries over a struct whose field `x` repeats a metadata key."""
    t = pa.struct([pa.field("x", pa.int64(), metadata=REPEATED)])
    for _ in range(depth):
        t = pa.dictionary(pa.int8(), t)
    return pa.schema([pa.field("d", t)])


# --- the differential's in-process cases --------------------------------------------------


def differential_cases() -> dict[str, tuple[pa.Schema, list[pa.RecordBatch]]]:
    """{case id: (schema, record batches)} for the differential against starfix-python 0.5.0.

    Not here: an empty dictionary without buffers under a dictionary column, which 0.5.0
    fails on with ``TypeError`` (``test_parity_edge_cases.py`` pins the binding's digest).
    """
    cases = {}
    for t in NESTED_LISTS:
        bufferless = bufferless_nested(t)
        batch = pa.RecordBatch.from_arrays([bufferless], names=["c"])
        other = pa.RecordBatch.from_arrays([pa.array([[], None], t)], names=["c"])
        cases[f"bufferless:{t}"] = (batch.schema, [batch, other, batch])
    for case_id, schema, *_ in extension_conflicts():
        cases[f"extension-conflict:{case_id}"] = (schema, [point_batch(schema)])
    own = {
        "ARROW:extension:name": POINT_NAME,
        "ARROW:extension:metadata": POINT_SERIALIZED,
    }
    for case_id, metadata in [("own", own), ("none", None)]:
        schema = pa.schema([pa.field("p", Point(), metadata=metadata)])
        cases[f"extension-registered:{case_id}"] = (schema, [point_batch(schema)])
    for depth in (1, 2, 3):
        cases[f"nested-dictionary-repeated-key:{depth}"] = (
            nested_dictionary_schema(depth),
            [],
        )
    return cases
