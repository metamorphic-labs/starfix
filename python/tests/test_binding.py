"""The Rust-backed package's own tests (ITL-1178).

Every chunk of a table is hashed, an empty table hashes, schema-level metadata reaches the
digester, the interface is starfix-python's (names, signatures, defaults, exception types),
and pyarrow inputs that arrow-rs imports badly or refuses are handled. The golden vectors were
produced by the Rust crate, so matching them is agreement with it; starfix-python's own test
suite and ``test_differential.py`` check agreement with starfix-python.

Run after ``maturin develop``: ``pytest python/tests``.
"""

import copy
import decimal
import inspect
import pickle
import random
import threading

import pyarrow as pa
import pytest

import starfix
from _support import OPTION_SETS, golden_vectors, option_id, options_of, read_ipc, rechunk
from starfix import ArrowDigester, PreparedSchema

VECTORS = golden_vectors()


def _schema(metadata=None):
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False, metadata={"unit": "count"}),
            pa.field("name", pa.string()),
            pa.field("tags", pa.list_(pa.struct([("k", pa.string()), ("v", pa.int32())]))),
        ],
        metadata=metadata,
    )


def _rows(start, count):
    return {
        "id": list(range(start, start + count)),
        "name": [None if i % 3 == 0 else f"n{i}" for i in range(start, start + count)],
        "tags": [[{"k": "a", "v": i}] if i % 2 else [] for i in range(start, start + count)],
    }


def _batch(start, count, schema=None):
    return pa.RecordBatch.from_pydict(_rows(start, count), schema=schema or _schema())


def _table(*chunks, schema=None):
    schema = schema or _schema()
    return pa.Table.from_batches([_batch(s, c, schema) for s, c in chunks], schema=schema)


def _streamed(digester, batches):
    for batch in batches:
        digester.update(batch)
    return digester.finalize()


# --- every chunk of a table is hashed ------------------------------------------------------


def test_a_later_chunk_changes_the_hash():
    first, later_differs = _table((0, 3), (3, 2)), _table((0, 3), (10, 2))
    assert ArrowDigester.hash_table(first) != ArrowDigester.hash_table(later_differs)
    expected = ArrowDigester.hash_record_batch(_batch(0, 5))
    assert ArrowDigester.hash_table(first) == expected
    assert ArrowDigester.prepare(first.schema).hash_table(first) == expected
    assert _streamed(ArrowDigester(first.schema), first.to_batches()) == expected


@pytest.mark.parametrize("sizes", [(5,), (1, 4), (2, 0, 3), (1, 1, 1, 1, 1), (4, 1), (0, 5, 0)])
def test_chunks_of_any_size_hash_as_one_chunk(sizes):
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    chunked = _table(*zip(starts, sizes))
    assert chunked.num_rows == 5
    assert len(chunked.to_batches()) > 1 or len([s for s in sizes if s]) == 1
    expected = ArrowDigester.hash_record_batch(_batch(0, 5))
    assert ArrowDigester.hash_table(chunked) == expected
    assert ArrowDigester.prepare(chunked.schema).hash_table(chunked) == expected
    assert _streamed(ArrowDigester(chunked.schema), chunked.to_batches()) == expected


def test_a_sliced_table_hashes_as_its_rows():
    table = _table((0, 3), (3, 4), (7, 3))
    for offset, length in [(1, 5), (3, 4), (4, 0), (2, 8), (9, 1)]:
        sliced = table.slice(offset, length)
        expected = ArrowDigester.hash_record_batch(_batch(offset, length))
        assert ArrowDigester.hash_table(sliced) == expected
        assert _streamed(ArrowDigester(sliced.schema), sliced.to_batches()) == expected


def test_a_chunked_array_hashes_as_its_chunks_combined():
    table = _table((0, 2), (2, 3), (5, 0), (5, 1))
    for name in table.column_names:
        chunked = table.column(name)
        assert chunked.num_chunks == 4
        expected = ArrowDigester.hash_array(chunked.combine_chunks())
        assert ArrowDigester.hash_array(chunked) == expected
        assert ArrowDigester.hash_array(chunked.slice(1, 3)) == ArrowDigester.hash_array(
            chunked.combine_chunks().slice(1, 3)
        )
    dictionaries = pa.chunked_array(
        [pa.array(["a", "b"]).dictionary_encode(), pa.array(["c", "a"]).dictionary_encode()]
    )
    assert ArrowDigester.hash_array(dictionaries) == ArrowDigester.hash_array(pa.array(["a", "b", "c", "a"]))
    assert ArrowDigester.hash_array(pa.chunked_array([], pa.int8())) == ArrowDigester.hash_array(pa.array([], pa.int8()))


def test_a_chunked_dictionary_whose_chunks_do_not_combine_hashes():
    # Two valid dictionary<int8, string> chunks of 100 values each: their combined dictionary
    # would need 200 entries, more than int8 keys index (combine_chunks raises). The chunks are
    # hashed one after the other, as their decoded concatenation.
    def chunk(start):
        values = [f"v{i}" for i in range(start, start + 100)]
        return pa.DictionaryArray.from_arrays(pa.array(range(100), pa.int8()), pa.array(values))

    chunked = pa.chunked_array([chunk(0), chunk(100)])
    with pytest.raises(pa.ArrowException):
        chunked.combine_chunks()
    decoded = pa.array([f"v{i}" for i in range(200)])
    assert ArrowDigester.hash_array(chunked) == ArrowDigester.hash_array(decoded)
    table = pa.table({"d": chunked})
    assert ArrowDigester.hash_table(table) == ArrowDigester.hash_table(pa.table({"d": decoded}))


@pytest.mark.parametrize(
    "v",
    [v for v in VECTORS if v["method"] in ("hash_record_batch", "hash_stream", "hash_array")],
    ids=lambda v: v["id"],
)
def test_golden_vector_in_several_chunks(v):
    schema, batches = read_ipc(v)
    table = rechunk(pa.Table.from_batches(batches, schema=schema), random.Random(v["id"]))
    if v["method"] == "hash_array":
        got = ArrowDigester.hash_array(
            table.column(0), ordered_columns=v["ordered_columns"], ordered_struct_fields=v["ordered_struct_fields"]
        )
    else:
        got = ArrowDigester.hash_table(table, **options_of(v))
        assert ArrowDigester.prepare(schema, **options_of(v)).hash_table(table) == got
        assert _streamed(ArrowDigester(schema, **options_of(v)), table.to_batches()) == got
    assert got.hex() == v["expected_hash"]


def test_a_slice_of_nested_structs_imports():
    # A struct inside a sliced struct or fixed-size list: arrow-rs's import applies the outer
    # offset twice (a panic) unless struct offsets are moved into the children.
    inner = pa.struct([("x", pa.int32()), ("y", pa.string())])
    schema = pa.schema([("s", pa.struct([("t", inner), ("n", pa.int8())])), ("f", pa.list_(inner, 2))])
    rows = {
        "s": [{"t": {"x": i, "y": str(i)}, "n": i} if i % 4 else None for i in range(8)],
        "f": [[{"x": i, "y": None}, {"x": -i, "y": "f"}] for i in range(8)],
    }
    table = pa.Table.from_pydict(rows, schema=schema)
    for offset, length in [(1, 3), (3, 5), (0, 8), (7, 1)]:
        fresh = pa.RecordBatch.from_pydict({k: v[offset : offset + length] for k, v in rows.items()}, schema=schema)
        expected = ArrowDigester.hash_record_batch(fresh)
        assert ArrowDigester.hash_record_batch(table.to_batches()[0].slice(offset, length)) == expected
        assert ArrowDigester.hash_table(table.slice(offset, length)) == expected
        assert ArrowDigester.hash_array(table.column("s").slice(offset, length)) == ArrowDigester.hash_array(
            fresh.column("s")
        )


def test_buffers_less_aligned_than_arrow_rs_requires():
    # pyarrow allows a decimal128 buffer at an 8-byte boundary (as read from IPC); arrow-rs
    # requires 16, so the import copies it.
    values = pa.array([decimal.Decimal("1.25"), decimal.Decimal("-3.50")], pa.decimal128(10, 2))
    raw = values.buffers()[1].to_pybytes()
    backing = pa.allocate_buffer(len(raw) + 16)
    shift = (8 - backing.address) % 16
    memoryview(backing).cast("B")[shift : shift + len(raw)] = raw
    misaligned = backing.slice(shift, len(raw))
    assert misaligned.address % 16 == 8
    array = pa.Array.from_buffers(values.type, len(values), [None, misaligned])
    assert array.equals(values)
    assert ArrowDigester.hash_array(array) == ArrowDigester.hash_array(values)
    batch = pa.RecordBatch.from_arrays([array], names=["d"])
    assert ArrowDigester.hash_record_batch(batch) == ArrowDigester.hash_record_batch(
        pa.RecordBatch.from_arrays([values], names=["d"])
    )


def test_an_empty_list_without_offsets_buffer_hashes_as_an_empty_list():
    # Arrow allows an empty list array with no offsets buffer.
    for t in (pa.list_(pa.int32()), pa.large_list(pa.int32())):
        arr = pa.Array.from_buffers(t, 0, [None, None], children=[pa.array([], pa.int32())])
        arr.validate(full=True)
        assert ArrowDigester.hash_array(arr) == ArrowDigester.hash_array(pa.array([], t))
        s = pa.StructArray.from_arrays([arr], names=["l"])
        assert ArrowDigester.hash_array(s) == ArrowDigester.hash_array(
            pa.StructArray.from_arrays([pa.array([], t)], names=["l"])
        )
        batch = pa.RecordBatch.from_arrays([arr], names=["l"])
        assert ArrowDigester.hash_record_batch(batch) == ArrowDigester.hash_record_batch(
            pa.RecordBatch.from_arrays([pa.array([], t)], names=["l"])
        )


# --- an empty table hashes ----------------------------------------------------------------


@pytest.mark.parametrize("options", OPTION_SETS, ids=option_id)
def test_an_empty_table_hashes_as_a_zero_row_batch(options):
    schema = _schema(metadata={"source": "test"})
    empty = schema.empty_table()
    assert empty.to_batches() == []
    expected = ArrowDigester.hash_record_batch(_batch(0, 0, schema), **options)
    assert ArrowDigester.hash_table(empty, **options) == expected
    assert ArrowDigester.prepare(schema, **options).hash_table(empty) == expected
    assert ArrowDigester(schema, **options).finalize() == expected


def _rows_of(v):
    try:
        return sum(b.num_rows for b in read_ipc(v)[1])
    except pytest.skip.Exception:
        return None  # this pyarrow cannot read the vector


def test_zero_row_golden_vectors_through_an_empty_table():
    zero_row = [v for v in VECTORS if v["method"] == "hash_record_batch" and _rows_of(v) == 0]
    assert zero_row, "the golden file has zero-row record batch vectors"
    for v in zero_row:
        schema, _ = read_ipc(v)
        assert ArrowDigester.hash_table(schema.empty_table(), **options_of(v)).hex() == v["expected_hash"], v["id"]


def test_an_empty_table_of_another_schema_is_refused_by_a_prepared_schema():
    prepared = ArrowDigester.prepare(_schema())
    with pytest.raises(ValueError, match="Table schema does not match the prepared schema"):
        prepared.hash_table(pa.schema([("q", pa.int8())]).empty_table())


# --- schema-level metadata reaches the digester -------------------------------------------


def test_schema_metadata_is_hashed_when_included():
    with_metadata = _table((0, 2), (2, 1), schema=_schema(metadata={"source": "a"}))
    other_metadata = with_metadata.replace_schema_metadata({"source": "b"})
    without = with_metadata.replace_schema_metadata(None)
    tables = (with_metadata, other_metadata, without)
    assert len({ArrowDigester.hash_table(t) for t in tables}) == 1
    assert len({ArrowDigester.hash_table(t, include_metadata=True) for t in tables}) == 3
    batch = with_metadata.to_batches()[0]
    assert ArrowDigester.hash_record_batch(batch, include_metadata=True) != ArrowDigester.hash_record_batch(
        batch.replace_schema_metadata({"source": "b"}), include_metadata=True
    )
    digester = ArrowDigester(with_metadata.schema, include_metadata=True)
    assert _streamed(digester, with_metadata.to_batches()) == ArrowDigester.hash_table(
        with_metadata, include_metadata=True
    )
    prepared = ArrowDigester.prepare(with_metadata.schema, include_metadata=True)
    assert prepared.hash_table(with_metadata) == ArrowDigester.hash_table(with_metadata, include_metadata=True)
    with pytest.raises(ValueError, match="does not match"):
        prepared.hash_record_batch(batch.replace_schema_metadata(None))


def test_repeated_metadata_keys_are_refused_with_include_metadata():
    # pyarrow keeps a repeated key in exported metadata; arrow-rs would keep one of the values.
    repeated = pa.KeyValueMetadata([(b"k", b"1"), (b"k", b"2")])
    field_level = pa.schema([pa.field("a", pa.int8(), metadata=repeated)])
    schema_level = pa.schema([pa.field("a", pa.int8())], metadata=repeated)
    nested = pa.schema([pa.field("s", pa.struct([pa.field("x", pa.int8(), metadata=repeated)]))])
    for schema, where in [(field_level, "Field 'a'"), (schema_level, "on the schema"), (nested, "Field 's/x'")]:
        with pytest.raises(ValueError, match="Duplicate metadata key 'k'") as refused:
            ArrowDigester.hash_schema(schema, include_metadata=True)
        assert where in str(refused.value)
        with pytest.raises(ValueError, match="Duplicate metadata key"):
            ArrowDigester.prepare(schema, include_metadata=True)
        ArrowDigester.hash_schema(schema)  # without the option, metadata is not read
        ArrowDigester.hash_table(schema.empty_table())


# --- starfix-python's interface -----------------------------------------------------------

_OPTIONS = ["include_metadata", "ordered_columns", "ordered_struct_fields"]


@pytest.mark.parametrize(
    "func, positional, names",
    [
        (ArrowDigester, ["schema"], _OPTIONS),
        (ArrowDigester.prepare, ["schema"], _OPTIONS),
        (ArrowDigester.hash_schema, ["schema"], _OPTIONS),
        (ArrowDigester.hash_record_batch, ["record_batch"], _OPTIONS),
        (ArrowDigester.hash_table, ["table"], _OPTIONS),
        (ArrowDigester.hash_array, ["array"], _OPTIONS[1:]),
    ],
    ids=lambda x: getattr(x, "__qualname__", "") if callable(x) else "",
)
def test_signature_matches_starfix_python(func, positional, names):
    params = inspect.signature(func).parameters
    assert [n for n, p in params.items() if p.kind == p.POSITIONAL_OR_KEYWORD and n != "self"] == positional
    assert {n: (params[n].kind, params[n].default) for n in names} == {
        n: (inspect.Parameter.KEYWORD_ONLY, False) for n in names
    }
    assert set(params) - {"self"} == {*positional, *names}


def test_the_public_names_are_starfix_pythons():
    assert set(starfix.__all__) == {"ArrowDigester", "PreparedSchema", "__version__"}
    assert isinstance(starfix.__version__, str)
    from starfix.arrow_digester import ArrowDigester as FromModule
    from starfix.legacy.v0_4 import ArrowDigester as Legacy

    assert FromModule is ArrowDigester
    assert Legacy.hash_schema(pa.schema([("a", pa.int8())]))[:3] == b"\x00\x00\x01"


def test_prepared_schema_behaves_as_starfix_pythons():
    schema = _schema(metadata={"m": "1"})
    prepared = ArrowDigester.prepare(schema, include_metadata=True, ordered_columns=True)
    assert (prepared.include_metadata, prepared.ordered_columns, prepared.ordered_struct_fields) == (True, True, False)
    assert prepared.schema is schema
    assert prepared.schema_digest == ArrowDigester.hash_schema(schema, include_metadata=True, ordered_columns=True)
    assert repr(prepared).startswith("PreparedSchema(include_metadata=True, ordered_columns=True")
    assert copy.copy(prepared) is prepared and copy.deepcopy(prepared) is prepared
    again = pickle.loads(pickle.dumps(prepared))
    assert again.schema_digest == prepared.schema_digest and again.ordered_columns
    with pytest.raises(TypeError):
        PreparedSchema(schema)
    with pytest.raises(AttributeError):
        prepared.include_metadata = False
    assert isinstance(prepared.digester(), ArrowDigester)


@pytest.mark.parametrize("v", VECTORS, ids=lambda v: v["id"])
def test_golden_vector_through_every_call(v):
    schema, batches = read_ipc(v)
    options = options_of(v)
    if v["method"] == "hash_schema":
        got = ArrowDigester.hash_schema(schema, **options)
        assert ArrowDigester.prepare(schema, **options).schema_digest == got
    elif v["method"] == "hash_record_batch":
        (batch,) = batches
        got = ArrowDigester.hash_record_batch(batch, **options)
        assert ArrowDigester.hash_table(pa.Table.from_batches([batch]), **options) == got
        assert _streamed(ArrowDigester(schema, **options), [batch]) == got
        assert ArrowDigester.prepare(schema, **options).hash_record_batch(batch) == got
    elif v["method"] == "hash_stream":
        got = _streamed(ArrowDigester(schema, **options), batches)
        assert ArrowDigester.hash_table(pa.Table.from_batches(batches, schema=schema), **options) == got
        assert _streamed(ArrowDigester.prepare(schema, **options).digester(), batches) == got
    else:  # hash_array
        (batch,) = batches
        array_options = {k: options[k] for k in _OPTIONS[1:]}
        got = ArrowDigester.hash_array(batch.column(0), **array_options)
    assert got.hex() == v["expected_hash"]


# --- errors ------------------------------------------------------------------------------


def test_wrong_container_types_raise_type_error():
    table = _table((0, 2))
    batch = table.to_batches()[0]
    prepared = ArrowDigester.prepare(table.schema)
    for call in (
        lambda: ArrowDigester.hash_record_batch(table),
        lambda: ArrowDigester.hash_table(batch),
        lambda: ArrowDigester(table.schema).update(table),
        lambda: ArrowDigester.hash_schema(table),
        lambda: ArrowDigester.hash_array(batch),
        lambda: ArrowDigester(batch),
        lambda: ArrowDigester.prepare(table),
        lambda: prepared.hash_record_batch(table),
        lambda: prepared.hash_table(batch),
    ):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(TypeError):
        ArrowDigester.hash_schema(table.schema, True)  # the options are keyword-only


def test_a_refused_update_leaves_the_digester_unchanged():
    schema = pa.schema([pa.field("a", pa.string()), pa.field("z", pa.int64(), nullable=False)])
    first = pa.RecordBatch.from_pydict({"a": ["p", None], "z": [1, 2]}, schema=schema)
    second = pa.RecordBatch.from_pydict({"a": ["q"], "z": [3]}, schema=schema)
    expected = ArrowDigester.hash_table(pa.Table.from_batches([first, second]))
    digester = ArrowDigester(schema)
    digester.update(first)
    with pytest.raises(ValueError, match="does not match ArrowDigester schema"):
        digester.update(pa.RecordBatch.from_pydict({"a": ["x"], "other": [1]}))
    # A null in a non-nullable column is found while the batch is traversed, after column "a".
    late_null = pa.RecordBatch.from_arrays([pa.array(["r", "s"]), pa.array([4, None], pa.int64())], schema=schema)
    with pytest.raises(ValueError, match="non-nullable|Non-nullable"):
        digester.update(late_null)
    digester.update(second)
    assert digester.finalize() == expected
    assert digester.finalize() == expected  # finalize does not end the digester


def test_refusals_raise_value_error():
    with pytest.raises(ValueError):
        ArrowDigester.hash_table(pa.table({"m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int8()))}))
    with pytest.raises(ValueError):
        ArrowDigester.hash_schema(pa.schema([("a", pa.int8()), ("a", pa.int8())]))
    with pytest.raises(ValueError):
        ArrowDigester.prepare(pa.schema([("n", pa.null())])).hash_table(pa.table({"n": pa.nulls(1)}))


def test_hidden_out_of_range_dictionary_keys_are_not_read():
    # pyarrow builds an INVALID array here: an all-null struct over a dictionary child gets
    # key 0 into an empty dictionary. The key is under a null struct, so it is never read
    # (byte-layout spec Section 3.2): the hash is that of any in-range key, as starfix-python
    # computes it (test_differential.py compares the two).
    array = pa.array([None, {"x": "a"}], type=pa.struct([("x", pa.dictionary(pa.int32(), pa.string()))]))
    hidden = pa.array([None], type=array.type)
    with pytest.raises(pa.ArrowInvalid):
        hidden.validate(full=True)
    valid_equivalent = array.slice(0, 1)  # the same null struct, its key in range
    valid_equivalent.validate(full=True)
    assert ArrowDigester.hash_array(hidden) == ArrowDigester.hash_array(valid_equivalent)
    batch = pa.RecordBatch.from_arrays([hidden], names=["s"])
    assert ArrowDigester.hash_record_batch(batch) == ArrowDigester.hash_record_batch(
        pa.RecordBatch.from_arrays([valid_equivalent], names=["s"])
    )


def test_invalid_arrow_data_raises_value_error():
    # A VISIBLE out-of-range dictionary key: invalid data the hasher must read. arrow-rs panics
    # decoding it; the binding raises ValueError, never PanicException.
    keys = pa.array([0, 5], pa.int32())
    invalid = pa.DictionaryArray.from_buffers(
        pa.dictionary(pa.int32(), pa.string()), 2, keys.buffers(), pa.array(["a"]), offset=0
    )
    with pytest.raises(pa.ArrowInvalid):
        invalid.validate(full=True)
    with pytest.raises(ValueError, match="Invalid Arrow data"):
        ArrowDigester.hash_array(invalid)
    with pytest.raises(ValueError, match="Invalid Arrow data"):
        ArrowDigester.hash_record_batch(pa.RecordBatch.from_arrays([invalid], names=["d"]))


# --- the GIL is released while hashing ------------------------------------------------------


def test_hashing_from_several_threads_agrees():
    table = _table((0, 300), (300, 200))
    expected = ArrowDigester.hash_table(table)
    prepared = ArrowDigester.prepare(table.schema)
    results = []

    def work():
        for _ in range(20):
            results.append(prepared.hash_table(table))
            results.append(ArrowDigester.hash_table(table))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results and set(results) == {expected}
