"""The binding's one-shot and streaming calls (ITL-1178).

Every chunk of a table is hashed, an empty table hashes, schema-level metadata reaches the
digester, and the hashing options are exposed with starfix-python's names and defaults. The
golden vectors were produced by the Rust crate, so matching them is agreement with it.

Run after ``maturin develop``: ``pytest python/tests``.
"""

import decimal
import inspect
import random

import pyarrow as pa
import pytest

import starfix
from _support import OPTION_SETS, golden_vectors, option_id, options_of, read_ipc, rechunk

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


# --- the reported failures, through the calls the binding already had ------------------------
# (``hash_record_batch(table)``, ``PyArrowDigester(schema).update(table)``, and the prepared
# schema, whose single-batch path was already right.)


def test_reported_later_chunks_were_ignored():
    first, later_differs = _table((0, 3), (3, 2)), _table((0, 3), (10, 2))
    assert starfix.hash_record_batch(first) != starfix.hash_record_batch(later_differs)
    expected = starfix.prepare(first.schema).hash_record_batch(_batch(0, 5))
    assert starfix.hash_record_batch(first) == expected
    digester = starfix.PyArrowDigester(first.schema)
    digester.update(first)
    assert digester.finalize() == expected


def test_reported_an_empty_table_failed():
    schema = _schema()
    expected = starfix.prepare(schema).hash_record_batch(_batch(0, 0))
    assert starfix.hash_record_batch(schema.empty_table()) == expected
    digester = starfix.PyArrowDigester(schema)
    digester.update(schema.empty_table())
    assert digester.finalize() == expected


def test_reported_schema_metadata_was_dropped_on_update():
    table = _table((0, 2), (2, 1), schema=_schema(metadata={"source": "a"}))
    prepared = starfix.prepare(table.schema, include_metadata=True)
    digester = prepared.digester()
    digester.update(table)  # a table, as PyArrowDigester.update takes
    assert digester.finalize() == prepared.hash_table(table)


# --- 1. every chunk of a table is hashed ---------------------------------------------------


def test_a_later_chunk_changes_the_hash():
    first = _table((0, 3), (3, 2))
    later_differs = _table((0, 3), (10, 2))
    assert starfix.hash_record_batch(first) != starfix.hash_record_batch(later_differs)
    assert starfix.ArrowDigester.hash_table(first) != starfix.ArrowDigester.hash_table(later_differs)
    for table in (first, later_differs):
        digester = starfix.PyArrowDigester(table.schema)
        digester.update(table)
        assert digester.finalize() == starfix.ArrowDigester.hash_table(table)


@pytest.mark.parametrize("sizes", [(5,), (1, 4), (2, 0, 3), (1, 1, 1, 1, 1), (4, 1), (0, 5, 0)])
def test_chunks_of_any_size_hash_as_one_chunk(sizes):
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    chunked = _table(*zip(starts, sizes))
    assert chunked.num_rows == 5
    assert len(chunked.to_batches()) > 1 or len([s for s in sizes if s]) == 1
    one = _batch(0, 5)
    expected = starfix.ArrowDigester.hash_record_batch(one)
    assert starfix.hash_record_batch(chunked) == expected
    assert starfix.ArrowDigester.hash_table(chunked) == expected
    assert starfix.ArrowDigester.hash_table(chunked) == expected
    digester = starfix.PyArrowDigester(chunked.schema)
    digester.update(chunked)
    assert digester.finalize() == expected
    assert _streamed(starfix.ArrowDigester(chunked.schema), chunked.to_batches()) == expected


def test_a_sliced_table_hashes_as_its_rows():
    table = _table((0, 3), (3, 4), (7, 3))
    for offset, length in [(1, 5), (3, 4), (4, 0), (2, 8), (9, 1)]:
        sliced = table.slice(offset, length)
        expected = starfix.ArrowDigester.hash_record_batch(_batch(offset, length))
        assert starfix.hash_record_batch(sliced) == expected
        assert starfix.ArrowDigester.hash_table(sliced) == expected
        digester = starfix.PyArrowDigester(sliced.schema)
        digester.update(sliced)
        assert digester.finalize() == expected


def test_a_chunked_array_hashes_as_its_chunks_combined():
    table = _table((0, 2), (2, 3), (5, 0), (5, 1))
    for name in table.column_names:
        chunked = table.column(name)
        assert chunked.num_chunks == 4
        expected = starfix.ArrowDigester.hash_array(chunked.combine_chunks())
        assert starfix.ArrowDigester.hash_array(chunked) == expected
        assert starfix.ArrowDigester.hash_array(chunked) == expected
        assert starfix.ArrowDigester.hash_array(chunked.slice(1, 3)) == starfix.ArrowDigester.hash_array(chunked.combine_chunks().slice(1, 3))
    dictionaries = pa.chunked_array(
        [pa.array(["a", "b"]).dictionary_encode(), pa.array(["c", "a"]).dictionary_encode()]
    )
    assert starfix.ArrowDigester.hash_array(dictionaries) == starfix.ArrowDigester.hash_array(pa.array(["a", "b", "c", "a"]))


def _rechunkable_vectors():
    return [v for v in VECTORS if v["method"] in ("hash_record_batch", "hash_stream", "hash_array")]


@pytest.mark.parametrize("v", _rechunkable_vectors(), ids=lambda v: v["id"])
def test_golden_vector_in_several_chunks(v):
    schema, batches = read_ipc(v)
    table = rechunk(pa.Table.from_batches(batches, schema=schema), random.Random(v["id"]))
    if v["method"] == "hash_array":
        got = starfix.ArrowDigester.hash_array(
            table.column(0), ordered_columns=v["ordered_columns"], ordered_struct_fields=v["ordered_struct_fields"]
        )
    else:
        got = starfix.ArrowDigester.hash_table(table, **options_of(v))
        assert starfix.hash_record_batch(table, **options_of(v)) == got
        digester = starfix.PyArrowDigester(schema, **options_of(v))
        digester.update(table)
        assert digester.finalize() == got
    assert got.hex() == v["expected_hash"]


def test_a_slice_of_nested_structs_imports():
    # A struct inside a sliced struct or fixed-size list: arrow-rs's import applied the outer
    # offset twice (a panic) until the binding moved struct offsets into the children.
    inner = pa.struct([("x", pa.int32()), ("y", pa.string())])
    schema = pa.schema([("s", pa.struct([("t", inner), ("n", pa.int8())])), ("f", pa.list_(inner, 2))])
    rows = {
        "s": [{"t": {"x": i, "y": str(i)}, "n": i} if i % 4 else None for i in range(8)],
        "f": [[{"x": i, "y": None}, {"x": -i, "y": "f"}] for i in range(8)],
    }
    table = pa.Table.from_pydict(rows, schema=schema)
    for offset, length in [(1, 3), (3, 5), (0, 8), (7, 1)]:
        fresh = pa.RecordBatch.from_pydict({k: v[offset : offset + length] for k, v in rows.items()}, schema=schema)
        expected = starfix.ArrowDigester.hash_record_batch(fresh)
        assert starfix.ArrowDigester.hash_record_batch(table.to_batches()[0].slice(offset, length)) == expected
        assert starfix.hash_record_batch(table.slice(offset, length)) == expected
        assert starfix.ArrowDigester.hash_array(table.column("s").slice(offset, length)) == starfix.ArrowDigester.hash_array(
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
    assert starfix.ArrowDigester.hash_array(array) == starfix.ArrowDigester.hash_array(values)
    batch = pa.RecordBatch.from_arrays([array], names=["d"])
    assert starfix.ArrowDigester.hash_record_batch(batch) == starfix.ArrowDigester.hash_record_batch(
        pa.RecordBatch.from_arrays([values], names=["d"])
    )


# --- 2. an empty table hashes ----------------------------------------------------------------


@pytest.mark.parametrize("options", OPTION_SETS, ids=option_id)
def test_an_empty_table_hashes_as_a_zero_row_batch(options):
    schema = _schema(metadata={"source": "test"})
    empty = schema.empty_table()
    assert empty.to_batches() == []
    expected = starfix.ArrowDigester.hash_record_batch(_batch(0, 0, schema), **options)
    assert starfix.hash_record_batch(empty, **options) == expected
    assert starfix.ArrowDigester.hash_table(empty, **options) == expected
    assert starfix.ArrowDigester.hash_table(empty, **options) == expected
    digester = starfix.PyArrowDigester(schema, **options)
    digester.update(empty)
    assert digester.finalize() == expected
    assert starfix.ArrowDigester(schema, **options).finalize() == expected


def test_zero_row_golden_vectors_through_an_empty_table():
    zero_row = [v for v in VECTORS if v["method"] == "hash_record_batch" and sum(b.num_rows for b in read_ipc(v)[1]) == 0]
    assert zero_row, "the golden file has zero-row record batch vectors"
    for v in zero_row:
        schema, _ = read_ipc(v)
        assert starfix.ArrowDigester.hash_table(schema.empty_table(), **options_of(v)).hex() == v["expected_hash"], v["id"]


# --- 3. schema-level metadata reaches the digester ------------------------------------------


def test_schema_metadata_is_hashed_when_included():
    with_metadata = _table((0, 2), (2, 1), schema=_schema(metadata={"source": "a"}))
    other_metadata = with_metadata.replace_schema_metadata({"source": "b"})
    without = with_metadata.replace_schema_metadata(None)
    for call in (starfix.hash_record_batch, starfix.ArrowDigester.hash_table):
        assert call(with_metadata) == call(other_metadata) == call(without)
        assert len({call(t, include_metadata=True) for t in (with_metadata, other_metadata, without)}) == 3
    batch = with_metadata.to_batches()[0]
    assert starfix.ArrowDigester.hash_record_batch(batch, include_metadata=True) != starfix.ArrowDigester.hash_record_batch(
        batch.replace_schema_metadata({"source": "b"}), include_metadata=True
    )
    # Streaming: the digester's schema and each batch's schema carry the metadata.
    digester = starfix.ArrowDigester(with_metadata.schema, include_metadata=True)
    assert _streamed(digester, with_metadata.to_batches()) == starfix.ArrowDigester.hash_table(with_metadata, include_metadata=True)


def test_a_batch_without_the_schema_metadata_is_refused_when_included():
    table = _table((0, 2), schema=_schema(metadata={"source": "a"}))
    digester = starfix.ArrowDigester(table.schema, include_metadata=True)
    with pytest.raises(ValueError, match="schema"):
        digester.update(table.to_batches()[0].replace_schema_metadata(None))
    # Without the option the metadata is not part of the schema check.
    digester = starfix.ArrowDigester(table.schema)
    digester.update(table.to_batches()[0].replace_schema_metadata(None))


# --- 4. the options, with starfix-python's names and defaults --------------------------------

_OPTIONS = ["include_metadata", "ordered_columns", "ordered_struct_fields"]


def _resolve(dotted):
    obj = starfix
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _keyword_only_options(func, names=_OPTIONS):
    params = inspect.signature(func).parameters
    return {name: (params[name].kind, params[name].default) for name in names if name in params}


@pytest.mark.parametrize(
    "func, positional, names",
    [
        ("ArrowDigester", ["schema"], _OPTIONS),
        ("ArrowDigester.prepare", ["schema"], _OPTIONS),
        ("ArrowDigester.hash_schema", ["schema"], _OPTIONS),
        ("ArrowDigester.hash_record_batch", ["record_batch"], _OPTIONS),
        ("ArrowDigester.hash_table", ["table"], _OPTIONS),
        ("ArrowDigester.hash_array", ["array"], _OPTIONS[1:]),
        ("hash_schema", ["schema"], _OPTIONS),
        ("hash_record_batch", ["table"], _OPTIONS),
        ("PyArrowDigester", ["schema"], _OPTIONS),
        ("prepare", ["schema"], _OPTIONS),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_signature_matches_starfix_python(func, positional, names):
    func = _resolve(func)
    params = inspect.signature(func).parameters
    assert [n for n, p in params.items() if p.kind == p.POSITIONAL_OR_KEYWORD and n != "self"] == positional
    assert _keyword_only_options(func, names) == {n: (inspect.Parameter.KEYWORD_ONLY, False) for n in names}
    assert set(params) - {"self"} == {*positional, *names}


def test_the_starfix_python_surface_is_present():
    for name in ("prepare", "hash_schema", "hash_record_batch", "hash_table", "hash_array", "update", "finalize"):
        assert callable(getattr(starfix.ArrowDigester, name)), name
    assert issubclass(starfix.PyArrowDigester, starfix.ArrowDigester)
    assert isinstance(starfix.ArrowDigester.prepare(_schema()).digester(), starfix.ArrowDigester)


@pytest.mark.parametrize("v", VECTORS, ids=lambda v: v["id"])
def test_golden_vector_through_the_one_shot_and_streaming_calls(v):
    schema, batches = read_ipc(v)
    options = options_of(v)
    if v["method"] == "hash_schema":
        got = starfix.ArrowDigester.hash_schema(schema, **options)
        assert starfix.hash_schema(schema, **options) == got
    elif v["method"] == "hash_record_batch":
        (batch,) = batches
        got = starfix.ArrowDigester.hash_record_batch(batch, **options)
        assert starfix.hash_record_batch(batch, **options) == got
        assert starfix.hash_record_batch(pa.Table.from_batches([batch]), **options) == got
        assert _streamed(starfix.ArrowDigester(schema, **options), [batch]) == got
    elif v["method"] == "hash_stream":
        got = _streamed(starfix.ArrowDigester(schema, **options), batches)
        assert starfix.ArrowDigester.hash_table(pa.Table.from_batches(batches, schema=schema), **options) == got
        digester = starfix.PyArrowDigester(schema, **options)
        for batch in batches:
            digester.update(batch)
        assert digester.finalize() == got
    else:  # hash_array
        (batch,) = batches
        array_options = {k: options[k] for k in _OPTIONS[1:]}
        got = starfix.ArrowDigester.hash_array(batch.column(0), **array_options)
        assert starfix.ArrowDigester.hash_array(batch.column(0), **array_options) == got
    assert got.hex() == v["expected_hash"]


def test_options_change_the_hash_as_in_the_crate():
    schema = pa.schema([("b", pa.int32()), ("a", pa.struct([("y", pa.int8()), ("x", pa.int8())]))])
    reordered = pa.schema([("a", pa.struct([("x", pa.int8()), ("y", pa.int8())])), ("b", pa.int32())])
    assert starfix.hash_schema(schema) == starfix.hash_schema(reordered)
    assert starfix.hash_schema(schema, ordered_columns=True) != starfix.hash_schema(reordered, ordered_columns=True)
    assert starfix.hash_schema(schema, ordered_struct_fields=True) != starfix.hash_schema(
        reordered, ordered_struct_fields=True
    )
    with pytest.raises(TypeError):
        starfix.hash_schema(schema, True)  # the options are keyword-only


# --- errors as starfix-python raises them ----------------------------------------------------


def test_wrong_container_types_raise_type_error():
    table = _table((0, 2))
    batch = table.to_batches()[0]
    with pytest.raises(TypeError):
        starfix.ArrowDigester.hash_record_batch(table)
    with pytest.raises(TypeError):
        starfix.ArrowDigester.hash_table(batch)
    with pytest.raises(TypeError):
        starfix.ArrowDigester(table.schema).update(table)
    with pytest.raises(TypeError):
        starfix.ArrowDigester.hash_schema(table)
    with pytest.raises(TypeError):
        starfix.ArrowDigester.hash_array(batch)


def test_rejections_raise_value_error():
    schema = pa.schema([pa.field("a", pa.string()), pa.field("z", pa.int64(), nullable=False)])
    batch = pa.RecordBatch.from_pydict({"a": ["p", None], "z": [1, 2]}, schema=schema)
    expected = starfix.ArrowDigester.hash_record_batch(batch)
    digester = starfix.ArrowDigester(schema)
    digester.update(batch)
    assert digester.finalize() == expected
    assert digester.finalize() == expected  # finalize does not end the digester
    with pytest.raises(ValueError, match="schema"):
        digester.update(pa.RecordBatch.from_pydict({"a": ["x"], "other": [1]}))
    # Unlike starfix-python, which restores the digester, a rejected update ends it.
    with pytest.raises(ValueError, match="rejected update"):
        digester.finalize()
    with pytest.raises(ValueError, match="rejected update"):
        digester.update(batch)
    # A null in a non-nullable column is found while the batch is hashed.
    late_null = pa.RecordBatch.from_arrays([pa.array(["r", "s"]), pa.array([4, None], pa.int64())], schema=schema)
    with pytest.raises(ValueError, match="non-nullable|null"):
        starfix.ArrowDigester.hash_record_batch(late_null)
    with pytest.raises(ValueError, match="non-nullable|null"):
        starfix.hash_record_batch(pa.Table.from_batches([batch, late_null]))
    # Data of a type the format does not hash.
    with pytest.raises(ValueError):
        starfix.hash_record_batch(pa.table({"m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int8()))}))
