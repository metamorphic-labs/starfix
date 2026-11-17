"""A rejected update leaves the digester usable and unchanged (ITL-1178), as in starfix-python.

For every refusal of ``update`` (a wrong container type, a schema mismatch of every kind, a null
in a non-nullable field found mid-batch, repeated metadata keys, invalid Arrow data found
mid-batch), a following valid update and ``finalize`` give the digest of the digester that never
saw the rejected batch, through ``ArrowDigester(...)`` and ``PreparedSchema.digester()``; and a
refusal of a prepared one-shot call changes nothing after it.
"""

import pyarrow as pa
import pytest

from starfix import ArrowDigester

OPTIONS = {"include_metadata": True, "ordered_columns": True}
DICT = pa.dictionary(pa.int32(), pa.string())
ITEMS = pa.list_(pa.field("item", pa.int64(), nullable=False))
SCHEMA = pa.schema(
    [
        pa.field("a", pa.string()),
        pa.field("b", pa.int64(), nullable=False),
        pa.field("d", DICT),
        pa.field("z", ITEMS),
    ],
    metadata={"k": "v"},
)


def _batch(schema, b, keys, items=None):
    """Columns a, b, d (dictionary keys, unvalidated) and z (one item per row; a None item is a
    null in the non-nullable item field, which only the hasher finds, after a, b and d)."""
    n = len(b)
    a = pa.array([f"r{i}" for i in range(n)])
    d = pa.DictionaryArray.from_buffers(DICT, n, pa.array(keys, pa.int32()).buffers(), pa.array(["x", "y"]))
    items = list(range(n)) if items is None else items
    z = pa.ListArray.from_arrays(pa.array(range(n + 1), pa.int32()), pa.array(items, pa.int64()), type=ITEMS)
    columns = [a, pa.array(b, pa.int64()), d, z][: len(schema)]
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _good(first):
    return _batch(SCHEMA, [first, first + 1], [0, 1])


def _duplicate_names():
    good = _good(1)
    return pa.RecordBatch.from_arrays(good.columns, names=["a", "a", "d", "z"]).replace_schema_metadata(SCHEMA.metadata)


def _reordered(schema):
    good = _good(1)
    return pa.RecordBatch.from_arrays([good.column(1), good.column(0), good.column(2), good.column(3)], schema=schema)


def _rejected():
    fields = list(SCHEMA)
    reordered = pa.schema([fields[1], fields[0], fields[2], fields[3]], metadata=SCHEMA.metadata)
    other_column = pa.schema([fields[0], fields[1], pa.field("q", DICT), fields[3]], metadata=SCHEMA.metadata)
    repeated = pa.KeyValueMetadata([(b"k", b"v"), (b"k", b"w")])
    union = pa.schema([fields[0], fields[1], pa.field("d", pa.sparse_union([pa.field("u", pa.int8())])), fields[3]])
    union_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["r"]),
            pa.array([1], pa.int64()),
            pa.UnionArray.from_sparse(pa.array([0], pa.int8()), [pa.array([1], pa.int8())], ["u"]),
            pa.ListArray.from_arrays(pa.array([0, 1], pa.int32()), pa.array([1], pa.int64()), type=ITEMS),
        ],
        schema=union,
    )
    return [
        ("wrong container type", pa.Table.from_batches([_good(5)]), TypeError),
        ("other column", _batch(other_column, [1], [0]), ValueError),
        ("column order (ordered_columns)", _reordered(reordered), ValueError),
        ("schema metadata (include_metadata)", _batch(SCHEMA.remove_metadata(), [1], [0]), ValueError),
        ("repeated metadata key (include_metadata)", _batch(SCHEMA.with_metadata(repeated), [1], [0]), ValueError),
        ("duplicate column names", _duplicate_names(), ValueError),
        ("union column", union_batch, ValueError),
        # Refused by the import (arrow-rs checks top-level nullability).
        ("null in a non-nullable column", _batch(SCHEMA, [1, None], [0, 1]), ValueError),
        # Found by the hasher, after columns a, b and d were hashed.
        ("null in a non-nullable list item", _batch(SCHEMA, [1, 2], [0, 1], items=[3, None]), ValueError),
        # An out-of-range dictionary key: invalid data, read while hashing (after a and b).
        ("invalid dictionary key", _batch(SCHEMA, [1, 2], [0, 7]), ValueError),
    ]


@pytest.mark.parametrize("make", ["ArrowDigester", "prepared.digester"])
@pytest.mark.parametrize("reason, batch, error", _rejected(), ids=[r[0] for r in _rejected()])
def test_a_rejected_update_leaves_the_digester_usable(make, reason, batch, error):
    prepared = ArrowDigester.prepare(SCHEMA, **OPTIONS)

    def fresh():
        return ArrowDigester(SCHEMA, **OPTIONS) if make == "ArrowDigester" else prepared.digester()

    reference = fresh()
    reference.update(_good(0))
    after_first = reference.finalize()
    reference.update(_good(10))
    expected = reference.finalize()

    digester = fresh()
    digester.update(_good(0))
    with pytest.raises(error):
        digester.update(batch)
    assert digester.finalize() == after_first, f"{reason}: the rejected batch changed the digester"
    digester.update(_good(10))
    assert digester.finalize() == expected


@pytest.mark.parametrize("reason, batch, error", _rejected(), ids=[r[0] for r in _rejected()])
def test_a_rejected_prepared_call_changes_nothing_after_it(reason, batch, error):
    prepared = ArrowDigester.prepare(SCHEMA, **OPTIONS)
    expected_batch = prepared.hash_record_batch(_good(0))
    expected_table = prepared.hash_table(pa.Table.from_batches([_good(0), _good(10)]))
    with pytest.raises(error):
        prepared.hash_record_batch(batch)
    if isinstance(batch, pa.RecordBatch) and batch.schema.remove_metadata() != SCHEMA.remove_metadata():
        # (A table has one schema: batches differing only in metadata take the table's.)
        with pytest.raises((ValueError, pa.ArrowException)):
            prepared.hash_table(pa.Table.from_batches([_good(0), batch]))
    assert prepared.hash_record_batch(_good(0)) == expected_batch
    assert prepared.hash_table(pa.Table.from_batches([_good(0), _good(10)])) == expected_table
