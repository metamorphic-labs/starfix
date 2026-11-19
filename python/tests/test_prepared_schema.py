"""The prepared-schema interface through the Python binding (ITL-1177).

Run after ``maturin develop``: ``pytest python/tests``.
"""

import json
from pathlib import Path

import pyarrow as pa
import pytest

import starfix
from _support import options_of, read_ipc

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "format2_v0.5.json"


def _vectors():
    return json.loads(GOLDEN.read_text())["vectors"]


def _options(v):
    return options_of(v)  # the float options included


def _read(v):
    return read_ipc(v)  # skips a vector this pyarrow cannot read


@pytest.mark.parametrize(
    "v",
    [v for v in _vectors() if v["method"] != "hash_array"],
    ids=lambda v: v["id"],
)
def test_golden_vector_through_prepared_path(v):
    schema, batches = _read(v)
    prepared = starfix.ArrowDigester.prepare(schema, **_options(v))
    if v["method"] == "hash_schema":
        got = prepared.schema_digest
    elif v["method"] == "hash_record_batch":
        (batch,) = batches
        got = prepared.hash_record_batch(batch)
        assert prepared.hash_table(pa.Table.from_batches([batch])) == got
    else:  # hash_stream
        digester = prepared.digester()
        for batch in batches:
            digester.update(batch)
        got = digester.finalize()
        assert prepared.hash_table(pa.Table.from_batches(batches, schema=schema)) == got
    assert got.hex() == v["expected_hash"]


def _batch(metadata=b"m"):
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False, metadata={"unit": metadata}),
            pa.field("name", pa.string()),
        ],
        metadata={"source": "test"},
    )
    return pa.RecordBatch.from_pydict({"id": [1, 2], "name": ["a", None]}, schema=schema)


def test_default_options_equal_the_one_shot_binding():
    batch = _batch()
    prepared = starfix.ArrowDigester.prepare(batch.schema)
    assert prepared.schema_digest == starfix.ArrowDigester.hash_schema(batch.schema)
    assert prepared.hash_record_batch(batch) == starfix.ArrowDigester.hash_table(
        pa.Table.from_batches([batch])
    )


def test_table_in_chunks_and_empty_table():
    batch = _batch()
    prepared = starfix.ArrowDigester.prepare(batch.schema)
    table = pa.Table.from_batches([batch.slice(0, 1), batch.slice(1, 1)])
    assert prepared.hash_table(table) == prepared.hash_record_batch(batch)
    empty = batch.schema.empty_table()
    assert prepared.hash_table(empty) == prepared.hash_record_batch(batch.slice(0, 0))


def test_other_schema_is_refused():
    batch = _batch()
    prepared = starfix.ArrowDigester.prepare(batch.schema)
    other = pa.RecordBatch.from_pydict({"id": [1], "other": ["x"]})
    with pytest.raises(Exception, match="does not match the prepared schema"):
        prepared.hash_record_batch(other)
    with pytest.raises(Exception, match="does not match"):
        prepared.hash_table(pa.Table.from_batches([other]))


def test_metadata_is_part_of_the_rule_only_when_included():
    prepared_from, changed = _batch(b"m"), _batch(b"cm")
    without = starfix.ArrowDigester.prepare(prepared_from.schema)
    assert without.hash_record_batch(changed) == without.hash_record_batch(prepared_from)
    with_metadata = starfix.ArrowDigester.prepare(prepared_from.schema, include_metadata=True)
    with_metadata.hash_record_batch(prepared_from)
    with pytest.raises(Exception, match="does not match the prepared schema"):
        with_metadata.hash_record_batch(changed)
    # schema-level metadata survives the C Data Interface
    no_schema_metadata = prepared_from.replace_schema_metadata(None)
    with pytest.raises(Exception, match="does not match the prepared schema"):
        with_metadata.hash_record_batch(no_schema_metadata)
