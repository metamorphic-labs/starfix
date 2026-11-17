"""Inputs on which the binding differed from starfix-python 0.5.0 (Copilot review of c85c512).

starfix-python 0.5.0 (PyPI) hashes a nonempty batch whose empty descendants have no offsets
buffers as the same batch with allocated buffers; with ``include_metadata`` it refuses explicit
``ARROW:extension:*`` metadata that conflicts with a registered extension type, and repeated
metadata keys below nested dictionaries. The binding raised, hashed and hashed. These inputs
are also in ``test_differential.py`` (built in process by ``_parity_cases``); the tests here pin
the expected outcomes without starfix-python. A prepared schema (not in 0.5.0) refuses an empty
table of another schema also when pyarrow cannot build an empty batch of it.
"""

import ctypes

import pyarrow as pa
import pytest

from _parity_cases import (
    NESTED_LISTS,
    POINT_NAME,
    POINT_SERIALIZED,
    Point,
    bufferless_empty,
    bufferless_nested,
    extension_conflicts,
    nested_dictionary_schema,
    point_batch,
    without_metadata,
)
from starfix import ArrowDigester

# --- empty descendants without buffers ----------------------------------------------------


def _bufferless_cases():
    """(id, bufferless array, the same values with allocated buffers)."""
    for t in NESTED_LISTS:
        yield str(t), bufferless_nested(t), pa.array([[]], t)
    # A dictionary column whose empty dictionary has no offsets buffer (every key null).
    # starfix-python 0.5.0 raises TypeError on it; the binding hashes it as allocated.
    keys = pa.array([None, None], pa.int32())
    yield (
        "dictionary-of-empty-string",
        pa.DictionaryArray.from_arrays(keys, bufferless_empty(pa.string())),
        pa.DictionaryArray.from_arrays(keys, pa.array([], pa.string())),
    )


BUFFERLESS = list(_bufferless_cases())


def _every_path(array: pa.Array) -> dict[str, bytes]:
    """The digest of `array` as a column through every entry point."""
    batch = pa.RecordBatch.from_arrays([array], names=["c"])
    table = pa.Table.from_batches([batch, batch])
    prepared = ArrowDigester.prepare(batch.schema)
    digester = ArrowDigester(batch.schema)
    digester.update(batch)
    from_prepared = prepared.digester()
    from_prepared.update(batch)
    return {
        "hash_array": ArrowDigester.hash_array(array),
        "hash_array_chunked": ArrowDigester.hash_array(pa.chunked_array([array, array])),
        "hash_record_batch": ArrowDigester.hash_record_batch(batch),
        "hash_table": ArrowDigester.hash_table(table),
        "stream": digester.finalize(),
        "prepared.hash_record_batch": prepared.hash_record_batch(batch),
        "prepared.hash_table": prepared.hash_table(table),
        "prepared.digester": from_prepared.finalize(),
    }


@pytest.mark.parametrize(
    ("bufferless", "allocated"),
    [c[1:] for c in BUFFERLESS],
    ids=[c[0] for c in BUFFERLESS],
)
def test_empty_descendants_without_offsets_buffers_hash_as_allocated_ones(bufferless, allocated):
    bufferless.validate(full=True)
    assert bufferless.equals(allocated)
    assert _every_path(bufferless) == _every_path(allocated)


# --- a prepared schema checks an empty table's schema -------------------------------------


def _imported_type(format_string: bytes) -> pa.DataType:
    """A type from its C Data Interface format: pyarrow has no factory for some types."""

    class ArrowSchema(ctypes.Structure):
        pass

    release_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowSchema))
    ArrowSchema._fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", release_type),
        ("private_data", ctypes.c_void_p),
    ]

    @release_type
    def release(schema):
        schema.contents.release = release_type()

    c_schema = ArrowSchema(format_string, b"", None, 2, 0, None, None, release, None)
    return pa.DataType._import_from_c(ctypes.addressof(c_schema))


@pytest.mark.parametrize("format_string", [b"tiM", b"tiD"], ids=["month_interval", "day_time_interval"])
def test_an_empty_table_pyarrow_cannot_batch_is_checked_by_a_prepared_schema(
    format_string,
):
    schema = pa.schema([("z", _imported_type(format_string))])
    with pytest.raises((KeyError, NotImplementedError, pa.ArrowException)):
        pa.nulls(0, schema.field("z").type)  # why this case is special
    table = pa.Table.from_batches([], schema=schema)
    with pytest.raises(ValueError, match="Table schema does not match the prepared schema"):
        ArrowDigester.prepare(pa.schema([("a", pa.int64())])).hash_table(table)
    for refused in (ArrowDigester.prepare(schema).hash_table, ArrowDigester.hash_table):
        with pytest.raises(ValueError, match="is not supported"):
            refused(table)


# --- explicit ARROW:extension:* metadata conflicting with a registered extension type -----

CONFLICTS = list(extension_conflicts())


@pytest.mark.parametrize(
    ("schema", "path", "key", "explicit", "own"),
    [c[1:] for c in CONFLICTS],
    ids=[c[0] for c in CONFLICTS],
)
def test_conflicting_extension_metadata_is_refused_with_include_metadata(schema, path, key, explicit, own):
    message = (
        f"Field {path!r}: {key}={explicit!r} is set, conflicting with its extension type ({own!r}): "
        "ambiguous, rejected with include_metadata"
    )
    batch = point_batch(schema)
    for call in (
        lambda: ArrowDigester.hash_schema(schema, include_metadata=True),
        lambda: ArrowDigester.prepare(schema, include_metadata=True),
        lambda: ArrowDigester(schema, include_metadata=True),
        lambda: ArrowDigester.hash_record_batch(batch, include_metadata=True),
        lambda: ArrowDigester.hash_table(pa.Table.from_batches([batch]), include_metadata=True),
    ):
        with pytest.raises(ValueError) as refused:
            call()
        assert str(refused.value) == message
    # Without the option metadata is not read: the field hashes as without the metadata.
    plain = pa.schema([without_metadata(schema.field(0))])
    assert ArrowDigester.hash_schema(schema) == ArrowDigester.hash_schema(plain)
    assert ArrowDigester.hash_record_batch(batch) == ArrowDigester.hash_record_batch(point_batch(plain))


def test_a_conflicting_batch_is_refused_by_a_digester_of_the_plain_schema():
    _, schema, *_ = CONFLICTS[0]
    plain = pa.schema([without_metadata(schema.field(0))])
    good = point_batch(plain)
    prepared = ArrowDigester.prepare(plain, include_metadata=True)
    for digester in (ArrowDigester(plain, include_metadata=True), prepared.digester()):
        with pytest.raises(ValueError, match="conflicting with its extension type"):
            digester.update(point_batch(schema))
        digester.update(good)
        assert digester.finalize() == ArrowDigester.hash_record_batch(good, include_metadata=True)
    with pytest.raises(ValueError, match="conflicting with its extension type"):
        prepared.hash_record_batch(point_batch(schema))


def test_extension_metadata_equal_to_the_types_own_is_accepted():
    own = {
        "ARROW:extension:name": POINT_NAME,
        "ARROW:extension:metadata": POINT_SERIALIZED,
    }
    explicit = pa.schema([pa.field("p", Point(), metadata=own)])
    implicit = pa.schema([pa.field("p", Point())])
    assert ArrowDigester.hash_schema(explicit, include_metadata=True) == ArrowDigester.hash_schema(
        implicit, include_metadata=True
    )
    assert ArrowDigester.hash_record_batch(point_batch(explicit), include_metadata=True) == (
        ArrowDigester.hash_record_batch(point_batch(implicit), include_metadata=True)
    )


# --- repeated metadata keys below nested dictionaries -------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_repeated_keys_below_nested_dictionaries_are_refused(depth):
    schema = nested_dictionary_schema(depth)
    for call in (ArrowDigester.hash_schema, ArrowDigester.prepare, ArrowDigester):
        with pytest.raises(ValueError) as refused:
            call(schema, include_metadata=True)
        assert str(refused.value) == (
            "Field 'd/x': Duplicate metadata key 'k' on the field: ambiguous, rejected with include_metadata"
        )
    ArrowDigester.hash_schema(schema)  # without the option, metadata is not read
