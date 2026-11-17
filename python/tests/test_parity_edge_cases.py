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
    binary_metadata_cases,
    dictionary_of_point_batches,
    extension_conflicts,
    nested_dictionary_schema,
    nul_name_batch,
    nul_name_cases,
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


def test_a_conflicting_batch_is_refused_by_a_prepared_schema_of_its_exported_twin():
    # pyarrow exports the conflicting field as its storage type with the explicit metadata: the
    # export of a valid schema, so a same-schema check on exports would accept the conflicting
    # batch (a finding on starfix-python PR 13; 0.5.0's one-shot calls refuse the batch).
    _, schema, _, key, explicit, _ = CONFLICTS[0]
    metadata = {key: explicit, "ARROW:extension:metadata": POINT_SERIALIZED}
    twin = pa.schema([pa.field("p", Point().storage_type, metadata=metadata)])
    assert pa.Schema._import_from_c_capsule(schema.__arrow_c_schema__()).equals(twin, check_metadata=True)
    prepared = ArrowDigester.prepare(twin, include_metadata=True)
    batch = point_batch(schema)
    for call in (
        lambda: prepared.hash_record_batch(batch),
        lambda: prepared.hash_table(pa.Table.from_batches([batch])),
        lambda: prepared.digester().update(batch),
    ):
        with pytest.raises(ValueError, match="conflicting with its extension type"):
            call()
    # The twin itself is valid: its batches hash, through the prepared path as one-shot.
    good = pa.RecordBatch.from_arrays([batch.column(0).storage], schema=twin)
    assert prepared.hash_record_batch(good) == ArrowDigester.hash_record_batch(good, include_metadata=True)
    assert prepared.hash_table(pa.Table.from_batches([good])) == prepared.hash_record_batch(good)


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


# --- field names containing NUL -----------------------------------------------------------

NUL_NAMES = nul_name_cases()
# List item and map key/value names do not enter the hash (in starfix-python 0.5.0 either).
NAMES_NOT_HASHED = {"list-item", "map-key-and-value"}
# Data of a map is not hashed (byte-layout spec Section 8); cut, these names repeat.
DATA_NOT_HASHED = {"map-key-and-value"}
CUT_NAMES_REPEAT = {"top-beside-its-prefix", "struct"}


def _hashes(schema: pa.Schema, batch: pa.RecordBatch, options: dict) -> dict[str, object]:
    """Every hash of the batch, or the exception type of the call."""

    def outcome(call):
        try:
            return call()
        except ValueError as error:
            return type(error)

    prepared = ArrowDigester.prepare(schema, **options)

    table = pa.Table.from_batches([batch, batch])

    def streamed(digester):
        for each in table.to_batches():
            digester.update(each)
        return digester.finalize()

    array_options = {k: v for k, v in options.items() if k != "include_metadata"}
    return {
        "hash_schema": outcome(lambda: ArrowDigester.hash_schema(schema, **options)),
        "prepared.schema_digest": prepared.schema_digest,
        "hash_record_batch": outcome(lambda: ArrowDigester.hash_record_batch(batch, **options)),
        "prepared.hash_record_batch": outcome(lambda: prepared.hash_record_batch(batch)),
        "hash_table": outcome(lambda: ArrowDigester.hash_table(table, **options)),
        "prepared.hash_table": outcome(lambda: prepared.hash_table(table)),
        "stream": outcome(lambda: streamed(ArrowDigester(schema, **options))),
        "prepared.digester": outcome(lambda: streamed(prepared.digester())),
        "hash_array": [outcome(lambda c=c: ArrowDigester.hash_array(c, **array_options)) for c in batch.columns],
    }


@pytest.mark.parametrize("case", sorted(NUL_NAMES))
@pytest.mark.parametrize(
    "options", [{}, {"include_metadata": True, "ordered_struct_fields": True}], ids=["default", "options"]
)
def test_names_containing_nul_are_hashed_in_full(case, options):
    schema, cut, rows = NUL_NAMES[case]
    batch = nul_name_batch(schema, rows)
    hashes = _hashes(schema, batch, options)
    # Every call hashes, except the data of a map, which no implementation hashes.
    data_calls = [k for k in hashes if k not in ("hash_schema", "prepared.schema_digest", "hash_array")]
    assert isinstance(hashes["hash_schema"], bytes) and isinstance(hashes["prepared.schema_digest"], bytes)
    if case in DATA_NOT_HASHED:
        assert all(hashes[k] is ValueError for k in data_calls), hashes
        assert all(h is ValueError for h in hashes["hash_array"]), hashes
    else:
        assert all(isinstance(hashes[k], bytes) for k in data_calls), hashes
        assert all(isinstance(h, bytes) for h in hashes["hash_array"]), hashes
    # One-shot, streaming and prepared agree (the prepared and streaming paths decide the names
    # once per schema, the one-shot calls on every call).
    for one_shot, others in [
        ("hash_schema", ["prepared.schema_digest"]),
        ("hash_record_batch", ["prepared.hash_record_batch"]),
        ("hash_table", ["prepared.hash_table", "stream", "prepared.digester"]),
    ]:
        assert all(hashes[other] == hashes[one_shot] for other in others), (case, one_shot, hashes)
    # The names are not cut at the NUL. Where the cut names repeat, the cut schema is refused;
    # elsewhere it hashes, the same as the full names only where names are not hashed.
    if case in CUT_NAMES_REPEAT:
        with pytest.raises(ValueError, match="Ambiguous"):
            ArrowDigester.hash_schema(cut, **options)
        return
    cut_digest = ArrowDigester.hash_schema(cut, **options)
    if case in NAMES_NOT_HASHED:
        assert hashes["hash_schema"] == cut_digest
    else:
        assert hashes["hash_schema"] != cut_digest


def test_a_prepared_schema_tells_names_cut_at_nul_apart():
    # Only "top" has data and cut names that do not repeat (the others are refused or not hashed).
    schema, cut, rows = NUL_NAMES["top"]
    cut_prepared = ArrowDigester.prepare(cut)
    with pytest.raises(ValueError, match="does not match"):
        cut_prepared.hash_record_batch(nul_name_batch(schema, rows))
    with pytest.raises(ValueError, match="does not match"):
        ArrowDigester.prepare(schema).hash_record_batch(nul_name_batch(cut, rows))


# --- metadata that is not UTF-8 ------------------------------------------------------------

BINARY_METADATA = binary_metadata_cases()


@pytest.mark.parametrize("case", sorted(BINARY_METADATA))
def test_metadata_that_is_not_utf8_is_ignored_unless_hashed(case):
    batch = BINARY_METADATA[case]
    plain = pa.schema([without_metadata(field) for field in batch.schema])
    plain_batch = pa.RecordBatch.from_arrays(batch.columns, schema=plain)
    table = pa.Table.from_batches([batch, batch])
    # Without include_metadata it is not read (it was decoded, and refused, on import).
    assert ArrowDigester.hash_schema(batch.schema) == ArrowDigester.hash_schema(plain)
    expected = ArrowDigester.hash_record_batch(plain_batch)
    assert ArrowDigester.hash_record_batch(batch) == expected
    assert ArrowDigester.prepare(batch.schema).hash_record_batch(batch) == expected
    assert ArrowDigester.hash_table(table) == ArrowDigester.hash_table(pa.Table.from_batches([plain_batch] * 2))
    assert ArrowDigester.hash_array(batch.column(0)) == ArrowDigester.hash_array(plain_batch.column(0))
    # With it, starfix-python's UnicodeDecodeError.
    for call in (
        lambda: ArrowDigester.hash_schema(batch.schema, include_metadata=True),
        lambda: ArrowDigester.prepare(batch.schema, include_metadata=True),
        lambda: ArrowDigester(batch.schema, include_metadata=True),
        lambda: ArrowDigester.hash_record_batch(batch, include_metadata=True),
        lambda: ArrowDigester.hash_table(table, include_metadata=True),
    ):
        with pytest.raises(UnicodeDecodeError):
            call()


# --- a dictionary of an extension type ----------------------------------------------------

DICTIONARY_OF_POINT = dictionary_of_point_batches()


@pytest.mark.parametrize("case", sorted(DICTIONARY_OF_POINT))
def test_a_dictionary_of_an_extension_type_hashes_its_extension_metadata(case):
    # The extension's keys are exported on the dictionary's value type, and were lost on import:
    # with include_metadata the field hashed as a dictionary of the storage type.
    batch = DICTIONARY_OF_POINT[case]
    storage = pa.schema([_dictionary_of_storage(field) for field in batch.schema])
    storage_batch = pa.RecordBatch.from_arrays(
        [column.cast(field.type) for column, field in zip(batch.columns, storage)], schema=storage
    )
    with_metadata = {"include_metadata": True}
    assert ArrowDigester.hash_schema(batch.schema) == ArrowDigester.hash_schema(storage)
    assert ArrowDigester.hash_schema(batch.schema, **with_metadata) != ArrowDigester.hash_schema(
        storage, **with_metadata
    )
    digest = ArrowDigester.hash_record_batch(batch, **with_metadata)
    assert digest != ArrowDigester.hash_record_batch(storage_batch, **with_metadata)
    prepared = ArrowDigester.prepare(batch.schema, **with_metadata)
    assert prepared.hash_record_batch(batch) == digest
    assert prepared.hash_table(pa.Table.from_batches([batch])) == digest
    with pytest.raises(ValueError, match="does not match"):
        ArrowDigester.prepare(storage, **with_metadata).hash_record_batch(batch)


def _dictionary_of_storage(field: pa.Field) -> pa.Field:
    """The field with each dictionary of an extension type made a dictionary of its storage."""
    t = field.type
    if pa.types.is_dictionary(t) and isinstance(t.value_type, pa.BaseExtensionType):
        t = pa.dictionary(t.index_type, t.value_type.storage_type)
    elif pa.types.is_struct(t):
        t = pa.struct([_dictionary_of_storage(child) for child in t])
    return field.with_type(t)
