"""The float options ``normalize_negative_zero`` and ``normalize_nan`` through the binding (ITL-1252).

With an option on, data hashes exactly as a copy of it whose float values are normalized (-0.0
to +0.0; every NaN to the quiet NaN with a clear sign and no payload) hashes with both options
off (byte-layout spec Sections 6 and 11). Checked on explicit inputs (each option alone and both,
several values in two orders, float16 / float32 / float64, top level, list, large list,
fixed-size list, nested fixed-size list, struct, dictionary, slices) through every call, and as a
property over the harness's seeded tables in float mode (``_harness.build_case(seed, floats)``).

Run after ``maturin develop``: ``pytest python/tests``.
"""

import os
import pickle
import struct

import pyarrow as pa
import pytest

from _harness import _FLOAT_BITS, _FLOAT_LAYOUT, build_case, normalized_float_bits, outcomes, same_outcome
from _support import OPTION_SETS
from starfix import ArrowDigester

OFF = (False, False)
ZERO_ONLY = (True, False)
NAN_ONLY = (False, True)
BOTH = (True, True)
SETTINGS = [ZERO_ONLY, NAN_ONLY, BOTH]
TYPES = {16: pa.float16(), 32: pa.float32(), 64: pa.float64()}

# Indices into _FLOAT_BITS[width]: 1.5, -0.0, payload NaN, +0.0, -NaN, signaling NaN, -2.25, NaN
MIXED = [2, 1, 8, 0, 7, 9, 3, 6]
MIXED_REORDERED = [6, 9, 0, 3, 1, 7, 8, 2]


def _kw(setting):
    return {"normalize_negative_zero": setting[0], "normalize_nan": setting[1]}


def _floats(width, indices, setting=OFF, valid=None):
    """A float array of the values at `indices`, normalized by `setting`."""
    code = _FLOAT_LAYOUT[width][0]
    bits = [normalized_float_bits(width, _FLOAT_BITS[width][i], *setting) for i in indices]
    data = pa.py_buffer(struct.pack(f"<{len(bits)}{code}", *bits))
    if valid is None:
        return pa.Array.from_buffers(TYPES[width], len(bits), [None, data])
    validity = pa.array(valid, pa.bool_()).buffers()[1]
    return pa.Array.from_buffers(TYPES[width], len(bits), [validity, data], valid.count(False))


def _forms(values):
    """`values` (8 values) as a column of each nesting form."""
    t = values.type
    pairs = pa.FixedSizeListArray.from_arrays(values, 2)
    return {
        "top level": values,
        "sliced": values.slice(1, 6),
        "list": pa.ListArray.from_arrays(pa.array([0, 3, 3, 8], pa.int32()), values),
        "large list with a null row": pa.LargeListArray.from_arrays(
            pa.array([0, 2, 4, 8], pa.int64()), values, mask=pa.array([False, True, False])
        ),
        "fixed-size list": pairs,
        "fixed-size list of fixed-size lists": pa.FixedSizeListArray.from_arrays(pairs, 2),
        "struct": pa.StructArray.from_arrays([values], fields=[pa.field("v", t)]),
        # the keys reach -0.0 and a NaN other than the canonical one in both value orders
        "dictionary values": pa.DictionaryArray.from_arrays(pa.array([3, None, 0, 7, 1, 4], pa.int32()), values),
    }


def _all_hashes(column, setting):
    """Every call's hash of a one-column table of `column` under `setting`."""
    table = pa.table({"x": column})
    (batch,) = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
    options = _kw(setting)
    digester = ArrowDigester(table.schema, **options)
    digester.update(batch)
    prepared = ArrowDigester.prepare(table.schema, **options)
    return {
        "hash_table": ArrowDigester.hash_table(table, **options),
        "hash_record_batch": ArrowDigester.hash_record_batch(batch, **options),
        "stream": digester.finalize(),
        "prepared": prepared.hash_table(table),
        "prepared_batch": prepared.hash_record_batch(batch),
        "prepared_digester": _streamed(prepared.digester(), [batch]),
        "hash_array": ArrowDigester.hash_array(column, **options),
        "hash_array_chunked": ArrowDigester.hash_array(pa.chunked_array([column.slice(0, 1), column.slice(1)]), **options),
    }


def _streamed(digester, batches):
    for batch in batches:
        digester.update(batch)
    return digester.finalize()


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("order", [MIXED, MIXED_REORDERED], ids=["order1", "order2"])
@pytest.mark.parametrize("setting", SETTINGS, ids=["zero", "nan", "both"])
def test_an_option_hashes_as_the_normalized_data_with_the_options_off(width, order, setting):
    raw, normalized = _forms(_floats(width, order)), _forms(_floats(width, order, setting))
    for form in raw:
        want = _all_hashes(normalized[form], OFF)
        assert _all_hashes(raw[form], setting) == want, form
        assert _all_hashes(normalized[form], setting) == want, f"{form}: normalized data, option on"
        assert _all_hashes(raw[form], OFF) != want, f"{form}: with the options off the values are hashed as is"


@pytest.mark.parametrize("width", [16, 32, 64])
def test_the_two_options_are_independent(width):
    for order in (MIXED, MIXED_REORDERED):
        for form, column in _forms(_floats(width, order)).items():
            hashes = {s: ArrowDigester.hash_table(pa.table({"x": column}), **_kw(s)) for s in [OFF, *SETTINGS]}
            assert len(set(hashes.values())) == 4, form


@pytest.mark.parametrize("width", [16, 32, 64])
def test_values_under_null_slots_never_count(width):
    valid = [True, False, True, False, False]
    first = _floats(width, [2, 1, 3, 8, 0], valid=valid)  # null slots hold -0.0, payload NaN, +0.0
    second = _floats(width, [2, 0, 3, 0, 7], valid=valid)  # +0.0, +0.0, -NaN
    mask = pa.array([False, False, False, True, False])
    for setting in [OFF, *SETTINGS]:
        for a, b in [
            (first, second),
            (
                pa.StructArray.from_arrays([first], names=["v"], mask=mask),
                pa.StructArray.from_arrays([second], names=["v"], mask=mask),
            ),
        ]:
            assert ArrowDigester.hash_array(a, **_kw(setting)) == ArrowDigester.hash_array(b, **_kw(setting))
            assert ArrowDigester.hash_table(pa.table({"x": a}), **_kw(setting)) == ArrowDigester.hash_table(
                pa.table({"x": b}), **_kw(setting)
            )


def test_other_types_are_untouched():
    columns = {
        "int16": pa.array([-32768, 0, -512], pa.int16()),  # the bits of -0.0 and -NaN as float16
        "int64": pa.array([-(2**63), 0, 7], pa.int64()),
        "uint64": pa.array([0x7FF0_0000_0000_0001, 0x8000_0000_0000_0000, 1], pa.uint64()),
        "binary8": pa.array([struct.pack("<Q", 0x8000_0000_0000_0000), struct.pack("<d", float("-nan"))], pa.binary(8)),
        "decimal": pa.array([0, -1], pa.decimal128(5, 0)),
    }
    for name, column in columns.items():
        off = _all_hashes(column, OFF)
        for setting in SETTINGS:
            assert _all_hashes(column, setting) == off, name


@pytest.mark.parametrize("width", [16, 32, 64])
def test_the_options_are_not_recorded_in_the_hash(width):
    # clean values only (1.5, -2.25, +0.0, +inf, NaN) in two orders: every setting hashes alike
    for order in ([2, 3, 0, 4, 6], [6, 0, 4, 3, 2]):
        column = _floats(width, order)
        for form in (column, pa.ListArray.from_arrays(pa.array([0, 2, 5], pa.int32()), column)):
            schema = pa.schema([pa.field("x", form.type)])
            off = _all_hashes(form, OFF)
            for setting in SETTINGS:
                assert _all_hashes(form, setting) == off
                assert ArrowDigester.hash_schema(schema, **_kw(setting)) == ArrowDigester.hash_schema(schema)
                assert ArrowDigester.prepare(schema, **_kw(setting)).schema_digest == ArrowDigester.hash_schema(schema)


def test_prepared_schema_keeps_the_float_options():
    schema = pa.schema([pa.field("x", pa.float64())])
    table = pa.table({"x": _floats(64, MIXED)}, schema=schema)
    for setting in [OFF, *SETTINGS]:
        prepared = ArrowDigester.prepare(schema, ordered_columns=True, **_kw(setting))
        assert (prepared.normalize_negative_zero, prepared.normalize_nan) == setting
        assert f"normalize_negative_zero={setting[0]}, normalize_nan={setting[1]}" in repr(prepared)
        again = pickle.loads(pickle.dumps(prepared))
        assert (again.normalize_negative_zero, again.normalize_nan) == setting
        expected = ArrowDigester.hash_table(table, ordered_columns=True, **_kw(setting))
        assert prepared.hash_table(table) == expected
        assert again.hash_table(table) == expected


# --- the property over the harness's seeded tables in float mode -----------------------------


def _float_seeds() -> range:
    start, count = os.environ.get("STARFIX_FLOAT_SEEDS", "0:150").split(":")
    return range(int(start), int(start) + int(count))


@pytest.mark.parametrize("setting", SETTINGS, ids=["zero", "nan", "both"])
def test_seeded_tables_hash_with_an_option_as_their_normalized_twin_with_the_options_off(setting):
    """``build_case(seed, OFF)`` and ``build_case(seed, setting)`` have the same structure (types,
    names, metadata, slices, nulls, batches, invalid data); only the float values the option
    normalizes differ. Every call under the first and last option sets must agree: equal digests,
    or a refusal of the same class on both."""
    option_sets = [OPTION_SETS[0], OPTION_SETS[-1]]
    with_option = [{**options, **_kw(setting)} for options in option_sets]
    failures, differing, compared = [], 0, 0
    for seed in _float_seeds():
        raw, normalized = build_case(seed, OFF), build_case(seed, setting)
        ours = outcomes(ArrowDigester, raw, with_option)
        twin = outcomes(ArrowDigester, normalized, option_sets)
        plain = outcomes(ArrowDigester, raw, option_sets)
        for (key, calls), twin_calls, plain_calls in zip(ours.items(), twin.values(), plain.values()):
            for call, outcome in calls.items():
                compared += 1
                if not same_outcome(outcome, twin_calls[call]):
                    failures.append(f"seed {seed} options {key} {call}: {outcome} / {twin_calls[call]}")
                if outcome[0] == "digest" and plain_calls[call] != twin_calls[call]:
                    differing += 1
    assert not failures, f"{len(failures)} of {compared} calls differ:\n" + "\n".join(failures[:15])
    # the seeded tables must exercise the option, not only clean values
    assert differing >= 30, f"only {differing} of {compared} calls had values to normalize"
