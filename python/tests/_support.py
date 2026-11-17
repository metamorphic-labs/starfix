"""Shared inputs for the binding's tests: the golden vectors and seeded random tables."""

from __future__ import annotations

import base64
import decimal
import itertools
import json
import random
from pathlib import Path

import pyarrow as pa

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "format2_v0.5.json"

OPTION_NAMES = ("include_metadata", "ordered_columns", "ordered_struct_fields")
OPTION_SETS = [dict(zip(OPTION_NAMES, values)) for values in itertools.product([False, True], repeat=3)]


def option_id(options: dict) -> str:
    return "".join("1" if options[name] else "0" for name in OPTION_NAMES)


def golden_vectors() -> list[dict]:
    return json.loads(GOLDEN.read_text())["vectors"]


def options_of(vector: dict) -> dict:
    return {name: vector[name] for name in OPTION_NAMES}


def read_ipc(vector: dict) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """The vector's input; skips the test when this pyarrow cannot decode it (older pyarrow
    lacks newer types such as decimal32 or string_view)."""
    try:
        reader = pa.ipc.open_stream(base64.b64decode(vector["ipc_b64"]))
        return reader.schema, list(reader)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        import pytest

        pytest.skip(f"pyarrow {pa.__version__} cannot read vector {vector['id']}: {e}")


def rechunk(table: pa.Table, rng: random.Random) -> pa.Table:
    """The same rows as ``table`` in random chunks (some empty, of different sizes), each a
    slice (non-zero offset) of one combined batch."""
    combined = table.combine_chunks()
    batches = combined.to_batches()
    batch = batches[0] if batches else None
    n = table.num_rows
    cuts = sorted(rng.randint(0, n) for _ in range(rng.randint(1, 4)))
    bounds = [0, *cuts, n]
    chunks = [batch.slice(a, b - a) for a, b in zip(bounds, bounds[1:])] if batch is not None else []
    return pa.Table.from_batches(chunks, schema=table.schema)


# --- seeded random tables ---------------------------------------------------------------

_LEAVES = [
    (pa.int8(), lambda r: r.randint(-128, 127)),
    (pa.int32(), lambda r: r.randint(-(2**31), 2**31 - 1)),
    (pa.int64(), lambda r: r.randint(-(2**63), 2**63 - 1)),
    (pa.uint16(), lambda r: r.randint(0, 2**16 - 1)),
    (pa.float32(), lambda r: r.choice([0.0, -0.0, 1.5, -2.25, float("inf")])),
    (pa.float64(), lambda r: r.uniform(-1e6, 1e6)),
    (pa.bool_(), lambda r: r.random() < 0.5),
    (pa.string(), lambda r: "".join(r.choice("abéz") for _ in range(r.randint(0, 5)))),
    (pa.large_string(), lambda r: "".join(r.choice("xyz") for _ in range(r.randint(0, 4)))),
    (pa.binary(), lambda r: bytes(r.randint(0, 255) for _ in range(r.randint(0, 4)))),
    (pa.date32(), lambda r: r.randint(0, 20000)),
    (pa.timestamp("us", tz="UTC"), lambda r: r.randint(0, 2**40)),
    (pa.decimal128(10, 2), lambda r: decimal.Decimal(r.randint(-(10**9), 10**9)) / 100),
    (pa.binary(3), lambda r: bytes(r.randint(0, 255) for _ in range(3))),
]


def _random_type(rng: random.Random, depth: int):
    """A random type and a generator of one non-null Python value of it."""
    kinds = ["leaf"] * 4 + (["list", "large_list", "struct", "fixed_list", "dict"] if depth < 2 else [])
    kind = rng.choice(kinds)
    if kind == "leaf":
        return rng.choice(_LEAVES)
    if kind == "dict":
        return pa.dictionary(pa.int32(), pa.string()), lambda r: r.choice(["red", "green", "blue"])
    if kind in ("list", "large_list", "fixed_list"):
        item_type, item_value = _random_type(rng, depth + 1)
        if kind == "fixed_list":
            size = rng.randint(1, 3)
            return pa.list_(item_type, size), lambda r: [_maybe_null(r, item_value) for _ in range(size)]
        make = pa.list_ if kind == "list" else pa.large_list
        return make(item_type), lambda r: [_maybe_null(r, item_value) for _ in range(r.randint(0, 3))]
    # struct
    names = rng.sample(["x", "y", "z", "w"], rng.randint(1, 3))
    children = [(name, *_random_type(rng, depth + 1)) for name in names]
    fields = [pa.field(name, t, nullable=True, metadata=_random_metadata(rng)) for name, t, _ in children]
    return pa.struct(fields), lambda r: {name: _maybe_null(r, value) for name, _, value in children}


def _maybe_null(rng: random.Random, value):
    return None if rng.random() < 0.2 else value(rng)


def _random_metadata(rng: random.Random):
    if rng.random() < 0.6:
        return None
    return {f"k{i}": rng.choice(["a", "b", "c"]) for i in range(rng.randint(1, 2))}


def random_table(seed: int) -> pa.Table:
    """A seeded random table: nested types, nullability, field and schema metadata, several
    chunks of different sizes (some empty, some sliced), sometimes no rows at all.

    Only valid Arrow data is returned: pyarrow can build an invalid array (an all-null struct
    over a dictionary child gets an out-of-range key), and such a draw is replaced by the next.
    """
    rng = random.Random(seed)
    while True:
        table = _random_table(rng)
        try:
            table.validate(full=True)
        except pa.ArrowInvalid:
            continue
        return rechunk(table, rng)


def _random_table(rng: random.Random) -> pa.Table:
    num_rows = rng.choice([0, 1, 2, 5, 9, 17])
    fields, columns = [], []
    for name in rng.sample(["a", "b", "c", "d", "e", "f"], rng.randint(1, 5)):
        t, value = _random_type(rng, 0)
        nullable = rng.random() < 0.7
        values = [_maybe_null(rng, value) if nullable else value(rng) for _ in range(num_rows)]
        fields.append(pa.field(name, t, nullable=nullable, metadata=_random_metadata(rng)))
        columns.append(pa.array(values, type=t))
    schema = pa.schema(fields, metadata=_random_metadata(rng))
    return pa.Table.from_arrays(columns, schema=schema)
