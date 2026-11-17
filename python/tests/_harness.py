"""Seeded inputs for the differential harness against starfix-python 0.5.0 (ITL-1178).

``build_case(seed)`` returns a table (schema and record batches) that depends only on the seed
and the pyarrow version. Both processes of the harness, this package's tests and the
starfix-python worker, build it in process: no IPC round trip, so layouts that IPC would change
(offsets at every depth, bufferless empty arrays, registered extension types) reach both
implementations as they are. ``outcomes`` runs the same calls on either implementation and
``same_outcome`` compares them. Imports pyarrow and ``_parity_cases`` only.

The axes, drawn together in each case:
- types: every leaf type of the format, list, large_list, fixed_size_list, struct, map,
  dictionaries (of leaves, structs, lists, extensions), registered extension types over any
  storage, and the types whose data or schema the format refuses (null, interval, union);
- names: plain, non-ASCII, empty, containing "/", containing NUL, repeated;
- metadata: UTF-8, not UTF-8, repeated keys, explicit ARROW:extension:* (conflicting or not);
- layout: slices (offsets at every depth), bufferless empty arrays, empty and many batches;
- invalid data: a visible out-of-range dictionary key, a null in a non-nullable field.
"""

from __future__ import annotations

import decimal
import random
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise

import pyarrow as pa

from _parity_cases import bufferless_empty

GENERATOR_VERSION = 1


class HarnessExtension(pa.ExtensionType):
    """One registered extension name over any storage; the serialization is its parameter."""

    def __init__(self, storage_type: pa.DataType, parameter: bytes = b""):
        self._parameter = parameter
        super().__init__(storage_type, "starfix.harness.extension")

    def __arrow_ext_serialize__(self):
        return self._parameter

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls(storage_type, serialized)


try:
    pa.register_extension_type(HarnessExtension(pa.int64()))
except pa.ArrowKeyError:
    pass  # registered by an earlier import


@dataclass
class Case:
    seed: int
    schema: pa.Schema
    batches: list[pa.RecordBatch]
    tags: set[str] = field(default_factory=set)


# --- types --------------------------------------------------------------------------------

_TIME_ZONES = [None, None, "UTC", "+05:30", "Europe/Paris", "UTC\x00junk"]
_PLAIN_NAMES = ["a", "b", "c", "d", "e", "f", "x", "y", "z", "id", "value", "é"]
_SPECIAL_NAMES = ["", "a/b", "n\x00ul", "a\x00", "ü\x00z"]


def _leaf_types() -> list[pa.DataType]:
    return [
        pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
        pa.float32(), pa.float64(), pa.bool_(), pa.string(), pa.large_string(), pa.binary(), pa.large_binary(),
        pa.binary(3), pa.date32(), pa.date64(), pa.time32("s"), pa.time32("ms"), pa.time64("us"), pa.time64("ns"),
        pa.duration("s"), pa.duration("ns"), pa.decimal128(9, 2), pa.decimal128(30, 0), pa.decimal256(40, 5),
    ]  # fmt: skip


class _Generator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.tags: set[str] = set()

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    # names and metadata

    def names(self, count: int) -> list[str]:
        """Distinct names, some special (empty, "/", NUL); rarely a repeated one (refused)."""
        names = self.rng.sample(_PLAIN_NAMES, count)
        for i in range(count):
            if self.chance(0.12):
                names[i] = self.rng.choice(_SPECIAL_NAMES)
        if count > 1 and self.chance(0.04):
            names[-1] = names[0]
        if len(set(names)) < count:
            self.tags.add("repeated-name")
        if any("\x00" in name for name in names):
            self.tags.add("name-nul")
        return names

    def metadata(self, extension: pa.DataType | None = None):
        if not self.chance(0.3):
            return None
        kind = self.rng.choice(["utf8"] * 15 + ["binary", "repeated", "extension"])
        self.tags.add(f"metadata-{kind}")
        if kind == "binary":
            return {b"k": b"\xff\xfe", b"u": b"v"}
        if kind == "repeated":
            return pa.KeyValueMetadata([(b"k", b"1"), (b"k", b"2")])
        if kind == "extension":
            name = "starfix.harness.extension" if extension is not None and self.chance(0.5) else "other"
            return {"ARROW:extension:name": name, "ARROW:extension:metadata": self.rng.choice(["", "p"])}
        return {f"k{i}": self.rng.choice(["a", "é", "1", ""]) for i in range(self.rng.randint(1, 2))}

    # types

    def data_type(self, depth: int) -> pa.DataType:
        kinds = ["leaf"] * 6
        if depth < 3:
            kinds += ["list", "large_list", "fixed_size_list", "struct", "struct", "dictionary", "extension"]
            kinds += ["map"] if self.chance(0.1) else []  # its data is refused: the whole case is
        kinds += ["refused"] if self.chance(0.05) else []
        kind = self.rng.choice(kinds)
        if kind == "leaf":
            if self.chance(0.08):
                tz = self.rng.choice(_TIME_ZONES)
                if tz and "\x00" in tz:
                    self.tags.add("time-zone-nul")
                return pa.timestamp(self.rng.choice(["s", "ms", "us", "ns"]), tz=tz)
            return self.rng.choice(_leaf_types())
        if kind == "refused":
            self.tags.add("refused-type")
            return self.rng.choice([pa.null(), pa.month_day_nano_interval()])
        if kind in ("list", "large_list"):
            item = self.field(depth + 1, name=self.rng.choice(["item"] * 6 + ["element", "i\x00", ""]))
            return (pa.list_ if kind == "list" else pa.large_list)(item)
        if kind == "fixed_size_list":
            return pa.list_(self.field(depth + 1, name="item"), self.rng.randint(0, 3))
        if kind == "struct":
            count = self.rng.randint(1, 3)
            return pa.struct([self.field(depth + 1, name=n) for n in self.names(count)])
        if kind == "map":
            self.tags.add("map")
            key = pa.field("key", self.rng.choice([pa.string(), pa.int32()]), nullable=False)
            return pa.map_(key, self.field(depth + 1, name="value"))
        if kind == "dictionary":
            self.tags.add("dictionary")
            value = self.rng.choice(
                [pa.string(), pa.int64(), pa.binary()] * 3 + [self.data_type(depth + 1), self.extension_type(depth + 1)]
            )
            index = self.rng.choice([pa.int8(), pa.int16(), pa.int32(), pa.uint8()])
            return pa.dictionary(index, value)
        return self.extension_type(depth)

    def extension_type(self, depth: int) -> pa.DataType:
        self.tags.add("extension")
        storage = self.rng.choice(
            [pa.int64(), pa.string(), pa.struct([("x", pa.int64())]), pa.list_(pa.int32())]
            + ([pa.dictionary(pa.int8(), pa.string())] if depth < 3 else [])
        )
        parameter = self.rng.choice([b"", b"p1", b"p2", b"\xff", b"a\x00b"])
        if parameter == b"\xff":
            self.tags.add("extension-binary-serialization")
        return HarnessExtension(storage, parameter)

    def field(self, depth: int, name: str) -> pa.Field:
        t = self.data_type(depth)
        nullable = self.chance(0.8) or pa.types.is_null(t)
        extension = t if isinstance(t, pa.BaseExtensionType) else None
        return pa.field(name, t, nullable=nullable, metadata=self.metadata(extension))

    # arrays

    def validity(self, length: int, nullable: bool | None) -> tuple[list[bool], int]:
        """Row validity; `nullable` False sometimes gets a null anyway (refused by both), None
        never (map keys: pyarrow aborts on a null key)."""
        if nullable is None or (not nullable and not self.chance(0.03)):
            return [True] * length, 0
        if not nullable:
            self.tags.add("null-in-non-nullable")
        valid = [not self.chance(0.25) for _ in range(length)]
        return valid, valid.count(False)

    def array(self, t: pa.DataType, length: int, nullable: bool | None = True, top: bool = False) -> pa.Array:
        """An array of `t`, sometimes a slice of a longer one (an offset at this depth). Only a
        top-level dictionary column, never sliced, may get an invalid key: one outside a slice
        is not part of the array (the binding hashes the array, starfix-python 0.5.0 raises)."""
        if top and pa.types.is_dictionary(t):
            valid, _ = self.validity(length, nullable)
            return self.dictionary_array(t, length, valid, invalid_key=True)
        if length == 0 and self.chance(0.15) and _bufferless_ok(t):
            self.tags.add("bufferless-empty")
            return bufferless_empty(t)
        if self.chance(0.25):
            self.tags.add("offset")
            before, after = self.rng.randint(0, 3), self.rng.randint(0, 2)
            return self._array(t, before + length + after, nullable).slice(before, length)
        return self._array(t, length, nullable)

    def _array(self, t: pa.DataType, length: int, nullable: bool | None) -> pa.Array:
        valid, null_count = self.validity(length, nullable)
        validity = pa.array(valid, pa.bool_()).buffers()[1] if null_count else None
        if isinstance(t, pa.BaseExtensionType):
            return pa.ExtensionArray.from_storage(t, self.array(t.storage_type, length, nullable))
        if pa.types.is_dictionary(t):
            return self.dictionary_array(t, length, valid)
        if pa.types.is_struct(t):
            children = [self.array(t.field(i).type, length, t.field(i).nullable) for i in range(t.num_fields)]
            return pa.Array.from_buffers(t, length, [validity], null_count, children=children)
        if pa.types.is_fixed_size_list(t):
            child = self.array(t.value_type, length * t.list_size, t.value_field.nullable)
            return pa.Array.from_buffers(t, length, [validity], null_count, children=[child])
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_map(t):
            sizes = [self.rng.randint(0, 3) if ok else self.rng.choice([0, 0, 1]) for ok in valid]
            offsets = [0]
            for size in sizes:
                offsets.append(offsets[-1] + size)
            width = pa.int64() if pa.types.is_large_list(t) else pa.int32()
            offsets_buffer = pa.array(offsets, width).buffers()[1]
            if pa.types.is_map(t):
                keys = self.array(t.key_type, offsets[-1], None)
                items = self.array(t.item_type, offsets[-1], t.item_field.nullable)
                entries = pa.StructArray.from_arrays([keys, items], fields=[t.key_field, t.item_field])
                return pa.Array.from_buffers(t, length, [validity, offsets_buffer], null_count, children=[entries])
            child = self.array(t.value_type, offsets[-1], t.value_field.nullable)
            return pa.Array.from_buffers(t, length, [validity, offsets_buffer], null_count, children=[child])
        if pa.types.is_null(t):
            return pa.nulls(length)
        return pa.array([self.value(t) if ok else None for ok in valid], t)

    def dictionary_array(self, t: pa.DataType, length: int, valid: list[bool], invalid_key: bool = False) -> pa.Array:
        size = self.rng.randint(1, 4)
        dictionary = self.array(t.value_type, size, True)
        keys = [self.rng.randrange(size) if ok else None for ok in valid]
        if invalid_key and length and self.chance(0.1):
            self.tags.add("invalid-dictionary-key")
            row = self.rng.randrange(length)
            keys[row] = size + 2  # visible and out of range: invalid data
            indices = pa.array([0 if k is None else k for k in keys], t.index_type)
            null_validity = pa.array([k is not None for k in keys], pa.bool_()).buffers()[1]
            null_count = keys.count(None)
            buffers = [null_validity if null_count else None, indices.buffers()[1]]
            return pa.DictionaryArray.from_buffers(t, length, buffers, dictionary, null_count=null_count)
        return pa.DictionaryArray.from_arrays(pa.array(keys, t.index_type), dictionary)

    def value(self, t: pa.DataType):
        r = self.rng
        if pa.types.is_boolean(t):
            return r.random() < 0.5
        if pa.types.is_integer(t):
            bits = t.bit_width
            low, high = (
                (0, 2**bits - 1) if pa.types.is_unsigned_integer(t) else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            )
            return r.choice([low, high, 0, 1, r.randint(low, high)])
        if pa.types.is_floating(t):
            return r.choice([0.0, -0.0, 1.5, -2.25, float("inf"), float("-inf"), float("nan"), r.uniform(-1e9, 1e9)])
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return "".join(r.choice("ab é\x00z") for _ in range(r.randint(0, 5)))
        if pa.types.is_fixed_size_binary(t):
            return bytes(r.randint(0, 255) for _ in range(t.byte_width))
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            return bytes(r.randint(0, 255) for _ in range(r.randint(0, 4)))
        if pa.types.is_date32(t):
            return r.randint(-100_000, 100_000)
        if pa.types.is_date64(t):
            return r.randint(-100_000, 100_000) * 86_400_000
        if pa.types.is_time32(t):
            return r.randint(0, 86_399 if t.unit == "s" else 86_399_999)
        if pa.types.is_time64(t):
            return r.randint(0, 86_399_999_999 if t.unit == "us" else 86_399_999_999_999)
        if pa.types.is_timestamp(t) or pa.types.is_duration(t):
            return r.randint(-(2**40), 2**40)
        if pa.types.is_decimal(t):
            digits = min(t.precision, 18)
            return decimal.Decimal(r.randint(-(10**digits) + 1, 10**digits - 1)).scaleb(-t.scale)
        if pa.types.is_interval(t):
            return (r.randint(-5, 5), r.randint(-5, 5), r.randint(-(10**9), 10**9))
        raise NotImplementedError(str(t))

    # cases

    def case(self, seed: int) -> Case:
        count = self.rng.randint(1, 4)
        fields = [self.field(0, name=name) for name in self.names(count)]
        schema = pa.schema(fields, metadata=self.metadata())
        rows = self.rng.choice([0, 1, 2, 3, 5, 9, 20])
        columns = [self.array(f.type, rows, f.nullable, top=True) for f in fields]
        batch = pa.RecordBatch.from_arrays(columns, schema=schema)
        batches = []
        if rows or self.chance(0.7):
            cuts = sorted(self.rng.randint(0, rows) for _ in range(self.rng.randint(0, 3)))
            bounds = [0, *cuts, rows]
            batches = [batch.slice(a, b - a) for a, b in pairwise(bounds)]
            if len(batches) > 1:
                self.tags.add("multi-chunk")
        else:
            self.tags.add("no-batches")
        return Case(seed, schema, batches, self.tags)


def _bufferless_ok(t: pa.DataType) -> bool:
    """Whether `bufferless_empty` builds `t`: leaves with data, lists and structs of them (a
    malformed request aborts pyarrow, so the check is exact)."""
    if pa.types.is_struct(t):
        return all(_bufferless_ok(t.field(i).type) for i in range(t.num_fields))
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _bufferless_ok(t.value_type)
    nested = t.num_fields or isinstance(t, (pa.DictionaryType, pa.BaseExtensionType))
    return not nested and not pa.types.is_null(t)


def build_case(seed: int) -> Case:
    """The case of `seed` (deterministic for a seed, GENERATOR_VERSION and pyarrow version)."""
    return _Generator(random.Random(f"starfix-harness-{GENERATOR_VERSION}-{seed}")).case(seed)


# --- running and comparing ----------------------------------------------------------------


def classify(error: BaseException) -> str:
    """The equivalence class of an exception: refusals, invalid data and wrong types compare
    equal across implementations whatever the exception type; anything else is a crash."""
    if not isinstance(error, Exception):
        return "crash"  # PanicException, KeyboardInterrupt, ...
    if isinstance(error, pa.ArrowException) or str(error).startswith("Invalid Arrow data"):
        return "invalid-data"
    if isinstance(error, (IndexError, KeyError)):
        return "invalid-data"
    if isinstance(error, TypeError):
        return "type"
    if isinstance(error, ValueError):
        return "refusal"
    return "crash"


def _outcome(call) -> list:
    try:
        return ["digest", call().hex()]
    except BaseException as error:  # noqa: BLE001 - every outcome is compared
        return ["raise", classify(error), type(error).__name__, str(error)[:160]]


def outcomes(digester_class, case: Case, option_sets: list[dict]) -> dict:
    """{option key: {call: outcome}} for one case: hash_schema, hash_table, a stream of the
    batches, hash_record_batch per batch and hash_array per column."""
    table = pa.Table.from_batches(case.batches, schema=case.schema)
    return {
        "".join("1" if options[k] else "0" for k in sorted(options)): _calls(digester_class, case, table, options)
        for options in option_sets
    }


def _calls(D, case: Case, table: pa.Table, options: dict) -> dict:
    array_options = {k: options[k] for k in ("ordered_columns", "ordered_struct_fields")}

    def stream():
        digester = D(case.schema, **options)
        for batch in case.batches:
            digester.update(batch)
        return digester.finalize()

    calls = {
        "hash_schema": _outcome(lambda: D.hash_schema(case.schema, **options)),
        "hash_table": _outcome(lambda: D.hash_table(table, **options)),
        "stream": _outcome(stream),
    }
    for i, batch in enumerate(case.batches):
        calls[f"hash_record_batch[{i}]"] = _outcome(partial(D.hash_record_batch, batch, **options))
    for i in range(table.num_columns):
        calls[f"hash_array[{i}]"] = _outcome(partial(D.hash_array, table.column(i), **array_options))
    return calls


def same_outcome(ours: list, theirs: list) -> bool:
    """Equal digests, or both raise in the same class (a crash never matches)."""
    if ours[0] == "digest" or theirs[0] == "digest":
        return ours == theirs
    return ours[1] == theirs[1] != "crash"
