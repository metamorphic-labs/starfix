"""Differential test: the binding against starfix-python (ITL-1178).

starfix-python (the pure-Python implementation, released on PyPI as ``starfix``) and this
binding are both imported as ``starfix``, so they cannot share an environment. This test
writes every input as an Arrow IPC stream (which keeps the chunk boundaries), runs
``starfix_python_worker.py`` with the interpreter named by ``STARFIX_PYTHON`` (an environment
with starfix-python installed), and requires every result of the binding to be identical:
schema, table, stream, per-batch and per-column array hashes, under all 8 option combinations.
Exceptions must match by type, and some inputs must raise on both sides. The worker must report
starfix 0.5.0, the version pinned in ``starfix-python-requirements.txt``.

The inputs are the golden vectors, each in several chunks, and seeded random multi-chunk
tables (nested types, dictionaries, metadata at every level, empty and sliced chunks, empty
tables), unregistered ``ARROW:extension:*`` metadata, and inputs an IPC round trip would change
(``_parity_cases``: empty descendants without buffers, registered extension types with
conflicting metadata or in a dictionary, repeated keys below nested dictionaries, field names
containing NUL, metadata that is not UTF-8),
built in each process. Skipped when ``STARFIX_PYTHON`` is unset, unless ``STARFIX_REQUIRE_DIFFERENTIAL`` is
set (as in CI), where it fails instead.
"""

import json
import os
import random
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest

import starfix
from _parity_cases import differential_cases
from _support import OPTION_SETS, dictionary_table, golden_vectors, random_table, read_ipc, rechunk

RANDOM_SEEDS = range(int(os.environ.get("STARFIX_DIFFERENTIAL_SEEDS", "150")))
WORKER = Path(__file__).with_name("starfix_python_worker.py")
STARFIX_PYTHON_VERSION = "0.5.0"  # the pin of starfix-python-requirements.txt


def _outcome(call, *args, **kwargs):
    try:
        return call(*args, **kwargs).hex()
    except Exception as e:  # noqa: BLE001 - outcomes compare by exception type
        return f"raises {type(e).__name__}"


def _stream(schema, batches, options):
    digester = starfix.ArrowDigester(schema, **options)
    for batch in batches:
        digester.update(batch)
    return digester.finalize()


def _binding_results(schema, batches, options):
    D = starfix.ArrowDigester
    table = pa.Table.from_batches(batches, schema=schema)
    array_options = {k: options[k] for k in ("ordered_columns", "ordered_struct_fields")}
    return {
        "hash_schema": _outcome(D.hash_schema, schema, **options),
        "hash_table": _outcome(D.hash_table, table, **options),
        "stream": _outcome(_stream, schema, batches, options),
        "hash_record_batch": [_outcome(D.hash_record_batch, b, **options) for b in batches],
        "hash_array": [_outcome(D.hash_array, table.column(i), **array_options) for i in range(table.num_columns)],
    }


def _cases():
    """(id, table) pairs: every golden input in several chunks, and seeded random tables."""
    for v in golden_vectors():
        try:
            schema, batches = read_ipc(v)
        except pytest.skip.Exception:
            continue  # this pyarrow cannot read the vector
        yield f"golden:{v['id']}", rechunk(pa.Table.from_batches(batches, schema=schema), random.Random(v["id"]))
    for seed in RANDOM_SEEDS:
        yield f"random:{seed}", random_table(seed)
    for seed in range(int(os.environ.get("STARFIX_DIFFERENTIAL_DICTIONARY_SEEDS", "12"))):
        yield f"dictionary:{seed}", dictionary_table(seed)
    # pyarrow's invalid array of a null struct over a dictionary child (an out-of-range key
    # that is never read): both implementations must hash it alike.
    hidden = pa.array([None, None], type=pa.struct([("x", pa.dictionary(pa.int32(), pa.string()))]))
    yield "special:hidden-dictionary-key", pa.table({"s": hidden, "n": pa.array([1, 2])})
    # Extension metadata of a type neither process registers: field metadata, hashed as such.
    unregistered = {"ARROW:extension:name": "starfix.test.unregistered", "ARROW:extension:metadata": "m"}
    for case_id, metadata in [("name-and-metadata", unregistered), ("name-only", {"ARROW:extension:name": "u"})]:
        field = pa.field("e", pa.int64(), metadata=metadata)
        schema = pa.schema([field, pa.field("s", pa.struct([field]))])
        columns = [pa.array([1, 2]), pa.array([{"e": 3}, None], pa.struct([field]))]
        yield f"special:extension-metadata-{case_id}", pa.table(columns, schema=schema)


def _starfix_python():
    interpreter = os.environ.get("STARFIX_PYTHON")
    if not interpreter:
        if os.environ.get("STARFIX_REQUIRE_DIFFERENTIAL"):
            pytest.fail("STARFIX_REQUIRE_DIFFERENTIAL is set but STARFIX_PYTHON is not")
        pytest.skip("set STARFIX_PYTHON to an interpreter with starfix-python installed")
    return interpreter


def test_binding_equals_starfix_python(tmp_path):
    interpreter = _starfix_python()
    cases = []
    tables = {}
    for i, (case_id, table) in enumerate(_cases()):
        path = tmp_path / f"case{i}.arrows"
        with pa.ipc.new_stream(path, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        cases.append({"id": case_id, "ipc": str(path)})
        tables[case_id] = path
    in_process = differential_cases()
    for key in in_process:
        cases.append({"id": f"in-process:{key}", "in_process": key})
        tables[f"in-process:{key}"] = key
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases, "option_sets": OPTION_SETS}))
    results_path = tmp_path / "results.json"
    subprocess.run([interpreter, "-I", str(WORKER), str(manifest), str(results_path)], check=True)
    worker = json.loads(results_path.read_text())
    assert worker["starfix_version"] == STARFIX_PYTHON_VERSION, "the worker must run starfix-python 0.5.0"
    theirs = worker["results"]

    compared = multi_chunk = hashed = raised = 0
    mismatches = []
    for case_id, source in tables.items():
        if isinstance(source, str):
            schema, batches = in_process[source]
        else:
            # The same bytes the worker read: same chunks, same offsets.
            with pa.ipc.open_stream(pa.OSFile(str(source))) as reader:
                schema, batches = reader.schema, list(reader)
        multi_chunk += len(batches) > 1
        for options in OPTION_SETS:
            key = case_id + "/" + "".join("1" if options[k] else "0" for k in sorted(options))
            ours = _binding_results(schema, batches, options)
            compared += 1
            hashed += not ours["hash_table"].startswith("raises")
            raised += ours["hash_schema"].startswith("raises")
            if ours != theirs[key]:
                mismatches.append((key, ours, theirs[key]))
    assert not mismatches, f"{len(mismatches)} of {compared} differ; first: {mismatches[0]}"
    assert compared == len(tables) * 8
    assert multi_chunk > len(tables) // 2, "most cases have several chunks"
    assert hashed > compared * 3 // 4, "most cases hash (an exception on both sides is a weak match)"
    # Refusals are compared too: with include_metadata, the 3 nested-dictionary and 6 extension
    # conflict cases raise on both sides.
    assert raised >= (3 + 6) * 4, "refusals are compared"
