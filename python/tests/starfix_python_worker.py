"""Hash the cases of a differential run with starfix-python (run by ``test_differential.py``).

Runs in its own environment, where ``import starfix`` is the pure-Python starfix-python
package (both packages are named ``starfix``, so they cannot share one environment). Reads the
manifest written by the test, hashes each case's IPC stream under each option set, and writes
the results as JSON. Every outcome is a hex digest or ``raises <ExceptionType>``.

Usage: python -I starfix_python_worker.py <manifest.json> <results.json>
"""

import json
import sys

import pyarrow as pa
import starfix


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


def main(manifest_path, results_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    D = starfix.ArrowDigester
    results = {}
    for case in manifest["cases"]:
        with pa.ipc.open_stream(pa.OSFile(case["ipc"])) as reader:
            schema, batches = reader.schema, list(reader)
        table = pa.Table.from_batches(batches, schema=schema)
        for options in manifest["option_sets"]:
            array_options = {k: options[k] for k in ("ordered_columns", "ordered_struct_fields")}
            key = case["id"] + "/" + "".join("1" if options[k] else "0" for k in sorted(options))
            results[key] = {
                "hash_schema": _outcome(D.hash_schema, schema, **options),
                "hash_table": _outcome(D.hash_table, table, **options),
                "stream": _outcome(_stream, schema, batches, options),
                "hash_record_batch": [_outcome(D.hash_record_batch, b, **options) for b in batches],
                "hash_array": [_outcome(D.hash_array, table.column(i), **array_options) for i in range(table.num_columns)],
            }
    with open(results_path, "w") as f:
        json.dump({"starfix_version": getattr(starfix, "__version__", None), "results": results}, f)


if __name__ == "__main__":
    main(*sys.argv[1:])
