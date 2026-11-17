"""The seeded differential harness: the binding against starfix-python 0.5.0 (ITL-1178).

Each seed is a table built by ``_harness.build_case`` (types, names, metadata, layouts and
invalid data drawn together), in this process and in starfix-python's (``STARFIX_PYTHON``, run
by ``starfix_python_harness_worker.py``), without IPC. Every call (hash_schema, hash_table, a
stream, hash_record_batch per batch, hash_array per column) under all 8 option sets must give
the same digest on both sides, or raise on both sides in the same class (``_harness.classify``:
refusal, invalid data, wrong type); a crash never matches.

``STARFIX_HARNESS_SEEDS`` is ``start:count`` (default ``0:200``): CI runs 2,000 seeds on every
pull request and 50,000, in shards, before a release. Skipped when ``STARFIX_PYTHON`` is unset,
unless ``STARFIX_REQUIRE_DIFFERENTIAL`` is set, where it fails instead.

A failure names the seed; ``PYTHONPATH=python/tests python -c "import _harness;
print(_harness.build_case(SEED))"`` rebuilds the input, and each finding becomes a permanent,
hand-reduced case in ``_parity_cases``.
"""

import collections
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa

from _harness import GENERATOR_VERSION, build_case, outcomes, same_outcome
from _support import OPTION_SETS
from starfix import ArrowDigester
from test_differential import STARFIX_PYTHON_VERSION, _starfix_python

WORKER = Path(__file__).with_name("starfix_python_harness_worker.py")


def _seeds() -> tuple[int, int]:
    start, count = os.environ.get("STARFIX_HARNESS_SEEDS", "0:200").split(":")
    return int(start), int(count)


def _report(failure) -> str:
    seed, tags, key, call, ours, theirs, schema = failure
    return f"seed {seed} options {key} {call}: ours {ours} / starfix-python {theirs}; tags {tags}; schema {schema}"


def test_binding_equals_starfix_python_on_seeded_cases(tmp_path):
    interpreter = _starfix_python()
    start, count = _seeds()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seeds": [start, count], "option_sets": OPTION_SETS}))
    results_path = tmp_path / "results.json"
    subprocess.run([interpreter, "-I", str(WORKER), str(manifest), str(results_path)], check=True)
    worker = json.loads(results_path.read_text())
    assert worker["starfix_version"] == STARFIX_PYTHON_VERSION, "the worker must run starfix-python 0.5.0"
    assert worker["generator_version"] == GENERATOR_VERSION
    assert pa.__version__, "both environments install the same pyarrow (CI pins it)"

    failures, counts = [], collections.Counter()
    for seed in range(start, start + count):
        case = build_case(seed)
        theirs = worker["results"][str(seed)]
        for key, calls in outcomes(ArrowDigester, case, OPTION_SETS).items():
            for call, ours in calls.items():
                other = theirs[key][call]
                counts["digest" if ours[0] == "digest" else f"raise:{ours[1]}"] += 1
                if not same_outcome(ours, other):
                    failures.append((seed, sorted(case.tags), key, call, ours, other, str(case.schema)[:300]))
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"harness seeds {start}..{start + count - 1}: {sum(counts.values())} calls compared ({summary})")
    seeds_failed = sorted({f[0] for f in failures})
    assert not failures, (
        f"{len(failures)} calls differ in {len(seeds_failed)} seeds (first seeds {seeds_failed[:20]}):\n"
        + "\n".join(_report(f) for f in failures[:15])
    )
    assert counts["digest"] > sum(counts.values()) // 3, f"most calls must hash, not raise: {summary}"
