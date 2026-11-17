"""Run the harness cases with starfix-python (run by ``test_harness.py``).

Runs in starfix-python's own environment (both packages import as ``starfix``). Builds each
seed's case in process with ``_harness`` (no IPC: layouts and registered extension types reach
starfix-python as they reach the binding), runs the harness calls under each option set and
writes the outcomes as JSON, with the installed starfix version and the generator version.

Usage: python -I starfix_python_harness_worker.py <manifest.json> <results.json>
"""

import json
import os
import sys
from importlib.metadata import version

import starfix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script's directory out
import _harness


def main(manifest_path, results_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    start, count = manifest["seeds"]
    results = {
        str(seed): _harness.outcomes(starfix.ArrowDigester, _harness.build_case(seed), manifest["option_sets"])
        for seed in range(start, start + count)
    }
    named = {
        name: _harness.outcomes(starfix.ArrowDigester, difference.build(), manifest["option_sets"])
        for name, difference in _harness.EXPECTED_DIFFERENCES.items()
    }
    with open(results_path, "w") as f:
        json.dump(
            {
                "starfix_version": version("starfix"),
                "generator_version": _harness.GENERATOR_VERSION,
                "results": results,
                "expected_differences": named,
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
