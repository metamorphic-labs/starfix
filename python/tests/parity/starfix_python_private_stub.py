"""pytest plugin for running starfix-python's own test suite against this package (ITL-1178).

starfix-python's tests also exercise its private helpers (``starfix._format2``: the canonical
JSON builders, the per-entry encoders and other internals of the pure-Python implementation),
which this Rust-backed package does not have. Some test modules import them at module level,
so without this plugin the public-behaviour tests in those modules would not even be
collected.

This plugin installs a stand-in ``starfix._format2`` whose every use (attribute value, call,
operator, iteration, ...) skips the test that uses it, with the reason ``PRIVATE``. Tests that
use only the public interface run unchanged; a test that touches a private helper is reported
as skipped, never as passed. Load it with ``-p starfix_python_private_stub`` (this directory on
``PYTHONPATH``). It is not part of the wheel.
"""

from __future__ import annotations

import sys
import types

import pytest

PRIVATE = "uses starfix-python's private starfix._format2 (not part of the public interface)"


def _skip(*_args, **_kwargs):
    raise pytest.skip.Exception(PRIVATE, allow_module_level=True)


class _Private:
    """A stand-in for any private helper or constant: every use skips."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<private starfix._format2.{self._name}>"

    __call__ = __getattr__ = __getitem__ = __iter__ = __len__ = __bool__ = __int__ = __index__ = _skip
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __hash__ = _skip
    __or__ = __ror__ = __and__ = __rand__ = __xor__ = __add__ = __radd__ = __sub__ = __mul__ = _skip
    __contains__ = __enter__ = __invert__ = __neg__ = __str__ = __bytes__ = _skip


class _PrivateModule(types.ModuleType):
    def __getattr__(self, name: str) -> _Private:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Private(name)


def _install() -> None:
    import starfix

    module = _PrivateModule("starfix._format2", PRIVATE)
    sys.modules["starfix._format2"] = module
    starfix._format2 = module
    # starfix-python's arrow_digester module imports _format2; tests that reach it through
    # the module (``arrow_digester._format2``) get the stand-in too.
    import starfix.arrow_digester

    starfix.arrow_digester._format2 = module


# Tests that exercise starfix-python's internals in ways the stand-in cannot reach, each with
# the reason it does not apply to this package (and where its public behaviour is covered).
EXCLUDED = {
    "test_format2.py::test_empty_list_without_offsets_buffer_does_not_crash": (
        "runs `from starfix import _format2` in a subprocess (private helper); the public "
        "behaviour (an empty list array without an offsets buffer hashes as an empty list) is "
        "tested in python/tests/test_binding.py"
    ),
    "test_prepared_schema.py::test_one_shot_calls_analyse_once": (
        "counts calls of starfix-python's private analysis functions; the crate's one-shot "
        "calls analyse the schema once by construction (ITL-1177)"
    ),
    "test_prepared_schema.py::test_prepared_hashes_of_an_identical_schema_do_not_analyse": (
        "counts calls of starfix-python's private analysis functions; the crate's prepared "
        "calls accept an identical or equal schema by pointer or structural comparison, without "
        "analysis (tests/prepared_schema.rs, ITL-1177)"
    ),
    "test_prepared_schema.py::test_streaming_update_of_an_identical_schema_does_not_analyse": (
        "counts calls of starfix-python's private analysis functions; the crate's streaming "
        "update accepts an equal schema by structural comparison, without analysis (ITL-1177)"
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        for suffix, reason in EXCLUDED.items():
            if item.nodeid.endswith(suffix):
                item.add_marker(pytest.mark.skip(reason=f"not applicable: {reason}"))


_install()
