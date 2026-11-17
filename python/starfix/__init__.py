"""starfix: deterministic logical hashing of Apache Arrow data (hash format 2).

``ArrowDigester`` and ``PreparedSchema`` have starfix-python's interface and results, computed
by the starfix Rust crate. ``starfix.legacy.v0_4`` is starfix-python's frozen pure-Python
implementation of the 0.4 format (format 1), shipped unchanged.
"""

from starfix._internal import __version__
from starfix.arrow_digester import ArrowDigester, PreparedSchema

__all__ = ["ArrowDigester", "PreparedSchema", "__version__"]
