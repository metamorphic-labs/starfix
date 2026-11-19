"""The Python samples of README.md run, and print what the README says they print.

Every fenced ``python`` block of README.md is executed, top to bottom, in one namespace (the
samples share variables such as ``table``). A comment line that directly follows a ``print(...)``
line (or another such comment) is the output the README shows for it: each block must print
exactly those lines, and a block without them must print nothing. So a sample that stops
running, or a hash shown in the README that stops being the real one, fails here.

The Rust samples are compiled and run as doctests (``src/lib.rs``, ``ReadmeDoctests``).

Run after ``maturin develop``: ``pytest python/tests``.
"""

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"

_OPENING = re.compile(r"^```(\S*)\s*$")


def _python_blocks(text: str) -> list[tuple[int, str]]:
    """Every fenced python block of `text`, as (line number of its first line, its code)."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opening = _OPENING.match(lines[i])
        if not opening:
            i += 1
            continue
        end = next(j for j in range(i + 1, len(lines)) if lines[j].rstrip() == "```")
        if opening.group(1) in ("python", "py", "python3"):
            blocks.append((i + 2, "\n".join(lines[i + 1 : end]) + "\n"))
        i = end + 1
    return blocks


def _expected_output(code: str) -> str:
    """The output the README shows for a block: the comment lines that follow a print line."""
    expected = []
    after_print = False
    for line in code.splitlines():
        if line.startswith("print("):
            after_print = True
        elif after_print and line.startswith("# "):
            expected.append(line[2:] + "\n")
        else:
            after_print = False
    return "".join(expected)


def test_readme_python_samples_run_and_print_what_the_readme_shows():
    text = README.read_text(encoding="utf-8")
    blocks = _python_blocks(text)
    # Every python fence is found (a fence the parser missed would go untested).
    assert len(blocks) == len(re.findall(r"^```(?:python|py|python3)\s*$", text, re.M)) > 0

    namespace: dict = {"__name__": "__readme__"}
    for line, code in blocks:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"README.md:{line}", "exec"), namespace)  # noqa: S102
        assert output.getvalue() == _expected_output(code), f"the python block at README.md:{line}"


def test_expected_output_reads_the_comments_after_a_print():
    code = 'x = 1\nprint(x)\n# 1\n# 2\n\n# not output\nprint("a")\n# a\n'
    assert _expected_output(code) == "1\n2\na\n"


def test_python_blocks_skips_other_languages():
    text = "```bash\nls\n```\n\n```python\nx = 1\n```\n\n```text\n00\n```\n"
    assert _python_blocks(text) == [(6, "x = 1\n")]
