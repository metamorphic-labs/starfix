"""Release gates of maturin-release.yml (RELEASING.md). Standard library only, Python >= 3.11.

  gates              Before anything is published: the tag is v<Cargo.toml version>, the tagged
                     commit is on main (an ancestor of refs/remotes/origin/main), and CHANGELOG.md
                     has the section "## <version>" (outside fenced code blocks). Every failing
                     gate is reported; exit 1 if any fails.
  dist-complete      Before each upload to TestPyPI or PyPI: the directory holds exactly the
                     release's files, one wheel of the version for each target the workflow
                     builds (RELEASE_WHEELS) and its sdist, and nothing else. Exit 1 otherwise.
  crate-published    Whether <name> <version> is already on crates.io (its sparse index, yanked
                     versions included), written as published=true|false to $GITHUB_OUTPUT, so a
                     re-run of the publish job skips an upload that already happened. Exit 1 when
                     the index cannot be read: an unknown answer never counts as "not published".
  wait-crate-listed  After the crate upload (or when it was already there): poll the sparse index
                     until it lists <name> <version>, so PyPI never starts before the crate is
                     available to Cargo. Exit 1 when it is not listed within --timeout seconds.

The crates.io sparse index answers, as read here: 200 with rows that are all JSON objects with a
string "vers": the versions listed; 404: the crate has never been published, so not published;
200 with no rows, a row that is not such an object, any other status, or no answer after the
retries of http_get: an error, never "not published".

Tests: test_release_checks.py, next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

CRATES_INDEX = "https://index.crates.io"
USER_AGENT = "starfix-release-checks (https://github.com/metamorphic-labs/starfix)"

Fetch = Callable[[str], "tuple[int, str]"]

# The wheels of one release, one per build job of maturin-release.yml (linux matrix, macos,
# windows), by target: a wheel belongs to a target when one of its platform tags (the last field
# of the file name, split at ".") matches. A target added to or removed from the workflow must be
# added to or removed from this table too; until then dist-complete refuses the upload.
RELEASE_WHEELS = {
    "x86_64-unknown-linux-gnu": r"manylinux_2_\d+_x86_64",
    "aarch64-unknown-linux-gnu": r"manylinux_2_\d+_aarch64",
    "x86_64-unknown-linux-musl": r"musllinux_1_\d+_x86_64",
    "aarch64-unknown-linux-musl": r"musllinux_1_\d+_aarch64",
    "universal2-apple-darwin": r"macosx_\d+_\d+_universal2",
    "x86_64-pc-windows-msvc": r"win_amd64",
}
# The extension is built against the stable ABI, Python >= 3.10 (abi3-py310, Cargo.toml).
WHEEL_PYTHON_ABI = ("cp310", "abi3")
DISTRIBUTION = "starfix"


class ReleaseCheckError(Exception):
    """A gate refused, or a check could not get an answer."""


def cargo_version(path: Path | str) -> str:
    """The [package] version of a Cargo.toml."""
    with open(path, "rb") as f:
        manifest = tomllib.load(f)
    version = manifest.get("package", {}).get("version")
    if not isinstance(version, str):
        raise ReleaseCheckError(f"{path} has no [package] version")
    return version


def check_tag(tag: str, version: str) -> None:
    """The tag must be exactly v<version>."""
    if tag != f"v{version}":
        raise ReleaseCheckError(f"the tag {tag} does not name Cargo.toml's version {version} (expected v{version})")


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def changelog_has_section(text: str, version: str) -> bool:
    """A heading "## <version>", "## <version> (<date>)" or "## [<version>] ..." at a line start,
    outside fenced code blocks (CommonMark: a fence of 3+ backticks or tildes, indented 0-3
    spaces, closed by a fence of the same character at least as long, or by the end of the text).
    An indented code block cannot match, as the heading must start at column 0."""
    heading = re.compile(rf"## \[?{re.escape(version)}\]?(?:[ \t].*)?")
    fence = ""  # the opening fence of the code block the line is in, if any
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence:
            run = match.group(1) if match else ""
            if run[:1] == fence[0] and len(run) >= len(fence) and not match.group(2).strip():
                fence = ""
            continue
        if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
            fence = match.group(1)
            continue
        if heading.fullmatch(line):
            return True
    return False


MAIN_REF = "refs/remotes/origin/main"


def is_on_main(sha: str, main_ref: str = MAIN_REF, cwd: str | None = None) -> bool:
    """Whether the commit is an ancestor of (or is) main_ref. An unknown commit or ref is an error.
    main_ref is named in full: git would resolve "origin/main" to a tag of that name first."""
    for rev in (sha, main_ref):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise ReleaseCheckError(f"git does not know the commit {rev}")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, main_ref], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise ReleaseCheckError(f"git merge-base failed ({result.returncode}): {result.stderr.strip()}")


def crates_index_path(name: str) -> str:
    """The path of a crate in the crates.io sparse index (lower case)."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[0:2]}/{name[2:4]}/{name}"


def http_get(url: str, attempts: int = 3) -> tuple[int, str]:
    """GET with a few retries on network errors and 5xx; returns (status, body)."""
    last = ""
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code, ""
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
        if attempt + 1 < attempts:
            time.sleep(2 * (attempt + 1))
    raise ReleaseCheckError(f"could not read {url}: {last}")


def crate_version_published(name: str, version: str, fetch: Fetch = http_get) -> bool:
    """Whether crates.io lists the version (yanked or not). Unknown answers raise.

    404 (the crate was never published) is "not published". 200 needs at least one row, and every
    row a JSON object with a string "vers"; anything else, and any other status, raises."""
    url = f"{CRATES_INDEX}/{crates_index_path(name)}"
    status, body = fetch(url)
    if status == 404:
        return False
    if status != 200:
        raise ReleaseCheckError(f"{url} answered HTTP {status}")
    rows = [line for line in body.splitlines() if line.strip()]
    if not rows:
        raise ReleaseCheckError(f"{url} answered HTTP 200 with no index rows")
    versions = set()
    for row in rows:
        try:
            vers = json.loads(row)["vers"]
        except (ValueError, KeyError, TypeError, IndexError) as e:
            raise ReleaseCheckError(f"{url} is not a sparse index file: {e!r}") from e
        if not isinstance(vers, str):
            raise ReleaseCheckError(f"{url} has a row whose version is not a string: {row[:200]}")
        versions.add(vers)
    return version in versions


def wait_for_crate_version(
    name: str,
    version: str,
    fetch: Fetch = http_get,
    timeout: float = 600,
    interval: float = 15,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll the sparse index until it lists the version; raise if it does not within timeout
    seconds. An error while polling is not a yes: it is retried, and named if time runs out."""
    start = clock()
    last = "the index does not list it"
    while True:
        try:
            if crate_version_published(name, version, fetch=fetch):
                return
            last = "the index does not list it"
        except ReleaseCheckError as e:
            last = str(e)
        if clock() - start + interval > timeout:
            raise ReleaseCheckError(
                f"{name} {version} is not listed by the crates.io index after {timeout:g} seconds ({last})"
            )
        sleep(interval)


def _parse_wheel(filename: str) -> tuple[str, str, str, str, list[str]] | None:
    """(distribution, version, python tag, abi tag, platform tags) of a wheel file name without a
    build tag, or None."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) != 5:
        return None
    distribution, version, python, abi, platform = parts
    return distribution, version, python, abi, platform.split(".")


def check_dist_complete(directory: Path | str, version: str) -> list[str]:
    """The release's files in directory, sorted: exactly one wheel per RELEASE_WHEELS target and
    the sdist, all of this version, and nothing else. Raises with every problem otherwise."""
    directory = Path(directory)
    problems: list[str] = []
    found: dict[str, list[str]] = {target: [] for target in RELEASE_WHEELS}
    sdist = f"{DISTRIBUTION}-{version}.tar.gz"
    names = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    if not directory.is_dir():
        problems.append(f"{directory} is not a directory")
    for name in names:
        if not Path(directory, name).is_file():
            problems.append(f"{name} is not a file")
            continue
        if name == sdist:
            continue
        wheel = _parse_wheel(name)
        if wheel is None or wheel[:4] != (DISTRIBUTION, version, *WHEEL_PYTHON_ABI):
            kind = f"{DISTRIBUTION} {version} {'-'.join(WHEEL_PYTHON_ABI)}"
            problems.append(f"{name} is not a {kind} wheel or its sdist")
            continue
        targets = [t for t, tag in RELEASE_WHEELS.items() if any(re.fullmatch(tag, p) for p in wheel[4])]
        if len(targets) != 1:
            problems.append(f"{name} is the wheel of {'no' if not targets else 'more than one'} release target")
            continue
        found[targets[0]].append(name)
    for target, wheels in found.items():
        if not wheels:
            problems.append(f"no wheel for {target}")
        elif len(wheels) > 1:
            problems.append(f"more than one wheel for {target}: {', '.join(wheels)}")
    if sdist not in names:
        problems.append(f"no sdist {sdist}")
    if problems:
        raise ReleaseCheckError("; ".join(problems))
    return names


def _error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def cmd_gates(args: argparse.Namespace) -> int:
    failures = 0
    try:
        version = cargo_version(args.cargo_toml)
    except (ReleaseCheckError, OSError, tomllib.TOMLDecodeError) as e:
        _error(f"{e}. Nothing was published.")
        return 1

    try:
        check_tag(args.tag, version)
    except ReleaseCheckError as e:
        _error(f"{e}. Nothing was published.")
        failures += 1

    try:
        if not is_on_main(args.sha, args.main_ref, cwd=args.repo):
            _error(
                f"the tagged commit {args.sha} is not on {args.main_ref}: release tags go on a commit "
                f"of main only. Nothing was published."
            )
            failures += 1
    except ReleaseCheckError as e:
        _error(f"could not check that {args.sha} is on {args.main_ref}: {e}. Nothing was published.")
        failures += 1

    try:
        changelog = Path(args.changelog).read_text(encoding="utf-8")
    except OSError as e:
        changelog = ""
        print(f"cannot read {args.changelog}: {e}")
    if not changelog_has_section(changelog, version):
        _error(f"CHANGELOG.md has no '## {version}' section. Nothing was published.")
        failures += 1

    if failures:
        return 1
    print(
        f"OK: the tag {args.tag} names Cargo.toml's version {version}, the commit {args.sha} is on "
        f"{args.main_ref}, and CHANGELOG.md has the section '## {version}'."
    )
    return 0


def cmd_dist_complete(args: argparse.Namespace) -> int:
    try:
        names = check_dist_complete(args.dir, args.version)
    except (ReleaseCheckError, OSError) as e:
        _error(
            f"{args.dir} is not the complete set of {DISTRIBUTION} {args.version}'s files: {e}. "
            f"Nothing was uploaded."
        )
        return 1
    print(f"{args.dir} holds the {len(names)} files of {DISTRIBUTION} {args.version}:")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_wait_crate_listed(args: argparse.Namespace, fetch: Fetch) -> int:
    try:
        wait_for_crate_version(args.name, args.version, fetch=fetch, timeout=args.timeout, interval=args.interval)
    except ReleaseCheckError as e:
        _error(f"{e}. PyPI was not started; re-run the failed jobs once crates.io lists it.")
        return 1
    print(f"crates.io lists {args.name} {args.version}.")
    return 0


def cmd_crate_published(args: argparse.Namespace, fetch: Fetch) -> int:
    try:
        published = crate_version_published(args.name, args.version, fetch=fetch)
    except ReleaseCheckError as e:
        _error(f"cannot tell whether {args.name} {args.version} is on crates.io: {e}. Re-run the job.")
        return 1
    if published:
        print(f"{args.name} {args.version} is already on crates.io: nothing to upload.")
    else:
        print(f"{args.name} {args.version} is not on crates.io yet.")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as f:
            f.write(f"published={'true' if published else 'false'}\n")
    return 0


def main(argv: list[str] | None = None, fetch: Fetch = http_get) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gates = sub.add_parser("gates", help="the tag, main and CHANGELOG.md gates")
    gates.add_argument("--tag", required=True, help="the tag name, e.g. v0.6.0")
    gates.add_argument("--sha", required=True, help="the tagged commit")
    gates.add_argument("--main-ref", default=MAIN_REF)
    gates.add_argument("--cargo-toml", default="Cargo.toml")
    gates.add_argument("--changelog", default="CHANGELOG.md")
    gates.add_argument("--repo", default=None, help="the git repository (default: the current directory)")

    crate = sub.add_parser("crate-published", help="is this crate version on crates.io")
    crate.add_argument("--name", required=True)
    crate.add_argument("--version", required=True)
    crate.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    dist = sub.add_parser("dist-complete", help="does the directory hold exactly the release's files")
    dist.add_argument("--dir", default="dist")
    dist.add_argument("--version", required=True)

    wait = sub.add_parser("wait-crate-listed", help="wait until crates.io lists this crate version")
    wait.add_argument("--name", required=True)
    wait.add_argument("--version", required=True)
    wait.add_argument("--timeout", type=float, default=600, help="seconds (default 600)")
    wait.add_argument("--interval", type=float, default=15, help="seconds between reads (default 15)")

    args = parser.parse_args(argv)
    if args.command == "gates":
        return cmd_gates(args)
    if args.command == "dist-complete":
        return cmd_dist_complete(args)
    if args.command == "wait-crate-listed":
        return cmd_wait_crate_listed(args, fetch)
    return cmd_crate_published(args, fetch)


if __name__ == "__main__":
    sys.exit(main())
