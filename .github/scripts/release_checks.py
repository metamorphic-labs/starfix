"""Release gates of maturin-release.yml (RELEASING.md). Standard library only, Python >= 3.11.

  gates            Before anything is published: the tag is v<Cargo.toml version>, the tagged
                   commit is on main (an ancestor of origin/main), and CHANGELOG.md has the
                   section "## <version>". Every failing gate is reported; exit 1 if any fails.
  crate-published  Whether <name> <version> is already on crates.io (its sparse index, yanked
                   versions included), written as published=true|false to $GITHUB_OUTPUT, so a
                   re-run of the publish job skips an upload that already happened. Exit 1 when
                   the index cannot be read: an unknown answer never counts as "not published".

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


def changelog_has_section(text: str, version: str) -> bool:
    """A heading "## <version>", "## <version> (<date>)" or "## [<version>] ..." at a line start."""
    pattern = rf"^## \[?{re.escape(version)}\]?(?:[ \t].*)?$"
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def is_on_main(sha: str, main_ref: str = "origin/main", cwd: str | None = None) -> bool:
    """Whether the commit is an ancestor of (or is) main_ref. An unknown commit or ref is an error."""
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
    """Whether crates.io lists the version (yanked or not). Unknown answers raise."""
    url = f"{CRATES_INDEX}/{crates_index_path(name)}"
    status, body = fetch(url)
    if status == 404:
        return False
    if status != 200:
        raise ReleaseCheckError(f"{url} answered HTTP {status}")
    try:
        versions = {json.loads(line)["vers"] for line in body.splitlines() if line.strip()}
    except (ValueError, KeyError, TypeError) as e:
        raise ReleaseCheckError(f"{url} is not a sparse index file: {e}") from e
    return version in versions


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
    gates.add_argument("--main-ref", default="origin/main")
    gates.add_argument("--cargo-toml", default="Cargo.toml")
    gates.add_argument("--changelog", default="CHANGELOG.md")
    gates.add_argument("--repo", default=None, help="the git repository (default: the current directory)")

    crate = sub.add_parser("crate-published", help="is this crate version on crates.io")
    crate.add_argument("--name", required=True)
    crate.add_argument("--version", required=True)
    crate.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    args = parser.parse_args(argv)
    if args.command == "gates":
        return cmd_gates(args)
    return cmd_crate_published(args, fetch)


if __name__ == "__main__":
    sys.exit(main())
