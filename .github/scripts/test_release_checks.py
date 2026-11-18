"""Unit tests of release_checks.py (the release gates of maturin-release.yml, RELEASING.md).

Run with: python -m unittest discover -s .github/scripts -v
Set STARFIX_LIVE_NETWORK=1 to also query the real crates.io index (CI does).
"""

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import release_checks as rc

CARGO_TOML = """\
[package]
name = "starfix"
version = "0.6.0"
edition = "2021"

[dependencies]
other = { version = "9.9.9" }
"""

CHANGELOG = """\
# Changelog

## Unreleased

## 0.6.0 (2026-11-20)

### Added

## 0.5.0
"""


class CargoVersionTest(unittest.TestCase):
    def test_reads_the_package_version_not_a_dependency_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "Cargo.toml")
            path.write_text(CARGO_TOML)
            self.assertEqual(rc.cargo_version(path), "0.6.0")

    def test_refuses_a_manifest_without_a_package_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "Cargo.toml")
            path.write_text("[workspace]\nmembers = []\n")
            with self.assertRaises(rc.ReleaseCheckError):
                rc.cargo_version(path)


class CheckTagTest(unittest.TestCase):
    def test_accepts_v_and_the_version(self):
        rc.check_tag("v0.6.0", "0.6.0")

    def test_refuses_another_version(self):
        with self.assertRaisesRegex(rc.ReleaseCheckError, "does not name"):
            rc.check_tag("v0.6.1", "0.6.0")

    def test_refuses_a_tag_without_the_v(self):
        with self.assertRaises(rc.ReleaseCheckError):
            rc.check_tag("0.6.0", "0.6.0")

    def test_refuses_a_suffix(self):
        with self.assertRaises(rc.ReleaseCheckError):
            rc.check_tag("v0.6.0-rc1", "0.6.0")


class ChangelogTest(unittest.TestCase):
    def test_accepts_the_headings_in_use(self):
        for heading in ("## 0.6.0 (2026-11-20)", "## 0.6.0", "## [0.6.0] - 2026-11-20", "## 0.6.0\t"):
            with self.subTest(heading=heading):
                self.assertTrue(rc.changelog_has_section(f"# Changelog\n\n{heading}\n\ntext\n", "0.6.0"))

    def test_refuses_what_is_not_that_section(self):
        for text in (
            "## Unreleased\n\n0.6.0 is coming\n",
            "### 0.6.0\n",
            "## 0.6.01\n",
            "## 0.6.0-rc1\n",
            "## 0x6x0\n",  # the dots are literal
            " ## 0.6.0\n",
            "## 10.6.0\n",
        ):
            with self.subTest(text=text):
                self.assertFalse(rc.changelog_has_section(text, "0.6.0"))


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class IsOnMainTest(unittest.TestCase):
    """A throwaway repository: origin/main is A <- B, and C is on another branch from A."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "test@example.invalid")
        git(self.repo, "config", "user.name", "test")
        git(self.repo, "config", "commit.gpgsign", "false")
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "A")
        self.a = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "switch", "-q", "-c", "other")
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "C")
        self.c = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "switch", "-q", "main")
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "B")
        self.b = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "update-ref", "refs/remotes/origin/main", self.b)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_head_of_main_is_on_main(self):
        self.assertTrue(rc.is_on_main(self.b, "origin/main", cwd=self.repo))

    def test_an_older_commit_of_main_is_on_main(self):
        self.assertTrue(rc.is_on_main(self.a, "origin/main", cwd=self.repo))

    def test_a_commit_of_another_branch_is_not(self):
        self.assertFalse(rc.is_on_main(self.c, "origin/main", cwd=self.repo))

    def test_an_unknown_commit_is_an_error_not_a_no(self):
        with self.assertRaises(rc.ReleaseCheckError):
            rc.is_on_main("0" * 40, "origin/main", cwd=self.repo)

    def test_a_missing_main_ref_is_an_error(self):
        with self.assertRaises(rc.ReleaseCheckError):
            rc.is_on_main(self.b, "origin/nope", cwd=self.repo)


class GatesCommandTest(IsOnMainTest):
    """The `gates` command as the workflow runs it; exit 0 only when every gate passes."""

    def run_gates(self, tag, sha, cargo=CARGO_TOML, changelog=CHANGELOG):
        Path(self.repo, "Cargo.toml").write_text(cargo)
        Path(self.repo, "CHANGELOG.md").write_text(changelog)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rc.main(
                [
                    "gates",
                    "--tag", tag,
                    "--sha", sha,
                    "--main-ref", "origin/main",
                    "--cargo-toml", str(Path(self.repo, "Cargo.toml")),
                    "--changelog", str(Path(self.repo, "CHANGELOG.md")),
                    "--repo", self.repo,
                ]
            )
        return status, out.getvalue()

    def test_passes(self):
        status, out = self.run_gates("v0.6.0", self.b)
        self.assertEqual(status, 0, out)
        self.assertIn("OK", out)

    def test_refuses_a_tag_that_is_not_the_cargo_version(self):
        status, out = self.run_gates("v0.6.1", self.b)
        self.assertEqual(status, 1)
        self.assertIn("::error::", out)

    def test_refuses_a_commit_not_on_main(self):
        status, out = self.run_gates("v0.6.0", self.c)
        self.assertEqual(status, 1)
        self.assertIn("not on origin/main", out)

    def test_refuses_without_the_changelog_section(self):
        status, out = self.run_gates("v0.6.0", self.b, changelog="# Changelog\n\n## Unreleased\n")
        self.assertEqual(status, 1)
        self.assertIn("CHANGELOG.md", out)

    def test_reports_every_failing_gate(self):
        status, out = self.run_gates("v9.9.9", self.c, changelog="")
        self.assertEqual(status, 1)
        self.assertEqual(out.count("::error::"), 3, out)


class CratesIndexPathTest(unittest.TestCase):
    def test_follows_the_sparse_index_layout(self):
        self.assertEqual(rc.crates_index_path("starfix"), "st/ar/starfix")
        self.assertEqual(rc.crates_index_path("a"), "1/a")
        self.assertEqual(rc.crates_index_path("ab"), "2/ab")
        self.assertEqual(rc.crates_index_path("abc"), "3/a/abc")
        self.assertEqual(rc.crates_index_path("Serde"), "se/rd/serde")


def fake_fetch(status, body=""):
    calls = []

    def fetch(url):
        calls.append(url)
        return status, body

    fetch.calls = calls
    return fetch


INDEX = (
    '{"name":"starfix","vers":"0.5.0","yanked":false}\n'
    '{"name":"starfix","vers":"0.6.0","yanked":true}\n'
)


class CrateVersionPublishedTest(unittest.TestCase):
    def test_a_listed_version_is_published(self):
        fetch = fake_fetch(200, INDEX)
        self.assertTrue(rc.crate_version_published("starfix", "0.5.0", fetch=fetch))
        self.assertEqual(fetch.calls, ["https://index.crates.io/st/ar/starfix"])

    def test_a_yanked_version_counts_as_published(self):
        self.assertTrue(rc.crate_version_published("starfix", "0.6.0", fetch=fake_fetch(200, INDEX)))

    def test_an_unlisted_version_is_not(self):
        self.assertFalse(rc.crate_version_published("starfix", "0.6.1", fetch=fake_fetch(200, INDEX)))

    def test_an_unknown_crate_is_not(self):
        self.assertFalse(rc.crate_version_published("starfix", "0.6.0", fetch=fake_fetch(404)))

    def test_any_other_answer_is_an_error_not_a_no(self):
        for status in (403, 500, 503):
            with self.subTest(status=status), self.assertRaises(rc.ReleaseCheckError):
                rc.crate_version_published("starfix", "0.6.0", fetch=fake_fetch(status))

    def test_an_unreadable_index_is_an_error(self):
        with self.assertRaises(rc.ReleaseCheckError):
            rc.crate_version_published("starfix", "0.6.0", fetch=fake_fetch(200, "not json\n"))


class CratePublishedCommandTest(unittest.TestCase):
    def run_command(self, fetch):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, "github_output")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = rc.main(
                    ["crate-published", "--name", "starfix", "--version", "0.5.0", "--github-output", str(output)],
                    fetch=fetch,
                )
            return status, out.getvalue(), output.read_text() if output.exists() else ""

    def test_writes_true_and_says_so(self):
        status, out, output = self.run_command(fake_fetch(200, INDEX))
        self.assertEqual(status, 0)
        self.assertEqual(output, "published=true\n")
        self.assertIn("already on crates.io", out)

    def test_writes_false(self):
        status, _, output = self.run_command(fake_fetch(404))
        self.assertEqual(status, 0)
        self.assertEqual(output, "published=false\n")

    def test_fails_and_writes_nothing_when_the_index_cannot_be_read(self):
        status, out, output = self.run_command(fake_fetch(500))
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("::error::", out)


@unittest.skipUnless(os.environ.get("STARFIX_LIVE_NETWORK") == "1", "set STARFIX_LIVE_NETWORK=1")
class LiveCratesIndexTest(unittest.TestCase):
    def test_starfix_0_5_0_is_published(self):
        self.assertTrue(rc.crate_version_published("starfix", "0.5.0"))

    def test_an_absurd_version_is_not(self):
        self.assertFalse(rc.crate_version_published("starfix", "999.0.0"))


if __name__ == "__main__":
    unittest.main()
