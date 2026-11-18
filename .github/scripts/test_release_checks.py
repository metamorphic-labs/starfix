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

    def test_refuses_the_heading_inside_a_fenced_code_block(self):
        pairs = (("```", "```"), ("~~~", "~~~"), ("````", "`````"), ("```markdown", "```"), ("  ```", "```"))
        for opening, closing in pairs:
            text = f"# Changelog\n\n## Unreleased\n\n{opening}\n## 0.6.0\n{closing}\n"
            with self.subTest(opening=opening):
                self.assertFalse(rc.changelog_has_section(text, "0.6.0"))

    def test_an_unclosed_fence_runs_to_the_end(self):
        self.assertFalse(rc.changelog_has_section("# Changelog\n\n```\n## 0.6.0\n", "0.6.0"))

    def test_a_shorter_or_other_fence_does_not_close_the_block(self):
        for inner in ("```", "~~~~"):
            text = f"````\n{inner}\n## 0.6.0\n````\n"
            with self.subTest(inner=inner):
                self.assertFalse(rc.changelog_has_section(text, "0.6.0"))

    def test_accepts_the_heading_after_a_closed_fence(self):
        text = "# Changelog\n\n```sh\n## 0.6.0\n```\n\n## 0.6.0 (2026-11-20)\n"
        self.assertTrue(rc.changelog_has_section(text, "0.6.0"))


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

    def test_a_tag_named_origin_main_does_not_shadow_the_branch(self):
        # git resolves the short name origin/main to refs/tags/origin/main before
        # refs/remotes/origin/main; the default names the remote-tracking branch in full.
        git(self.repo, "tag", "origin/main", self.c)
        self.assertTrue(rc.is_on_main(self.b, cwd=self.repo))
        self.assertFalse(rc.is_on_main(self.c, cwd=self.repo))


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

    def test_the_default_main_ref_is_the_remote_tracking_branch_in_full(self):
        git(self.repo, "tag", "origin/main", self.c)
        Path(self.repo, "Cargo.toml").write_text(CARGO_TOML)
        Path(self.repo, "CHANGELOG.md").write_text(CHANGELOG)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rc.main(
                [
                    "gates",
                    "--tag", "v0.6.0",
                    "--sha", self.b,
                    "--cargo-toml", str(Path(self.repo, "Cargo.toml")),
                    "--changelog", str(Path(self.repo, "CHANGELOG.md")),
                    "--repo", self.repo,
                ]
            )
        self.assertEqual(status, 0, out.getvalue())
        self.assertIn("refs/remotes/origin/main", out.getvalue())

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

    def test_an_empty_index_file_is_an_error_not_a_no(self):
        for body in ("", "\n", "  \n\n"):
            with self.subTest(body=body), self.assertRaisesRegex(rc.ReleaseCheckError, "no index rows"):
                rc.crate_version_published("starfix", "0.6.1", fetch=fake_fetch(200, body))

    def test_a_row_without_a_string_version_is_an_error_not_a_no(self):
        rows = ('{"name":"starfix","vers":null}', '{"name":"starfix","vers":6}', '["starfix","0.6.1"]', '"0.6.1"')
        for row in rows:
            body = INDEX + row + "\n"
            with self.subTest(row=row), self.assertRaises(rc.ReleaseCheckError):
                rc.crate_version_published("starfix", "0.6.1", fetch=fake_fetch(200, body))

    def test_the_not_found_answers_of_the_registry_protocol_other_than_404_are_errors(self):
        # Cargo reads 404, 410 and 451 as "no such crate". starfix exists, so only the 404 the
        # index serves for an unknown crate is read as "not published"; anything else fails.
        for status in (410, 451):
            with self.subTest(status=status), self.assertRaises(rc.ReleaseCheckError):
                rc.crate_version_published("starfix", "0.6.0", fetch=fake_fetch(status))


def fetch_sequence(*answers):
    """A fetch that gives the answers in order (an exception is raised), then the last one again."""
    calls = []

    def fetch(url):
        calls.append(url)
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    fetch.calls = calls
    return fetch


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class WaitForCrateVersionTest(unittest.TestCase):
    INDEX_WITHOUT = '{"name":"starfix","vers":"0.5.0","yanked":false}\n'

    def wait(self, fetch, clock, timeout=60, interval=10):
        rc.wait_for_crate_version(
            "starfix", "0.6.0", fetch=fetch, timeout=timeout, interval=interval, sleep=clock.sleep, clock=clock.time
        )

    def test_returns_at_once_when_the_version_is_listed(self):
        clock = FakeClock()
        fetch = fetch_sequence((200, INDEX))
        self.wait(fetch, clock)
        self.assertEqual(len(fetch.calls), 1)
        self.assertEqual(clock.sleeps, [])

    def test_polls_until_the_version_is_listed(self):
        clock = FakeClock()
        fetch = fetch_sequence((200, self.INDEX_WITHOUT), (200, self.INDEX_WITHOUT), (200, INDEX))
        self.wait(fetch, clock)
        self.assertEqual(len(fetch.calls), 3)
        self.assertEqual(clock.sleeps, [10, 10])

    def test_an_unreadable_answer_while_waiting_is_not_a_yes(self):
        clock = FakeClock()
        fetch = fetch_sequence(rc.ReleaseCheckError("HTTP 503"), (200, ""), (200, INDEX))
        self.wait(fetch, clock)
        self.assertEqual(len(fetch.calls), 3)

    def test_fails_when_the_version_never_appears_within_the_bound(self):
        clock = FakeClock()
        fetch = fetch_sequence((200, self.INDEX_WITHOUT))
        with self.assertRaisesRegex(rc.ReleaseCheckError, "not listed"):
            self.wait(fetch, clock, timeout=60, interval=10)
        self.assertLessEqual(clock.now, 60)
        self.assertEqual(len(fetch.calls), 7)  # at 0, 10, ..., 60 seconds

    def test_the_failure_names_the_last_error(self):
        clock = FakeClock()
        fetch = fetch_sequence(rc.ReleaseCheckError("HTTP 503"))
        with self.assertRaisesRegex(rc.ReleaseCheckError, "HTTP 503"):
            self.wait(fetch, clock, timeout=20, interval=10)


class WaitCrateCommandTest(unittest.TestCase):
    def test_exit_0_when_listed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rc.main(
                ["wait-crate-listed", "--name", "starfix", "--version", "0.5.0", "--timeout", "0"],
                fetch=fake_fetch(200, INDEX),
            )
        self.assertEqual(status, 0)
        self.assertIn("crates.io lists starfix 0.5.0", out.getvalue())

    def test_exit_1_with_an_error_when_never_listed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = rc.main(
                ["wait-crate-listed", "--name", "starfix", "--version", "0.6.1", "--timeout", "0"],
                fetch=fake_fetch(200, INDEX),
            )
        self.assertEqual(status, 1)
        self.assertIn("::error::", out.getvalue())


# The files of one release, as maturin-release.yml builds them (the names maturin 1.x gave in
# this workflow's run of 2026-11-18, version changed).
RELEASE_FILES = [
    "starfix-0.6.0-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "starfix-0.6.0-cp310-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
    "starfix-0.6.0-cp310-abi3-musllinux_1_2_x86_64.whl",
    "starfix-0.6.0-cp310-abi3-musllinux_1_2_aarch64.whl",
    "starfix-0.6.0-cp310-abi3-macosx_10_12_universal2.macosx_10_12_x86_64.macosx_11_0_arm64.whl",
    "starfix-0.6.0-cp310-abi3-win_amd64.whl",
    "starfix-0.6.0.tar.gz",
]


class DistCompleteTest(unittest.TestCase):
    def check(self, names, version="0.6.0"):
        with tempfile.TemporaryDirectory() as tmp:
            for name in names:
                Path(tmp, name).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, name).write_bytes(b"x")
            rc.check_dist_complete(tmp, version)

    def test_accepts_the_complete_set(self):
        self.check(RELEASE_FILES)

    def test_refuses_a_missing_wheel(self):
        for missing in RELEASE_FILES[:-1]:
            with self.subTest(missing=missing), self.assertRaisesRegex(rc.ReleaseCheckError, "no wheel"):
                self.check([n for n in RELEASE_FILES if n != missing])

    def test_refuses_a_missing_sdist(self):
        with self.assertRaisesRegex(rc.ReleaseCheckError, "sdist"):
            self.check(RELEASE_FILES[:-1])

    def test_refuses_an_empty_directory(self):
        with self.assertRaises(rc.ReleaseCheckError):
            self.check([])

    def test_refuses_another_version(self):
        with self.assertRaises(rc.ReleaseCheckError):
            self.check([n.replace("0.6.0", "0.5.0") for n in RELEASE_FILES])
        with self.assertRaises(rc.ReleaseCheckError):
            self.check(RELEASE_FILES[:-1] + ["starfix-0.5.0.tar.gz"])
        with self.assertRaises(rc.ReleaseCheckError):
            self.check(RELEASE_FILES[1:] + [RELEASE_FILES[0].replace("0.6.0", "0.6.1")])

    def test_refuses_an_extra_or_unknown_file(self):
        for extra in (
            "starfix-0.6.0-cp310-abi3-win32.whl",
            "starfix-0.6.0-cp312-cp312-win_amd64.whl",
            "other-0.6.0-cp310-abi3-win_amd64.whl",
            "starfix-0.6.0.zip",
            "README.txt",
            "sub/starfix-0.6.0-cp310-abi3-win_amd64.whl",
        ):
            with self.subTest(extra=extra), self.assertRaises(rc.ReleaseCheckError):
                self.check(RELEASE_FILES + [extra])

    def test_refuses_two_wheels_of_one_target(self):
        second = "starfix-0.6.0-cp310-abi3-manylinux_2_28_x86_64.whl"
        with self.assertRaisesRegex(rc.ReleaseCheckError, "more than one"):
            self.check(RELEASE_FILES + [second])

    def test_reports_every_problem(self):
        with self.assertRaises(rc.ReleaseCheckError) as caught:
            self.check(RELEASE_FILES[2:-1] + ["README.txt"])
        message = str(caught.exception)
        self.assertIn("x86_64-unknown-linux-gnu", message)
        self.assertIn("aarch64-unknown-linux-gnu", message)
        self.assertIn("sdist", message)
        self.assertIn("README.txt", message)

    def test_the_command_lists_the_files_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in RELEASE_FILES:
                Path(tmp, name).write_bytes(b"x")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(rc.main(["dist-complete", "--dir", tmp, "--version", "0.6.0"]), 0)
            self.assertIn("starfix-0.6.0.tar.gz", out.getvalue())
            Path(tmp, RELEASE_FILES[0]).unlink()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(rc.main(["dist-complete", "--dir", tmp, "--version", "0.6.0"]), 1)
            self.assertIn("::error::", out.getvalue())


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
