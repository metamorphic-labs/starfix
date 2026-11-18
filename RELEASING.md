# Releasing starfix

`metamorphic-labs/starfix` is the release source of both published forms of starfix:

| Registry | Package | Built from | Workflow | GitHub environment |
|---|---|---|---|---|
| crates.io | `starfix` (crate) | this repository | `release.yml` | `crates` |
| TestPyPI | `starfix` (wheels + sdist) | this repository (`python/starfix`, PyO3) | `maturin-release.yml` | `testpypi` |
| PyPI | `starfix` (wheels + sdist) | this repository (`python/starfix`, PyO3) | `maturin-release.yml` | `pypi` |

From 0.6.0 the PyPI package `starfix` is built from this repository. `metamorphic-labs/starfix-python`
(the pure-Python implementation, which published `starfix` up to 0.5.x) is frozen at 0.5.x and
publishes nothing new under that name. starfix-python 0.5.0 and this crate both produce hash
format 2 (`00 00 02`) and share the golden vectors in `tests/golden`. Every hash of 0.6.0 must be
byte-identical to starfix 0.5.0's for the same input and options, except where the recorded,
named differences say otherwise (`python/tests/_harness.py`, `EXPECTED_DIFFERENCES`; for example
0.5.0 is wrong where pyarrow's take() is, fixed in 0.5.1: ITL-1223). Two tests prove this
against the published 0.5.0: `python/tests/test_differential.py` (in `ci.yml`, job
`python-binding (newest pyarrow)`), and the seeded harness `python/tests/test_harness.py`
(`differential-harness.yml`). The harness runs 2,000 seeds per leg on every pull request,
100,000 nightly on `main`, and 100,000 in `maturin-release.yml` (job `differential-harness`),
which the PyPI and TestPyPI publish jobs need: a tag that fails it publishes no wheels. The
crate is published by `release.yml` before the tag exists, and is not gated on that run.

## Who releases

Only the maintainer (@eywalker) releases. Nobody else, and no automation or agent, pushes a release
tag, runs `release.yml`, or publishes to crates.io, TestPyPI or PyPI.

## Publishing is trusted publishing (OIDC), with no stored token

No API token for crates.io, TestPyPI or PyPI is stored in this repository. Each registry has a
trusted publisher registered for this repository, the workflow file and the environment named in
the table above. Only the job that publishes declares that environment and gets
`permissions: id-token: write`; every other job has `contents: read`.

## Release steps

A release never pushes a commit to `main`. `main` accepts changes only through a pull request
with the required checks, and `release.yml` only publishes the version that `main` already has
and pushes a tag.

1. **Bump pull request.** A pull request against `main` that sets the version in `Cargo.toml`
   (`[package] version`; the Python package takes its version from it) and turns the
   `## Unreleased` section of `CHANGELOG.md` into `## <version> (<date>)`. `Cargo.lock` is not
   committed, so there is nothing else to change.
2. **Merge it** once the required checks pass.
3. **Run the workflow.** Actions → **release** → *Run workflow* on `main`, with the version (for
   example `0.6.0`). The job runs in the environment `crates`. Before anything is published, it
   refuses, with a message, when:
   - it does not run on `main`;
   - the version in `Cargo.toml` is not the version given;
   - `CHANGELOG.md` has no `## <version>` section;
   - the tag `v<version>` already exists on GitHub;
   - that version is already on crates.io (yanked versions included);
   - the App token cannot be created, or is not accepted for a push to this repository.

   It then publishes the crate to crates.io (`cargo publish`), and pushes the tag `v<version>`,
   on the commit the run was dispatched on, and nothing else.
4. The tag `v<version>` triggers `ci.yml`, which checks that the tag matches `Cargo.toml`.
5. The same tag triggers `maturin-release.yml`. It checks that the tag matches `Cargo.toml`,
   builds and tests the wheels and the sdist, runs the 100,000-seed harness against starfix
   0.5.0, publishes them to TestPyPI (environment `testpypi`), and then to PyPI (environment
   `pypi`).

### The one remaining window: the tag push after the publication

A version on crates.io cannot be removed, so the publication is the point of no return, and the
tag push is the one step after it. If the tag push fails (for example the App lost its access,
or a tag ruleset refuses it), the crate is published, the tag does not exist, and no wheels are
built. The job fails, and its output and summary give the commit; the maintainer then creates the
tag on that commit by hand:

```sh
git fetch origin
git tag -a v<version> -m "starfix <version>" <commit>
git push origin v<version>
```

The tag pushed by the maintainer triggers the tag workflows as above. Do not run `release` again
for that version: it refuses, as the version is on crates.io. If `cargo publish` itself fails,
the output says to check whether the version reached crates.io: if it did not, fix the cause and
run the workflow again; if it did, create the tag by hand as above.

## The maintainer's checklist (not done by any workflow or pull request)

- **Environments** `crates`, `testpypi` and `pypi` exist in *Settings → Environments*. Each one
  should allow deployments only from `main` and `v*` tags, and may require the maintainer's
  approval.
- **Trusted publishers**, registered for owner `metamorphic-labs`, repository `starfix`:
  - crates.io, crate `starfix`: workflow `release.yml`, environment `crates`;
  - TestPyPI and PyPI, project `starfix`: workflow `maturin-release.yml`, environments `testpypi`
    and `pypi`. The publishers of `starfix-python` for the PyPI project are removed once 0.6.0 is
    out.
- **The release App.** `release.yml` reads the secrets `RELEASE_APP_ID` and
  `RELEASE_APP_PRIVATE_KEY`, the ID and private key of a GitHub App. They are not among this
  repository's own secrets; they may be organization secrets, and if so they must be visible to
  this repository (*Organization settings → Secrets and variables → Actions*, repository access).
  The App must be installed on this repository with the permission **Contents: read and write**
  (to push the tag). The App token is needed so that the pushed tag triggers the tag workflows:
  a tag pushed with `GITHUB_TOKEN` would not.
- **Nothing may refuse the tag push.** No tag ruleset applies to `v*` tags today (the rulesets
  of this repository and of the organization all target the default branch). If a tag ruleset
  is added that restricts creating `v*` tags, the release App must be one of its bypass actors.
- **`STARFIX_CRATES`.** The repository secret `STARFIX_CRATES` is not read by any workflow:
  publishing to crates.io uses trusted publishing. The maintainer can delete it.
