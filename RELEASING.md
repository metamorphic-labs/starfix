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
byte-identical to starfix 0.5.0's for the same input and options. Do not release the Python
package unless the test that proves this against the published 0.5.0 passes on the release
commit: `python/tests/test_differential.py`, in the job `python-binding (newest pyarrow)` of
`ci.yml`.

## Who releases

Only the maintainer (@eywalker) releases. Nobody else, and no automation or agent, pushes a release
tag, runs `release.yml`, or publishes to crates.io, TestPyPI or PyPI.

## Publishing is trusted publishing (OIDC), with no stored token

No API token for crates.io, TestPyPI or PyPI is stored in this repository. Each registry has a
trusted publisher registered for this repository, the workflow file and the environment named in
the table above. Only the job that publishes declares that environment and gets
`permissions: id-token: write`; every other job has `contents: read`.

## Release steps

1. `main` is green, and `CHANGELOG.md` has the release's section.
2. Actions → **release** → *Run workflow* on `main`, with the version (for example `0.6.0`).
   The job runs in the environment `crates`. cargo-release bumps `Cargo.toml`, commits
   `chore: release v<version>`, pushes the commit and the tag `v<version>`, and publishes the
   crate to crates.io.
3. The pushed tag `v<version>` triggers `ci.yml`, which checks that the tag matches `Cargo.toml`.
4. The same tag triggers `maturin-release.yml`. It checks that the tag matches `Cargo.toml`,
   builds and tests the wheels and the sdist, publishes them to TestPyPI (environment
   `testpypi`), and then to PyPI (environment `pypi`).

## Repository setup the maintainer does (not done by any workflow or pull request)

- Environments `crates`, `testpypi` and `pypi` exist in *Settings → Environments*. Each one should
  allow deployments only from `main` and `v*` tags, and may require the maintainer's approval.
- Trusted publishers registered for owner `metamorphic-labs`, repository `starfix`:
  - crates.io, crate `starfix`: workflow `release.yml`, environment `crates`;
  - TestPyPI and PyPI, project `starfix`: workflow `maturin-release.yml`, environments `testpypi` and
    `pypi`. The publishers of `starfix-python` for the PyPI project are removed once 0.6.0 is out.
- The secrets `RELEASE_APP_ID` and `RELEASE_APP_PRIVATE_KEY` (a GitHub App that may push to this
  repository) are used by `release.yml`. The App token is needed so that the tag pushed by
  cargo-release triggers the tag workflows.
