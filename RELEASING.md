# Releasing starfix

`metamorphic-labs/starfix` is the release source of both published forms of starfix. One
workflow, `.github/workflows/maturin-release.yml`, publishes all three, when the maintainer pushes
a release tag:

| Order | Registry | Package | Job | GitHub environment |
|---|---|---|---|---|
| 1 | TestPyPI | `starfix` (wheels + sdist) | `publish to TestPyPI` | `testpypi` |
| 2 | crates.io | `starfix` (crate) | `publish to crates.io` | `crates` |
| 3 | PyPI | `starfix` (wheels + sdist) | `publish to PyPI` | `pypi` |

From 0.6.0 the PyPI package `starfix` is built from this repository (`python/starfix`, PyO3).
`metamorphic-labs/starfix-python` (the pure-Python implementation, which published `starfix` up
to 0.5.x) is frozen at 0.5.x and publishes nothing new under that name. starfix-python 0.5.0 and
this crate both produce hash format 2 (`00 00 02`) and share the golden vectors in
`tests/golden`. Every hash of 0.6.0 must be byte-identical to starfix 0.5.0's for the same input
and options, except where the recorded, named differences say otherwise
(`python/tests/_harness.py`, `EXPECTED_DIFFERENCES`; for example 0.5.0 is wrong where pyarrow's
take() is, fixed in 0.5.1: ITL-1223). Two tests prove this against the published 0.5.0:
`python/tests/test_differential.py` (in `ci.yml`, job `python-binding (newest pyarrow)`), and the
seeded harness `python/tests/test_harness.py` (`differential-harness.yml`). The harness runs 2,000
seeds per leg on every pull request, 100,000 nightly on `main`, and 100,000 in
`maturin-release.yml` (job `differential-harness`) on a release tag, before anything is published.

## Who releases

Only the maintainer (@eywalker) pushes a release tag. Nobody else, and no automation or agent,
pushes a tag `v*` or publishes to crates.io, TestPyPI or PyPI. No workflow commits, pushes a
branch or pushes a tag.

## Release steps

1. **Bump pull request.** A pull request against `main` sets the version in `Cargo.toml`
   (`[package] version`; the Python package takes its version from it) and turns the
   `## Unreleased` section of `CHANGELOG.md` into `## <version> (<date>)`. `Cargo.lock` is not
   committed, so there is nothing else to change. It is merged like any other pull request, once
   the required checks pass.
2. **Tag.** The maintainer tags the commit the bump pull request was merged as, and pushes the
   tag, and only that tag. Take that commit from the pull request, not from the tip of `main`:
   a pull request merged after the bump would otherwise be released with it, and the gates
   would not notice (the version and the changelog are the same).

   ```sh
   git fetch origin
   commit=$(gh pr view <bump pull request> --json mergeCommit --jq .mergeCommit.oid)
   git merge-base --is-ancestor "$commit" refs/remotes/origin/main && echo "on main"
   git show --stat "$commit"   # the bump: Cargo.toml and CHANGELOG.md
   git tag -a v0.6.0 -m "starfix 0.6.0" "$commit"
   git push origin v0.6.0
   ```

   Push the one tag by name. Do not use `git push --tags`: GitHub creates no workflow event when
   more than three tags are pushed at once.
3. **The tag workflow.** The push of `v0.6.0` runs `maturin-release.yml` (and `ci.yml`, whose
   `verify-version-tag-sync` checks the tag too). The maintainer approves the deployments to the
   environments `crates` and `pypi` when the run asks (*Review deployments*), and follows the run
   to the end.

### Not a GitHub Release

Creating a GitHub Release with a new tag in the web interface is **not** a supported way to start
a release. GitHub's documentation of the `push` event does not say that it fires for a tag created
that way (community reports disagree), and it could not be tried here without creating a tag in
this repository. The tag pushed with `git push`, as above, is the trigger. A GitHub Release can be
created afterwards for the existing tag, for its notes; that sends a `release` event, which this
workflow does not listen to, so it starts nothing.

## What the tag workflow does

Nothing is published unless every gate passed. The gates run on every release tag:

| Gate | Job |
|---|---|
| The tag is `v` + the version in `Cargo.toml` | `verify-tag` |
| The tagged commit is on `main` (an ancestor of `refs/remotes/origin/main`, named in full so that a tag called `origin/main` cannot stand in for it); a tag on any other commit publishes nothing | `verify-tag` |
| `CHANGELOG.md` has the section `## <version>`, as a heading (a line inside a fenced code block does not count) | `verify-tag` |
| The 100,000-seed differential harness against starfix 0.5.0 passes | `differential-harness` |
| The wheels build and their tests pass in a clean environment (Linux glibc x86_64 and aarch64, macOS universal2, Windows x64; the musllinux wheels build but cannot be tested, as pyarrow has no musllinux wheels) | `linux …`, `macos universal2`, `windows x64` |
| The sdist builds, and a wheel built from it alone passes the tests | `sdist` |
| The crate packages and builds from the package alone (`cargo publish --dry-run`), and `cargo test` passes | `crate-package` |

The gate checks of `verify-tag` are in `.github/scripts/release_checks.py`, unit-tested by the job
`release-checks` on every pull request.

Then the three publications, each needing the previous one:

1. **TestPyPI**, the rehearsal: the first upload of the release (trusted publishing, the files'
   metadata accepted by the index) happens where a mistake costs nothing.
2. **crates.io**, the crate. The job ends by reading the crates.io index until it lists the
   version (every 15 seconds, for at most 10 minutes), and fails if it never does.
3. **PyPI**, last, so PyPI never has a version whose crate is not on crates.io.

Before each upload to TestPyPI and to PyPI, the job checks that `dist/` holds exactly the
release's files: one wheel of the tagged version for each target the run builds (the table
`RELEASE_WHEELS` of `release_checks.py`) and the sdist, and nothing else. A missing artifact
fails the job before anything is uploaded, rather than publishing part of the release. (The
check compares the file names with the tag's version as written, so a pre-release version, which
maturin writes in its PEP 440 form, is refused there.)

## Publishing is trusted publishing (OIDC), with no stored token

No API token for crates.io, TestPyPI or PyPI is stored in this repository. Each registry has a
trusted publisher registered for this repository, the workflow file `maturin-release.yml` and the
environment named in the table above. Only the job that publishes declares that environment and
gets `permissions: id-token: write`; every other job has `contents: read`.

## If the run fails: re-run the failed jobs

A publication cannot be undone, and a run can fail half way (a network error, an index that
answers 503). Each publish job is safe to run again for the same tag:

- **crates.io.** The job first reads the crates.io sparse index
  (`https://index.crates.io/st/ar/starfix`, which lists every published version, yanked ones too).
  When it lists the version, the job says *"starfix <version> is already on crates.io: nothing to
  upload"* and goes on without authenticating or uploading. Only a 404 (a crate never published)
  counts as "not on crates.io"; any other status, an empty answer or a row without a version
  fails the job rather than guess.
- **TestPyPI and PyPI.** `uv publish --check-url <the index>` compares each file with the index
  before uploading it: a file already there with the same SHA-256 is skipped (*"File … already
  exists, skipping"*), a missing file is uploaded, and a file there with other contents is an
  error. So a re-run uploads only what a failed run left out.

To complete a half-done release, open the run and choose **Re-run failed jobs**. It reuses the
wheels and the sdist the first attempt built, so their hashes match what was already uploaded.
Do not choose *Re-run all jobs* once something is published: the wheels would be built again, may
differ byte for byte, and the index would then refuse them as different files of the same name.

When a gate fails, nothing was published. First decide whether the tagged commit is at fault:

- **A transient failure** (the crates.io index answering 503 to the live tests of
  `release-checks`, a network error, a lost runner): the same commit can pass. Open the run and
  choose **Re-run failed jobs**. The tag stays where it is.
- **A fix the commit needs** (a gate that refuses the commit itself, a failing test): fix the
  cause through a pull request; then, as the tag must name a commit with the fix, delete the tag
  and tag the new commit of `main`, as in step 2 (the version is unchanged, since nothing was
  published):

```sh
git push origin :refs/tags/v0.6.0   # delete the remote tag
git tag -d v0.6.0
git fetch origin
git tag -a v0.6.0 -m "starfix 0.6.0" <new commit>
git push origin v0.6.0
```

Once any publication has happened (even TestPyPI only), the version's files exist on that registry
and cannot be replaced: a fix in the code then needs a new version (a new bump pull request).

## What can no longer happen

- **Published but untagged.** The tag exists before anything is published; every publication
  belongs to the tagged commit.
- **A commit pushed to `main` by a workflow.** The version bump reaches `main` through a pull
  request, and no workflow pushes anything.
- **A publication from a commit that is not on `main`**, or of a version other than the tag's, or
  without the `CHANGELOG.md` section.
- **A crate published without the harness.** The crate is gated on the same run's 100,000-seed
  harness and on the wheels' tests, like the wheels.

## Repository setup the maintainer does

Not done by any workflow or pull request.

- **Environment `crates`** (*Settings → Environments → crates*): add the deployment tag rule `v*`
  (today it allows the branch `main` only, which no workflow deploys from any more; that rule can
  be removed), and keep its required reviewer.
- **Environments `testpypi` and `pypi`** allow deployments from `v*` tags, as they do today; `pypi`
  keeps its required reviewer.
- **The crates.io trusted publisher** of the crate `starfix` (*crates.io → starfix → Settings →
  Trusted Publishing*): owner `metamorphic-labs`, repository `starfix`, workflow
  `maturin-release.yml`, environment `crates`. A publisher registered for `release.yml` no longer
  matches any workflow and can be removed.
- **The TestPyPI and PyPI trusted publishers** of the project `starfix` are unchanged: owner
  `metamorphic-labs`, repository `starfix`, workflow `maturin-release.yml`, environments
  `testpypi` and `pypi`. The publishers of `starfix-python` for the PyPI project are removed once
  0.6.0 is out.
- **Unused secrets.** The repository secret `STARFIX_CRATES` and the secrets `RELEASE_APP_ID` and
  `RELEASE_APP_PRIVATE_KEY` of the release GitHub App are read by no workflow. They can be
  deleted, and the App's access to this repository removed if nothing else uses it.
- **Nothing refuses the tag push.** On 2026-11-18 the rulesets that apply to this repository (the
  organization rulesets `standard-code-review-rule` and `standard-rust-project`, the repository
  ruleset `required-ci-checks`) all target the default branch only; no ruleset restricts creating
  `v*` tags. If a tag ruleset is added, the maintainer must still be allowed to create `v*` tags.
