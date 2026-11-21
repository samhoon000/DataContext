# Contributing to Oribi

Oribi is an open-source Python library for turning pandas DataFrames into structured analytical context. Contributions are welcome, including bug fixes, analytical rules, tests, documentation, performance improvements, and ideas.

Analytical quality matters because Oribi's output can influence downstream analysis. Contributions should make the evidence and uncertainty behind an interpretation clear.

## Before you start

Open a [GitHub issue](https://github.com/samhoon000/Oribi/issues) before implementation when a change could affect how users interpret their data. This is especially useful for:

- New or changed semantic inference rules
- Dataset grain or candidate-key detection
- Aggregation recommendations or analytical warnings
- Public API or architecture changes
- New runtime dependencies

Small fixes, focused tests, and documentation improvements can generally go straight to a pull request. If a proposed approach is uncertain, an issue is a good place to discuss it before investing in implementation.

## Development setup

Oribi supports Python 3.10 through 3.14 and pandas 1.5.3 through the latest pandas 3.x release. The `dev` extra installs pytest, build, ruff, and mypy.

From a clone of the repository, create and activate a virtual environment, then install the development dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite, linter, and type checker from the repository root:

```bash
pytest
ruff check .
mypy oribi
```

GitHub Actions runs the test suite on representative supported Python/pandas combinations, runs `ruff` and `mypy`, and builds wheel and source distributions.

To run the installed package tests locally, build a wheel and install it in a clean virtual environment, then run pytest from outside the checkout. For the release validation used in CI, run:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python scripts/validate_package.py
```

### Running the quickstart with a legacy Windows stdout encoding

The end-to-end example prints Unicode symbols, including the `⚠` warning marker in compact output:

```bash
python -m examples.quickstart
```

On Windows, when stdout uses a legacy encoding such as `cp1252` and the output is redirected or captured (for example, piped through PowerShell), Python can raise `UnicodeEncodeError` before the summary and compact report finish printing. This is an output-encoding issue, not a problem with the example itself, and no global Windows setting needs to change.

To reproduce the failure in PowerShell, temporarily override the stdout encoding, run the default command with its output captured, and then clear the override:

```powershell
$env:PYTHONIOENCODING = 'cp1252'
python -m examples.quickstart | Out-String
Remove-Item Env:PYTHONIOENCODING
```

`PYTHONIOENCODING` takes precedence over the interpreter's UTF-8 mode, so clear that override before running the UTF-8 command. With the override removed, enable UTF-8 for this invocation only:

```powershell
python -X utf8 -m examples.quickstart
```

The command completes with the full summary and compact report, and it leaves the system-wide Windows encoding unchanged.

## Releasing

Set the next PEP 440 version in `project.version` in `pyproject.toml` and `oribi.__version__` in `oribi/__init__.py`. Add a matching `CHANGELOG.md` entry in the same commit. Merge that change, then create a GitHub Release with an annotated tag exactly matching `v<version>`. The release workflow checks the tag against the package metadata, reruns the full test matrix, builds and validates the wheel and source distribution, and publishes those exact artifacts to PyPI only when all prerequisites pass. A version already present on PyPI is immutable. PyPI rejects duplicate uploads and the publish job fails.

Before the first release, configure PyPI Trusted Publishing for this GitHub repository (no PyPI token is stored in GitHub): on PyPI, add a pending publisher for owner `samhoon000`, repository `Oribi`, workflow file `publish.yml`, and environment name `pypi`. In GitHub, create an environment named `pypi` under **Settings → Environments**. Publish a GitHub Release to trigger deployment. The publish job receives only the `id-token: write` permission needed for OIDC.

For example, after the release PR is merged and CI is green, tag the tested release commit and publish its GitHub Release:

```bash
# Replace 0.2.0 with the new version after confirming it is unused on PyPI.
git tag -a v0.2.0 <tested-release-commit> -m "Release Oribi 0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --verify-tag --title "Oribi v0.2.0" --generate-notes
```

If the version is already registered on PyPI, choose a higher unused version and use it consistently in the files, commit, tag, and release. Create the GitHub Release only after its commit is on the default branch and CI is green. The release workflow checks out the release tag.

## Project structure

The implementation separates observations, analytical reasoning, the canonical context models, and output rendering:

| Area | Responsibility |
| --- | --- |
| `oribi/profiling/` | Collects DataFrame observations, column statistics, and conservative within-frame value mappings. |
| `oribi/reasoning/` | Infers semantic roles, candidate keys and grain, warnings, and aggregation recommendations from observations. |
| `oribi/models/` | Defines the typed context, statistics, evidence, warning, key, grain, and recommendation models. |
| `oribi/serialization/` | Renders an existing context as a dictionary, JSON, Markdown, or compact AI-oriented text. |
| `oribi/api.py` | Orchestrates the public `inspect(df)` pipeline. |
| `tests/` | Exercises the public API, profiling, inference, reasoning, and serialization behavior. |

Detection records what the frame contains. Reasoning interprets those observations and records evidence and confidence. The context models hold the result, and serializers present that result. Keep these responsibilities separate when extending the library.

## Contribution workflow

1. Open an issue first for substantial analytical, API, dependency, or architecture changes.
2. Fork the repository and create a branch for your change.
3. Implement the change and add or update behavior-focused tests when appropriate.
4. Run `pytest`, `ruff check .`, and `mypy oribi` from the repository root.
5. Commit your changes and open a pull request describing the change and its effect on analytical behavior.
6. Respond to review feedback and update the pull request as needed.

The repository does not document branch protection or require a specific branch naming scheme. Avoid pushing directly to `main` when the repository's GitHub settings protect it. A readable branch name can use a category prefix such as `feat/measure-detection`, `fix/grain-inference`, `test/semantic-rules`, or `docs/contributing-guide`.

## Commit messages

Conventional Commit-style messages are a useful convention, but the repository does not enforce them. Use a concise type and description, for example:

```text
feat: add measure detection
fix: correct grain confidence calculation
test: cover repeated identifier values
docs: explain analytical evidence
refactor: separate profiling from reasoning
perf: reduce repeated cardinality scans
```

Use `!` or a `BREAKING CHANGE:` footer to call out a change that breaks public API or serialized output compatibility. Avoid vague messages such as `updated stuff` or `final`.

## Analytical logic and evidence

Keep three kinds of statements distinct:

| Kind | Meaning | Example |
| --- | --- | --- |
| Observation | A value measured directly from the input. | `order_id` has 500 distinct non-null values across 1,000 rows. |
| Inference | A qualified interpretation of observations. | `order_id` is likely an identifier, with confidence and supporting evidence. |
| Recommendation | Suggested analytical use based on observations and inference. | Use `COUNT(DISTINCT order_id)` when counting orders. |

An inference should not be stated as a fact, and a recommendation should be traceable to its evidence where practical. Prefer multiple supporting signals over a single weak hint. Confidence expresses heuristic strength; it is not a guarantee of business meaning. Keep rules deterministic and make uncertainty visible.

Changes to analytical behavior need particular care. A small change to a score or threshold can affect AI-oriented context, aggregation guidance, warnings, and grain estimates. Describe the behavior change and its rationale in the pull request, with a before/after example when useful.

## Tests and regression coverage

Add or update tests when changing profiling or reasoning behavior. Relevant areas include semantic roles, candidate keys, grain estimates, recommendations, warnings, relationships, confidence, missing values, and serialization.

Use both ordinary and edge-case data where it helps explain the behavior. Consider unique and repeated identifiers, duplicate rows, ambiguous grain, missing values, negative measures, numeric dimensions, datetime columns, high-cardinality columns, and empty frames. A bug fix should ideally include a regression test that reproduces the original issue.

Tests should assert observable behavior and structured evidence where possible. Avoid relying only on exact prose if a stable model field or warning code describes the behavior more directly.

## Dependencies and public API

The runtime dependency set is intentionally small: pandas is the only required dependency. Before adding a runtime dependency, explain why it is needed, whether pandas or the standard library can meet the need, and the effect on installation and maintenance.

The public entry point is `oribi.inspect(df)`. It returns a `DatasetContext` with `summary()`, `to_dict()`, `to_json()`, `to_markdown()`, and `to_context()` methods. `ProfileOptions` is also exported for controlling a small set of profiling options. Treat public method signatures and serialized fields as compatibility-sensitive. Before changing them, consider existing callers, tests, documentation, and a migration path.

## Pull requests

Pull requests should explain the user-visible change and include relevant validation details. Use this outline if helpful:

```markdown
## What does this PR do?

Describe the change.

## Why?

Explain the problem or use case.

## Changes

- Summarize the implementation.
- Note any public API or dependency changes.

## Analytical behavior

Describe any change to roles, grain, keys, recommendations, warnings, or relationships.

## Testing

List the checks run and any relevant new test coverage.
```

Before opening a pull request, check the items that apply:

- [ ] I have added or updated tests for behavior changes.
- [ ] `pytest` passes locally.
- [ ] I have described changes to analytical behavior.
- [ ] I have documented public API or dependency changes.
- [ ] I have updated relevant documentation or docstrings.
- [ ] I have considered compatibility with existing users.

## Documentation and community

Update documentation or docstrings when a contribution changes user-visible behavior. Explain analytical behavior and its limitations as well as the API details.

The repository currently has no `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue templates, or discussion configuration. Use [GitHub Issues](https://github.com/samhoon000/Oribi/issues) to report bugs, propose changes, or ask project questions.
