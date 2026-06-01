# Contributing to `aiida-agents`

**`uv` is the main local workflow.**
Hatch works too if you prefer it (the template ships Hatch envs); it just isn't required locally.

## One-time setup

```bash
uv sync                       # create .venv from uv.lock + dev dependencies
uv run pre-commit install     # install git hooks (pre-commit + commit-msg)
```

## Everyday commands

| Task | Command | When |
|---|---|---|
| Add / remove a dependency | `uv add <pkg>` / `uv remove <pkg>` | adding/removing a dependency — updates `pyproject.toml`, `uv.lock`, and the env in one step |
| Sync the environment | `uv sync` | after `git pull`, or to (re)create the env from `uv.lock` |
| Run all checks | `uv run pre-commit run --all-files` | before pushing / opening a PR |
| Run one check | `uv run pre-commit run ruff --all-files` | iterating on a single hook (`ruff`, `mypy`, `mdformat`, `uv-lock`, …) |
| Run tests | `uv run pytest` | while developing |
| Type-check only | `uv run mypy` | iterating on types (also runs in pre-commit) |
| Build docs | `cd docs && uv run myst build` | checking docs render |
| Serve docs live | `cd docs && uv run myst start` | writing docs |

Hooks also run automatically on `git commit`.
`uv run pre-commit run --all-files` is the manual full pass.

## Running the MCP server locally

These commands assume the project venv is active (`.venv`); otherwise prefix each with `uv run`.

The server loads your **default** AiiDA profile on startup (a FastMCP lifespan calls `load_profile()`), so it always talks to whichever profile is default in whichever config directory AiiDA resolves to.
That directory is `~/.aiida` unless `AIIDA_PATH` is set, in which case AiiDA uses the first `AIIDA_PATH` entry that contains a `.aiida` folder (appending `.aiida` for you): for a setup at `/path/to/project/.aiida`, `export AIIDA_PATH=/path/to/project`.
Set it in the same shell that launches the server.
Then make the profile you want the default: `verdi profile setdefault <name>` (check with `verdi profile list`, the default is marked `*`).

One command launches the server and the MCP Inspector together, over stdio:

```bash
fastmcp dev src/aiida_agents/mcp/server.py:mcp     # needs node/npx on PATH
```

If the Inspector shows `ECONNREFUSED 127.0.0.1:8000`, it's reusing a cached "Streamable HTTP → :8000" connection from an earlier run instead of the stdio server `fastmcp dev` just started: switch the Inspector's transport to **STDIO** (or clear its stored connection).

To use streamable-http instead (e.g. a long-lived server), run the server and the Inspector in two terminals:

```bash
# terminal 1 — serve over streamable-http
python -m aiida_agents.mcp.server          # serves http://127.0.0.1:8000/mcp/

# terminal 2 — launch the Inspector, then set Transport = "Streamable HTTP",
#              URL = http://127.0.0.1:8000/mcp/, and Connect
npx @modelcontextprotocol/inspector
```

For tool metadata without a GUI: `fastmcp inspect src/aiida_agents/mcp/server.py:mcp`.

## Lockfile (`uv.lock`)

`uv.lock` is committed.
`uv add` / `uv remove` update it (and `pyproject.toml`) automatically.
If you edit `pyproject.toml` dependencies by hand, the `uv-lock` pre-commit hook regenerates it (or run `uv lock`).
Either way, stage and commit the updated `uv.lock` with your change.
Optional-group deps: `uv add --optional <extra> <pkg>`; dev-group deps: `uv add --group dev <pkg>`.

## Markdown

Authored Markdown under `docs/adr/` and `docs/gsoc/` is auto-formatted by `mdformat` via pre-commit (Mermaid blocks and tables preserved, prose never hard-wrapped).
Other (copier-managed) Markdown is not auto-formatted.

**Write one sentence per line** in prose paragraphs (semantic line breaks) — the same style as our blog posts.
It keeps PR diffs to one changed sentence per line.
This is a convention, not enforced by a tool: `mdformat wrap="keep"` preserves the breaks but cannot create them (reliable prose sentence-splitting isn't possible — abbreviations, versions, decimals, URLs), and review catches the rest.

## Commit messages

Each commit subject starts with a leading emoji indicating the type of change.
Enforced locally by the `commit-msg` hook (`dev/check_commit_msg.py`) and in CI by a `commit-msgs` job that checks every commit in a PR.
`dev/update_changelog.py` uses the same emojis to sort commits into changelog sections at release time, so the type is captured while the change is fresh.
Full specification and emoji table: the [`python-copier` commit conventions](https://mbercx.github.io/python-copier/dev-standards/#specifying-the-type-of-change).

## Linting (Ruff) — ignored rules

From the [Ruff ruleset](https://docs.astral.sh/ruff/rules/) we ignore, globally:

| Code | Rule | Rationale |
|---|---|---|
| `TRY003` | [raise-vanilla-args](https://docs.astral.sh/ruff/rules/raise-vanilla-args/) | Pre-formatting exception messages hurts readability for a minor gain. |
| `EM101` | [raw-string-in-exception](https://docs.astral.sh/ruff/rules/raw-string-in-exception/) | Same as `TRY003`. |
| `EM102` | [f-string-in-exception](https://docs.astral.sh/ruff/rules/f-string-in-exception/) | Same as `TRY003`. |
| `PLR2004` | [magic-value-comparison](https://docs.astral.sh/ruff/rules/magic-value-comparison/) | Scientific code has many magic values; naming them all reduces readability for little benefit. |
| `FBT002` | [boolean-default-value-positional-argument](https://docs.astral.sh/ruff/rules/boolean-default-value-positional-argument/) | Understood, but adhering is not a small change; disabled for now. |
| `TID252` | [relative-imports](https://docs.astral.sh/ruff/rules/relative-imports/) | Relative imports (not going up a level) are more readable. |

Additionally, in `tests/`: `INP001` (no `__init__.py` needed) and `S101` (`assert` is fine in tests).
In `dev/`: `INP001` and `T201` (`print()` is fine in dev scripts).

## Coverage

`fail_under = 80` (`pyproject.toml`) is a floor, not the goal — and the global number is a blunt instrument here, so target coverage by *what the code is*, not by chasing the percentage.
The deterministic, safety-critical core — the Validator (schema/range checks), HITL enforcement, `QueryBuilderDict`/tool-input construction — should be ~100% covered with real objects.
External-IO and LLM/agent boundaries (network, model calls) are marked `# pragma: no cover` and validated by the eval harness, not by mock-to-hit-lines tests — which would contradict our "real objects, test the contract" philosophy.
Codecov is configured `target: auto` / `threshold: 0%` (no-regression), so coverage can never drop; the floor only sets the bar.

## Release

Releases are cut by pushing a `vX.Y.Z` tag; `.github/workflows/cd.yaml` then builds an sdist+wheel with Hatch and publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

One-time setup (per project): register the repo as a PyPI Trusted Publisher and create a `pypi` GitHub environment — see the [`python-copier` first-publication guide](https://mbercx.github.io/python-copier/publishing/).

To cut a release:

1. Bump the version and draft the changelog: `hatch run bump <new-version>`.
   This runs `hatch version` to update `src/aiida_agents/__about__.py`, then `dev/update_changelog.py` to prepend a `CHANGELOG.md` section with commits sorted by type.
   Review/edit the changelog and commit the bump on `main` (via PR).
2. Tag and push: `git tag -a v<new-version> -m '🚀 Release v<new-version>'` then `git push origin v<new-version>`.
3. `cd.yaml` picks up the tag, builds, and publishes to PyPI.

The git tag and the version in `__about__.py` must agree — PyPI only sees the version baked into the built distribution, so a mismatch silently publishes under the wrong version or is rejected as a duplicate.

## Notes

- CI uses Hatch and runs the test matrix on Python 3.10–3.14 (via `hatch test`).
  `uv` is the main local workflow, but you can use Hatch locally too (`hatch run …`, `hatch test`) if you prefer — it just isn't required.
- `uv run pytest` tests a single Python version.
  A failure specific to another version (e.g. 3.14) will surface in CI; to reproduce locally: `uv run --python 3.14 pytest`.
