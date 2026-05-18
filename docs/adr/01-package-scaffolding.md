# ADR-01: Standalone `aiida-agents` package, scaffolded from `python-copier`

## Context

`aiida-agents` is a new, exploratory package — a natural-language, multi-agent interface to AiiDA — and needs an engineering foundation.
Constraints: zero `aiida-core` changes; the design must allow external AiiDA plugins (e.g. `aiida-quantumespresso`) to contribute agents, MCP tools and RAG knowledge later; engineering scaffolding (tests, linting, typing, CI) must exist _before_ any feature code, so quality is enforced and the work is reproducible and reviewable from the first commit.

The AiiDA ecosystem is entry-point based and uses an idiomatic packaging toolchain; `mbercx/python-copier` already encodes most of that toolchain.

## Decision

We will develop the work as a **new standalone package `aiida-agents`** under the `aiidateam` GitHub organisation, with **no changes to `aiida-core`**, and generate it via `copier` from the original `mbercx/python-copier` template (pinned to tag `v0.16.0`).
This is Marnik Bercx's own template (`mbercx`, a core AiiDA developer) — not an official AiiDA standard, but its choices are sound and we deliberately adopt and follow it.
`_src_path` records the upstream directly — for due credit, and so `copier update` tracks his tagged releases (the template author tags releases; an optional in-repo bot can open update PRs, all PR-gated and reviewed).
With these answers:

| Copier question | Answer         | Rationale                                                                                                                               |
| --------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `package_name`  | `aiida-agents` | AiiDA naming convention; the package is about _agents_, MCP/RAG are implementation details                                              |
| `docs`          | `myst`         | AiiDA-ecosystem-idiomatic MyST docs                                                                                                     |
| `doc_deploy`    | `nowhere`      | No docs to deploy yet; revisit when there are (note: the template only offers `rtd` with `mkdocs`, so MyST + RTD is not an option here) |
| `type_check`    | `strict`       | `mypy --strict` from commit one is the highest-leverage quality guardrail; the initial friction is accepted deliberately                |
| `coverage`      | `codecov`      | Consistency with `aiida-core`                                                                                                           |

Generated toolchain: Hatch build backend, Ruff lint+format via pre-commit, pytest (+ Codecov), MyST docs, GitHub Actions for pre-commit / commit-message lint / tests, and PyPI publishing via Trusted Publishing on `vX.Y.Z` tags.

Reproducibility: the exact answers are recorded in `.copier-answers.yml`; template updates flow via `copier update`.

**Lockfile:** `uv.lock` is tracked in version control, replicating the `aiida-core` workflow, with an `astral-sh/uv-pre-commit` `uv-lock` hook keeping it current.
We track it because a pinned, reproducible dev/CI environment is high-value for reproducible development and debugging.
A committed lockfile never affects downstream `pip install aiida-agents` resolution, and CI runs via Hatch (resolving against the `pyproject.toml` ranges, not the lock), so the lock does not mask dependency range-rot — no separate "unlocked"/compat CI job is needed.
One wrinkle: the template's task runner is Hatch while the pinned env is `uv` (`uv sync`/`uv.lock`); these coexist (Hatch for tasks/CI, `uv.lock` for reproducible local dev).

**Markdown formatting:** authored Markdown is formatted with `mdformat` (+ `mdformat-myst`, `mdformat-tables`) via a pre-commit hook, configured in `.mdformat.toml`.
The AiiDA ecosystem runs no Markdown tooling, but consistent style is worth the deliberate divergence here.
`mdformat` is the only safe choice for this repo: Python (no Node added), MyST-aware (directives preserved), leaves fenced code — including the Mermaid diagrams in ADRs — verbatim, and does not hard-wrap prose (so it preserves our one-sentence-per-line convention).
The hook is scoped to `docs/adr/` and `docs/gsoc/` (authored docs) so copier-managed template Markdown is untouched and `copier update` stays conflict-free.
`mdformat-frontmatter` is deliberately excluded — it conflicts with `mdformat-myst` (both render `front_matter`).
Pinned to mdformat `0.7.22` because the plugins constrain mdformat `<1.0`.
`mdformat` does not read `pyproject.toml` (verified), so `.mdformat.toml` is the lone unavoidable standalone tool-config file (Ruff and mypy live in `pyproject.toml`); it is kept minimal — only the non-default, project-critical settings — and is the single config source shared by the hook and any editor run.

**Local dev workflow:** `uv` is the main, documented local workflow (`uv sync`, then `uv run pre-commit` / `pytest` / `mypy`), matching aiida-core.
The template's Hatch envs and Hatch-based CI are kept template-shaped (untouched) so `copier update` stays clean; Hatch therefore remains fully usable locally (`hatch run …`, `hatch test`) for anyone who prefers it — it is simply not required.
Practical commands live in `CONTRIBUTING.md`.
Caveat: local `uv run pytest` is single-Python; the 3.10–3.14 matrix runs in CI.

## Consequences

- Quality gates (tests + types green in CI) are enforced from the first commit, not deferred to a late cleanup.
- `type_check=strict` and `coverage` (`fail_under` in `pyproject.toml`) are reversible single-line toggles, or re-run via `copier update`, if the strictness proves counterproductive.
- `doc_deploy=nowhere` means enabling a docs site later is a `copier update` (or a switch to `mkdocs` if RTD is wanted).
- `aiida-core` is never modified — `aiida-agents` is its own repository.
  Within it, work lands via feature-branch pull requests, reviewed on every PR.
- The plugin-extensibility requirement is **not** addressed by scaffolding; it is deferred to a later ADR and is deliberately built _after_ a working concretion, not up front.

## Alternatives considered

- **Hand-rolled package layout / different template.** Rejected: re-deriving an idiomatic AiiDA toolchain wastes time and diverges from ecosystem norms; `python-copier` is maintained by a core AiiDA developer.
- **Name `aiida-mcp`.** Rejected: the package is about agents, not one transport (MCP may be swapped or complemented), and `aiida-mcp` is already taken by an internal colleague's project — reusing it would collide and confuse.
  The naming should not frame this as "the MCP package" anywhere (docs, entry-point groups, CLI).
- **Module inside `aiida-core`.** Rejected: violates the zero-core-changes constraint and the plugin-extensibility goal.
- **`type_check=loose`.** Rejected: strict typing is a primary quality lever from the outset.
  Revisit only if it proves counterproductive.
- **Do not track `uv.lock`** (the `python-copier` and `aiida-restapi` default). Rejected: a reproducible pinned dev/CI environment outweighs the small lockfile-churn cost, and it matches `aiida-core`.
  The usual library objection (CI masking range-rot) does not apply here because CI resolves via Hatch against `pyproject.toml` ranges, not the lock.
- **No Markdown formatter** (the ecosystem/template default). Rejected: the consistency is worth a deliberate divergence.
- **Standardize everything on `uv` (strip Hatch), or force Hatch on devs.** Rejected: stripping template-owned Hatch envs/CI breaks clean `copier update`; forcing Hatch on devs adds a second task runner with no functional gain over `uv`.
  `uv` is the human interface; Hatch stays untouched CI/build plumbing.
