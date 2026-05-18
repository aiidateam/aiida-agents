# Architecture Decision Records

This directory holds the Architecture Decision Records (ADRs) for `aiida-agents`.

An ADR records one architecturally significant decision: its context, the decision, and its consequences.
To change a decision, add a new ADR that supersedes it and note that in the log below.
ADRs cover **tooling and architecture only**; project/program planning lives in [`docs/gsoc/`](/docs/gsoc/).

New ADR: create `NN-short-title.md` (increment `NN`) with `Context`, `Decision`, `Consequences`, and `Alternatives considered`.
No formal status/author/date header — we keep it lightweight (two main maintainers).
Use [`01-package-scaffolding.md`](/docs/adr/01-package-scaffolding.md) as the worked example.
Diagrams are embedded as [Mermaid](https://mermaid.js.org/) fenced blocks (and exported UML where a static image is clearer); MyST renders both in the docs site.

## Log

Numbering follows build/dependency order, not chronology: **01–06** are the path to the first milestone (a natural-language agent that reads a real AiiDA database); **07–08** are the write path; **09–10** expansion; **11** exploratory.

| ADR                                                | Title                                                                                                                   |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [01](/docs/adr/01-package-scaffolding.md)          | Standalone `aiida-agents` package, scaffolded from `python-copier`                                                      |
| [02](/docs/adr/02-mcp-tools-wrap-aiida-restapi.md) | MCP tool layer wraps `aiida-restapi` (not hand-rolled)                                                                  |
| [03](/docs/adr/03-llm-library.md)                  | Adopt an existing provider-agnostic LLM library (don't hand-roll); local + cloud                                        |
| 04                                                 | Read-only provenance-exploration agent first — the first milestone                                                      |
| 05                                                 | RAG over AiiDA docs: local embeddings, minimal first (hybrid/cross-encoder deferred)                                    |
| 06                                                 | Agent-behaviour evaluation harness (golden NL → expected tool-calls/answers)                                            |
| 07                                                 | Validator: deterministic checks (schema + ranges) plus an optional LLM pass                                             |
| 08                                                 | Enforced human-in-the-loop confirmation before any write/submit                                                         |
| 09                                                 | Agent orchestration: single-agent first, then orchestrator + ≤3 specialists; A2A vs. function-calls decided empirically |
| 10                                                 | Plugin extensibility via `aiida.*` entry points (concretion before abstraction)                                         |
| 11                                                 | Agent-run provenance: persist agent decisions/traces in AiiDA's provenance graph (exploratory)                          |

ADR-01 is in effect; ADR-02 and ADR-03 are bare-bones seeds; 04–11 are planned and written up (with diagrams) as each decision is finalized.
