# Architecture Decision Records

This directory holds the Architecture Decision Records (ADRs) for `aiida-agents`.

An ADR records one architecturally significant decision: its context, the decision, and its consequences.
To change a decision, add a new ADR that supersedes it and note that in the log below.
ADRs cover **tooling and architecture only**; project/program planning lives in [`docs/gsoc/`](/docs/gsoc/).

New ADR: create `NN-short-title.md` (increment `NN`) with `Context`, `Decision`, `Consequences`, and `Alternatives considered`.
No formal status/author/date header: we keep it lightweight (two main maintainers).
Use [`01-package-scaffolding.md`](/docs/adr/01-package-scaffolding.md) as the worked example.
Diagrams are embedded as [Mermaid](https://mermaid.js.org/) fenced blocks (and exported UML where a static image is clearer); MyST renders both in the docs site.

## Log

Numbering follows build/dependency order, not chronology: **01–06** are the path to the first milestone (a natural-language agent that reads a real AiiDA database); **07–08** are the write path; **09–10** expansion; **11** exploratory.

| ADR                                                   | Title                                                                                          |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [01](/docs/adr/01-package-scaffolding.md)             | Standalone `aiida-agents` package, scaffolded from `python-copier`                             |
| [02](/docs/adr/02-mcp-tools-wrap-aiida-restapi.md)    | MCP tool layer wraps `aiida-restapi` (not hand-rolled)                                         |
| [03](/docs/adr/03-llm-library.md)                     | Adopt an existing provider-agnostic LLM library (don't hand-roll); local + cloud               |
| [04](/docs/adr/04-multi-agent-architecture.md)        | Read-only provenance-exploration agent first: the first milestone                              |
| [05](/docs/adr/05-rag-over-aiida-docs.md)             | RAG over AiiDA docs: local embeddings, minimal first (hybrid/cross-encoder deferred)           |
| [06](/docs/adr/06-eval-harness.md)                    | Agent-behaviour evaluation harness, deterministic in CI plus an opt-in real-model tier         |
| [07](/docs/adr/07-validator.md)                       | Validator: schema validation delegated to AiiDA; the range/physics tier deferred               |
| [08](/docs/adr/08-human-in-the-loop-before-writes.md) | Enforced human-in-the-loop confirmation before any write/submit                                |
| [09](/docs/adr/09-agent-orchestration.md)             | Agent orchestration: a planner over two specialists; specialists are never wrapped             |
| [10](/docs/adr/10-plugin-extensibility.md)            | Plugin extensibility through one `aiida_agents.plugins` entry point                            |
| [11](/docs/adr/11-code-execution.md)                  | Executing generated code against a write-refusing database role                                |
| 11                                                    | Agent-run provenance: persist agent decisions/traces in AiiDA's provenance graph (exploratory) |

ADR-01 is in effect; ADR-02 and ADR-03 are seeds with direction confirmed (2026-05-22).
ADR-04 through ADR-10 are written; ADR-11 is still exploratory and has no record yet.
ADR-09 supersedes ADR-04's future-architecture table and settles the agent-to-agent question ADR-04 left open; ADR-06 and ADR-07 carry Revision sections where reality diverged from the original decision.

For how the pieces fit together rather than why each was chosen, see [Architecture](/docs/architecture.md); to add your own, see [Extending](/docs/extending.md).
