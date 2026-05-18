# ADR-03: Adopt an existing provider-agnostic LLM library (don't hand-roll)

> Bare-bones seed — the concrete library is chosen during implementation; this records the decision *not* to build our own abstraction.

## Context

Agents need to turn natural language into tool calls, and the project targets both local and cloud models (local is a hard gate; see the timeline).
A unified way to talk to OpenAI, Anthropic, and local models is required.

Ollama is **not** that layer: it is a local-model runner that serves models and exposes its own API plus an OpenAI-compatible `/v1` endpoint.
It covers local serving (and an OpenAI-shaped shim), not Anthropic or arbitrary providers.

Provider-agnostic LLM libraries already exist and are mature; building our own abstraction would be reinventing a solved problem (and contradicts the project's reuse ethos — cf. ADR-01, ADR-02).

## Decision

Adopt an existing provider-agnostic LLM library rather than hand-rolling an abstraction.
The concrete choice is made during implementation; candidates:

- **LiteLLM** — one `completion()` over OpenAI, Anthropic, Ollama, 100+.
- **Pydantic AI** — typed, model-agnostic, by the Pydantic team; fits the project's Pydantic-heavy stack (`aiida-restapi` models, the validator).
- **`llm`** (Simon Willison) — pluggable models, simple.

The local-model requirement still applies regardless of the library chosen.

## Consequences

- No bespoke LLM abstraction to build or maintain; provider/model swaps come from the library.
- A library dependency is added; the local-model gate must be verified against whichever library is selected.

## Alternatives considered

- **Hand-roll a `LLMClient` abstraction.** Rejected: reinvents a solved problem.
- **Ollama only.** Rejected: local serving + OpenAI shim only; no Anthropic / multi-provider.
- **Raw per-provider SDKs side by side.** Rejected: duplicates the normalization a single library already provides.
