# ADR-03: Adopt Pydantic AI as the provider-agnostic LLM library

> Status: accepted — Pydantic AI adopted and implemented as of Weeks 1–2 (May 2026).

## Context

Agents need to turn natural language into tool calls, and the project targets
both local and cloud models — local is a hard gate given HPC cluster constraints.

A unified way to talk to OpenAI, Anthropic, and local Ollama models is required.
Ollama is not that layer: it is a local-model runner with an OpenAI-compatible
`/v1` endpoint, not a multi-provider abstraction.

Provider-agnostic LLM libraries already exist and are mature; hand-rolling
an abstraction would reinvent a solved problem.

## Decision

**Adopt Pydantic AI** (`pydantic-ai-slim[openai,anthropic]`) as the agent and
tool layer. The slim variant pulls only the OpenAI and Anthropic extras,
avoiding unnecessary SDK bloat.

### Why Pydantic AI over the alternatives

- **LiteLLM** — swaps the model call but provides no agent or tool structure.
  We need both; LiteLLM alone is half a solution.
- **`llm` (Simon Willison)** — simple and pluggable but no typed tool schema,
  no structured output, no native HITL support. Too minimal for this use case.
- **Pydantic AI** — typed agents, structured tool calls, native
  `requires_approval` HITL support (ADR-08), and fits the project's
  Pydantic-heavy stack. The team behind it actively maintains it.

### Provider support

Four providers are supported via `agents/_models.py`, selected at runtime
from `AIIDA_AGENT_PROVIDER`:

| Provider            | Model class                          | Notes                                                                                            |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `ollama` (default)  | `OpenAIChatModel` + `OllamaProvider` | Local; `OLLAMA_BASE_URL` sets endpoint                                                           |
| `openai`            | `OpenAIChatModel`                    | Reads `OPENAI_API_KEY`                                                                           |
| `anthropic`         | `AnthropicModel`                     | Reads `ANTHROPIC_API_KEY`                                                                        |
| `openai-compatible` | `OpenAIChatModel` + `OpenAIProvider` | Any OpenAI-compatible endpoint (DeepSeek, Together, vLLM, etc.); requires `AIIDA_AGENT_BASE_URL` |

### Model strategy: local + cloud, dual-path

Two tracks run in parallel:

- **Local models via Ollama** — offline-capable, runs on HPC clusters with no
  outbound internet. Default dev model is `qwen3.5:2b` (placeholder only —
  too small for reliable tool calling). For real use, `qwen3.6:27b` or
  `gemma3:12b` are recommended.
- **Cloud models** — capable out of the box, no extra infra. Useful for
  evaluating answer quality against local models.

The provider abstraction means switching between tracks is a `.env` change,
not a code change.

### No module-level side effects

The model is constructed in a `get_model()` factory called from `get_agent()`,
which is called from `cli.main()` after `load_dotenv()`. Importing the agents
package is inert — no filesystem access, no environment mutation, no model
construction at import time.

## Consequences

- No bespoke LLM abstraction to build or maintain.
- Provider and model swaps are `.env` changes; no code changes required.
- `pydantic-ai-slim[openai,anthropic]` keeps the dependency surface minimal.
- The local-model gate is verified: the agent runs end-to-end against
  `qwen3.5:2b` via Ollama with no cloud dependency.

## Alternatives considered

- **LiteLLM.** Rejected: model-call abstraction only, no agent/tool structure.
- **`llm` (Simon Willison).** Rejected: no typed tool schema or HITL support.
- **Hand-roll a `LLMClient`.** Rejected: reinvents a solved problem.
- **Ollama only.** Rejected: local serving + OpenAI shim only, no Anthropic.
- **Raw per-provider SDKs side by side.** Rejected: duplicates what Pydantic AI
  already provides.
