# ADR-06: Agent-behaviour evaluation harness

> Status: accepted — initial harness implemented in Weeks 3–4 (June 2026).

## Context

The agent's correctness cannot be verified by unit tests alone. A unit test
can confirm that a tool returns the right data given a PK; it cannot confirm
that the agent calls the right tool for a given natural language query, or that
it chains tools correctly for multi-step diagnostics.

We need a way to verify agent behaviour — tool selection and response quality
— against real AiiDA fixture nodes, without requiring a live LLM in CI.

## Decision

Build a lightweight eval harness as a pytest suite that drives the agent
against real fixture nodes (no hardcoded PKs, no mocks of the agent itself)
and asserts on tool selection and output structure.

### What the harness tests

**Structural tests** (in `tests/agents/analysis/test_analysis.py`) — verified
without any LLM call or DB fixture:

- `get_agent()` returns an agent with exactly the expected tool set
- `get_model()` selects the correct provider class per `AIIDA_AGENTS_PROVIDER`
- Unsupported providers raise `ValueError` with a clear message

**Tool-execution tests** (same module) — use
`pydantic_ai.models.function.FunctionModel` to script tool calls deterministically,
exercising real tool logic against real AiiDA fixture nodes without a live LLM:

- Each test scripts the model to call a specific tool with fixture-derived
  arguments, then asserts the tool ran and returned expected data
- Fixture PKs are never hardcoded; they come from session-scoped AiiDA fixtures
  (`add_calc`, `multiply_add_workchain`, `silicon_structure`)

**What the harness does not test** — tool selection quality (does the model
pick `list_processes` for "show recent calcs"?) depends on the model and
belongs in a separate, opt-in evaluation suite run against a real model.
Mocking `agent.run` to assert on canned tool names is explicitly rejected
(see Alternatives).

### Test infrastructure

AiiDA fixtures in `tests/conftest.py` run real calculations in-process
against a temporary `core.sqlite_dos` profile — no daemon, no broker, no
external services. Each fixture is session-scoped so the calculations run
once per test session, not per test.

The LLM is replaced by `FunctionModel` in all harness tests, so CI requires
no Ollama instance or API keys.

## Consequences

- Tool registration, tool logic, and multi-step tool chaining are all covered
  by deterministic tests that run in CI.
- Semantic quality (does the model give good answers?) is explicitly deferred
  to a manual or opt-in eval run against a real model.
- The harness grows incrementally: new tools get a new test, new fixture nodes
  extend the existing session fixtures.

## Alternatives considered

- **Mock `agent.run` with `AsyncMock` and assert on canned tool names.**
  Rejected: this tests the mock, not the agent. The assertion checks a value
  the test itself injected; the agent's tools, system prompt, and wiring never
  run. Looks like coverage, verifies nothing.
- **Full end-to-end eval with a live LLM in CI.**
  Rejected: non-deterministic, slow, costs money, requires API keys or a
  running Ollama instance. Better suited to a separate opt-in eval suite.
- **No harness; rely on manual testing.**
  Rejected: regressions go undetected; refactors have no safety net.
