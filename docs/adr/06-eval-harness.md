# ADR-06: Agent-behaviour evaluation harness

> Status: accepted. Initial harness implemented in Weeks 3–4 (June 2026).

## Context

The agent's correctness cannot be verified by unit tests alone. A unit test
can confirm that a tool returns the right data given a PK; it cannot confirm
that the agent calls the right tool for a given natural language query, or that
it chains tools correctly for multi-step diagnostics.

We need a way to verify agent behaviour: tool selection and response quality
against real AiiDA fixture nodes, without requiring a live LLM in CI.

## Decision

Build a lightweight eval harness as a pytest suite that drives the agent
against real fixture nodes (no hardcoded PKs, no mocks of the agent itself)
and asserts on tool selection and output structure.

### What the harness tests

**Structural tests** (in `tests/agents/analysis/test_analysis.py`): verified
without any LLM call or DB fixture:

- `get_agent()` returns an agent with exactly the expected tool set

Provider selection lives in `tests/agents/test_models.py`: `get_model()` builds
the right class per `AIIDA_AGENTS_PROVIDER`, and bad config fails fast: an
unsupported provider raises `ValidationError` at settings load, while a missing
`AIIDA_AGENTS_BASE_URL` for `openai-compatible` raises `ValueError` in
`get_model()`.

**Tool-execution tests** (same module): use
`pydantic_ai.models.function.FunctionModel` to script tool calls deterministically,
exercising real tool logic against real AiiDA fixture nodes without a live LLM:

- Each test scripts the model to call a specific tool with fixture-derived
  arguments, then asserts the tool ran and returned expected data
- Fixture PKs are never hardcoded; they come from session-scoped AiiDA fixtures
  (`add_calc`, `multiply_add_workchain`, `silicon_structure`)

**What the harness does not test**: tool selection quality (does the model
pick `list_processes` for "show recent calcs"?) depends on the model and
belongs in a separate, opt-in evaluation suite run against a real model.
Mocking `agent.run` to assert on canned tool names is explicitly rejected
(see Alternatives).

That opt-in suite now exists; see the Revision section.

### Test infrastructure

AiiDA fixtures in `tests/conftest.py` run real calculations in-process
against a temporary `core.sqlite_dos` profile: no daemon, no broker, no
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

## Revision (2026-07): the opt-in tier, and what it measures

The suite this ADR deferred is now `tests/evals/`, in two tiers.

**`test_harness.py` runs in CI.** It records a run into a `RunTrace`: the tool
calls in order, what each returned, and the final text, and asserts on it. Three
checks, each corresponding to a failure that reached users:

- did it consult the documentation before answering a knowledge question,
- does every physical quantity in the answer appear in some tool output,
- is an answer built on retrieved docs actually cited.

Critically, this tier tests **the assertions themselves**. It scripts a
fabricating agent with `FunctionModel`: one that searches, gets a real excerpt,
then states a value appearing nowhere in it, and requires each check to catch
that, then requires correct behaviour to pass. An eval that quietly degrades into
always-passing is worse than none, because it certifies nothing while looking
green.

**`test_grounding.py` is opt-in** (`AIIDA_AGENTS_EVAL=1`, marked `llm`). It needs
a real model and a built index, so it never runs in CI. It covers routing and
planning quality, grounding under a real model, and the honest reporting of a
missing index.

Two things this ADR did not anticipate:

**The quantity check moved into the package.** `aiida_agents.grounding` runs on
every CLI reply, not only in tests, because prompt rules asking a model not to
fabricate were measured failing: one explicit instruction was ignored in five
runs out of five. The test tier imports the same function, so the check guarding
a shipped answer and the check guarding the suite cannot drift apart.

**Model choice is part of the measurement.** An auto-routing provider that
selects a different backend per request makes results unattributable; every case
must log the resolved provider and model, and a specific model must be pinned for
a result to mean anything.

The opt-in tier has not yet produced a complete baseline run.
