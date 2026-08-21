# ADR-04: Package-by-feature agents subpackage; single Analysis Agent first

> Status: accepted. Analysis Agent implemented as of Weeks 3–4 (June 2026).
> Execution Agent added and the tool layer re-grouped per agent (see the
> Revision section). Orchestrator deferred to Weeks 7–8.

## Context

The project targets a range of user intents with very different risk profiles:
read-only provenance queries, workflow submission, failure diagnosis. A single
monolithic agent conflates these concerns and makes the system prompt
unmanageable as the tool surface grows.

Breaking work into specialised agents is the standard pattern. The question is
what the package structure, agent boundaries, and inter-agent protocol should
be, and critically, when to introduce that complexity.

## Decision

### Build one agent first; earn complexity before adding it

The Analysis Agent is the first and only agent in the first milestone. It is
a read-only provenance-exploration agent over the MCP tools and RAG pipeline.
The Orchestrator and specialist agents (Diagnostic, Config, Workflow) are
post-midterm work. Adding multi-agent routing before a single agent works
end-to-end would be complexity without a working concretion to validate it.

### Package-by-feature: `agents/` subpackage

Each agent is its own subpackage under `src/aiida_agents/agents/`:

```
src/aiida_agents/agents/
    __init__.py          # public API: get_agent()
    _models.py           # shared get_model() factory (all agents share one model)
    analysis/
        __init__.py      # Analysis agent: get_agent(), _TOOLS, system prompt
        prompt.md        # agent's system prompt: co-located, plain Markdown
    validator/           # ADR-07: deterministic validation before any write
        __init__.py
        _schema.py
        _ranges.py
```

Key design decisions:

- **Prompt co-location**: each agent's `prompt.md` lives with its agent, not
  in a shared `prompts/` directory. One prompt per agent, no shared prompts yet.
- **Shared model factory**: `_models.py` provides `get_model()` for all agents.
  ADR-03's provider abstraction lives here, not in any individual agent.
- **No module-level agent instance**: `get_agent()` is a factory called from
  `cli.main()`, not a module-level `agent = Agent(...)`. Importing the package
  is inert.
- **CLI separated**: `ask()` and `main()` live in `aiida_agents/cli.py`, not
  in any agent module. The CLI drives whichever agent is active.

### Analysis Agent tool set

The Analysis Agent exposes seven tools:

> Paths below are as of this ADR's original writing; see the Revision section
> for the current per-agent layout (`tools/analysis/…`), and note that
> `submit_workflow` has since moved off this agent entirely.

| Tool                    | Source                | Type                            |
| ----------------------- | --------------------- | ------------------------------- |
| `get_process_status`    | `tools/processes.py`  | Read                            |
| `list_recent_processes` | `tools/processes.py`  | Read                            |
| `query_nodes`           | `tools/nodes.py`      | Read                            |
| `get_node_inputs`       | `tools/nodes.py`      | Read                            |
| `get_node_outputs`      | `tools/nodes.py`      | Read                            |
| `search_structures`     | `tools/structures.py` | Read                            |
| `search_aiida_docs`     | `rag/__init__.py`     | Read (RAG)                      |
| `submit_workflow`       | `tools/submit.py`     | Write: `requires_approval=True` |

A write tool is registered with Pydantic AI's native `requires_approval=True`,
which pauses the agent run and returns a `DeferredToolRequests` object for the
CLI to handle (ADR-08).

### Future multi-agent architecture (Weeks 7–8)

> **Superseded by [ADR-09](/docs/adr/09-agent-orchestration.md).** The table below
> planned three specialists behind an Orchestrator whose tools are the specialist
> `run()` calls. Two specialists were built, with diagnosis folded into the
> Analysis agent, and the orchestration layer is a *planner* with no tools:
> wrapping the specialists in tools would have broken the human-in-the-loop
> guarantee. ADR-09 gives the reasoning.

Once the single-agent foundation is stable, the architecture expands to:

| Agent                | Responsibility                       | AiiDA access        |
| -------------------- | ------------------------------------ | ------------------- |
| **Orchestrator**     | Routes intent to specialist agents   | No tools            |
| **Analysis Agent**   | Provenance queries, structure search | Read-only MCP + RAG |
| **Diagnostic Agent** | Interpret failures, map exit codes   | Read-only MCP + RAG |
| **Workflow Agent**   | Submit workflows                     | Write tools + HITL  |

Each specialist agent will be a sibling subpackage under `agents/`. The
Orchestrator will be a `pydantic_ai.Agent` whose only tools are the specialist
`run()` calls. A2A vs. plain function calls will be decided empirically.

## Consequences

- The read/write split maps cleanly onto agent boundaries: the write tool
  is gated by `requires_approval` regardless of which agent holds it.
- Adding a new agent means adding a new sibling subpackage; no changes to
  existing agents.
- The single-agent-first approach meant a working, testable system at the
  end of Weeks 3–4 rather than a partially-working multi-agent system.
- `_models.py` as shared infrastructure means model selection is changed in
  one place for all agents.

## Alternatives considered

- **Build Orchestrator + specialists first.**
  Rejected: multi-agent routing before a single working agent is complexity
  without a concretion to validate it. Julian's timeline explicitly sequences
  single agent first, Orchestrator post-midterm.
- **Single monolithic agent with all tools.**
  Rejected: system prompt grows unboundedly; read/write risk split cannot be
  structurally enforced.
- **Shared `prompts/` directory.**
  Rejected: each agent owns its prompt; a shared directory implies shared
  prompts that don't exist yet. Refactor when a common preamble emerges.
- **LangGraph or dedicated orchestration framework.**
  Rejected: heavy dependency, framework-specific abstractions; Pydantic AI's
  native agent composition is sufficient and keeps the stack minimal.

## Revision (2026-07): tools grouped per agent

The Execution Agent landed as the second sibling under `agents/`, as this ADR
planned (it is the "Workflow Agent" of the table above). With two agents, the
flat `tools/` directory no longer said who owned what: `tools/workflows/` and
`tools/execution/` were both Execution's, while `nodes.py`, `processes.py`,
`query_builder.py`, and `structures.py` were all Analysis's, with nothing in
the layout saying so.

`tools/` now mirrors `agents/`: each agent's tools live in `tools/<agent>/`,
and only genuinely shared infrastructure stays at the top level:

```
src/aiida_agents/tools/
    _errors.py, _orm.py, _types.py    # shared by every agent
    analysis/
        nodes.py, processes.py, query_builder.py, structures.py
    execution/
        run_context.py, codes.py, introspection.py,
        protocol.py, schemas.py, spec_execution.py, submit.py
```

A new agent adds a sibling package under both `agents/` and `tools/` rather
than more flat modules. `tests/tools/` mirrors the same split.

Two consequences worth stating:

- **One write path.** `submit_workflow` was registered on the Analysis Agent
  back when it was the only agent. It is not any more: Execution reaches the
  database through its own HITL-gated `execute_workflow_spec` (which delegates
  to `submit_workflow` after building and validating a spec), and Analysis is
  read-only. A plugin-contributed write tool is still gated the same way on
  whichever agent registers it (ADR-08).
- **The MCP server's read-only guarantee is unchanged**, but now excludes two
  names rather than one (`submit_workflow` and `execute_workflow_spec`); the
  server's discovery test walks the subpackages recursively, so a new tool in a
  new agent package is still caught automatically.
