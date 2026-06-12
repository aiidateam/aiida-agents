# ADR-04: Package-by-feature agents subpackage; single Analysis Agent first

> Status: accepted — Analysis Agent implemented as of Weeks 3–4 (June 2026).
> Orchestrator and specialist agents deferred to Weeks 7–8.

## Context

The project targets a range of user intents with very different risk profiles:
read-only provenance queries, workflow submission, failure diagnosis. A single
monolithic agent conflates these concerns and makes the system prompt
unmanageable as the tool surface grows.

Breaking work into specialised agents is the standard pattern. The question is
what the package structure, agent boundaries, and inter-agent protocol should
be — and critically, when to introduce that complexity.

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
        prompt.md        # agent's system prompt — co-located, plain Markdown
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

| Tool                 | Source                    | Type                             |
| -------------------- | ------------------------- | -------------------------------- |
| `get_process_status` | `mcp/tools/processes.py`  | Read                             |
| `list_processes`     | `mcp/tools/processes.py`  | Read                             |
| `query_nodes`        | `mcp/tools/nodes.py`      | Read                             |
| `get_node_inputs`    | `mcp/tools/nodes.py`      | Read                             |
| `get_node_outputs`   | `mcp/tools/nodes.py`      | Read                             |
| `search_structures`  | `mcp/tools/structures.py` | Read                             |
| `search_aiida_docs`  | `rag/__init__.py`         | Read (RAG)                       |
| `submit_workflow`    | `mcp/tools/submit.py`     | Write — `requires_approval=True` |

`submit_workflow` is registered with Pydantic AI's native `requires_approval=True`,
which pauses the agent run and returns a `DeferredToolRequests` object for the
CLI to handle (ADR-08).

### Future multi-agent architecture (Weeks 7–8)

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

- The read/write split maps cleanly onto agent boundaries — the write tool
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
