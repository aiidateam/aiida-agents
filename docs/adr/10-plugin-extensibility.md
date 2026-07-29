# ADR-10: Plugin extensibility through one entry point

## Context

The agents are useful in proportion to how much they know about the codes a user actually runs.
That knowledge lives in AiiDA plugins — `aiida-quantumespresso` knows its own conventions, ships its own documentation, and could offer tools no general agent could write.

Three things a plugin might contribute: tools, a documentation corpus for retrieval, and domain guidance for the system prompt.

The constraint that shapes the answer is dependency direction.
`aiida-agents` must not depend on any plugin — installing it should not drag in one plugin's stack.
And a plugin should not have to depend on pydantic-ai or chromadb to describe what it offers; a plugin author writing a provider should not be installing an agent framework to do it.

This ADR was written after two agents and the RAG pipeline existed, deliberately.
An extension point designed before there was anything to extend describes a guess.

## Decision

A plugin declares **one entry point** in the `aiida_agents.plugins` group, pointing at a provider object:

```toml
[project.entry-points."aiida_agents.plugins"]
quantumespresso = "my_plugin.agents:PROVIDER"
```

The provider implements `AgentPlugin`, a structural `Protocol` with three hooks, **all optional**:

| Hook                | Contributes                                               |
| ------------------- | --------------------------------------------------------- |
| `tools()`           | `AgentTool` objects registered on the agent               |
| `rag_corpora()`     | `RagCorpus` objects indexed and cited as the plugin's own |
| `prompt_fragment()` | Domain guidance appended to the system prompt             |

### The contract imports nothing heavy

`AgentTool`, `RagCorpus` and `AgentPlugin` are plain dataclasses and a `Protocol`.
They import nothing from pydantic-ai or chromadb, so a plugin can build a provider without either in its dependency tree.
A tool is a plain function: its name, signature and docstring become the tool the model sees, exactly as for a built-in one.

### `writes=True` is the entire declaration

A contributed tool that changes state sets `writes=True` and is registered behind the human-in-the-loop gate (ADR-08).
The plugin does not implement the gate, and cannot opt out of it.

This is the point of declaring rather than registering: a plugin says *what a tool is*, and this package decides *how it is exposed*.
A plugin cannot introduce an ungated write, however it is written.

### Failure is isolated per hook

Every hook is read defensively. A provider that omits a hook, or whose hook raises, is skipped **for that hook alone**.
A broken plugin degrades to the agent running without its contribution — never to an agent that will not start.

The same isolation applies to corpora: one corpus failing to build does not stop the others.

### The prompt fragment is bounded

A fragment says what only the plugin knows — its conventions, its units, its physics.
It is budgeted by character count, and the core prompt wins on any conflict.
A plugin cannot rewrite the agent's behaviour by contributing a longer fragment.

### Corpora are versioned by the plugin

A `RagCorpus` is keyed by name *and* version, and the version is expected to come from the plugin's own distribution version.
A version bump resolves to a different collection and rebuilds, so an index can never silently disagree with the installed code.

## Consequences

- A plugin can extend the agents without this package knowing it exists, and without depending on the agent stack.
- The write-gating guarantee holds across contributed tools, because it is applied at registration rather than requested by the contributor.
- Discovery cost is paid at agent construction: every installed provider is read once when an agent is built.
- A plugin's contributions are only as good as its docstrings — a contributed tool the model cannot understand from its signature and docstring is one it will not use correctly.
- Three hooks is a small surface. Anything a plugin wants to contribute that is not a tool, a corpus, or prompt text needs this ADR revisited.

## Alternatives considered

- **A separate entry-point group per contribution kind** (`aiida_agents.tools`, `aiida_agents.corpora`, …).
  Rejected: three registrations to keep in step instead of one object, and no way for a plugin to share state between them.
- **Requiring plugins to import and register against pydantic-ai directly.**
  Rejected: it puts the agent framework in every plugin's dependency tree, and it hands plugins the ability to register a write tool without the gate.
- **Reading configuration files rather than entry points.**
  Rejected: entry points are how AiiDA already discovers plugin contributions, and they resolve through the same environment the profile does.
- **Designing the extension point before the agents existed.**
  Rejected on principle — concretion before abstraction. The hooks here are the three things a plugin turned out to have, not the three that seemed likely.

## Status

Implemented in `src/aiida_agents/plugins/`.
`dev/qe_rag_stub/` is a working example, written because `aiida-quantumespresso` does not ship this entry point yet — it registers that plugin's documentation as a corpus so cross-corpus retrieval and attribution can be exercised end to end.
