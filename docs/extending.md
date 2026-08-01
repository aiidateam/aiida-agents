---
title: Extending
---

# Extending

Four ways to add to the system, from smallest to largest.
Read [Architecture](/docs/architecture.md) first if you have not — the constraints below follow from it.

If you maintain an AiiDA plugin, start at [From your own plugin](#from-your-plugin): you can contribute tools, documentation and prompt guidance without this package depending on yours, or yours on this one.

## A tool on an existing agent

A tool is a plain Python function. Its **name, signature and docstring are its entire interface to the model** — a model choosing between tools has nothing else to go on, so write them for someone who has never seen your code.

1. Put it in `tools/analysis/` or `tools/execution/` — whichever agent owns it. If both agents need it, put it at the top of `tools/`.
1. Export it from `tools/__init__.py`.
1. Add it to that agent's `_READ_TOOLS` list.
1. Say when to reach for it in that agent's `prompt.md`. A registered tool the prompt never mentions is a tool the model will not use.
1. Register it on the MCP server in `mcp/tools/__init__.py` **if it is read-only**.

A pinned test asserts each agent exposes exactly the expected tool set, so adding one is a deliberate edit in two places rather than a silent change.

### What a good tool returns

Return a typed dict (add the shape to `tools/_types.py`), and hold to three rules that this project learned the hard way:

**Never invent a value the caller could mistake for a queried one.** A default that gets echoed back — a `structure_type` nobody asked for, a "recommended" item nothing ranked — is indistinguishable from real data once it reaches the model. If you cannot support a value, do not return a field for it.

**Carry the units of anything physical.** A bare `60.0` invites the model to supply "Ry" or "eV", and it has done both. The tool knows; the caller does not.

**Say when you found nothing, and why.** "This workflow is not installed" and "this workflow has no runs" lead to different actions. Reporting an empty result as an absence of data is how a caller ends up proceeding confidently on nothing.

### If it writes

A tool that changes state is registered with `requires_approval=True` and **not** exposed on the MCP server:

```python
agent.tool_plain(requires_approval=True)(import_structure)
```

The run then pauses and returns the proposed call; the CLI shows it and asks. Do not ask for confirmation in your prompt as well — the gate already does it, and a second prompt trains users to dismiss both.

## A RAG corpus

Documentation the agents should be able to search and cite. See [ADR-05](/docs/adr/05-rag-over-aiida-docs.md); to contribute one from a plugin, see below.

A corpus is keyed by its name *and version*, so bumping the version rebuilds only that corpus rather than serving a stale index. Set the version from your distribution's own version and the corpus can never silently disagree with the installed code.

(from-your-plugin)=

## From your own plugin

A plugin contributes through one entry point in the `aiida_agents.plugins` group, pointing at a provider object. Nothing else is required, and this package never imports yours.

```toml
# your plugin's pyproject.toml
[project.entry-points."aiida_agents.plugins"]
quantumespresso = "my_plugin.agents:PROVIDER"
```

Every hook is optional — implement only what you have:

```python
from aiida_agents.plugins import AgentPlugin, AgentTool, RagCorpus


class Provider:
    name = "quantumespresso"

    def tools(self):
        return [
            AgentTool(fn=suggest_pseudo_family),
            AgentTool(fn=submit_relaxation, writes=True),   # approval-gated
        ]

    def rag_corpora(self):
        return [
            RagCorpus(
                name="quantumespresso",
                version="5.0.0",
                docs_repo="https://github.com/aiidateam/aiida-quantumespresso",
                docs_subdir="docs",
                docs_url="https://aiida-quantumespresso.readthedocs.io/en/{version}/{page}.html",
            )
        ]

    def prompt_fragment(self):
        return "Cutoffs for this plugin are in Ry, following Quantum ESPRESSO's own input format."


PROVIDER = Provider()
```

Three things worth knowing:

**`writes=True` is the whole declaration.** A tool marked that way is registered behind the approval gate automatically. You do not implement the gate and you cannot opt out of it.

**Hooks are read defensively.** A provider that omits a hook, or whose hook raises, is skipped for that hook alone. A broken plugin degrades to the agent working without it, never to an agent that will not start.

**A prompt fragment says what only you know.** Your conventions, your physics, your units. Do not restate the agent's workflow: the core prompt wins on any conflict, and the fragment has a character budget.

A corpus needs exactly one source — either `text_dir` (pre-rendered text you ship) or `docs_repo` (cloned and rendered with your own `docs` extra in an isolated build).

**Give `docs_url` if your documentation is published.** It is a template taking `{version}` and `{page}`, and it is what lets an answer cite your docs with a link the reader can open rather than a path they have to go and find. `{version}` is filled from your `docs_ref`, so the page linked is the one the corpus was rendered from — a citation cannot quote one release and link another. Leave it unset and passages from your corpus are still retrieved and still attributed, just unlinked; a guessed URL would be worse than none.

`dev/qe_rag_stub/` in this repository is a working example of all three hooks, written because aiida-quantumespresso does not ship this entry point yet. It contributes its documentation as a corpus, a `read_scf_convergence` tool that parses pw.x's own electronic-convergence trace, and a fragment saying when that tool applies. The tool is the reason the split matters: reading a Quantum ESPRESSO output format is exactly the knowledge that belongs to the plugin and not to `aiida-agents`, and a plugin for another code contributes its own equivalent without either package learning about the other.

One limit to know: plugin tools are registered on the **Analysis agent only**. A tool that belongs on the Execution agent has nowhere to go today.

## A new specialist agent

The largest change, and the one to be most sceptical about. Two specialists have been enough so far, and diagnosis was folded into the Analysis agent rather than becoming a third.

**Before adding one, check the alternative.** A new agent is justified when a domain needs its own tool surface *and* its own prompt. If it needs neither — if you are really adding capability — a tool on an existing agent is cheaper, has no routing cost, and cannot be mis-routed to.

If it is genuinely warranted:

1. Add `agents/<name>/` with `__init__.py` (a `get_agent()` factory, never a module-level instance) and `prompt.md` alongside it.
1. Add `tools/<name>/` for the tools it owns.
1. Add it to `Specialist` and `_SPECIALISTS` in `agents/planner/__init__.py`, and describe it in the planner's `prompt.md` — the planner can only choose specialists it has been told about.
1. Add it to `_AGENT_CHOICES` in `cli/agent.py` so `--agent` can name it.
1. Add routing cases to the eval tier, in both directions: requests that should reach it, and neighbouring requests that should not.

**Keep the read/write boundary legible.** If the new agent writes, its write tools are approval-gated like every other. If it does not, say so in its prompt and let the planner fall back to it safely.

## Conventions worth keeping

These are not style preferences; each one is here because its absence caused a real bug.

**Name things for what they do.** A tool called `query_analysis_agent` that ran database queries directly cost two debugging sessions, because the name was believed over the code.

**Fail loud, not open.** A lookup that finds nothing must say so. Silently substituting a plausible default — a store path, an embedding model, a workflow type — produces a system that looks like it is working while doing something else, which is far harder to notice than an error.

**Make a check catch its own bug.** After writing a test for a fix, revert the fix and confirm the test fails. Several tests in this repository were written against real transcripts and *still* missed the failure they were for until this was done.

**One implementation of one idea.** Where a check guards both a shipped answer and the test suite, both call the same function. Two expressions of one rule drift, and the drift is the bug.
