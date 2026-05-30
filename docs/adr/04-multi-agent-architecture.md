# ADR-04: Multi-agent architecture with a routing Orchestrator

<!-- > Seed — direction confirmed 2026-05-25; agent boundaries and inter-agent protocol still to be specified during implementation. -->

## Context

The project targets a range of user intents that differ in risk, domain knowledge, and AiiDA access pattern.
A single monolithic agent would need to handle all of these at once: read-only provenance queries, workflow parameter construction, failure diagnosis, and write/submit operations.
That conflates concerns that have very different risk profiles (ADR-08) and makes the system prompt unmanageable as the tool surface grows.

Breaking the work into specialised agents is the standard pattern for multi-step, multi-domain agentic systems.
The question is how to route between them and what the boundaries should be.

## Decision

Adopt a **routed multi-agent architecture** with a lightweight Orchestrator agent at the entry point and a set of specialised sub-agents, each owning a bounded slice of the tool surface and domain knowledge.

### Orchestrator

Receives every user message.
Classifies intent and delegates to exactly one sub-agent per turn.
Holds no domain tools itself — its only job is routing and context hand-off.
Runs on the same model as the sub-agents; no separate model is required.

### Sub-agents (first milestone scope)

| Agent | Responsibility | AiiDA access |
|---|---|---|
| **Analysis Agent** | Provenance queries, process status, structure search | Read-only MCP tools |
| **Diagnostic Agent** | Interpret calculation failures, map exit codes to causes | Read-only MCP tools + RAG (ADR-05) |
| **Config Agent** | Build and validate workflow input parameters | Validator (ADR-07), no DB writes |
| **Workflow Agent** | Submit workflows to AiiDA | Write tools, gated by HITL (ADR-08) |

The Analysis Agent is the first-milestone deliverable.
The remaining agents are introduced in later milestones once the read-only foundation is stable.

### Inter-agent protocol

Sub-agents receive a structured context object from the Orchestrator: the original user message, the resolved intent class, and any identifiers (PKs, UUIDs) already extracted.
Sub-agents return a structured result; the Orchestrator composes the final user-facing response.
All inter-agent calls are in-process — no network hops, no message queues.

### Library

Pydantic AI (ADR-03) provides the agent/tool abstraction.
Each sub-agent is a `pydantic_ai.Agent` instance with its own tool set and system prompt.
The Orchestrator is also a `pydantic_ai.Agent` whose only "tools" are the sub-agent `run()` calls.

## Consequences

- Each agent's system prompt and tool surface stays small and auditable.
- The read/write split (ADR-02, ADR-08) maps cleanly onto agent boundaries — the Workflow Agent is the only one with write tools, making the HITL gate structurally enforceable.
- Adding a new capability means adding a new sub-agent, not growing a monolithic prompt.
- The Orchestrator is a single point of failure for routing; misclassification sends the query to the wrong agent. Mitigation: the Orchestrator prompt is tested with a representative query suite (the eval harness).
- Inter-agent calls are synchronous and in-process; parallel sub-agent execution is not supported in the first milestone and deferred.

## Alternatives considered

- **Single monolithic agent with all tools.**
  Rejected: the system prompt grows unboundedly, tool selection degrades as the surface widens, and the read/write risk split cannot be structurally enforced.
- **Separate processes / microservices per agent.**
  Rejected: adds network, serialisation, and deployment complexity for no benefit at this scale; in-process calls are sufficient.
- **LangGraph or a dedicated orchestration framework.**
  Rejected: introduces a heavy dependency and framework-specific abstractions; Pydantic AI's native agent composition is sufficient for the first milestone and keeps the stack minimal (ADR-01 reuse ethos).
- **Static routing (if/else on keywords).**
  Rejected: brittle against natural language variation; an LLM classifier generalises better and requires no hand-maintained keyword lists.