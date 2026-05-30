# ADR-02: MCP tool layer wraps `aiida-restapi` (not hand-rolled)

> Seed — direction confirmed 2026-05-22; concrete tool list and diagrams still to be added.

## Context

`aiida-core`'s internal `restapi` is being deprecated.
The external `aiida-restapi` package already exposes the Pydantic models and query/submit handlers the agent tools need, so the AiiDA-access layer does not have to be written from scratch.

## Decision

MCP tools wrap `aiida-restapi`'s models and handlers rather than re-implementing AiiDA access.
This reuses validated, typed building blocks, keeps the tool surface consistent with the REST layer, and avoids duplicating logic that `aiida-restapi` already maintains.

We use these models and handlers **in-process** — `aiida-restapi`'s Pydantic models are imported and called directly; we do **not** stand up a running REST API server.
The "REST" here is only the source of the typed models and the read/write split, not a network layer.
We retain that read vs. write split to gate agent permissions: read-only tools run unguarded, write/submit tools go through the deterministic Validator and enforced human confirmation (ADR-07, ADR-08).

### Refinement (2026-05-30)

"Wrap `aiida-restapi`" applies where it earns its place: the collection/query surface, where the service layer centralises filtering, projection, and pagination (e.g. `list_processes`, `query_nodes`). For simple single-node operations addressed by a known identifier (load one node, then read its attributes or traverse its links), use `aiida-core`'s ORM directly (`orm.load_node`, `node.base.links`). That is the canonical AiiDA access path, not a re-implementation, and it avoids the round-trips the service layer incurs for a single node: resolving the identifier to a uuid, then an N+1 follow-up query per linked node. Rule of thumb: `NodeService` for queries over many nodes; plain ORM where you already hold, or load, a single node.

## Consequences

- The agent tool layer depends on `aiida-restapi` (a deliberate, well-scoped dependency) instead of touching `aiida-core` internals.
- The concrete tool list, request/response schemas, and the wrapping boundary are still to be specified when this ADR is fleshed out.

## Alternatives considered

- **Hand-roll AiiDA access in the tools.** Rejected: duplicates logic `aiida-restapi` already provides and tracks against deprecated `aiida-core` internals.
