# ADR-02: MCP tool layer wraps `aiida-restapi` (not hand-rolled)

> Bare-bones seed — to be expanded with the concrete tool list and diagrams as the design is finalized.

## Context

`aiida-core`'s internal `restapi` is being deprecated.
The external `aiida-restapi` package already exposes the Pydantic models and query/submit handlers the agent tools need, so the AiiDA-access layer does not have to be written from scratch.

## Decision

MCP tools wrap `aiida-restapi`'s models and handlers rather than re-implementing AiiDA access.
This reuses validated, typed building blocks, keeps the tool surface consistent with the REST layer, and avoids duplicating logic that `aiida-restapi` already maintains.

## Consequences

- The agent tool layer depends on `aiida-restapi` (a deliberate, well-scoped dependency) instead of touching `aiida-core` internals.
- The concrete tool list, request/response schemas, and the wrapping boundary are still to be specified when this ADR is fleshed out.

## Alternatives considered

- **Hand-roll AiiDA access in the tools.** Rejected: duplicates logic `aiida-restapi` already provides and tracks against deprecated `aiida-core` internals.
