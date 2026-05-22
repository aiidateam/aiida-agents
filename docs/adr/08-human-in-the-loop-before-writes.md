# ADR-08: Enforced human-in-the-loop confirmation before any write/submit

## Context

Agents translate natural language into tool calls against a real AiiDA database.
Read and write operations are not symmetric in risk: a wrong query wastes nothing, but a wrong submission can burn thousands of core-hours on an HPC cluster and pollute the provenance graph.
LLMs can produce inputs that look correct but are physically nonsensical, and the smaller local models we target (ADR-03) are more prone to this.

`aiida-restapi` already encodes a read vs. write split (ADR-02), which gives a natural seam to gate on.

## Decision

Gate agent permissions on the read/write split:

- **Read-only operations run unguarded** — the agent may issue queries, status lookups, and provenance traversals freely, with no human confirmation. The cost of a wrong read is negligible.
- **Any write/submit operation requires explicit human confirmation**, preceded by the deterministic Validator (ADR-07). The agent echoes the concrete action (the resolved top-level workflow and its inputs); a human approves before anything is submitted.

A regression test enforces the invariant: there is no code path that submits without passing through confirmation.

## Consequences

- The dangerous path is structurally gated, not mitigated by prompt-engineering or hope.
- Read-path agents stay fast and frictionless (the first-milestone use case), while the write path is deliberately slower and supervised.
- The agent layer must surface a clear, reviewable description of a pending write for the human to confirm — a UX requirement, not just a flag.

## Alternatives considered

- **Confirm everything (reads too).** Rejected: kills the read-path UX for no risk reduction; reads are cheap and reversible.
- **Confirm nothing; rely on the Validator alone.** Rejected: the Validator catches malformed or out-of-range inputs, but cannot judge scientific intent — a physically valid but unwanted submission still wastes compute.
- **Prompt the model to "be careful" before writes.** Rejected: not enforceable; a non-deterministic safeguard for a deterministic requirement.
