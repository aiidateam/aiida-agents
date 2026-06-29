# ADR-08: Enforced human-in-the-loop confirmation before any write/submit

## Context

Agents translate natural language into tool calls against a real AiiDA database.
Read and write operations are not symmetric in risk: a wrong query wastes nothing, but a wrong submission can burn thousands of core-hours on an HPC cluster and pollute the provenance graph.
LLMs can produce inputs that look correct but are physically nonsensical, and the smaller local models we target (ADR-03) are more prone to this.

`aiida-restapi` already encodes a read vs. write split (ADR-02), which gives a natural seam to gate on.

## Decision

Gate agent permissions on the read/write split:

- **Read-only operations run unguarded** — the agent may issue queries, status lookups, and provenance traversals freely, with no human confirmation. The cost of a wrong read is negligible.
- **Any write/submit operation requires explicit human confirmation**, preceded by deterministic input validation against the process spec (ADR-07). The agent echoes the concrete action (the resolved top-level workflow and its inputs); a human approves before anything is submitted.

Once approved, the write path is **submit-only**: the process is handed to the daemon (`engine.submit`) and the tool returns immediately with the new pk and an initial state, it never runs the process in-process. A profile with no broker is refused with an actionable error rather than silently blocking the caller on a long run (the ZMQ broker needs no system services, so requiring one is cheap). One path, one meaning: the tool reports a queued submission, never a "finished" one.

Regression tests enforce the invariant on both exposed surfaces: in the agent the write tool is registered only behind human approval (never in the unguarded read toolset), and the standalone MCP server exposes read-only tools (the write tool is not registered there at all). No exposed surface submits without confirmation.

## Consequences

- The dangerous path is structurally gated, not mitigated by prompt-engineering or hope.
- Read-path agents stay fast and frictionless (the first-milestone use case), while the write path is deliberately slower and supervised.
- The agent layer must surface a clear, reviewable description of a pending write for the human to confirm — a UX requirement, not just a flag.
- Submit-only means the tool reports a queued pk and initial state, not a finished result; the process runs on the daemon and is polled later via the read tools (the natural submit/poll model for an agent).

## Alternatives considered

- **Confirm everything (reads too).** Rejected: kills the read-path UX for no risk reduction; reads are cheap and reversible.
- **Confirm nothing; rely on the Validator alone.** Rejected: the Validator catches malformed or out-of-range inputs, but cannot judge scientific intent — a physically valid but unwanted submission still wastes compute.
- **Prompt the model to "be careful" before writes.** Rejected: not enforceable; a non-deterministic safeguard for a deterministic requirement.
