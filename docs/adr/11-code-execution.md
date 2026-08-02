# ADR-11: Executing generated code

## Status

Accepted.

## Context

The tool surface is fixed, and some questions do not fit it. "Which structures in group `screening-2026` contain Ti, and what were their final energies" is three filters and a projection; the QueryBuilder expresses it in six lines and no tool enumerates it. Adding a tool per combination does not converge: the first combination nobody anticipated breaks immediately, and the tool list becomes unreadable to the model long before it becomes complete.

Writing the query is therefore the feature. Running it is what makes the feature trustworthy: a snippet nobody executed is a guess with syntax highlighting.

That raises the obvious objection. Executing model-written Python against a research group's provenance database is the most dangerous thing this project could do, and provenance is not editable after the fact in the way ordinary data is.

## Decision

**Generated code runs against a profile whose PostgreSQL role holds no write privilege.**

Containment is at the database. The role can `SELECT` and nothing else, so a write is refused by Postgres rather than caught by us. `aiida-agents sandbox init` prints the SQL and the `verdi` invocation; `sandbox check` asks Postgres whether the role can insert into `db_dbnode` and treats every unclear answer as a failure.

The sandbox profile points at **the same database** as the user's own. A scratch profile would be safer and useless: an empty database cannot answer any question worth asking about someone's data.

Two further layers sit above it, and neither is containment:

- A **static guard** rejects imports outside an allowlist, calls that write, and the builtins (`getattr`, `exec`, `open`) that would step around either rule. It turns the common mistakes into a readable message instead of a permission error.
- A **subprocess with a timeout**, so an unbounded query is a timeout rather than a hung CLI.

**The execution tool is not approval-gated**, which is the decision most worth arguing with. It holds no capability to write, so there is nothing for an approval to protect. And the alternative is worse than it looks: an approval prompt showing twenty lines of unexecuted Python asks a researcher to audit code under time pressure, which is a far weaker check than the one it appears to be. Letting the code run where it can do no harm and showing what it *returned* converts the same decision into one made from evidence.

**Everything that genuinely writes stays where it was**: on the Execution agent, behind `requires_approval=True`, where the preview is a resolved input a researcher can judge in seconds.

**It is a third specialist**, on both grounds [`extending.md`](/docs/extending.md) requires. Its tool surface includes code execution, whose blast radius differs in kind from a twelve-tool read-only surface; and its prompt describes a loop (write, run, read the traceback, try again) that is a different job from explaining what is in a database. Diagnosis, by contrast, needed neither and remained a tool.

**It is not exported over MCP.** The safety of `run_aiida_code` rests entirely on `AIIDA_AGENTS_SANDBOX_PROFILE` naming a profile someone verified. An MCP client cannot check that, and we cannot see whether it holds; a client that wants to run Python can run it itself, with its own consent.

## Consequences

- A question needing an unanticipated combination of filters is answerable without a new tool.
- Answers come with output that was actually produced, and the agent fixes its own mistakes from real tracebacks before the user sees them.
- **The feature is inert until someone runs `sandbox init`.** `run_aiida_code` refuses to run when no sandbox profile is configured, and says the snippet is unverified rather than falling back to the user's writable profile. Silently falling back is the one failure that would make all of the above worthless.
- The guarantee is only as good as the Postgres grants. `sandbox check` exists because that is worth re-verifying after a database migration, and because nothing else in the system can tell a read-only profile from a writable one.
- Writes remain impossible from this path. A user who wants to submit is told so and routed to the Execution agent, which asks first.

## Alternatives considered

**Approval-gate the execution tool.** Rejected: it protects nothing the database does not already refuse, and it trains users to approve code they have not read, which devalues the approval prompt that guards real submissions.

**A scratch profile with copied data.** Rejected: copying enough provenance to answer real questions is its own project, and a partial copy produces answers that are wrong in a way nobody can see.

**A restricted interpreter instead of a database role.** Rejected: Python is not sandboxable in-process, and a guard that claimed to be one would be believed.

**No execution: generate code and let the user run it.** Rejected as the whole product. It leaves the correctness problem entirely with the user, and correctness is the thing the feature exists to provide.
