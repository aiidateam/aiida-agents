# ADR-11: Executing generated code

## Status

Accepted; the containment mechanism revised once. (2026-08) The sandbox was a profile pointing at the user's own database through a read-only role. It is now a disposable **copy**, after the original cost a maintainer his database ([#73](https://github.com/aiidateam/aiida-agents/issues/73)). The original reasoning is kept below, marked, because the mistake in it is the instructive part.

## Context

The tool surface is fixed, and some questions do not fit it. "Which structures in group `screening-2026` contain Ti, and what were their final energies" is three filters and a projection; the QueryBuilder expresses it in six lines and no tool enumerates it. Adding a tool per combination does not converge: the first combination nobody anticipated breaks immediately, and the tool list becomes unreadable to the model long before it becomes complete.

Writing the query is therefore the feature. Running it is what makes the feature trustworthy: a snippet nobody executed is a guess with syntax highlighting.

That raises the obvious objection. Executing model-written Python against a research group's provenance database is the most dangerous thing this project could do, and provenance is not editable after the fact in the way ordinary data is.

## Decision

**Generated code runs against a disposable copy of the user's storage.**

`aiida-agents sandbox init` copies the profile's storage and registers a profile pointing at the copy; `sandbox check` verifies the copy shares no database and no repository with any real profile; `refresh` rebuilds it when it has drifted, and `teardown` removes it. A read-only PostgreSQL role over the copy is available as a second layer, and is no longer the mechanism.

> **Revised (2026-08).** This originally read: *"Generated code runs against a profile whose PostgreSQL role holds no write privilege. The sandbox profile points at **the same database** as the user's own. A scratch profile would be safer and useless: an empty database cannot answer any question worth asking about someone's data."*
>
> The second sentence is true and the conclusion did not follow from it. The choice was framed as shared-versus-empty, and a copy is neither — it holds the user's real data and none of their risk. Framing it as a two-way choice is what made the wrong answer look forced.
>
> The cost was a maintainer's database. Deleting the sandbox profile and agreeing to delete its data deletes the storage under both profiles, and a read-only role does not help: the destructive command is run by the user, as themselves, against a profile they were told was disposable ([#73](https://github.com/aiidateam/aiida-agents/issues/73)).
>
> Two things the original also got wrong, both discovered by using it. The printed `verdi profile setup` could never complete, because it creates a default user and the read-only role refuses the insert — the setup path had never been run end to end. And SQLite, the default backend for `verdi presto`, has no roles at all, so the whole design had nothing to offer most users.
>
> The rule that replaces it is one sentence: **a sandbox profile must never share deletable storage with a real one.** It lives in `sandbox/copy.py` as `shares_storage`, it fails closed, and `init`, `check`, `teardown` and `doctor` all ask it rather than each re-deciding.

Two further layers sit above it, and neither is containment:

- A **static guard** rejects imports outside an allowlist, calls that write, and the builtins (`getattr`, `exec`, `open`) that would step around either rule. It turns the common mistakes into a readable message instead of a permission error. It is a pre-check and must never be relied on: a one-line bypass reaching `os` through an allowed module survived review and was found by dogfooding.
- A **subprocess** with a timeout, a scrubbed environment (it inherited the user's API keys until #73), resource limits, and its own process group so a timeout takes with it whatever the snippet spawned.

**The execution tool is not approval-gated**, which is the decision most worth arguing with. It holds no capability to write, so there is nothing for an approval to protect. And the alternative is worse than it looks: an approval prompt showing twenty lines of unexecuted Python asks a researcher to audit code under time pressure, which is a far weaker check than the one it appears to be. Letting the code run where it can do no harm and showing what it *returned* converts the same decision into one made from evidence.

**Everything that genuinely writes stays where it was**: on the Execution agent, behind `requires_approval=True`, where the preview is a resolved input a researcher can judge in seconds.

**It is a third specialist**, on both grounds [`extending.md`](/docs/extending.md) requires. Its tool surface includes code execution, whose blast radius differs in kind from a twelve-tool read-only surface; and its prompt describes a loop (write, run, read the traceback, try again) that is a different job from explaining what is in a database. Diagnosis, by contrast, needed neither and remained a tool.

**It is not exported over MCP.** The safety of `run_aiida_code` rests entirely on `AIIDA_AGENTS_SANDBOX_PROFILE` naming a profile someone verified. An MCP client cannot check that, and we cannot see whether it holds; a client that wants to run Python can run it itself, with its own consent.

## Consequences

- A question needing an unanticipated combination of filters is answerable without a new tool.
- Answers come with output that was actually produced, and the agent fixes its own mistakes from real tracebacks before the user sees them.
- **The feature is inert until someone runs `sandbox init`.** `run_aiida_code` refuses to run when no sandbox profile is configured, and says the snippet is unverified rather than falling back to the user's writable profile. Silently falling back is the one failure that would make all of the above worthless.
- **The copy drifts.** It is a snapshot, so anything the user has run since is missing until `sandbox refresh`. That is the price of not sharing their storage, and it is the right way round: a stale answer is visible and correctable, a destroyed database is not.
- **Copying is not free.** A large provenance database takes time and disk to copy, which is why the copy has a lifecycle (`init`/`refresh`/`teardown`) rather than being made per query.
- **The containment covers the database and nothing else.** Code that gets past the guard can still read the filesystem and reach the network. The environment scrub and the rlimits narrow that; OS-level isolation (`bwrap`, `nsjail`, a container) would close it, and is not implemented.
- Writes remain impossible from this path. A user who wants to submit is told so and routed to the Execution agent, which asks first.

## Alternatives considered

**Approval-gate the execution tool.** Rejected: it protects nothing the database does not already refuse, and it trains users to approve code they have not read, which devalues the approval prompt that guards real submissions.

**A scratch profile with copied data.** Originally rejected: *"copying enough provenance to answer real questions is its own project, and a partial copy produces answers that are wrong in a way nobody can see."* **This is now the decision.** The objection assumed a *partial* copy; a whole one has neither problem, and on SQLite it is a directory copy. The real cost is time and disk on a large database, which is a lifecycle question, not a correctness one.

**Keeping the read-only role as the mechanism, and warning about deletion instead.** Rejected: the warning would have to compete with the word "sandbox", which tells the user their whole life that the thing is disposable. A design that needs a warning to be safe is not safe.

**A restricted interpreter instead of a database role.** Rejected: Python is not sandboxable in-process, and a guard that claimed to be one would be believed.

**No execution: generate code and let the user run it.** Rejected as the whole product. It leaves the correctness problem entirely with the user, and correctness is the thing the feature exists to provide.
