# ADR-07: Validator, with deterministic schema and range checks before writes

> Status: revised twice. (2026-06) The standalone validator subpackage is
> removed; schema validation is delegated to AiiDA's own
> `spec.inputs.validate()`. (2026-07) The range/physics tier is built, but as
> a warning at the approval prompt rather than the gate this ADR specified:
> see the second Revision section.

## Context

The agent can now submit AiiDA workflows via `submit_workflow` (ADR-08).
Before any submission reaches the database, the inputs must be validated
deterministically: without involving the LLM, which can produce inputs that
look plausible but are type-incorrect or physically nonsensical.

A wrong submission on an HPC cluster wastes thousands of core-hours and
pollutes the provenance graph. The cost asymmetry (wrong read = nothing,
wrong submit = expensive) demands a hard gate, not a soft prompt.

AiiDA's process spec already encodes everything needed for schema validation:
which ports are required, what types they accept, and what the help text says.
There is no reason to duplicate this knowledge in the validator.

## Decision

Build a deterministic Validator as a subpackage
(`agents/validator/`) with two tiers, executed in order before
`submit_workflow` calls `aiida.engine.submit`.

### Tier 1: Schema validation (`_schema.py`)

Checks inputs against the process class's own `spec().inputs`:

- Required ports must be present.
- Provided values must be instances of the port's `valid_type`.
- Metadata ports (scheduler options) are skipped: AiiDA validates those
  at submit time.

The process class is loaded by entry point string (`"core.arithmetic.add"`,
`"core.arithmetic.multiply_add"`, etc.) by trying `aiida.calculations` then
`aiida.workflows`. No workflow-specific knowledge is hardcoded; the spec is
the source of truth. This makes the validator generic: it works for any
AiiDA process without modification.

Port attributes are read directly (`port.required`, `port.valid_type`,
`port.is_metadata`) rather than via `port.serialize()`, which requires a
value argument and is not appropriate for this use.

### Tier 2: Range and physics checks (`_ranges.py`)

Placeholder for Weeks 7–8. Will enforce sensible value ranges and physics
constraints (positive k-point meshes, non-negative energies, reasonable
wallclock limits). Returns an empty error list until workflow-specific
rules are defined.

### Public API

```python
from aiida_agents.agents.validator import validate, ValidationError

validate(entry_point, inputs)  # raises ValidationError if any tier fails
```

`ValidationError` carries a list of error strings, one per failing check,
so the agent can surface all failures at once rather than one at a time.

### Integration with submit_workflow

`submit_workflow` calls `validate()` before `aiida.engine.submit`. If
validation raises, a `ToolError` is raised instead: the submission never
reaches the database. This is enforced in tests:
`test_no_submit_without_valid_inputs` monkeypatches `aiida.engine.submit`
and asserts it is never called when validation fails.

## Consequences

- Type and presence errors are caught before any database write.
- The validator is generic: no per-workflow hardcoding required.
- Adding range checks in Weeks 7–8 requires only adding rules to
  `_ranges.py`; no changes to the public API or `submit_workflow`.
- The schema tier depends on AiiDA's port spec being accurate, which it
  is for all core workflows; plugin workflows may have less precise specs.

## Alternatives considered

- **Hardcode checks per workflow.**
  Rejected: requires updating the validator for every new workflow;
  AiiDA's spec is already the source of truth, so duplicating it is waste.
- **LLM-based validation.**
  Rejected: non-deterministic; an LLM cannot reliably catch type errors or
  missing required ports. The validator is explicitly deterministic Python.
- **Rely on AiiDA's own validation at submit time.**
  Rejected: AiiDA raises at submit, which means the database write has
  already been attempted. We want to catch errors before touching the DB.
- **Single flat module instead of subpackage.**
  Rejected: the two-tier design (schema + ranges) benefits from separate
  files that can evolve independently; a subpackage also mirrors the
  `agents/analysis/` pattern established in ADR-04.

## Revision (2026-06)

The two-tier subpackage in this ADR was implemented and then removed. The
schema tier (`_schema.py`) re-implemented checks that AiiDA's port spec already
performs: `process_class.spec().inputs.validate(inputs)` runs the full
required/type/nested-namespace validation, returns the first error, and (the
point the "rely on AiiDA" alternative above missed) writes nothing to the
database. It is pre-submit, not submit-time, so it meets the "catch before any
DB write" requirement that drove this ADR while honoring its own stated
principle: there is no reason to duplicate the spec's knowledge.

Validation now lives in `_prepare_submission` (`tools/submit.py`), the
single seam every submission passes through: resolve the agent's JSON inputs to
unstored nodes, then call `spec.inputs.validate()`; on failure raise
`SubmissionInputError`. The CLI (`_triage_submissions`) runs this before the
user is prompted and denies invalid submissions straight back to the model
(pydantic-ai `ToolDenied`), so the agent corrects its own inputs and only valid
submissions reach the confirmation prompt. The range/physics tier, if pursued,
attaches as an extra check in `_prepare_submission`, or as port validators on
the workflow (which `spec.inputs.validate()` runs for free), not as a separate
subpackage.

Trade-off accepted: `spec.inputs.validate()` reports the first error, not all
at once. For an agent loop this is fine, arguably better: one fix per turn.

Agent-scope policy: on top of the spec check, `_prepare_submission` requires a
`code` for any compute CalcJob. AiiDA makes `code` optional on the base CalcJob
on purpose, import/parse CalcJobs ingest a `RemoteData` and run no `Code`, but
the agent only submits compute jobs, so requiring one is a deliberate
agent-scope decision (not the duplicated workflow knowledge this ADR swears
off) and fails loudly rather than queueing a job that cannot run.

Scope: resolution and validation operate on top-level inputs only. A nested
input namespace (a real multi-step workflow) is passed through unresolved,
which suits the flat-input demo processes targeted here (arithmetic add /
multiply_add). Nested support can extend `_resolve_inputs` later without
touching the seam or the HITL layer.

## Revision (2026-07): the range tier is advisory, not a gate

Tier 2 is now implemented, in `tools/execution/ranges.py`. It departs from what
this ADR specified in one respect that matters, so the decision is recorded
rather than left as a discrepancy between the document and the code.

**This ADR said the range tier would block a submission. It warns instead.**

The original framing carried over the cost-asymmetry argument that justifies
the schema tier: a wrong submission wastes thousands of core-hours, so gate it
hard. That argument holds for a *type-incorrect* input, which is wrong under
every interpretation. It does not hold for a cutoff.

A cutoff below the pseudopotential family's recommendation is under-converged
for a production run and entirely reasonable for a smoke test, a convergence
study, or a five-minute check that a workflow is wired correctly. There is no
value at which "reject this" is right in general, so a gate would be wrong
about as often as it was right, and a validator that refuses legitimate work
teaches people to route around it, which costs more than the check earns.

What was actually missing was never enforcement. It was the *fact* reaching the
person approving, at the moment they approve. So:

- `check_input_ranges` is a read tool the Execution agent can call before
  proposing a submission, and the prompt tells it to whenever it has set a
  cutoff itself;
- the same check runs unconditionally in `_print_previews`, so a finding
  appears at the approval prompt whether or not the model bothered to look.

That pairing is the one the grounding check already uses (ADR-06's revision):
the prompt asks the model to comply, and the code verifies independently,
because an instruction has a measured failure rate and a post-hoc check does
not depend on compliance.

### What it compares against, and what it will not

The recommendation comes from the pseudopotential family the spec itself names,
through `aiida-pseudo`. This package ships no table of its own: the numbers
belong to the family's authors, who converged them, and a finding cites the
family and elements it came from. Where no family can be identified the check
reports nothing rather than falling back on a general-purpose number.

Cutoffs only. No installed package publishes a recommended k-point spacing per
structure, so a bound on `kpoints_distance` would be a number this project
invented and then presented with a validator's authority: the failure mode the
grounding work exists to prevent. It is left unchecked, and the prompt says so,
so that an empty result is not read as a clean bill of health.

### Consequence

The write path is now guarded at three levels, and only two of them can stop a
submission: AiiDA's own spec validation (hard), the human approval gate (hard),
and the range check (advisory, feeding the second). Nothing in the system
refuses a physically unusual submission on its own authority, and that is
deliberate.
