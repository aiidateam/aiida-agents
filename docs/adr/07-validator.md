# ADR-07: Validator — deterministic schema and range checks before writes

> Status: revised (2026-06). The standalone validator subpackage is removed;
> schema validation is delegated to AiiDA's own `spec.inputs.validate()` (see
> the Revision section). The range/physics tier remains deferred.

## Context

The agent can now submit AiiDA workflows via `submit_workflow` (ADR-08).
Before any submission reaches the database, the inputs must be validated
deterministically — without involving the LLM, which can produce inputs that
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

### Tier 1 — Schema validation (`_schema.py`)

Checks inputs against the process class's own `spec().inputs`:

- Required ports must be present.
- Provided values must be instances of the port's `valid_type`.
- Metadata ports (scheduler options) are skipped — AiiDA validates those
  at submit time.

The process class is loaded by entry point string (`"core.arithmetic.add"`,
`"core.arithmetic.multiply_add"`, etc.) by trying `aiida.calculations` then
`aiida.workflows`. No workflow-specific knowledge is hardcoded; the spec is
the source of truth. This makes the validator generic — it works for any
AiiDA process without modification.

Port attributes are read directly (`port.required`, `port.valid_type`,
`port.is_metadata`) rather than via `port.serialize()`, which requires a
value argument and is not appropriate for this use.

### Tier 2 — Range and physics checks (`_ranges.py`)

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
validation raises, a `ToolError` is raised instead — the submission never
reaches the database. This is enforced in tests:
`test_no_submit_without_valid_inputs` monkeypatches `aiida.engine.submit`
and asserts it is never called when validation fails.

## Consequences

- Type and presence errors are caught before any database write.
- The validator is generic — no per-workflow hardcoding required.
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

Validation now lives in `_prepare_submission` (`mcp/tools/submit.py`), the
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
