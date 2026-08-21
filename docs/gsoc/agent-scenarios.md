# Agent scenarios: what the system should serve, and what it can serve today

Empirical input for [ADR-09](/docs/adr/09-agent-orchestration.md).

ADR-04 said the choice between agent-to-agent protocols and plain function calls should be "decided empirically", and no empirical input was ever gathered.
This document is that input: a set of requests a computational materials scientist would plausibly make, each checked against the tools the two agents actually hold today.

## Caveat, stated up front

**These scenarios were written by inference, not gathered from users.**
They are derived from the tool surface, the AiiDA/Quantum ESPRESSO domain, and the project's own testing sessions: not from interviews with the MSD group or from logs of real usage.
That makes them a reasonable starting point and weak evidence.

The meeting notes record that JG "and MSD group, possibly" would refine the system prompts using their scientific domain knowledge.
The same people should sanity-check this list before it is treated as requirements.
Where a scenario below is wrong or missing, the conclusion it feeds may change.

## The tools each agent holds

| Analysis (read-only)    | Execution                    |
| ----------------------- | ---------------------------- |
| `get_process_status`    | `list_process_entry_points`  |
| `get_process_report`    | `describe_process`           |
| `list_recent_processes` | `build_process_inputs`       |
| `query_nodes`           | `list_codes`                 |
| `get_node_inputs`       | `query_run_context`          |
| `get_node_outputs`      | `get_process_status`         |
| `search_structures`     | `search_aiida_docs`          |
| `search_aiida_docs`     | `submit_process_spec` (HITL) |
|                         | `import_structure` (HITL)    |

Note the overlap: both agents hold `search_aiida_docs` and `get_process_status`.

## Scenarios

Verdict key: **A**: Analysis alone. **E**: Execution alone. **A→E**: needs both, in sequence. **GAP**: not servable today.

| #   | Request                                                                           | Verdict       | Notes                                                                                                                                                                            |
| --- | --------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | "How many `PwBandsWorkChain`s finished successfully last month?"                  | A             | `query_nodes` with a filter on `exit_status` and `ctime`.                                                                                                                        |
| 2   | "Show me the five structures with the highest band gap."                          | A             | `query_nodes` sorting an extras field with a numeric cast.                                                                                                                       |
| 3   | "Is pk 1234 still running?"                                                       | A or E        | Both hold `get_process_status`; routing is unambiguous in effect because either answer is correct.                                                                               |
| 4   | "What is a `CalcJobNode`?" / "How do I write a WorkChain?"                        | A or E        | Both hold `search_aiida_docs`. Same observation as #3.                                                                                                                           |
| 5   | "Why did pk 1234 fail?"                                                           | A (partial)   | `get_process_status` then `get_process_report` gives AiiDA's log. When the real cause is in the code's own stdout inside the `retrieved` folder, nothing can reach it: see Gaps. |
| 6   | "What workflows can I run, and what inputs does `PwRelaxWorkChain` need?"         | E             | `list_process_entry_points` then `describe_process`.                                                                                                                             |
| 7   | "Relax the silicon structure at pk 512."                                          | E             | Full discover → describe → `list_codes` → `build_process_inputs` → `submit_process_spec` chain.                                                                                  |
| 8   | "Relax the structure in `~/si.cif`."                                              | E             | Adds `import_structure` at the front. Both writes are HITL-gated.                                                                                                                |
| 9   | "What `ecutwfc` did my successful Si relaxations use? Use that for this new one." | **E**         | **Already served inside Execution**: `query_run_context` returns `median_ecutwfc`, `median_kpoints_distance`, `common_parameters` and success rate. No Analysis call needed.     |
| 10  | "Why did pk 1234 fail, and resubmit it with a longer wallclock."                  | **A→E**       | Diagnosis (`get_process_report`) must inform the resubmission. Not servable today by either agent alone.                                                                         |
| 11  | "Which of my relaxations failed this week? Resubmit them with a higher cutoff."   | **A→E** + GAP | Cross-agent, and additionally needs iteration over a result set: no tool submits in batch.                                                                                       |
| 12  | "Compare the final energies of pk 1234 and pk 1240."                              | A             | `get_node_outputs` on both.                                                                                                                                                      |

## Findings

**1. Most single requests are single-agent.**
Of twelve, eight are served by one agent, two are served equally well by either, and two genuinely need both.
The coordinator is therefore not load-bearing for the common case.

**2. But routing is needed for all twelve.**
Today the user must know the taxonomy and pass `-a execution`.
Every scenario above requires that choice to be made by someone, and there is no scenario in which asking the user to make it improves the answer.

**3. Routing is lower-risk than expected.**
The two most ambiguous request classes (status checks and documentation questions (#3, #4)) are ones *both* agents can serve.
Mis-routing there costs nothing.
The requests where routing genuinely matters (#7, #8 to Execution; #1, #2 to Analysis) are strongly signalled by their own wording.

**4. One of the coordinator's motivating examples is already solved.**
"Use history to inform a new submission" (#9) is handled inside Execution by `query_run_context` (called `query_analysis_agent` when this exercise was written).
The old name was misleading (it queries the database rather than the Analysis agent) but the capability was there.
This weakens the case that cross-agent state-passing is broadly needed: the most common instance of it was already built as a plain tool.

**5. The genuine cross-agent case is diagnose-then-act.**
Scenario #10 is the one thing here that no single agent can do and no renaming would fix: a failure diagnosis has to shape the inputs of the next submission.
It is also, plausibly, the most valuable thing an AiiDA assistant could do.

## What this means for ADR-09

The evidence supports building the orchestrator, but **it did not support the strength of the argument ADR-09's first draft made.**

That draft dismissed the router as "automating a flag" and rested its case on multi-step coordination.
On this evidence, routing is the part that pays off across all twelve scenarios, while true coordination pays off in one or two: one of which (#9) turned out to be already handled by a plain tool.

ADR-09 has been amended to lead with routing accordingly.
The framing it now takes:

- Build the routing layer, because every request needs it and users should not have to know the agent taxonomy.
- Allow multi-step plans, because #10 is real and valuable, but treat it as the second increment, not the justification.
- Keep the ADR's structural decisions unchanged, since none of them depend on which framing wins: one routing path, specialists that do not call each other, approval enforced by `requires_approval` and propagated through the coordinator, two specialists rather than three.

None of ADR-09's structural decisions depended on which framing won, so they stand unchanged.

## Capability gaps found, independent of orchestration

Two gaps surfaced that no amount of orchestration addresses.

**Reading a calculation's own output (#5).**
Nothing in `tools/` opens a `retrieved` `FolderData`.
`get_process_report` returns AiiDA's log and the scheduler's stdout/stderr, but when a Quantum ESPRESSO run fails for a physics reason, the reason is in the code's output file.
This blocks the most common diagnostic question, and it also blocks the valuable half of scenario #10: a diagnosis that cannot see the real error cannot inform a good resubmission.

**Batch operations (#11).**
Every write tool acts on one process.
"Resubmit all of these" has no path, and HITL for a batch is an unanswered design question: one approval for the set, or one per item?

Of these, the first is small, self-contained, and testable offline against the existing `add_calc` fixture, which already produces a `retrieved` output.
It is arguably a better next increment than the coordinator, because scenario #10 (the coordinator's own strongest justification) is only half-useful without it.
