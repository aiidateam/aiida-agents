[![Templated from python-copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mbercx/python-copier/refs/heads/main/docs/img/badge.json)](https://github.com/mbercx/python-copier)

# `aiida-agents`

> ⚠️ **Alpha. No release yet.**


A natural-language interface to [AiiDA](https://www.aiida.net).
Ask in plain language what you want to know or run, and specialized agents call typed Python tools against a real AiiDA profile.

<!-- Illustrative: the pks, exit code and handler name are from a real run
     against a Quantum ESPRESSO profile; the prose is abridged. -->

```console
$ aiida-agents ask "why did pk 334599 fail?"
→ analysis agent

PwBaseWorkChain 334599 exited 501, but that is the work chain reporting that a
sub-process failed. The actual failure is PwCalculation 334407, also exit 501:
"The ionic minimization cycle converged but the thresholds are exceeded in the
final SCF."

The work chain already tried its one applicable remedy:
handle_vcrelax_converged_except_final_scf fired on iteration 1. It still
landed on this exit code, so simply restarting is unlikely to help.
```

## Demo

Where things stood at the beginning of July 2026 (the interface has grown since):

<p align="center">
  <video src="https://github.com/user-attachments/assets/6d2a108c-bf6d-43fc-a4fd-21252939c7c2" width="100%" controls></video>
</p>

## What it can do

**Explore what you already have.** Count and rank nodes, follow provenance, search structures by formula, summarise what past runs of a workflow actually used.

**Explain a failure.** Walk from a work chain's exit code down to the calculation that actually broke, read what that exit code means from the process class itself, and report which of the workflow's own restart handlers already fired, so a remedy that has been tried twice is not recommended a third time. With `aiida-quantumespresso` installed, it also reads pw.x's SCF trace to tell a cycle that ran out of iterations from one that never settled.

**Explain a job that never started.** A failure has an exit code to explain; a process waiting on a stopped daemon has nothing wrong with it at all, and every status tool will call it "waiting" indefinitely. The agents check the daemon when a process is not progressing, and say which of the two it is: a job still running normally, or a queue nothing is draining.

**Set a calculation up and run it.** Discover installed workflows, inspect their input schemas, build inputs from a workflow's protocol builder, or draft them from the declared ports for the many processes that have none. Cutoffs are checked against what the spec's pseudopotential family was converged for.

**Run things in sequence, or in bulk.** Wait for a submission and feed its output to the next one; or rebuild a past run's inputs, change one parameter, and resubmit a whole set under a single approval.

**Nothing is written without your say-so.** Every tool that touches the database is approval-gated: the CLI shows you the resolved inputs and waits. That guarantee lives on the tool, not in a prompt.

**Nothing is quoted that no tool produced.** Every reply is checked for physical quantities (cutoffs, spacings, percentages) that appear in no tool output, and anything unsupported is flagged. This runs on every answer, because an instruction not to invent numbers has a measured failure rate.

## Quickstart

Needs Python ≥ 3.10, a working AiiDA profile, and a model provider.

```bash
pip install "aiida-agents[rag] @ git+https://github.com/aiidateam/aiida-agents.git"
```

Point it at a model, either a cloud provider or Ollama for a local one:

```bash
export AIIDA_AGENTS_PROVIDER=openai        # or anthropic, openrouter, ollama
export OPENAI_API_KEY=...
export AIIDA_AGENTS_MODEL=gpt-4o        # provider-specific; see .env.example
```

Check the wiring, then index the documentation so the agents can look things up:

```bash
aiida-agents doctor        # profile, model reachability, RAG index, docs toolchain
aiida-agents rag build     # clones and renders the AiiDA docs; takes a few minutes
```

Then ask it something:

```bash
aiida-agents ask "how many workchains finished successfully?"
aiida-agents chat          # interactive, and the only mode that can approve a write
```

`ask` is one-shot and cannot approve anything, so a request that wants to submit will tell you to use `chat`.

Other commands: `doctor` (diagnose the whole setup; `--warm` also proves the model generates), `config` (effective settings), `rag search`, `mcp` (serve the read-only tools over the Model Context Protocol).

## Where to look

- **[Architecture](/docs/architecture.md)**: how a request travels through the system, and why the pieces are arranged as they are.
- **[Extending](/docs/extending.md)**: adding a tool, a documentation corpus, or a whole specialist; and how an AiiDA plugin contributes to the agents without either package depending on the other.
- **[Architecture Decision Records](/docs/adr/README.md)**: the reasoning behind each decision, including the ones later revised.
