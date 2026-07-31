"""Read a Quantum ESPRESSO SCF cycle out of a calculation's own output.

``diagnose_process_failure`` resolves a failure as far as the plugin *declares*
it: the exit code, its meaning, the handlers that address it. For a convergence
failure that is where the declared knowledge stops. "The electronic
minimization cycle did not reach self-consistency" is true of a run that was
one iteration short and of a run that oscillated from the start, and those need
opposite responses --- more steps in the first case, a different mixing scheme
or smearing in the second.

Telling them apart means reading the SCF trace pw.x prints as it runs. That is
Quantum ESPRESSO's own output format, so it is parsed here, in the plugin, and
not in ``aiida-agents``: nothing in the core tool layer knows what an ``INCAR``
tag or a Davidson iteration is, and this is the mechanism by which it stays
that way (ADR-10). A VASP plugin would contribute its own equivalent, keyed to
its own output, and the agent would gain it the same way.

The parsing is split from the node handling on purpose. ``parse_scf_trace``
takes text and returns numbers, so it can be tested against real pw.x output
without a profile, a calculation, or Quantum ESPRESSO itself being installed.
"""

from __future__ import annotations

import re
import typing as t

from aiida import orm

__all__ = ["parse_scf_trace", "read_scf_convergence"]

# pw.x prints one of these per electronic iteration, in Ry. The number format
# varies with magnitude ("0.06", "4.9E-09", and the Fortran "1.0D-08"), so the
# character class is deliberately loose and the conversion normalises it.
_ACCURACY = re.compile(
    r"estimated\s+scf\s+accuracy\s*<\s*([0-9.]+(?:[eEdD][+-]?\d+)?)\s*Ry",
    re.IGNORECASE,
)
_ITERATION = re.compile(r"iteration\s*#\s*(\d+)", re.IGNORECASE)
_ACHIEVED = re.compile(
    r"convergence\s+has\s+been\s+achieved\s+in\s+(\d+)\s+iterations", re.IGNORECASE
)
_NOT_ACHIEVED = re.compile(
    r"convergence\s+NOT\s+achieved\s+after\s+(\d+)\s+iterations", re.IGNORECASE
)

# pw.x prints this once per SCF cycle. A single-point calculation has one; a
# relax or vc-relax has one per ionic step, each starting its accuracy series
# over from a large value. Splitting on it is what stops those restarts from
# being read as one long, badly behaved cycle.
_CYCLE_START = re.compile(r"^\s*Self-consistent Calculation\s*$", re.MULTILINE)

# Below this ratio between the largest and smallest of the last few iterations,
# the cycle is treated as having stopped making progress rather than as
# converging slowly. Two is generous: a healthy SCF cycle drops by an order of
# magnitude or more per iteration near the end.
_STALL_RATIO = 2.0

# How many trailing iterations the stall check looks at.
_STALL_WINDOW = 3

# Default name of pw.x's stdout as AiiDA retrieves it, when the calculation
# does not say otherwise.
_DEFAULT_OUTPUT_FILE = "aiida.out"


def _to_float(raw: str) -> float:
    """Parse a number that may carry a Fortran ``D`` exponent."""
    return float(raw.replace("D", "E").replace("d", "e"))


def _trend(series: t.Sequence[float]) -> str:
    """How the accuracy moved across the cycle.

    A model reading two dozen numbers reports the shape of the series
    unreliably; the shape is what distinguishes "nearly there" from "never
    going to get there", so it is computed rather than described.
    """
    if len(series) < 2:
        return "too_few_iterations"
    increases = sum(1 for a, b in zip(series, series[1:]) if b > a)
    if increases == 0:
        return "decreasing"
    if increases >= 2:
        return "oscillating"
    return "mostly_decreasing"


def _stalled(series: t.Sequence[float]) -> bool | None:
    """True if the last few iterations barely moved.

    Orthogonal to the trend: a cycle can decrease monotonically and still stall,
    which is the case that wants a different mixing scheme rather than a larger
    iteration budget.
    """
    if len(series) < _STALL_WINDOW:
        return None
    window = [value for value in series[-_STALL_WINDOW:] if value > 0]
    if len(window) < _STALL_WINDOW:
        return None
    return max(window) / min(window) < _STALL_RATIO


def _parse_cycle(text: str) -> dict[str, t.Any]:
    """One SCF cycle: its accuracy series, verdict, and shape."""
    accuracies = [_to_float(match) for match in _ACCURACY.findall(text)]
    iterations = [int(match) for match in _ITERATION.findall(text)]

    achieved = _ACHIEVED.search(text)
    not_achieved = _NOT_ACHIEVED.search(text)
    if achieved:
        converged: bool | None = True
        iterations_run = int(achieved.group(1))
    elif not_achieved:
        converged = False
        iterations_run = int(not_achieved.group(1))
    else:
        converged = None
        iterations_run = max(iterations) if iterations else 0

    return {
        "converged": converged,
        "iterations_run": iterations_run,
        "scf_accuracy": accuracies,
        "final_accuracy_ry": accuracies[-1] if accuracies else None,
        "trend": _trend(accuracies),
        "stalled": _stalled(accuracies),
    }


def parse_scf_trace(text: str) -> dict[str, t.Any]:
    """Extract the electronic convergence trace from pw.x stdout.

    A relax or vc-relax runs one SCF cycle *per ionic step*, and each starts
    its accuracy over from a large value. Read as a single series those
    restarts look exactly like a cycle that keeps blowing up, so the trace is
    segmented per cycle first and the shape is computed within each.

    That distinction decides the advice: a relax whose every cycle converged
    cleanly, reported as "oscillating", sends a reader after the mixing scheme
    when nothing was wrong with the electronic minimisation at all.

    Args:
        text: The contents of the calculation's output file.

    Returns:
        A mapping describing the **final** cycle --- ``converged``,
        ``iterations_run``, ``trend``, ``stalled``, ``final_accuracy_ry`` ---
        since that is the one the run ended on and the one any remedy would
        act on. ``scf_accuracy`` is the whole run's series in order, across
        every cycle; ``cycles`` holds the same breakdown per ionic step, and
        ``ionic_steps`` counts them.
    """
    segments = _CYCLE_START.split(text)
    # Text before the first marker is pw.x's header, not a cycle. A file with
    # no marker at all (a truncated job, or an output this pattern does not
    # match) falls back to being read as one cycle, as it was before.
    bodies = segments[1:] if len(segments) > 1 else segments
    cycles = [_parse_cycle(body) for body in bodies]
    cycles = [cycle for cycle in cycles if cycle["scf_accuracy"]] or cycles[-1:]

    if not cycles:
        return {
            "converged": None,
            "iterations_run": 0,
            "scf_accuracy": [],
            "final_accuracy_ry": None,
            "trend": _trend([]),
            "stalled": None,
            "ionic_steps": 0,
            "cycles": [],
        }

    last = cycles[-1]
    return {
        **last,
        "scf_accuracy": [value for cycle in cycles for value in cycle["scf_accuracy"]],
        "ionic_steps": len(cycles),
        "cycles": cycles,
    }


def _output_filename(node: orm.CalcJobNode) -> str:
    """The calculation's own output filename, or pw.x's usual one."""
    try:
        name = node.get_option("output_filename")
    except Exception:  # noqa: BLE001 - an option that is simply not set
        name = None
    return str(name) if name else _DEFAULT_OUTPUT_FILE


def _electrons_input(node: orm.CalcJobNode, key: str) -> t.Any:
    """One value from the calculation's ``ELECTRONS`` parameter namespace.

    Read from the inputs rather than the output, so the threshold reported next
    to the achieved accuracy is the one that run was actually asked for.
    """
    try:
        parameters = node.inputs.parameters.get_dict()
    except Exception:  # noqa: BLE001 - no parameters input, or an unstored node
        return None
    electrons = parameters.get("ELECTRONS") or parameters.get("electrons") or {}
    return electrons.get(key)


def read_scf_convergence(identifier: str) -> dict[str, t.Any]:
    """Read a Quantum ESPRESSO calculation's SCF convergence trace.

    Use this when a calculation failed to reach self-consistency and you need
    to say *why* rather than *that*. ``diagnose_process_failure`` gets you the
    exit code and the remedies the work chain knows; this gets you the shape of
    the cycle those remedies would be applied to, which is what decides between
    them.

    A relax or vc-relax runs one SCF cycle per ionic step. ``ionic_steps`` says
    how many, ``cycles`` breaks the run down per step, and the top-level
    ``trend`` / ``stalled`` / ``converged`` describe the **final** cycle --- the
    one the run ended on. Check ``ionic_steps`` before generalising: "the SCF
    oscillated" is a statement about one cycle, not about a relaxation whose
    earlier cycles all converged.

    Read ``trend`` and ``stalled`` together with the numbers:

    - ``decreasing`` and not ``stalled`` --- the cycle was converging and ran
      out of iterations. Compare ``final_accuracy_ry`` against
      ``conv_thr_ry``: if they are within an order of magnitude, more steps
      (``electron_maxstep``) is the proportionate fix.
    - ``oscillating`` --- the cycle is not settling. More steps will not help;
      this is the mixing/smearing case.
    - ``stalled`` --- progress stopped regardless of direction. Again not an
      iteration-budget problem.

    Quote the numbers you are given and do not estimate one that is absent: a
    ``final_accuracy_ry`` of ``None`` means the output never printed one, not
    that convergence was poor.

    Args:
        identifier: The calculation's pk or uuid. This must be the CalcJob that
            ran pw.x --- a work chain retrieves nothing of its own.
            ``diagnose_process_failure`` reports the right pk as
            ``retrieved_files_pk``.

    Returns:
        The parsed trace (final cycle at the top level, per-cycle detail under
        ``cycles``), plus ``conv_thr_ry`` and ``electron_maxstep`` from the
        calculation's own inputs so the accuracy can be read against what it was
        asked for.

    Raises:
        ValueError: If the identifier is not a calculation, retrieved nothing,
            or its output file is absent.
    """
    node = orm.load_node(identifier)
    if not isinstance(node, orm.CalcJobNode):
        msg = (
            f"Node {identifier} is a {type(node).__name__}, not a calculation. "
            "Only a CalcJob retrieves output files; for a work chain, diagnose "
            "it first and use the 'retrieved_files_pk' it reports."
        )
        raise ValueError(msg)

    try:
        retrieved = node.outputs.retrieved
    except Exception as exc:
        msg = (
            f"Calculation {identifier} has no 'retrieved' output, so it brought "
            "back no files -- it likely failed before or during retrieval."
        )
        raise ValueError(msg) from exc

    filename = _output_filename(node)
    available = retrieved.list_object_names()
    if filename not in available:
        msg = (
            f"Calculation {identifier} retrieved no {filename!r}. "
            f"Files it did bring back: {sorted(available)}."
        )
        raise ValueError(msg)

    parsed = parse_scf_trace(retrieved.get_object_content(filename))
    return {
        "pk": node.pk,
        "filename": filename,
        **parsed,
        "conv_thr_ry": _electrons_input(node, "conv_thr"),
        "electron_maxstep": _electrons_input(node, "electron_maxstep"),
    }
