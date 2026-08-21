"""Draft a workflow's inputs from its own ``Process.spec()``.

``build_workflow_inputs`` covers the workflows that ship a
``get_builder_from_protocol``, which is an aiida-quantumespresso / AiiDA Common
Workflows convention rather than an aiida-core one. Everything else --- most
calculations, many older workchains, and whole plugin packages that never
adopted protocols --- had no tool at all. ``build_workflow_inputs`` raised and
told the agent to build the inputs "by hand", which in practice means a model
reconstructing a port tree from prose and getting the names, the nesting, or
which ports are required wrong; the failure only surfaces at submission, if it
surfaces at all.

This tool closes that gap without knowing anything about any particular plugin.
It walks the process's declared input ports, fills in every default the spec
itself declares, keeps whatever the caller already knows, and returns the
required ports it could *not* fill **as data** --- so the next step is to supply
exactly those, rather than to guess a whole tree.

Four behaviours here are judgement calls rather than mechanics, and each is
chosen so that a wrong draft fails loudly and early instead of quietly:

*An unknown port name is an error, not a passthrough.* A port the spec does not
declare is the signature of an invented input, and inventing input names is the
exact failure this tool exists to prevent. The error names the ports that do
exist at that namespace, so the retry is informed. Dynamic namespaces (AiiDA's
own escape hatch for genuinely open-ended inputs, e.g. a ``pseudos`` mapping
keyed by element) accept anything, because there the spec is explicitly not the
authority on what the keys may be.

*A supplied node reference is resolved, then handed back as a reference.*
Resolving proves the node exists while the agent can still do something about
it; keeping the reference form means the drafted spec stays a plain, printable
document that ``execute_workflow_spec`` accepts unchanged.

*An optional namespace nobody has touched is left out.* AiiDA validates such a
namespace only once something in it is supplied, so its inner required ports are
not obligations until then. Reporting them anyway would bury the ports that
genuinely need a decision under the requirements of every optional sub-workflow
the process can run but this submission will not.

*Scheduler metadata is left to AiiDA.* A ``metadata`` namespace at any depth is
not default-filled: it is dozens of ports of scheduler noise that AiiDA fills
perfectly well itself, and copying them into the spec buries the handful of
physics inputs that actually need a decision. Required metadata ports with no
default --- ``metadata.options.resources``, most importantly --- are still
reported as missing, because a submission without them fails.
"""

from __future__ import annotations

import logging
import typing as t

from plumpy.ports import PortNamespace
from pydantic import Field
from typing_extensions import TypedDict

from aiida_agents.tools.execution._spec import (
    is_reference,
    load_process_class,
    port_accepts_node,
    resolve_reference,
    to_spec_value,
    valid_type_names,
)
from aiida_agents.tools.execution.schemas import WorkflowSpec

logger = logging.getLogger(__name__)

__all__ = ["draft_workflow_inputs"]

# A namespace by this name holds scheduler/bookkeeping options rather than
# physics, at whatever depth it appears (a workchain exposes one per sub-process
# it wraps). Its defaults are not copied into the draft; see the module docstring.
_METADATA_NAMESPACE = "metadata"

# Distinguishes "the caller supplied nothing for this port" from "the caller
# supplied None", which for an optional port is a meaningful value.
_UNSET = object()


class MissingInput(TypedDict):
    """A required port the draft could not fill, and what it would accept."""

    port: str
    """Dotted path to the port, e.g. ``'base.pw.structure'``."""

    valid_types: list[str]
    """Node types the port accepts, e.g. ``['StructureData']``."""

    help: str | None
    """The port's own help text, verbatim from the spec, when it has one."""


def _default_of(port: t.Any) -> t.Any:
    """A port's declared default, or ``_UNSET`` if it has none or it fails.

    A default can be a callable the spec evaluates on demand, and that call can
    raise for a port whose default depends on context this draft does not have.
    Such a port is treated as having no default, so a *required* one is reported
    as missing rather than silently dropped.
    """
    if not (hasattr(port, "has_default") and port.has_default()):
        return _UNSET
    try:
        return port.default() if callable(port.default) else port.default
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable default"
        logger.debug("Default for port %r could not be evaluated: %s", port, exc)
        return _UNSET


def _supplied_value(path: str, value: t.Any, port: t.Any) -> t.Any:
    """Validate one caller-supplied value and return it in spec form.

    A reference dict is resolved to prove the node exists --- raising here, while
    the agent can still correct it, rather than at submission --- and then handed
    back unchanged, so the draft stays a plain document. Anything else passes
    through: a nested plain dict is a ``Dict`` port's contents, and a bare value
    is wrapped by ``submit_workflow`` against the port's own type.

    Reference *shape* is not enough: only a port that accepts a node can hold
    one, so in a dict-valued port such as ``metadata.options.environment_variables``
    a key called ``label`` is the user's own data rather than a code to load.
    """
    if port_accepts_node(port) and is_reference(value):
        resolve_reference(value, path)
        return value
    return to_spec_value(value)


def _draft_namespace(
    namespace: PortNamespace,
    supplied: dict[str, t.Any],
    path: str,
    missing: list[MissingInput],
    *,
    fill_defaults: bool,
) -> dict[str, t.Any]:
    """Draft one port namespace, recursing into nested ones.

    Appends to ``missing`` in place, so a required port deep in a namespace is
    reported by its full dotted path rather than being lost in the recursion.
    """
    declared = set(namespace.keys())
    is_dynamic = bool(getattr(namespace, "dynamic", False))

    unknown = [name for name in supplied if name not in declared]
    if unknown and not is_dynamic:
        where = f" in namespace {path!r}" if path else ""
        msg = (
            f"Unknown input port(s) {sorted(unknown)}{where}: this process does "
            f"not declare them. Ports declared here: {sorted(declared)}. Check "
            f"describe_workflow's inputs_schema for the exact nesting."
        )
        raise ValueError(msg)

    drafted: dict[str, t.Any] = {}

    for name, port in namespace.items():
        full = f"{path}.{name}" if path else name
        value = supplied.get(name, _UNSET)

        if isinstance(port, PortNamespace):
            # An optional namespace nobody has touched stays out of the draft
            # entirely. AiiDA validates such a namespace only once something in
            # it is supplied, so demanding its inner required ports would invent
            # obligations the process does not have -- on a real workchain with
            # several optional sub-namespaces, that is most of the report.
            if not getattr(port, "required", False) and not (
                value is not _UNSET and value
            ):
                continue
            if value is _UNSET:
                nested_supplied: dict[str, t.Any] = {}
            elif isinstance(value, dict):
                nested_supplied = value
            else:
                msg = (
                    f"Input {full!r} is a namespace and takes a mapping of its "
                    f"own ports, not {type(value).__name__}."
                )
                raise ValueError(msg)
            nested = _draft_namespace(
                port,
                nested_supplied,
                full,
                missing,
                fill_defaults=fill_defaults and name != _METADATA_NAMESPACE,
            )
            if nested:
                drafted[name] = nested
            continue

        if value is not _UNSET:
            drafted[name] = _supplied_value(full, value, port)
            continue

        default = _default_of(port) if fill_defaults else _UNSET
        if default is not _UNSET:
            drafted[name] = to_spec_value(default)
            continue

        if getattr(port, "required", False):
            missing.append(
                MissingInput(
                    port=full,
                    valid_types=valid_type_names(port),
                    help=getattr(port, "help", None),
                )
            )

    # Reached only for a dynamic namespace, whose keys the spec deliberately
    # does not constrain; anything else raised above.
    for name in unknown:
        full = f"{path}.{name}" if path else name
        drafted[name] = _supplied_value(full, supplied[name], namespace)

    return drafted


def draft_workflow_inputs(
    entry_point: t.Annotated[
        str,
        Field(
            description="The AiiDA process entry point to draft inputs for, e.g. 'core.arithmetic.add'."
        ),
    ],
    inputs: t.Annotated[
        dict[str, t.Any] | None,
        Field(
            description=(
                "Whatever you already know, in the same nested shape the process "
                'declares: node-valued ports as reference dicts ({"pk": N}, '
                '{"uuid": "..."}, {"label": "name@computer"}), everything else as '
                "a bare value. Anything you leave out that the spec has a default "
                "for is filled in for you; anything required that neither you nor "
                "the spec supplies comes back under 'missing_required'. Omit "
                "entirely on a first call to see what the process asks for."
            )
        ),
    ] = None,
) -> WorkflowSpec:
    """Draft a WorkflowSpec from a process's own input ports, for any process.

    Use this whenever ``describe_workflow`` reports ``has_protocol_builder:
    false``. It reads the same ``Process.spec()`` that validates the eventual
    submission, so the port names and nesting in the returned draft are correct
    by construction --- which is exactly what assembling the inputs from the
    schema by hand cannot guarantee.

    The draft is built to be called more than once: pass what you know, read
    ``missing_required`` from the returned ``metadata``, find those values with
    the tools that know where they live (``list_codes`` for a code,
    ``query_run_context`` for a pseudopotential family or a proven parameter,
    ``search_aiida_docs`` for what a port means), and call again with them
    added. When ``metadata['ready_to_submit']`` is true, the spec has every
    required input and can go to ``execute_workflow_spec``.

    What this tool does *not* do is choose physics for you. It fills in only the
    defaults the process itself declares; a required cutoff with no default
    comes back in ``missing_required``, not filled with a plausible number.
    Where a protocol builder exists, ``build_workflow_inputs`` is still the
    better starting point, because its defaults were tuned by people who ran the
    underlying simulations.

    Args:
        entry_point: AiiDA entry point string, resolved against
            ``aiida.workflows`` then ``aiida.calculations``.
        inputs: Values you already have, nested as the process declares them.

    Returns:
        A ``WorkflowSpec`` whose ``metadata`` carries ``missing_required`` (the
        required ports still unfilled, each with its accepted types and help
        text) and ``ready_to_submit``.

    Raises:
        ValueError: If the entry point is unknown, a supplied port name is not
            declared by the process, a namespace was given a non-mapping, or a
            supplied node reference does not resolve.
    """
    logger.debug(
        "draft_workflow_inputs(entry_point=%r, inputs=%r)", entry_point, inputs
    )
    if not entry_point or not isinstance(entry_point, str):
        msg = "entry_point must be a non-empty string."
        raise ValueError(msg)
    if inputs is not None and not isinstance(inputs, dict):
        msg = f"inputs must be a mapping of port names to values, not {type(inputs).__name__}."
        raise ValueError(msg)

    process_class = load_process_class(entry_point)
    missing: list[MissingInput] = []
    drafted = _draft_namespace(
        process_class.spec().inputs,
        inputs or {},
        path="",
        missing=missing,
        fill_defaults=True,
    )

    return {
        "workflow_type": entry_point,
        "inputs": drafted,
        "metadata": {
            "source": "process_spec",
            "ready_to_submit": not missing,
            "missing_required": missing,
        },
    }
