"""Refuse generated code that could change anything, before it runs.

The first line of defence for executing model-written Python, and the cheapest:
a static pass over the AST that rejects whole categories of code outright. It
runs before a subprocess is spawned, so a rejection costs nothing.

**This is a pre-check, never the boundary.** Python cannot be contained
in-process, and a determined generation can defeat any pattern match. The point
of this layer is to turn the obvious mistakes into a clear message instead of a
Postgres permission error; the containment is what the code runs *under*, in
:mod:`aiida_agents.sandbox.runner`.

That distinction was not academic. Dogfooding turned up a one-line bypass
(`#73 <https://github.com/aiidateam/aiida-agents/issues/73>`_)::

    import aiida.common.folders
    aiida.common.folders.os.system("...")

``aiida`` is an allowed import, and the check was on the root module only, so
this passed --- and ``os`` here is the real standard-library ``os``. It was not
specific to ``aiida`` either: ``numpy.os`` and ``json.codecs.sys`` were
reachable the same way, because nearly every module imports ``os`` and
re-exports it as an attribute. :data:`FORBIDDEN_MODULE_ATTRIBUTES` closes the
route by naming the destinations rather than the routes, but the lesson is the
one above: assume this layer is porous and put nothing behind it that the layers
below do not also refuse.

The rules are deliberately narrow and deliberately dumb. Anything cleverer
would be a static analyser we would then have to trust, and the value here is
that a reader can check the whole policy in one sitting.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = ["Rejection", "check_code"]

#: Modules the generated code may import. AiiDA itself, plus the handful of
#: standard-library modules a query or a summary legitimately reaches for.
#: Everything else --- os, sys, subprocess, shutil, socket, pathlib, requests
#: --- is absent rather than blocked by name, so a module nobody thought of is
#: refused by default instead of allowed by oversight.
ALLOWED_IMPORTS = frozenset(
    {
        "aiida",
        "collections",
        "datetime",
        "itertools",
        "json",
        "math",
        "numpy",
        "pprint",
        "re",
        "statistics",
        "typing",
    }
)

#: Attribute names that write. Reading is the entire supported use, so a call
#: to any of these is a mistake worth naming rather than a permission error to
#: decode. ``append`` is conspicuously absent: it is how QueryBuilder is used,
#: and blocking it would make the guard useless for the main case it serves.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "delete",
        "delete_attribute",
        "delete_extra",
        "delete_group",
        "delete_nodes",
        "delete_property",
        "run",
        "run_get_node",
        "run_get_pk",
        "seal",
        "set_attribute",
        "set_extra",
        "set_extra_many",
        "store",
        "store_all",
        "submit",
    }
)

#: Builtins that would step around every other rule here. ``getattr`` and
#: friends are on the list because ``getattr(node, "delete")()`` reaches a
#: forbidden attribute without ever writing its name.
FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)

#: Modules that must not be reached **as an attribute of something else**, and
#: must not be pulled out of an allowed package by ``from ... import``.
#:
#: Refusing ``import os`` while allowing ``import aiida.common.folders`` then
#: ``aiida.common.folders.os.system(...)`` is not a policy, it is a formality:
#: a module that imports ``os`` re-exports it as an attribute, and almost every
#: module imports ``os``. The same held for every entry in
#: :data:`ALLOWED_IMPORTS`, not only ``aiida`` --- ``numpy.os`` and
#: ``json.codecs.sys`` were both reachable.
#:
#: These are the *terminal* names, not every module that contains one. Reaching
#: ``os.system`` means writing ``os`` somewhere; reading ``sys.modules['os']``
#: means writing ``sys``. Blocking the destinations covers every route to them
#: without trying to enumerate the routes.
#:
#: Deliberately not the whole of ``sys.stdlib_module_names``: ``uuid``, ``code``
#: and ``time`` are all stdlib modules *and* ordinary AiiDA attributes, so a
#: blanket rule would refuse ``node.uuid`` and be abandoned within a day.
FORBIDDEN_MODULE_ATTRIBUTES = frozenset(
    {
        "atexit",
        "builtins",
        "ctypes",
        "fcntl",
        "fileinput",
        "ftplib",
        "gc",
        "glob",
        "http",
        "importlib",
        "inspect",
        "io",
        "marshal",
        "mmap",
        "multiprocessing",
        "nt",
        "os",
        "pathlib",
        "pickle",
        "posix",
        "pty",
        "requests",
        "resource",
        "runpy",
        "shutil",
        "site",
        "socket",
        "subprocess",
        "sys",
        "sysconfig",
        "tempfile",
        "urllib",
        "webbrowser",
    }
)


@dataclass(frozen=True)
class Rejection:
    """Why a snippet was refused, in terms its author can act on."""

    reason: str
    line: int

    def __str__(self) -> str:
        return f"line {self.line}: {self.reason}"


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


class _Inspector(ast.NodeVisitor):
    """Collect every rule violation rather than stopping at the first.

    A model fixing one rejection at a time costs a round trip each; reporting
    all of them lets a single retry address the lot.
    """

    def __init__(self) -> None:
        self.rejections: list[Rejection] = []

    def _reject(self, node: ast.AST, reason: str) -> None:
        self.rejections.append(Rejection(reason, getattr(node, "lineno", 0)))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _root_module(alias.name) not in ALLOWED_IMPORTS:
                self._reject(node, f"importing {alias.name!r} is not allowed here")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if _root_module(module) not in ALLOWED_IMPORTS:
            self._reject(node, f"importing from {module!r} is not allowed here")
        # The names matter as much as the module. ``from aiida.common.folders
        # import os`` passes the module check --- the root is ``aiida`` --- and
        # then ``os`` is a plain local name that no later rule looks at.
        for alias in node.names:
            if alias.name in FORBIDDEN_MODULE_ATTRIBUTES:
                self._reject(
                    node,
                    f"importing {alias.name!r} is not allowed here, "
                    f"whichever module it is taken from",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self._reject(node, f"{node.attr}() would modify the database")
        elif node.attr in FORBIDDEN_MODULE_ATTRIBUTES:
            self._reject(
                node,
                f"reaching {node.attr!r} through another module is not allowed here",
            )
        elif node.attr.startswith("__"):
            # ``obj.__class__.__subclasses__()`` is the standard way out of a
            # restricted namespace. Generated code has no business here.
            self._reject(node, f"access to {node.attr!r} is not allowed here")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._reject(node, f"{node.id!r} is not available here")
        self.generic_visit(node)


def check_code(code: str) -> list[Rejection]:
    """Every reason ``code`` must not be run, or an empty list.

    Args:
        code: The Python to inspect. Not executed, imported or compiled.

    Returns:
        One :class:`Rejection` per violation, in source order. A snippet that
        does not parse yields a single rejection naming the syntax error ---
        unrunnable and unanalysable are the same answer to the caller.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [Rejection(f"not valid Python: {exc.msg}", exc.lineno or 0)]

    inspector = _Inspector()
    inspector.visit(tree)
    return sorted(inspector.rejections, key=lambda r: r.line)
