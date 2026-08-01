"""Tests for the static pass that refuses code before it runs.

The interesting half is what gets *through*. A guard that rejects real
QueryBuilder usage is worse than no guard: it makes the feature unusable and
the pressure to loosen it lands on the rules that matter. So the allowed cases
here are as load-bearing as the refused ones.
"""

from __future__ import annotations

import pytest

from aiida_agents.sandbox.guard import check_code


def _reasons(code: str) -> str:
    return " | ".join(r.reason for r in check_code(code))


class TestAllowed:
    """Real code that must run."""

    def test_a_plain_querybuilder_query_passes(self) -> None:
        code = """
from aiida.orm import QueryBuilder, CalcJobNode
qb = QueryBuilder()
qb.append(CalcJobNode, filters={'attributes.exit_status': 0})
print(qb.all())
"""
        assert check_code(code) == []

    def test_append_is_not_treated_as_a_write(self) -> None:
        """It is how QueryBuilder is used; blocking it would gut the guard."""
        assert check_code("qb.append(CalcJobNode)") == []

    def test_loading_a_node_passes(self) -> None:
        code = "from aiida.orm import load_node\nprint(load_node(334407).exit_status)"

        assert check_code(code) == []

    @pytest.mark.parametrize(
        "module", ["json", "math", "collections", "datetime", "itertools", "re"]
    )
    def test_a_safe_stdlib_import_passes(self, module: str) -> None:
        assert check_code(f"import {module}") == []

    def test_an_aiida_submodule_import_passes(self) -> None:
        assert check_code("from aiida.orm.nodes.data import StructureData") == []

    def test_reading_an_attribute_passes(self) -> None:
        assert check_code("print(node.exit_status, node.pk, node.uuid)") == []


class TestRefusedImports:
    """A module nobody vetted is refused by absence, not by blocklist."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "import socket",
            "from pathlib import Path",
            "import requests",
        ],
    )
    def test_an_unlisted_import_is_refused(self, code: str) -> None:
        assert check_code(code), f"{code!r} should have been refused"

    def test_the_refusal_names_the_module(self) -> None:
        """So a retry can fix it rather than guess at the rule."""
        assert "'subprocess'" in _reasons("import subprocess")

    def test_a_dotted_unlisted_import_is_refused(self) -> None:
        assert check_code("import os.path")


class TestRefusedWrites:
    """Reading is the whole supported use."""

    @pytest.mark.parametrize(
        "code",
        [
            "node.delete()",
            "node.store()",
            "node.set_extra('x', 1)",
            "node.delete_extra('x')",
            "builder.submit()",
            "engine.run(builder)",
            "node.seal()",
        ],
    )
    def test_a_write_call_is_refused(self, code: str) -> None:
        assert check_code(code), f"{code!r} should have been refused"

    def test_the_refusal_says_it_would_modify_the_database(self) -> None:
        assert "modify the database" in _reasons("node.delete()")


class TestEscapeHatches:
    """Ways around every other rule on this page."""

    def test_getattr_is_refused(self) -> None:
        """getattr(node, 'delete')() reaches a blocked name without writing it."""
        assert check_code("getattr(node, 'delete')()")

    @pytest.mark.parametrize(
        "code",
        ["exec('x=1')", "eval('1+1')", "__import__('os')", "open('/etc/passwd')"],
    )
    def test_a_dangerous_builtin_is_refused(self, code: str) -> None:
        assert check_code(code), f"{code!r} should have been refused"

    def test_dunder_attribute_access_is_refused(self) -> None:
        """The classic way out of a restricted namespace."""
        assert check_code("().__class__.__base__.__subclasses__()")


class TestReporting:
    """What the caller is told."""

    def test_every_violation_is_reported_not_just_the_first(self) -> None:
        """One rejection per round trip would cost a retry each."""
        code = "import os\nimport socket\nnode.delete()"

        assert len(check_code(code)) == 3

    def test_rejections_come_in_source_order(self) -> None:
        code = "x = 1\nimport os\nnode.delete()"

        assert [r.line for r in check_code(code)] == [2, 3]

    def test_the_line_number_is_reported(self) -> None:
        assert check_code("x = 1\nimport os")[0].line == 2

    def test_unparseable_code_is_refused_with_the_syntax_error(self) -> None:
        """Unrunnable and unanalysable are the same answer to the caller."""
        rejections = check_code("qb = QueryBuilder(")

        assert len(rejections) == 1
        assert "not valid Python" in rejections[0].reason

    def test_a_rejection_reads_as_a_sentence(self) -> None:
        assert str(check_code("import os")[0]).startswith("line 1: ")

    def test_empty_code_is_not_a_violation(self) -> None:
        assert check_code("") == []
