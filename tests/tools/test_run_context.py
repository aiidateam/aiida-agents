"""Test suite for query_run_context tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiida_agents.tools.run_context import query_run_context


class TestAnalysisQueries:
    """Verify query_run_context queries return expected structure and data."""

    def test_query_past_workflows_returns_expected_structure(self) -> None:
        """Querying past workflows must return required structured fields."""
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "PwRelaxWorkChain"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["workflow_type"] == "PwRelaxWorkChain"
        assert "count" in res
        assert "success_rate" in res
        assert "median_ecutwfc" in res
        assert "common_parameters" in res
        assert "common_failure_modes" in res
        assert "example_structures" in res

    def test_query_available_codes_returns_list(self, arithmetic_add_code: Any) -> None:
        """Querying available codes returns a list of codes with expected structure.

        Uses the ``arithmetic_add_code`` session fixture to guarantee at least
        one real code exists in the in-memory test profile.
        """
        res = query_run_context(
            query_type="available_codes",
            filters={},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res.get("codes"), list)
        # The conftest registers a 'bash' code — at least that one must appear
        assert len(res["codes"]) > 0
        # Every entry must have the expected schema fields
        for code in res["codes"]:
            assert "label" in code
            assert "plugin" in code

    def test_query_failed_attempts_structured(self) -> None:
        """Querying failed attempts should return structured failure modes."""
        res = query_run_context(
            query_type="failed_attempts",
            filters={"workflow_type": "PwRelaxWorkChain"},
        )
        assert res["query_type"] == "failed_attempts"
        assert "attempts" in res
        assert isinstance(res["attempts"], list)

    def test_invalid_query_type_raises_error(self) -> None:
        """Unknown query_type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown query_type"):
            query_run_context("invalid_query_type", {})


class TestAnalysisQueriesEdgeCases:
    """Verify behavior on empty results and unknown filters."""

    def test_query_past_workflows_empty_results(self) -> None:
        """A workflow with no matching runs returns the full shape, not an error.

        The workflow named here is not installed, so it exercises the
        no-matches path. This test owns the *structural* contract -- every
        statistics key present, defaults filled in, nothing raised -- so a
        caller can read the result without branching.

        What the note should *say* in that situation is asserted by
        ``TestEntryPointSpellingsAllFindTheSameRuns`` instead. It used to be
        checked here as "Using defaults", which encoded the very conflation
        that hid a real bug: an unregistered name and a registered workflow
        with no history are different facts and were reported identically.
        """
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "aiida.workflows:ExoticWorkChain"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["count"] == 0
        assert res["success_rate"] == 0.0
        assert res["median_ecutwfc"] is None
        assert res["common_parameters"] == {}
        assert res["note"]

    def test_query_available_codes_empty_results(self) -> None:
        """Querying an unknown code should return an empty codes list."""
        res = query_run_context(
            query_type="available_codes",
            filters={"code": "unknown-abinit-code"},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res["codes"], list)
        assert len(res["codes"]) == 0


class TestEntryPointSpellingsAllFindTheSameRuns:
    """A workflow's history must be found however the agent names the workflow.

    Regression from executing the scenarios in ``docs/gsoc/agent-scenarios.md``
    against real nodes. Nodes store ``process_label`` as the class name, but the
    lookup derived it with ``workflow_type.split(":")[-1]``, which only works
    for the legacy ``aiida.workflows:PwRelaxWorkChain`` spelling used in the
    prompt's examples. ``list_process_entry_points()`` hands the agent modern entry points
    (``core.arithmetic.multiply_add``), which have no colon, were passed through
    whole, and matched nothing.

    The result was silent: "No prior runs of this workflow found. Using
    defaults." on a database containing them, so the agent built inputs
    believing there was no history to draw on.
    """

    @pytest.mark.parametrize(
        "workflow_type",
        [
            "core.arithmetic.multiply_add",  # what list_process_entry_points() reports
            "aiida.workflows:core.arithmetic.multiply_add",  # what process_type stores
            "MultiplyAddWorkChain",  # the class name itself
        ],
    )
    def test_every_spelling_finds_the_run(
        self,
        multiply_add_workchain: Any,
        workflow_type: str,
    ) -> None:
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": workflow_type},
        )

        assert res["count"] >= 1, (
            f"{workflow_type!r} found no runs of "
            f"{multiply_add_workchain.process_label!r}; note was: {res.get('note')!r}"
        )
        assert res["success_rate"] > 0

    def test_unresolvable_workflow_says_so_instead_of_claiming_no_history(
        self,
    ) -> None:
        """An unregistered name must not be reported as "no prior runs".

        The two are different facts and lead the agent to different actions:
        one means proceed with defaults, the other means the request itself was
        wrong. Conflating them is what made the original bug invisible.
        """
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "definitely.not.installed"},
        )

        assert res["count"] == 0
        assert "not a registered entry point" in res["note"]


class TestNestedWorkflowParametersAreFound:
    """Statistics must come from nested ports, not only top-level ones.

    Found by running this tool against a real database. It matched three real
    successful ``PwRelaxWorkChain`` runs and still reported
    ``median_ecutwfc: None`` for all of them, which reads as "no historical
    data" rather than "we looked in the wrong place".

    AiiDA stores a nested port in a flat link label joined by ``__``, and
    ``node.inputs`` rebuilds the nesting from those labels -- so
    ``"parameters" in node.inputs`` asks only about a top-level port.
    ``PwRelaxWorkChain`` puts its Quantum ESPRESSO settings at
    ``base.pw.parameters``, so the check was always False and both statistics
    were silently skipped for exactly the workflows this tool summarises.

    Built here rather than tested against aiida-quantumespresso so the shape is
    pinned in CI without that plugin installed.
    """

    @staticmethod
    def _finished_workchain(label: str, inputs: dict[str, Any]) -> Any:
        """A stored, finished WorkChainNode with the given input link labels."""
        from aiida import orm
        from aiida.common.links import LinkType
        from aiida.engine import ProcessState

        node = orm.WorkChainNode()
        node.base.attributes.set("process_label", label)
        for link_label, source in inputs.items():
            node.base.links.add_incoming(
                source, link_type=LinkType.INPUT_WORK, link_label=link_label
            )
        node.store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(0)
        node.seal()
        return node

    def test_cutoff_nested_under_a_subnamespace_is_found(self) -> None:
        from aiida import orm

        params = orm.Dict({"SYSTEM": {"ecutwfc": 60.0}}).store()
        spacing = orm.Float(0.15).store()
        self._finished_workchain(
            "NestedParamsWorkChain",
            {"base__pw__parameters": params, "base__kpoints_distance": spacing},
        )

        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "NestedParamsWorkChain"},
        )

        assert res["count"] == 1
        assert res["median_ecutwfc"] == 60.0
        assert res["median_kpoints_distance"] == 0.15
        assert res["common_parameters"] == {"ecutwfc": 60.0}

    def test_top_level_ports_still_work(self) -> None:
        """The flat case must not regress -- some workflows really are flat."""
        from aiida import orm

        params = orm.Dict({"SYSTEM": {"ecutwfc": 45.0}}).store()
        self._finished_workchain(
            "FlatParamsWorkChain",
            {"parameters": params, "kpoints_distance": orm.Float(0.3).store()},
        )

        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "FlatParamsWorkChain"},
        )

        assert res["median_ecutwfc"] == 45.0
        assert res["median_kpoints_distance"] == 0.3

    def test_lowercase_system_card_is_accepted(self) -> None:
        """QE input is case-insensitive about card names; plugins use both."""
        from aiida import orm

        self._finished_workchain(
            "LowerCardWorkChain",
            {"base__pw__parameters": orm.Dict({"system": {"ecutwfc": 80.0}}).store()},
        )

        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "LowerCardWorkChain"},
        )

        assert res["median_ecutwfc"] == 80.0

    def test_several_namespaces_contribute_the_most_demanding_setting(self) -> None:
        """One value per run: a multi-step workflow sets the port more than once.

        PwBands has scf__pw__parameters and bands__pw__parameters, which can
        differ. Taking the highest cutoff (and the densest mesh) keeps one
        value per run and reports the setting that governed its hardest step --
        the safer number for the agent to reuse.
        """
        from aiida import orm

        self._finished_workchain(
            "MultiStepWorkChain",
            {
                "scf__pw__parameters": orm.Dict({"SYSTEM": {"ecutwfc": 40.0}}).store(),
                "bands__pw__parameters": orm.Dict(
                    {"SYSTEM": {"ecutwfc": 90.0}}
                ).store(),
                "scf__kpoints_distance": orm.Float(0.4).store(),
                "bands__kpoints_distance": orm.Float(0.1).store(),
            },
        )

        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "MultiStepWorkChain"},
        )

        assert res["count"] == 1
        assert res["median_ecutwfc"] == 90.0
        assert res["median_kpoints_distance"] == 0.1


class TestStatisticsCarryTheirUnits:
    """A bare number invites the caller to supply a unit, and one did.

    Across three real-model runs the same 60.0 cutoff was reported as "Ry"
    twice and "eV" once -- a factor-of-twenty error in a value someone would
    configure a calculation with. The tool knows the unit and the caller does
    not, so the tool states it rather than leaving a gap for a guess.
    """

    def test_units_are_reported_alongside_the_values(
        self, multiply_add_workchain: Any
    ) -> None:
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "MultiplyAddWorkChain"},
        )

        assert res["units"]["ecutwfc"] == "Ry"
        assert "1/A" in res["units"]["kpoints_distance"]

    def test_units_are_present_even_when_there_are_no_runs(self) -> None:
        """The caller must not have to branch on whether data was found.

        An empty result that omits the units invites exactly the improvisation
        this field exists to prevent, on the next result that does have values.
        """
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "definitely.not.installed"},
        )

        assert res["count"] == 0
        assert res["units"]["ecutwfc"] == "Ry"


class TestFailedAttemptsAreActuallyFound:
    """A failed run in the profile must come back from ``failed_attempts``.

    The query filtered ``exit_status`` with ``{"!=": 0}``, which the sqlite
    storage backends do not implement for a JSON attribute: they raise
    ``ValueError: SQLite does not support JSON query``. Every call therefore
    failed on a ``core.sqlite_dos`` profile and reported ``attempts: []``
    alongside the note "Real introspection of failed WorkChainNode records."

    Placed above the classes that take ``aiida_profile_clean``: that fixture
    closes the storage the session-scoped node fixtures are bound to, so
    anything requesting one after it errors out with ``ClosedStorage``.
    """

    def test_a_failed_run_is_reported(
        self, failed_multiply_add: Any, arithmetic_add_code: Any
    ) -> None:
        res = query_run_context(query_type="failed_attempts", filters={})

        pks = [attempt["pk"] for attempt in res["attempts"]]
        assert failed_multiply_add.pk in pks, (
            f"the failed run was not reported; suggestion was: {res['suggestion']!r}"
        )

    def test_a_successful_run_is_not_reported(
        self, multiply_add_workchain: Any, failed_multiply_add: Any
    ) -> None:
        """Nothing that finished with exit status 0 is a failed attempt."""
        res = query_run_context(query_type="failed_attempts", filters={})

        pks = [attempt["pk"] for attempt in res["attempts"]]
        assert multiply_add_workchain.pk not in pks
        assert all(attempt["exit_code"] != 0 for attempt in res["attempts"])


class TestTheToolStatesNothingItCannotSupport:
    """No invented subject, no invented qualifier, no unfounded endorsement.

    Audit finding. Every item below was a value the tool supplied itself and
    the caller could not tell apart from a queried one -- which is exactly what
    the agents' grounding rules forbid the *model* from doing, enforced
    everywhere except in the tool feeding it.
    """

    def test_missing_workflow_type_is_an_error_not_a_guess(self) -> None:
        """It used to default to PwRelaxWorkChain and report that as the subject.

        On a profile with 35 entry points, a caller who omitted the filter got
        confident statistics about a workflow they never named.
        """
        with pytest.raises(ValueError, match="needs filters"):
            query_run_context(query_type="past_successful_workflows", filters={})

    def test_no_structure_type_is_echoed_back(
        self, multiply_add_workchain: Any
    ) -> None:
        """It used to default to "metallic" and return it as if asked.

        Nothing filters on it and nothing verifies it, so a model could report
        "for metallic structures, the median is ..." on a qualifier that came
        from the tool rather than from the user or the database.
        """
        res = query_run_context(
            query_type="past_successful_workflows",
            filters={"workflow_type": "MultiplyAddWorkChain"},
        )

        assert "structure_type" not in res
        assert "structure_type_filter_note" not in res

    def test_codes_are_not_ranked(self, arithmetic_add_code: Any) -> None:
        """`recommended_version` was "the first code, or any whose label has 'pw'".

        That is not a recommendation, and the field name asserted an
        endorsement the tool has no basis for -- while also holding a label
        rather than a version.
        """
        res = query_run_context(query_type="available_codes", filters={})

        assert res["codes"], "fixture code should be found"
        assert "recommended_version" not in res

    def test_no_pseudo_family_is_named_when_none_is_installed(
        self, aiida_profile_clean: Any
    ) -> None:
        """The worst of them: it returned "SSSP/1.3/PBE/efficiency (needs installation)".

        A family label for something not in the profile, in a field a model
        would reasonably pass on as a workflow input. The note still tells the
        user how to install one -- guidance in prose, not a usable-looking
        reference.
        """
        res = query_run_context(query_type="available_pseudos", filters={})

        assert "recommended_family" not in res
        assert res["installed_families"] == []
        assert "aiida-pseudo install" in res["note"]


def _write_upf(directory: Path, element: str) -> None:
    """A UPF file aiida-pseudo can parse, minimal enough to build a family."""
    (directory / f"{element}.upf").write_text(
        f'<UPF version="2.0.1">\n'
        f'<PP_HEADER element="{element}" pseudo_type="NC" z_valence="4.0"/>\n'
        "</UPF>\n"
    )


class TestInstalledPseudosAreFound:
    """A stocked profile must not report as empty.

    Both queries looked in the wrong place, so the execution agent refused to
    submit any PW workflow while the protocol builder resolved a pseudo fine.
    Family groups are ``pseudo.family`` and its subtypes, and the pseudos are
    aiida-pseudo's own classes under ``data.pseudo.``, which is not where
    aiida-core's deprecated ``UpfData`` lives.
    """

    def test_an_installed_family_and_its_pseudos_are_reported(
        self, aiida_profile_clean: Any, tmp_path: Path
    ) -> None:
        """The family label reaches the caller, and so does the pseudo count."""
        from aiida_pseudo.groups.family.sssp import SsspFamily

        _write_upf(tmp_path, "Si")
        _write_upf(tmp_path, "Ge")
        SsspFamily.create_from_folder(tmp_path, "SSSP/1.3/PBE/efficiency")

        res = query_run_context(query_type="available_pseudos", filters={})

        labels = [family["label"] for family in res["installed_families"]]
        assert labels == ["SSSP/1.3/PBE/efficiency"]
        assert res["pseudo_count"] == 2
        assert "Found 1 pseudopotential family (2 pseudopotentials)" in res["note"]
        assert "aiida-pseudo install" not in res["note"]

    def test_a_non_upf_format_counts_too(
        self, aiida_profile_clean: Any, tmp_path: Path
    ) -> None:
        """The count is of pseudopotentials, in any of the six formats.

        Naming it after UPF was the other half of the original bug: a PSF
        family must not read as "no pseudopotentials installed" either.
        """
        from aiida_pseudo.data.pseudo.psf import PsfData
        from aiida_pseudo.groups.family.pseudo import PseudoPotentialFamily

        # PsfData reads the element off the first whitespace-delimited token.
        (tmp_path / "Si.psf").write_text(" Si  pseudopotential\n")
        PseudoPotentialFamily.create_from_folder(
            tmp_path, "psf-family", pseudo_type=PsfData
        )

        res = query_run_context(query_type="available_pseudos", filters={})

        assert res["pseudo_count"] == 1
        assert "(1 pseudopotential)" in res["note"]
        assert "aiida-pseudo install" not in res["note"]

    def test_an_empty_family_does_not_read_as_stocked(
        self, aiida_profile_clean: Any
    ) -> None:
        """A family holding no pseudos is a label a workflow would reject.

        Reporting it as installed, next to a count of 0, is the same
        families-say-yes/count-says-no contradiction this query is fixed for.
        """
        from aiida_pseudo.groups.family.pseudo import PseudoPotentialFamily

        PseudoPotentialFamily(label="empty-family").store()
        PseudoPotentialFamily(label="another-empty-family").store()

        res = query_run_context(query_type="available_pseudos", filters={})

        assert res["pseudo_count"] == 0
        assert "Found 2 pseudopotential families" in res["note"]
        assert "no pseudopotentials at all" in res["note"]
        assert "aiida-pseudo install" in res["note"]

    def test_family_descriptions_drop_the_checksums(
        self, aiida_profile_clean: Any, tmp_path: Path
    ) -> None:
        """Only the human-readable first line of a description is passed on.

        aiida-pseudo appends two md5 lines the caller can neither act on nor
        verify, and they dominate the payload once several families exist.
        """
        from aiida_pseudo.groups.family.sssp import SsspFamily

        _write_upf(tmp_path, "Si")
        family = SsspFamily.create_from_folder(tmp_path, "SSSP/1.3/PBE/efficiency")
        family.description = (
            "SSSP v1.3 PBE efficiency installed with aiida-pseudo v1.5.0\n"
            "Archive pseudos md5: a58f1b3373f330179fd0832c48bb9a52\n"
            "Pseudo metadata md5: 3153c4b20fc90a44fba0236627525644"
        )

        res = query_run_context(query_type="available_pseudos", filters={})

        (reported,) = res["installed_families"]
        assert reported["description"] == (
            "SSSP v1.3 PBE efficiency installed with aiida-pseudo v1.5.0"
        )

    def test_the_count_survives_aiida_pseudo_being_absent(
        self, aiida_profile_clean: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No entry point is resolved to run the count.

        The families come back from a raw ``type_string`` filter, which holds
        whether or not aiida-pseudo is importable -- an archive import is
        enough to have them. A count going through ``DataFactory("pseudo")``
        would not, so the payload would name families beside zero pseudos.
        """
        import aiida.plugins
        from aiida.common.exceptions import MissingEntryPointError
        from aiida_pseudo.groups.family.sssp import SsspFamily

        _write_upf(tmp_path, "Si")
        SsspFamily.create_from_folder(tmp_path, "SSSP/1.3/PBE/efficiency")

        # Signature matched to the real DataFactory, so a future arity change
        # fails here rather than being swallowed by ``*args``.
        def no_such_entry_point(entry_point_name: str, load: bool = True) -> Any:
            msg = f"Entry point '{entry_point_name}' not found in group 'aiida.data'"
            raise MissingEntryPointError(msg)

        monkeypatch.setattr(aiida.plugins, "DataFactory", no_such_entry_point)
        with pytest.raises(MissingEntryPointError):
            aiida.plugins.DataFactory("pseudo")  # the patch is live

        res = query_run_context(query_type="available_pseudos", filters={})

        assert res["pseudo_count"] == 1
        assert [family["label"] for family in res["installed_families"]] == [
            "SSSP/1.3/PBE/efficiency"
        ]
