"""Test suite for workflow schemas and parameter validation rules."""

from __future__ import annotations


from aiida_agents.tools.workflows.schemas import (
    KNOWN_WORKFLOWS,
    PARAMETER_SCHEMA,
    validate_parameter,
    check_parameter_compatibility,
)


class TestSchemas:
    """Test schema constants and basic parameter validation logic."""

    def test_known_workflows_structure(self):
        """Ensure KNOWN_WORKFLOWS has expected profiles and keys."""
        assert "aiida.workflows:PwRelaxWorkChain" in KNOWN_WORKFLOWS
        assert "aiida.workflows:PwCalculation" in KNOWN_WORKFLOWS
        assert "aiida.workflows:PwBandsWorkChain" in KNOWN_WORKFLOWS
        assert len(KNOWN_WORKFLOWS) >= 6
        for name, spec in KNOWN_WORKFLOWS.items():
            assert "description" in spec
            assert "required_inputs" in spec
            assert "optional_inputs" in spec
            assert "for_structure_types" in spec

    def test_parameter_schema_structure(self):
        """Ensure PARAMETER_SCHEMA defines type and ranges for critical cutoffs."""
        assert "ecutwfc" in PARAMETER_SCHEMA
        assert PARAMETER_SCHEMA["ecutwfc"]["type"] in ("float", float)
        assert PARAMETER_SCHEMA["ecutwfc"]["min"] > 0

    def test_validate_parameter_valid(self):
        """Test validate_parameter accepts valid floats and ints for float parameters."""
        valid, msg = validate_parameter("ecutwfc", 65.0)
        assert valid, f"Expected valid float ecutwfc, got error: {msg}"

        valid_int, msg_int = validate_parameter("ecutwfc", 65)
        assert valid_int, (
            f"Expected int accepted for float ecutwfc, got error: {msg_int}"
        )

    def test_validate_parameter_invalid_type(self):
        """Test validate_parameter rejects string where float is expected."""
        valid, msg = validate_parameter("ecutwfc", "65")
        assert not valid
        assert "expected float, got str" in msg

    def test_validate_parameter_out_of_range(self):
        """Test validate_parameter rejects out of range numeric values."""
        valid_low, msg_low = validate_parameter("ecutwfc", -10.0)
        assert not valid_low
        assert "minimum" in msg_low.lower()


class TestParameterCompatibilityRules:
    """Test all parameter compatibility checks thoroughly across structure types."""

    def test_ecutrho_ecutwfc_ratio_rule(self):
        """Test ecutrho ≈ 8 × ecutwfc rule."""
        # Good ratio
        params_good = {"ecutwfc": 65.0, "ecutrho": 520.0}
        warnings = check_parameter_compatibility(
            params_good, structure_type="insulator"
        )
        assert not any("ecutrho" in w.lower() for w in warnings)

        # Bad ratio
        params_bad = {"ecutwfc": 65.0, "ecutrho": 200.0}
        warnings = check_parameter_compatibility(params_bad, structure_type="insulator")
        assert any("ecutrho" in w.lower() for w in warnings)

    def test_smearing_and_degauss_rule(self):
        """Test rule requiring degauss when smearing is specified."""
        # smearing without degauss
        params_bad = {"smearing": "gaussian"}
        warnings = check_parameter_compatibility(params_bad, structure_type="metallic")
        assert any("degauss" in w.lower() for w in warnings)

        # smearing with degauss
        params_good = {"smearing": "gaussian", "degauss": 0.02}
        warnings = check_parameter_compatibility(params_good, structure_type="metallic")
        assert not any("degauss" in w.lower() for w in warnings)

    def test_metallic_smearing_type_rule(self):
        """Test smearing type rule for metallic structures."""
        params_bad = {"smearing": "methfessel-paxton"}
        warnings = check_parameter_compatibility(params_bad, structure_type="metallic")
        assert any(
            "smearing should be gaussian or fermi-dirac" in w.lower() for w in warnings
        )

        # Should not warn for insulator
        warnings_insulator = check_parameter_compatibility(
            params_bad, structure_type="insulator"
        )
        assert not any(
            "smearing should be gaussian or fermi-dirac" in w.lower()
            for w in warnings_insulator
        )

    def test_k_points_distance_metallic_vs_insulator(self):
        """k_points_distance rule triggers specifically for metallic structures when > 0.3 Å^-1."""
        params = {"kpoints_distance": 0.4}
        # Metallic should warn because 0.4 > 0.3
        warnings_metallic = check_parameter_compatibility(
            params, structure_type="metallic"
        )
        assert any("k_points_distance" in w.lower() for w in warnings_metallic)

        # Insulator should NOT trigger this metallic check
        warnings_insulator = check_parameter_compatibility(
            params, structure_type="insulator"
        )
        assert not any("k_points_distance" in w.lower() for w in warnings_insulator)

        # Good metallic value (<= 0.3)
        params_good = {"kpoints_distance": 0.25}
        warnings_good = check_parameter_compatibility(
            params_good, structure_type="metallic"
        )
        assert not any("k_points_distance" in w.lower() for w in warnings_good)

    def test_conv_thr_vs_scf_maxiter_rule(self):
        """Stricter conv_thr (< 1e-8) requires high scf_maxiter (>= 150)."""
        params_strict_low_iter = {"conv_thr": 1e-10, "scf_maxiter": 80}
        warnings = check_parameter_compatibility(
            params_strict_low_iter, structure_type="insulator"
        )
        assert any("scf_maxiter >= 150" in w.lower() for w in warnings)

        params_strict_high_iter = {"conv_thr": 1e-10, "scf_maxiter": 200}
        warnings_ok = check_parameter_compatibility(
            params_strict_high_iter, structure_type="insulator"
        )
        assert not any("scf_maxiter >= 150" in w.lower() for w in warnings_ok)

    def test_geometry_optimization_forces_rule(self):
        """Test rule requiring conv_thr_forces when conv_thr is set for geometry optimization."""
        params_missing_forces = {"conv_thr": 1e-8}
        warnings = check_parameter_compatibility(
            params_missing_forces, structure_type="insulator"
        )
        assert any("conv_thr_forces" in w.lower() for w in warnings)

        params_with_forces = {"conv_thr": 1e-8, "conv_thr_forces": 1e-4}
        warnings_ok = check_parameter_compatibility(
            params_with_forces, structure_type="insulator"
        )
        assert not any("conv_thr_forces" in w.lower() for w in warnings_ok)
