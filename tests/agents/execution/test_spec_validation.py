"""Test suite for workflow specification validation tool."""

from __future__ import annotations


from aiida_agents.tools.workflows.spec_validation import validate_workflow_spec


class TestSpecValidationCatchesErrors:
    """Verify validate_workflow_spec catches missing inputs and parameter errors."""

    def test_catches_missing_required_structure(self):
        """Spec missing required input should fail validation."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                # structure intentionally missing
                "parameters": {"system": {"ecutwfc": 65, "ecutrho": 520}},
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert not result["valid"]
        assert any("structure" in err["error"] for err in result["errors"])

    def test_catches_ecutwfc_too_low(self):
        """Spec with too-low ecutwfc (< 20 Ry) should fail validation."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": 15,  # Below minimum (20)
                        "ecutrho": 120,
                    }
                },
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert not result["valid"]
        assert any("ecutwfc" in err["error"] for err in result["errors"])

    def test_catches_type_mismatch_string_instead_of_float(self):
        """Should catch string value where float expected, while still running checks safely without TypeError."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": "65",  # String instead of float
                        "ecutrho": 520,
                    }
                },
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert not result["valid"]
        assert any(
            "expected float, got str" in err["error"] for err in result["errors"]
        )

    def test_catches_bad_ecutrho_ratio(self):
        """Should warn when ecutrho != 8x ecutwfc."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": 65,
                        "ecutrho": 200,  # Far too low (should be ~520)
                    }
                },
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert any("ecutrho" in w.lower() for w in result["warnings"])

    def test_passes_valid_spec(self):
        """A properly structured spec should pass validation cleanly."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": 65,
                        "ecutrho": 520,
                    }
                },
                "kpoints_distance": 0.2,
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert result["valid"]
        assert len(result["errors"]) == 0


class TestSpecValidationEdgeCases:
    """Test validation behavior under unusual or unknown parameters."""

    def test_unknown_structure_type_handled_gracefully(self):
        """If structure_type is unknown or non-standard, validation should complete cleanly using defaults."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": 55,
                        "ecutrho": 440,
                    }
                },
            },
        }
        result = validate_workflow_spec(spec, structure_type="exotic_crystal")
        assert result["valid"]
        assert isinstance(result["warnings"], list)
