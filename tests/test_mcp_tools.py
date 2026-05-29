"""Tests for AiiDA Agents MCP tools."""

from __future__ import annotations
from unittest.mock import MagicMock, patch

from aiida_agents.mcp.server import mcp
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.nodes import query_nodes, get_node_inputs, get_node_outputs
from aiida_agents.mcp.tools.structures import search_structures


def test_mcp_registration() -> None:
    """Verify that all tools are successfully registered on the FastMCP instance."""
    registered_tools = set(mcp._tool_manager._tools.keys())
    expected_tools = {
        "get_process_status",
        "list_processes",
        "query_nodes",
        "get_node_inputs",
        "get_node_outputs",
        "search_structures",
    }
    assert expected_tools.issubset(registered_tools)


@patch("aiida_agents.mcp.tools.processes.NodeService")
def test_get_process_status_success(mock_service_class: MagicMock) -> None:
    """Test get_process_status tool logic on a successful node lookup."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service

    mock_service.get_one.return_value = {
        "pk": 42,
        "uuid": "uuid-42",
        "process_type": "aiida.calculations:arithmetic.add",
    }
    mock_service.get_field.return_value = {
        "process_label": "ArithmeticAddCalculation",
        "process_state": "finished",
        "exit_status": 0,
        "exit_message": "Completed successfully",
    }

    result = get_process_status(42)

    assert result["pk"] == 42
    assert result["process_label"] == "ArithmeticAddCalculation"
    assert result["state"] == "finished"
    assert result["exit_status"] == 0
    assert result["exit_message"] == "Completed successfully"
    mock_service.get_one.assert_called_once_with(42)
    mock_service.get_field.assert_called_once_with(42, "attributes")


@patch("aiida_agents.mcp.tools.processes.NodeService")
def test_get_process_status_error(mock_service_class: MagicMock) -> None:
    """Test get_process_status tool logic handles exceptions gracefully."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service
    mock_service.get_one.side_effect = ValueError("Node with pk 999 does not exist")

    result = get_process_status(999)

    assert "error" in result
    assert "Node with pk 999 does not exist" in str(result["error"])


@patch("aiida_agents.mcp.tools.processes.NodeService")
def test_list_processes(mock_service_class: MagicMock) -> None:
    """Test list_processes tool successfully queries and formats process entries."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service

    mock_paginated = MagicMock()
    mock_paginated.data = [
        {
            "pk": 10,
            "uuid": "uuid-10",
            "node_type": "node.process.calc.job.CalcJobNode.",
            "process_type": "some_type",
        },
        {
            "pk": 9,
            "uuid": "uuid-9",
            "node_type": "node.process.workflow.workchain.WorkChainNode.",
            "process_type": "some_type",
        },
    ]
    mock_service.get_many.return_value = mock_paginated
    mock_service.get_field.side_effect = [
        {"process_state": "finished", "exit_status": 0},
        {"process_state": "running", "exit_status": None},
    ]

    result = list_processes(limit=2)

    assert len(result) == 2
    assert result[0]["pk"] == 10
    assert result[0]["state"] == "finished"
    assert result[0]["exit_status"] == 0
    assert result[1]["pk"] == 9
    assert result[1]["state"] == "running"
    assert result[1]["exit_status"] is None


@patch("aiida_agents.mcp.tools.nodes.NodeService")
def test_query_nodes(mock_service_class: MagicMock) -> None:
    """Test query_nodes tool successfully queries generic nodes and returns results."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service

    mock_paginated = MagicMock()
    mock_paginated.data = [
        {
            "pk": 42,
            "uuid": "uuid-42",
            "node_type": "node.data.dict.Dict.",
            "ctime": "2026-05-27 12:00:00",
        }
    ]
    mock_service.get_many.return_value = mock_paginated

    result = query_nodes(node_type="Dict", limit=1)

    assert len(result) == 1
    assert result[0]["pk"] == 42
    assert result[0]["node_type"] == "node.data.dict.Dict."
    assert result[0]["created"] == "2026-05-27 12:00:00"


@patch("aiida_agents.mcp.tools.nodes.NodeService")
def test_get_node_inputs(mock_service_class: MagicMock) -> None:
    """Test get_node_inputs tool successfully retrieves incoming links of a node."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service

    mock_service.get_one.side_effect = [
        {"uuid": "uuid-10"},  # Target node uuid lookup
        {"pk": 5, "node_type": "node.data.int.Int."},  # Source node uuid lookup
    ]

    mock_paginated = MagicMock()
    mock_paginated.data = [
        {
            "source": "uuid-5",
            "target": "uuid-10",
            "link_label": "x",
            "link_type": "input",
        }
    ]
    mock_service.get_links.return_value = mock_paginated

    result = get_node_inputs(10)

    assert len(result) == 1
    assert result[0]["pk"] == 5
    assert result[0]["link_label"] == "x"
    assert result[0]["link_type"] == "input"


@patch("aiida_agents.mcp.tools.nodes.NodeService")
def test_get_node_outputs(mock_service_class: MagicMock) -> None:
    """Test get_node_outputs tool successfully retrieves outgoing links of a node."""
    mock_service = MagicMock()
    mock_service_class.return_value = mock_service

    mock_service.get_one.side_effect = [
        {"uuid": "uuid-10"},  # Query node uuid lookup
        {"pk": 12, "node_type": "node.data.float.Float."},  # Target node uuid lookup
    ]

    mock_paginated = MagicMock()
    mock_paginated.data = [
        {
            "source": "uuid-10",
            "target": "uuid-12",
            "link_label": "result",
            "link_type": "output",
        }
    ]
    mock_service.get_links.return_value = mock_paginated

    result = get_node_outputs(10)

    assert len(result) == 1
    assert result[0]["pk"] == 12
    assert result[0]["link_label"] == "result"
    assert result[0]["link_type"] == "output"


@patch("aiida_agents.mcp.tools.structures.orm.load_node")
@patch("aiida_agents.mcp.tools.structures.orm.QueryBuilder")
def test_search_structures(mock_qb_class: MagicMock, mock_load_node: MagicMock) -> None:
    """Test search_structures tool successfully searches structure data with/without formula filter."""
    mock_qb = MagicMock()
    mock_qb_class.return_value = mock_qb
    # Query builder returns structure nodes
    mock_qb.all.return_value = [
        [20, "uuid-20", "2026-05-27 12:00:00", [{"symbols": ["Fe"]}]]
    ]

    mock_structure_node = MagicMock()
    mock_structure_node.get_formula.return_value = "Fe2O3"
    mock_structure_node.sites = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    mock_load_node.return_value = mock_structure_node

    # 1. Search with matching formula
    result_match = search_structures(formula="Fe", limit=1)
    assert len(result_match) == 1
    assert result_match[0]["pk"] == 20
    assert result_match[0]["formula"] == "Fe2O3"
    assert result_match[0]["num_sites"] == 5

    # 2. Search with non-matching formula
    result_no_match = search_structures(formula="Si", limit=1)
    assert len(result_no_match) == 0
