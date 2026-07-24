"""Surface-agnostic tools for walking a node's provenance links."""

from __future__ import annotations

import logging
import typing as t

from .._orm import load_node
from .._types import Identifier, NodeLink


logger = logging.getLogger(__name__)


def _node_links(
    identifier: Identifier, direction: t.Literal["incoming", "outgoing"]
) -> list[NodeLink]:
    """Return a node's incoming or outgoing links as serialisable dicts."""
    node = load_node(identifier)
    links = (
        node.base.links.get_incoming()
        if direction == "incoming"
        else node.base.links.get_outgoing()
    )
    return [
        {
            "pk": t.cast(int, entry.node.pk),  # a linked node is always stored
            "uuid": entry.node.uuid,
            "node_type": entry.node.node_type,
            "link_label": entry.link_label,
            "link_type": entry.link_type.value,
        }
        for entry in links.all()
    ]


def get_node_inputs(identifier: Identifier) -> list[NodeLink]:
    """Get the incoming links of any AiiDA node by its pk or uuid.

    Works for data and processes alike: a data node's incoming link is the
    process that created it; a process's incoming links are its input data.
    """
    logger.debug("get_node_inputs(identifier=%r)", identifier)
    results = _node_links(identifier, "incoming")
    logger.debug("get_node_inputs: found %d incoming links", len(results))
    return results


def get_node_outputs(identifier: Identifier) -> list[NodeLink]:
    """Get the outgoing links of any AiiDA node by its pk or uuid.

    Works for data and processes alike: a data node's outgoing links are the
    processes that consumed it; a process's outgoing links are the data it
    produced (and any sub-processes it called).
    """
    logger.debug("get_node_outputs(identifier=%r)", identifier)
    results = _node_links(identifier, "outgoing")
    logger.debug("get_node_outputs: found %d outgoing links", len(results))
    return results
