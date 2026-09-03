"""
qdice.utils
============

Small shared helpers that don't belong to a single methodology stage.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx

from .qpu_backend import QPUBackend


def describe_backend(backend: QPUBackend) -> str:
    """Return a short, human-readable summary of a QPUBackend."""
    lines = [
        f"QPUBackend: {backend.num_qubits} qubits, "
        f"{len(backend.coupling_map)} coupling-map edges",
    ]
    if backend.partitions:
        lines.append("Partitions:")
        for label, qubits in backend.partitions.items():
            lines.append(f"  {label}: {qubits}")
        lines.append(f"Communication ancillas: {backend.communication_ancillas}")
        lines.append(f"Nonlocal (virtual) links: {backend.nonlocal_edges}")
    else:
        lines.append("Monolithic (no partitions defined).")
    return "\n".join(lines)


def draw_backend_graph(backend: QPUBackend, ax=None):
    """
    Draw a backend's coupling map with local edges and nonlocal edges
    colored differently, using networkx + matplotlib.

    Returns the matplotlib Axes used for the drawing.
    """
    import matplotlib.pyplot as plt

    graph = backend.as_graph()
    pos = nx.spring_layout(graph, seed=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    nonlocal_set = set(backend.nonlocal_edges)
    local_edges = [e for e in backend.coupling_map if e not in nonlocal_set]

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#7aa6c2", node_size=500)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=local_edges, edge_color="gray")
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=list(nonlocal_set), edge_color="green", width=2.5
    )
    ax.set_axis_off()
    return ax


def qubit_partition_map(backend: QPUBackend) -> Dict[int, str]:
    """Invert `backend.partitions` into a per-qubit partition lookup."""
    return {q: label for label, qubits in backend.partitions.items() for q in qubits}
