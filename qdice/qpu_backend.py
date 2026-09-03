"""
qdice.qpu_backend
==================

Implements Section III-A of the Q-DICE methodology: a hardware-abstraction
object for QPUs, plus the two techniques used to turn monolithic backends
into distributed multi-QPU systems:

  * QPU *slicing*   -- partition an existing monolithic backend's coupling
                        map into disjoint "virtual QPUs" by cutting edges,
                        preserving only user-designated edges as virtual
                        nonlocal links. Intended for hardware execution,
                        where the physical qubit layout cannot change.

  * QPU *stitching* -- take several independently defined QPU topologies
                        (each with its own qubits and local coupling map)
                        and connect them via newly created edges between
                        user-designated "communication ancilla" qubits.
                        Intended for classical simulation, where QPU
                        topology is not constrained by real hardware.

A `QPUBackend` is a light, framework-agnostic description of a processor:
qubit count, coupling map (undirected edges), per-qubit noise parameters,
and (optionally) per-partition metadata once it has been sliced or stitched
into a distributed system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

Edge = Tuple[int, int]


def _normalize_edge(edge: Edge) -> Edge:
    """Store edges as an undirected, order-independent tuple."""
    a, b = edge
    return (a, b) if a <= b else (b, a)


@dataclass
class QubitProperties:
    """Per-qubit calibration data, analogous to a real backend's calibration table."""

    t1_us: float = 100.0          # relaxation time (microseconds)
    t2_us: float = 80.0           # dephasing time (microseconds)
    readout_error: float = 0.01   # probability of a readout bit-flip
    single_gate_error: float = 0.001
    two_qubit_gate_error: float = 0.01


@dataclass
class QPUBackend:
    """
    A framework-agnostic representation of a (possibly distributed) QPU.

    Attributes
    ----------
    num_qubits:
        Total number of qubits in this backend.
    coupling_map:
        List of undirected edges (i, j) describing which qubit pairs support
        a native two-qubit gate.
    qubit_properties:
        Per-qubit calibration data, keyed by qubit index. Any qubit not
        present falls back to `QubitProperties()` defaults.
    basis_gates:
        Native gate set supported by the backend.
    partitions:
        Maps a partition/virtual-QPU label to the list of qubit indices that
        belong to it. Populated by `slice_qpu` / `stitch_qpus`. Empty for a
        plain monolithic backend.
    communication_ancillas:
        Qubit indices that act as ports for nonlocal (inter-partition)
        operations. Populated by `slice_qpu` / `stitch_qpus`.
    nonlocal_edges:
        Subset of `coupling_map` that represents virtual nonlocal links
        (i.e., edges that cross a partition boundary).
    """

    num_qubits: int
    coupling_map: List[Edge]
    qubit_properties: Dict[int, QubitProperties] = field(default_factory=dict)
    basis_gates: Sequence[str] = ("id", "rz", "sx", "x", "cx")
    partitions: Dict[str, List[int]] = field(default_factory=dict)
    communication_ancillas: List[int] = field(default_factory=list)
    nonlocal_edges: List[Edge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.coupling_map = [_normalize_edge(e) for e in self.coupling_map]
        self.nonlocal_edges = [_normalize_edge(e) for e in self.nonlocal_edges]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def properties_of(self, qubit: int) -> QubitProperties:
        return self.qubit_properties.get(qubit, QubitProperties())

    def as_graph(self) -> nx.Graph:
        """Return the coupling map as a networkx Graph for inspection/plotting."""
        g = nx.Graph()
        g.add_nodes_from(range(self.num_qubits))
        g.add_edges_from(self.coupling_map)
        return g

    def restricted_coupling_map(self) -> List[Edge]:
        """
        The coupling map with all cross-partition edges removed except the
        user-designated nonlocal links. This is `G_restricted` from the
        boundary-aware mapping algorithm (Section III-C): a transpiler given
        this coupling map physically cannot insert an operation across a
        partition boundary except through a legitimate nonlocal link.
        """
        if not self.partitions:
            # Monolithic backend: nothing to restrict.
            return list(self.coupling_map)

        qubit_to_partition = {
            q: label for label, qubits in self.partitions.items() for q in qubits
        }
        allowed = set(self.nonlocal_edges)
        restricted = []
        for edge in self.coupling_map:
            a, b = edge
            same_partition = qubit_to_partition.get(a) == qubit_to_partition.get(b)
            if same_partition or edge in allowed:
                restricted.append(edge)
        return restricted

    def port_qubits(self) -> List[int]:
        """Physical qubits that terminate at least one nonlocal edge."""
        ports = set()
        for a, b in self.nonlocal_edges:
            ports.add(a)
            ports.add(b)
        return sorted(ports)


# ----------------------------------------------------------------------
# QPU slicing (hardware implementation path)
# ----------------------------------------------------------------------
def slice_qpu(
    monolithic: QPUBackend,
    partitions: Dict[str, Sequence[int]],
    nonlocal_edges: Iterable[Edge],
) -> QPUBackend:
    """
    Partition a monolithic backend's coupling map into virtual QPUs by
    removing ("slicing") every edge that crosses a partition boundary,
    except the edges the user explicitly designates as virtual nonlocal
    links.

    Parameters
    ----------
    monolithic:
        The physical, single-QPU backend to be partitioned (e.g. a backend
        object built from a real device's calibration data).
    partitions:
        Mapping from a virtual-QPU label to the qubit indices (on the
        monolithic device) assigned to that virtual QPU. Every qubit must
        belong to exactly one partition.
    nonlocal_edges:
        Coupling-map edges of `monolithic` that should be preserved as
        virtual nonlocal links connecting two different partitions. Every
        such edge becomes a "port" for inter-QPU communication.

    Returns
    -------
    A new `QPUBackend` with the same qubits and calibration data as
    `monolithic`, but whose coupling map only contains local (intra-
    partition) edges plus the designated nonlocal edges, and whose
    `partitions` / `nonlocal_edges` / `communication_ancillas` fields
    describe the resulting distributed structure.
    """
    qubit_to_partition: Dict[int, str] = {}
    for label, qubits in partitions.items():
        for q in qubits:
            if q in qubit_to_partition:
                raise ValueError(f"Qubit {q} assigned to multiple partitions")
            qubit_to_partition[q] = label

    missing = set(range(monolithic.num_qubits)) - set(qubit_to_partition)
    if missing:
        raise ValueError(f"Qubits not assigned to any partition: {sorted(missing)}")

    nonlocal_edges = [_normalize_edge(e) for e in nonlocal_edges]
    nonlocal_set = set(nonlocal_edges)

    for a, b in nonlocal_edges:
        if _normalize_edge((a, b)) not in {
            _normalize_edge(e) for e in monolithic.coupling_map
        }:
            raise ValueError(
                f"Designated nonlocal edge {(a, b)} is not part of the "
                "monolithic device's coupling map"
            )
        if qubit_to_partition[a] == qubit_to_partition[b]:
            raise ValueError(
                f"Nonlocal edge {(a, b)} connects two qubits in the same "
                "partition; nonlocal edges must cross a partition boundary"
            )

    # "Slice": keep an edge only if both endpoints share a partition, or the
    # edge is an explicitly designated nonlocal link.
    sliced_edges: List[Edge] = []
    for edge in monolithic.coupling_map:
        a, b = edge
        if qubit_to_partition[a] == qubit_to_partition[b] or edge in nonlocal_set:
            sliced_edges.append(edge)

    ancillas = sorted({q for edge in nonlocal_edges for q in edge})

    return QPUBackend(
        num_qubits=monolithic.num_qubits,
        coupling_map=sliced_edges,
        qubit_properties=dict(monolithic.qubit_properties),
        basis_gates=monolithic.basis_gates,
        partitions={label: list(qubits) for label, qubits in partitions.items()},
        communication_ancillas=ancillas,
        nonlocal_edges=nonlocal_edges,
    )


# ----------------------------------------------------------------------
# QPU stitching (classical simulation path)
# ----------------------------------------------------------------------
def stitch_qpus(
    qpus: Dict[str, QPUBackend],
    stitch_links: Iterable[Tuple[str, int, str, int]],
) -> QPUBackend:
    """
    Combine several independently defined QPU backends into a single larger
    distributed backend by adding new edges ("stitches") between designated
    communication-ancilla qubits in different processors.

    Because this path targets classical simulation, each input QPU can have
    an arbitrary, user-defined topology (it is not constrained to match any
    real device).

    Parameters
    ----------
    qpus:
        Mapping from a virtual-QPU label to a `QPUBackend` describing that
        processor's local qubits, local coupling map, and local noise
        properties. Backends are assumed to use locally-indexed qubits
        (0..num_qubits-1); this function re-indexes them into one global
        qubit space.
    stitch_links:
        Iterable of `(label_a, local_qubit_a, label_b, local_qubit_b)`
        tuples, each describing a virtual nonlocal link between a
        communication ancilla in QPU `label_a` and one in QPU `label_b`.

    Returns
    -------
    A single `QPUBackend` spanning all qubits of all input processors, with
    local coupling maps preserved, new nonlocal edges added between the
    stitched ancillas, and `partitions` describing which global qubit
    indices belong to which original QPU.
    """
    global_offset: Dict[str, int] = {}
    offset = 0
    for label, qpu in qpus.items():
        global_offset[label] = offset
        offset += qpu.num_qubits
    total_qubits = offset

    combined_edges: List[Edge] = []
    combined_props: Dict[int, QubitProperties] = {}
    partitions: Dict[str, List[int]] = {}
    basis_gates = None

    for label, qpu in qpus.items():
        base = global_offset[label]
        global_qubits = [base + q for q in range(qpu.num_qubits)]
        partitions[label] = global_qubits

        for a, b in qpu.coupling_map:
            combined_edges.append(_normalize_edge((base + a, base + b)))

        for local_q, props in qpu.qubit_properties.items():
            combined_props[base + local_q] = props

        # Basis gates: take the union, keep behavior simple/predictable.
        if basis_gates is None:
            basis_gates = set(qpu.basis_gates)
        else:
            basis_gates |= set(qpu.basis_gates)

    nonlocal_edges: List[Edge] = []
    for label_a, qubit_a, label_b, qubit_b in stitch_links:
        if label_a not in qpus or label_b not in qpus:
            raise ValueError(f"Unknown QPU label in stitch link: {label_a}, {label_b}")
        global_a = global_offset[label_a] + qubit_a
        global_b = global_offset[label_b] + qubit_b
        edge = _normalize_edge((global_a, global_b))
        combined_edges.append(edge)
        nonlocal_edges.append(edge)

    ancillas = sorted({q for edge in nonlocal_edges for q in edge})

    return QPUBackend(
        num_qubits=total_qubits,
        coupling_map=combined_edges,
        qubit_properties=combined_props,
        basis_gates=sorted(basis_gates) if basis_gates else ("id", "rz", "sx", "x", "cx"),
        partitions=partitions,
        communication_ancillas=ancillas,
        nonlocal_edges=nonlocal_edges,
    )
