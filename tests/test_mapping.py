"""Tests for qdice.mapping: the boundary-aware distributed circuit mapping algorithm."""

import pytest
from qiskit import QuantumCircuit

from qdice.qpu_backend import QPUBackend, stitch_qpus
from qdice.mapping import distributed_aware_mapping


def _two_node_backend():
    qpu_a = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    qpu_b = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    return stitch_qpus({"A": qpu_a, "B": qpu_b}, stitch_links=[("A", 1, "B", 0)])


def test_mapping_fixes_communication_ancillas_to_their_ports():
    backend = _two_node_backend()
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)

    virtual_partitions = {0: "A", 1: "A", 2: "B", 3: "B"}
    communication_ancillas = {1: 1, 2: 2}

    result = distributed_aware_mapping(
        circuit, backend, virtual_partitions, communication_ancillas,
        num_iterations=10, seed=0,
    )

    # Ancillas must land exactly on their designated physical ports.
    assert result.layout[1] == 1
    assert result.layout[2] == 2
    # Computational qubits must stay within their own partition's physical
    # subgraph.
    assert result.layout[0] in backend.partitions["A"]
    assert result.layout[3] in backend.partitions["B"]


def test_mapped_circuit_never_uses_a_removed_edge():
    backend = _two_node_backend()
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)

    virtual_partitions = {0: "A", 1: "A", 2: "B", 3: "B"}
    communication_ancillas = {1: 1, 2: 2}

    result = distributed_aware_mapping(
        circuit, backend, virtual_partitions, communication_ancillas,
        num_iterations=10, seed=1,
    )

    allowed_edges = {frozenset(e) for e in backend.restricted_coupling_map()}
    for instruction in result.circuit.data:
        if instruction.operation.num_qubits == 2:
            qubits = [result.circuit.find_bit(q).index for q in instruction.qubits]
            assert frozenset(qubits) in allowed_edges


def test_mapping_raises_when_no_valid_routing_exists():
    # Two partitions with *no* nonlocal link at all between them: the
    # restricted coupling map has two disconnected components, so any
    # cross-partition two-qubit gate is unroutable under every layout,
    # regardless of how many SWAPs a transpiler could otherwise insert.
    qpu_a = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    qpu_b = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    backend = stitch_qpus({"A": qpu_a, "B": qpu_b}, stitch_links=[])

    circuit = QuantumCircuit(4)
    circuit.cx(0, 2)  # crosses partitions with no path in the coupling map

    virtual_partitions = {0: "A", 1: "A", 2: "B", 3: "B"}

    with pytest.raises(RuntimeError):
        distributed_aware_mapping(
            circuit, backend, virtual_partitions, communication_ancillas={},
            num_iterations=5, seed=2,
        )
