"""Tests for qdice.qpu_backend: QPU slicing and QPU stitching."""

import pytest

from qdice.qpu_backend import QPUBackend, slice_qpu, stitch_qpus


def test_slice_qpu_removes_cross_partition_edges():
    mono = QPUBackend(num_qubits=6, coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    sliced = slice_qpu(
        mono,
        partitions={"A": [0, 1, 2], "B": [3, 4, 5]},
        nonlocal_edges=[(2, 3)],
    )
    # All original edges should be preserved: local ones stay local, and
    # (2, 3) survives only because it was explicitly designated nonlocal.
    assert set(sliced.coupling_map) == set(mono.coupling_map)
    assert sliced.nonlocal_edges == [(2, 3)]
    assert sliced.partitions == {"A": [0, 1, 2], "B": [3, 4, 5]}
    assert sliced.port_qubits() == [2, 3]


def test_slice_qpu_actually_cuts_an_edge():
    mono = QPUBackend(num_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    sliced = slice_qpu(
        mono,
        partitions={"A": [0, 1], "B": [2, 3]},
        nonlocal_edges=[],  # no nonlocal edges designated
    )
    # (1, 2) crosses the A/B boundary and was not designated nonlocal, so it
    # must be removed from the sliced coupling map.
    assert (1, 2) not in sliced.coupling_map
    assert set(sliced.coupling_map) == {(0, 1), (2, 3)}


def test_slice_qpu_rejects_unassigned_qubits():
    mono = QPUBackend(num_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    with pytest.raises(ValueError):
        slice_qpu(mono, partitions={"A": [0, 1]}, nonlocal_edges=[])


def test_slice_qpu_rejects_nonlocal_edge_within_same_partition():
    mono = QPUBackend(num_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    with pytest.raises(ValueError):
        slice_qpu(
            mono,
            partitions={"A": [0, 1, 2], "B": [3]},
            nonlocal_edges=[(0, 1)],  # (0, 1) is inside partition A
        )


def test_stitch_qpus_reindexes_and_connects():
    qpu1 = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    qpu2 = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    stitched = stitch_qpus({"Q1": qpu1, "Q2": qpu2}, stitch_links=[("Q1", 1, "Q2", 0)])

    assert stitched.num_qubits == 4
    assert stitched.partitions == {"Q1": [0, 1], "Q2": [2, 3]}
    # Local edges preserved, reindexed into the global qubit space.
    assert (0, 1) in stitched.coupling_map
    assert (2, 3) in stitched.coupling_map
    # New stitching edge between Q1's qubit 1 (global 1) and Q2's qubit 0 (global 2).
    assert (1, 2) in stitched.nonlocal_edges


def test_restricted_coupling_map_only_keeps_designated_nonlocal_edges():
    mono = QPUBackend(num_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    sliced = slice_qpu(mono, partitions={"A": [0, 1], "B": [2, 3]}, nonlocal_edges=[(1, 2)])
    restricted = sliced.restricted_coupling_map()
    assert set(restricted) == {(0, 1), (1, 2), (2, 3)}
