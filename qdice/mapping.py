"""
qdice.mapping
==============

Implements Section III-C of the Q-DICE methodology: a constrained,
boundary-aware layout-selection algorithm for mapping a "virtual" (abstract,
partition-labeled) circuit onto a distributed backend without letting a
standard monolithic transpiler break the partition boundaries.

Standard transpilers are built for monolithic hardware: their SWAP-based
routing and free-form layout selection will happily place a nonlocal
operation on a local link (or vice versa) if doing so reduces circuit
depth, silently destroying the distributed architecture. To prevent this,
the algorithm:

  1. Fixes every communication-ancilla qubit to a specific physical "port"
     qubit at the boundary of its home virtual QPU, so every intended
     nonlocal link corresponds to a real edge in the coupling map.
  2. Randomly assigns the remaining (purely computational) circuit qubits
     to free physical qubits, but only within the physical subgraph of
     their assigned virtual QPU -- never across a partition boundary.
  3. Transpiles the circuit with this partially-fixed initial layout and a
     *restricted* coupling map (all cross-partition edges removed except
     the designated nonlocal links), so the transpiler is structurally
     unable to insert an illegal cross-boundary operation.

This is a constrained random search: step 2 is repeated for a user-supplied
number of iterations, and the lowest-depth valid transpilation found across
all iterations is returned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from .qpu_backend import QPUBackend

Edge = Tuple[int, int]


@dataclass
class MappingResult:
    """Result of the boundary-aware mapping search."""

    circuit: QuantumCircuit
    layout: Dict[int, int]     # virtual circuit qubit -> physical qubit
    depth: int
    iterations_run: int


def _physical_subgraph_qubits(backend: QPUBackend, partition_label: str) -> List[int]:
    """All physical qubits belonging to a given virtual QPU partition."""
    return list(backend.partitions[partition_label])


def distributed_aware_mapping(
    circuit: QuantumCircuit,
    backend: QPUBackend,
    virtual_qubit_partitions: Dict[int, str],
    communication_ancillas: Dict[int, int],
    num_iterations: int = 50,
    seed: Optional[int] = None,
) -> MappingResult:
    """
    Map a virtual (partition-agnostic) circuit onto a distributed backend
    while enforcing partition boundaries, following Algorithm 1 of the
    Q-DICE methodology.

    Parameters
    ----------
    circuit:
        The abstract circuit to map, expressed on virtual qubit indices
        0..circuit.num_qubits-1.
    backend:
        A distributed `QPUBackend` produced by `slice_qpu` or `stitch_qpus`
        (i.e. one with populated `partitions`, `communication_ancillas`,
        and `nonlocal_edges`).
    virtual_qubit_partitions:
        Maps each virtual circuit qubit index to the label of the virtual
        QPU partition (in `backend.partitions`) it is assigned to.
    communication_ancillas:
        Maps a virtual circuit qubit index that acts as a communication
        ancilla to the specific physical port qubit (in
        `backend.communication_ancillas`) it must be pinned to. Every key
        must also appear in `virtual_qubit_partitions`, and the physical
        qubit given must be a valid port for that qubit's home partition.
    num_iterations:
        Number of random-layout trials (N in Algorithm 1). Each iteration
        produces a full initial layout and a full transpilation; the
        lowest-depth valid result is kept.
    seed:
        Optional seed for reproducible random layout search.

    Returns
    -------
    A `MappingResult` containing the best transpiled circuit found, the
    corresponding virtual-to-physical layout, its depth, and the number of
    iterations actually run.
    """
    rng = random.Random(seed)
    restricted_coupling = CouplingMap(couplinglist=backend.restricted_coupling_map())

    # Group virtual qubits by their home partition for per-partition random
    # assignment (Algorithm 1's inner loop over partitions).
    partitions_to_virtual_qubits: Dict[str, List[int]] = {}
    for vq, label in virtual_qubit_partitions.items():
        partitions_to_virtual_qubits.setdefault(label, []).append(vq)

    best_depth = float("inf")
    best_circuit: Optional[QuantumCircuit] = None
    best_layout: Optional[Dict[int, int]] = None
    iterations_run = 0

    for _ in range(num_iterations):
        iterations_run += 1
        layout: Dict[int, int] = {}

        # --- Step 1: fix every communication ancilla to its designated port.
        for vq, physical_port in communication_ancillas.items():
            layout[vq] = physical_port

        fixed_physical_qubits = set(layout.values())

        # --- Step 2: randomly place remaining qubits within their own
        # virtual QPU's physical subgraph, avoiding qubits already used as
        # ports.
        valid_iteration = True
        for label, virtual_qubits in partitions_to_virtual_qubits.items():
            remaining_virtual = [vq for vq in virtual_qubits if vq not in layout]
            available_physical = [
                q
                for q in _physical_subgraph_qubits(backend, label)
                if q not in fixed_physical_qubits
            ]

            if len(remaining_virtual) > len(available_physical):
                # Not enough free physical qubits in this partition; skip
                # this iteration (can happen with pathological inputs).
                valid_iteration = False
                break

            rng.shuffle(available_physical)
            for vq, physical_q in zip(remaining_virtual, available_physical):
                layout[vq] = physical_q
                fixed_physical_qubits.add(physical_q)

        if not valid_iteration:
            continue

        # --- Step 3: transpile with the fixed layout and restricted
        # coupling map, so the transpiler cannot cross partition boundaries.
        initial_layout = [layout[vq] for vq in range(circuit.num_qubits)]
        try:
            candidate = transpile(
                circuit,
                coupling_map=restricted_coupling,
                initial_layout=initial_layout,
                basis_gates=list(backend.basis_gates),
                optimization_level=1,
                seed_transpiler=rng.randint(0, 2**31 - 1),
            )
        except Exception:
            # An unroutable random layout (e.g. no path within the
            # restricted coupling map) simply loses this iteration.
            continue

        if candidate.depth() < best_depth:
            best_depth = candidate.depth()
            best_circuit = candidate
            best_layout = dict(layout)

    if best_circuit is None:
        raise RuntimeError(
            "No valid layout found in the given number of iterations; "
            "consider increasing num_iterations or checking that the "
            "restricted coupling map admits a feasible routing."
        )

    return MappingResult(
        circuit=best_circuit,
        layout=best_layout,
        depth=best_depth,
        iterations_run=iterations_run,
    )
