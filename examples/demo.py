"""
examples/demo.py
==================

A minimal, end-to-end walkthrough of the reimplemented Q-DICE methodology:

  1. QPU slicing   -- partition a small generic monolithic backend into two
                        virtual QPUs for hardware-mode emulation.
  2. QPU stitching -- build a small generic multi-QPU backend from scratch
                        for simulation-mode emulation.
  3. Noise models   -- construct Hamiltonian-mediated and entanglement-
                        mediated nonlocal link noise.
  4. Boundary-aware mapping -- map a toy distributed circuit onto the
                        stitched backend without violating partition
                        boundaries.
  5. Simulation     -- run the mapped circuit through Qiskit Aer with the
                        injected nonlocal noise model.

All numeric parameters below (T1/T2, link durations, error probabilities)
are illustrative placeholder values chosen for demonstration purposes only.
They are not drawn from, and are not intended to reproduce, any specific
experimental result reported in the literature.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdice.qpu_backend import QPUBackend, slice_qpu, stitch_qpus
from qdice.noise_models import (
    HamiltonianLinkParams,
    EntanglementLinkParams,
    build_density_matrix_noise_model,
    hardware_delay_duration,
    sample_hardware_pauli_twirl,
)
from qdice.mapping import distributed_aware_mapping
from qdice.utils import describe_backend


def demo_qpu_slicing():
    print("=" * 70)
    print("1) QPU SLICING (hardware-execution path)")
    print("=" * 70)

    # A generic 8-qubit monolithic device with a simple ladder topology.
    # (Not modeled on any specific vendor's real device.)
    monolithic = QPUBackend(
        num_qubits=8,
        coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
    )

    # Cut the ladder into two virtual QPUs of 4 qubits each, keeping the
    # (3, 4) edge as the sole virtual nonlocal link between them.
    distributed = slice_qpu(
        monolithic,
        partitions={"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]},
        nonlocal_edges=[(3, 4)],
    )
    print(describe_backend(distributed))
    print()
    return distributed


def demo_qpu_stitching():
    print("=" * 70)
    print("2) QPU STITCHING (simulation path)")
    print("=" * 70)

    # Two independent 2-qubit QPUs: qubit 0 is a computational qubit,
    # qubit 1 is a communication ancilla.
    qpu_north = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])
    qpu_south = QPUBackend(num_qubits=2, coupling_map=[(0, 1)])

    stitched = stitch_qpus(
        {"north": qpu_north, "south": qpu_south},
        stitch_links=[("north", 1, "south", 1)],
    )
    print(describe_backend(stitched))
    print()
    return stitched


def demo_noise_models():
    print("=" * 70)
    print("3) NONLOCAL NOISE MODELS")
    print("=" * 70)

    # Illustrative Hamiltonian-mediated link parameters.
    h_params = HamiltonianLinkParams(t1_us=60.0, t2_us=40.0, t_link_ns=250.0)
    delay = hardware_delay_duration(t_link_ns=h_params.t_link_ns, t_local_gate_ns=120.0)
    print(f"Hardware-mode delay to insert for Hamiltonian-mediated link: {delay} ns")

    # Illustrative entanglement-mediated link parameters.
    e_params = EntanglementLinkParams(link_error_probability=0.08, t_link_ns=900.0)
    twirl = sample_hardware_pauli_twirl(e_params.link_error_probability)
    print(f"Hardware-mode sampled Pauli twirl outcome (or None): {twirl}")

    noise_model = build_density_matrix_noise_model(
        hamiltonian_edges=[((0, 1), h_params)],
        entanglement_edges=[((2, 3), e_params)],
        nonlocal_gate_name="cx",
    )
    print(noise_model)
    print()
    return noise_model, h_params, e_params


def demo_boundary_aware_mapping(backend):
    print("=" * 70)
    print("4) BOUNDARY-AWARE CIRCUIT MAPPING")
    print("=" * 70)

    # A toy virtual circuit: a Bell pair prepared on "north", teleported
    # across the stitched link, then a local gate on "south".
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)   # local gate within "north"
    circuit.cx(1, 2)   # nonlocal gate across the stitched link
    circuit.h(3)
    circuit.cx(2, 3)   # local gate within "south"
    circuit.measure_all()

    virtual_partitions = {0: "north", 1: "north", 2: "south", 3: "south"}
    communication_ancillas = {1: 1, 2: 2}  # virtual qubit -> physical port

    result = distributed_aware_mapping(
        circuit,
        backend,
        virtual_partitions,
        communication_ancillas,
        num_iterations=25,
        seed=7,
    )
    print(f"Best depth found: {result.depth} (over {result.iterations_run} iterations)")
    print(f"Layout (virtual -> physical): {result.layout}")
    print()
    return result


def demo_simulation(mapped_circuit, noise_model):
    print("=" * 70)
    print("5) SIMULATING THE MAPPED, NOISY CIRCUIT")
    print("=" * 70)

    simulator = AerSimulator(noise_model=noise_model)
    job = simulator.run(mapped_circuit, shots=1024)
    counts = job.result().get_counts()
    print("Measurement counts (illustrative only):")
    print(counts)


if __name__ == "__main__":
    demo_qpu_slicing()
    stitched_backend = demo_qpu_stitching()
    noise_model, h_params, e_params = demo_noise_models()
    mapping_result = demo_boundary_aware_mapping(stitched_backend)
    demo_simulation(mapping_result.circuit, noise_model)
