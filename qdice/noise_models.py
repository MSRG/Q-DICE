"""
qdice.noise_models
====================

Implements Section III-B of the Q-DICE methodology: physically motivated
noise models for the two dominant classes of nonlocal (inter-QPU) links.

1. Hamiltonian-mediated links
   Direct physical interactions between processors (e.g. a shared resonator
   bus or phonon-mediated coupling). Their noise is dominated by amplitude
   relaxation (T1) and dephasing (T2) accrued over the nonlocal gate
   duration `t_link`.

   * Simulation mode: apply single-qubit amplitude-damping and
     phase-damping Kraus channels to the communication-ancilla qubits at
     every nonlocal operation.
   * Hardware mode: since NISQ hardware already relaxes/dephases qubits
     naturally over time, emulate the link by inserting an idle `delay`
     instruction of duration `t_delay = max(0, t_link - t_local_gate)` so
     that native decoherence approximates the target noise budget.

2. Entanglement-mediated links
   Links that rely on remote entanglement generation + teleportation
   (e.g. photonic interconnects). Their dominant error mechanism is
   depolarizing loss in the photonic channel, parameterized by a link
   error probability `p`.

   * Simulation mode: apply a two-qubit depolarizing channel to the pair of
     qubits involved in the nonlocal operation.
   * Hardware mode: Monte-Carlo sample whether an error occurs with
     probability `p`; if so, apply a uniformly random Pauli from {X, Y, Z}
     (a stochastic twirl that drives the state towards maximally mixed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error, depolarizing_error

Edge = Tuple[int, int]


# ----------------------------------------------------------------------
# Parameter containers
# ----------------------------------------------------------------------
@dataclass
class HamiltonianLinkParams:
    """User-defined physical parameters for a Hamiltonian-mediated link."""

    t1_us: float          # relaxation time, T1 (microseconds)
    t2_us: float          # dephasing time, T2 (microseconds)
    t_link_ns: float      # nonlocal ("remote") gate duration (nanoseconds)


@dataclass
class EntanglementLinkParams:
    """User-defined physical parameters for an entanglement-mediated link."""

    link_error_probability: float  # p: probability of a depolarizing event
    t_link_ns: float               # full teleportation-protocol duration (nanoseconds)


# ----------------------------------------------------------------------
# Simulation-mode channels (density-matrix level)
# ----------------------------------------------------------------------
def hamiltonian_link_kraus(params: HamiltonianLinkParams) -> Tuple[Kraus, Kraus]:
    """
    Build the amplitude-damping and phase-damping Kraus channels for a
    Hamiltonian-mediated nonlocal link, following the standard open-system
    decoherence model.

    Amplitude damping probability:  gamma = 1 - exp(-t_link / T1)
    Dephasing probability:          lam   = 1 - exp(-t_link / T2)

    Returns
    -------
    (amplitude_damping_kraus, phase_damping_kraus) -- single-qubit Kraus
    channels, meant to be applied to each communication-ancilla qubit
    involved in the nonlocal operation.
    """
    t_link_us = params.t_link_ns / 1000.0
    gamma = 1 - np.exp(-t_link_us / params.t1_us)
    lam = 1 - np.exp(-t_link_us / params.t2_us)

    # Amplitude damping Kraus operators.
    e0_amp = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    e1_amp = np.array([[0, np.sqrt(gamma)], [0, 0]])
    amplitude_damping = Kraus([e0_amp, e1_amp])

    # Phase damping Kraus operators.
    e0_phase = np.array([[1, 0], [0, np.sqrt(1 - lam)]])
    e1_phase = np.array([[0, 0], [0, np.sqrt(lam)]])
    phase_damping = Kraus([e0_phase, e1_phase])

    return amplitude_damping, phase_damping


def entanglement_link_kraus(params: EntanglementLinkParams) -> Kraus:
    """
    Build the two-qubit depolarizing channel for an entanglement-mediated
    nonlocal link:

        E(rho) = (1 - p) * rho + p * I / 4

    equivalently written in the two-qubit Pauli basis as an even mixture of
    all 15 nontrivial Pauli-pair operators. This channel is applied to the
    pair of communication-ancilla qubits at each nonlocal operation.
    """
    p = params.link_error_probability
    # qiskit_aer's depolarizing_error already implements this exact channel
    # for an n-qubit operation; converting to Kraus form keeps the return
    # type consistent with hamiltonian_link_kraus for downstream code that
    # wants to apply the channel manually (e.g. via Statevector/DensityMatrix).
    aer_error = depolarizing_error(p, num_qubits=2)
    return Kraus(aer_error.to_quantumchannel())


# ----------------------------------------------------------------------
# Building a full Aer-compatible noise model for simulation mode
# ----------------------------------------------------------------------
def build_density_matrix_noise_model(
    hamiltonian_edges: Sequence[Tuple[Edge, HamiltonianLinkParams]] = (),
    entanglement_edges: Sequence[Tuple[Edge, EntanglementLinkParams]] = (),
    nonlocal_gate_name: str = "cx",
    local_gate_time_ns: float = 0.0,
) -> NoiseModel:
    """
    Assemble a `qiskit_aer.noise.NoiseModel` that injects nonlocal-link noise
    on specific two-qubit gate instances, keyed by the physical qubit pair
    they act on. This lets a density-matrix simulator (e.g. Qiskit Aer)
    reproduce distributed-link noise for whichever edges are nonlocal, while
    leaving purely local operations untouched.

    Parameters
    ----------
    hamiltonian_edges:
        List of ((qubit_a, qubit_b), HamiltonianLinkParams) pairs. For each,
        an amplitude-damping + phase-damping channel (composed) is attached
        to `nonlocal_gate_name` acting on that qubit pair.
    entanglement_edges:
        List of ((qubit_a, qubit_b), EntanglementLinkParams) pairs. For
        each, a two-qubit depolarizing channel is attached to
        `nonlocal_gate_name` acting on that qubit pair.
    nonlocal_gate_name:
        The instruction name used to represent a nonlocal two-qubit
        operation in the transpiled circuit (commonly "cx" or "cz").
    local_gate_time_ns:
        Reserved for symmetry with the hardware-mode delay calculation; not
        used when building a simulation-mode noise model, since noise here
        is applied analytically rather than through elapsed device time.

    Returns
    -------
    A `NoiseModel` with one quantum error entry per nonlocal edge.
    """
    noise_model = NoiseModel()

    for (q_a, q_b), params in hamiltonian_edges:
        amp_damp, phase_damp = hamiltonian_link_kraus(params)
        # Apply both single-qubit channels to each of the two qubits
        # involved in the nonlocal gate, then compose into one two-qubit
        # error attached to that specific (qubit_a, qubit_b) gate instance.
        single_qubit_channel = amp_damp.compose(phase_damp)
        two_qubit_channel = single_qubit_channel.tensor(single_qubit_channel)
        noise_model.add_quantum_error(
            two_qubit_channel, nonlocal_gate_name, [q_a, q_b]
        )

    for (q_a, q_b), params in entanglement_edges:
        channel = entanglement_link_kraus(params)
        noise_model.add_quantum_error(channel, nonlocal_gate_name, [q_a, q_b])

    return noise_model


# ----------------------------------------------------------------------
# Hardware-mode helpers
# ----------------------------------------------------------------------
def hardware_delay_duration(t_link_ns: float, t_local_gate_ns: float) -> float:
    """
    Compute the idle-delay duration to insert when emulating a
    Hamiltonian-mediated nonlocal link on real NISQ hardware:

        t_delay = max(0, t_link - t_local_gate)

    The physical two-qubit gate that stands in for the nonlocal operation
    already consumes `t_local_gate` of the `t_link` noise budget; any
    remainder is made up with an explicit delay so that native device
    decoherence approximates the requested link noise.
    """
    return max(0.0, t_link_ns - t_local_gate_ns)


def sample_hardware_pauli_twirl(p: float, rng: random.Random | None = None) -> str | None:
    """
    Monte-Carlo hardware-mode emulation of an entanglement-mediated link.

    With probability `p`, a random Pauli error (uniformly chosen from
    {X, Y, Z}) is deemed to occur and should be injected into the physical
    circuit (i.e. as an explicit gate on the target qubit) to emulate
    depolarizing loss. With probability (1 - p), no error is injected.

    Parameters
    ----------
    p:
        Probability that a depolarizing event occurs on this nonlocal
        operation instance.
    rng:
        Optional `random.Random` instance for reproducible sampling.

    Returns
    -------
    One of "X", "Y", "Z" if an error should be injected, otherwise `None`.
    """
    rng = rng or random
    if rng.random() >= p:
        return None
    return rng.choice(["X", "Y", "Z"])
