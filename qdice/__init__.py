"""
Q-DICE: Quantum Distributed Interconnect Compiler and Emulator
================================================================

A lightweight, from-scratch reimplementation of the *methodology* described
in the Q-DICE paper: a hardware-aware emulation layer that turns monolithic
quantum backends (real or simulated) into distributed multi-QPU systems,
injects physically motivated nonlocal-link noise, and maps circuits onto
those systems while respecting virtual QPU boundaries.

The package is organized into four modules that mirror the paper's system
methodology section:

    qdice.qpu_backend   -- QPU representation, QPU slicing, QPU stitching
    qdice.noise_models   -- Hamiltonian- and entanglement-mediated link noise
    qdice.mapping         -- boundary-aware distributed circuit mapping
    qdice.utils           -- small shared helpers (graph construction, etc.)

Nothing in this package attempts to reproduce any specific numerical result,
figure, or experimental comparison from the paper -- it only implements the
general-purpose mechanisms (slicing, stitching, noise injection, boundary-
aware mapping) so that they can be applied to arbitrary backends and circuits
of the user's choosing.
"""

from .qpu_backend import QPUBackend, slice_qpu, stitch_qpus
from .noise_models import (
    HamiltonianLinkParams,
    EntanglementLinkParams,
    hamiltonian_link_kraus,
    entanglement_link_kraus,
    build_density_matrix_noise_model,
    hardware_delay_duration,
    sample_hardware_pauli_twirl,
)
from .mapping import distributed_aware_mapping, MappingResult
from .reference_parameters import (
    REFERENCE_NOISE_PARAMETERS,
    get_reference_link_params,
    list_references,
)

__all__ = [
    "QPUBackend",
    "slice_qpu",
    "stitch_qpus",
    "HamiltonianLinkParams",
    "EntanglementLinkParams",
    "hamiltonian_link_kraus",
    "entanglement_link_kraus",
    "build_density_matrix_noise_model",
    "hardware_delay_duration",
    "sample_hardware_pauli_twirl",
    "distributed_aware_mapping",
    "MappingResult",
    "REFERENCE_NOISE_PARAMETERS",
    "get_reference_link_params",
    "list_references",
]
