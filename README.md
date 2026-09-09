# qdice-emulator

[![tests](https://github.com/MSRG/Q-DICE/actions/workflows/tests.yml/badge.svg)](https://github.com/MSRG/Q-DICE/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Q-DICE (Quantum Distributed Interconnect Compiler and Emulator): a hardware-aware emulation layer that turns monolithic quantum backends into distributed multi-QPU systems, injects physically motivated nonlocal-link noise, and maps circuits onto those systems while respecting virtual QPU boundaries. Take a look at the original whitepaper [here](https://arxiv.org/abs/2606.11340).


## Features

- **QPU slicing** — partition a monolithic backend's coupling map into
  virtual QPUs for hardware-mode distributed execution.
- **QPU stitching** — compose independently defined QPU topologies into a
  larger distributed backend for classical simulation.
- **Nonlocal link noise models** — Hamiltonian-mediated (amplitude damping  + dephasing, from T1/T2/link duration) and entanglement-mediated
  (two-qubit depolarizing, from a link error probability) noise channels,
  plus their hardware-mode analogues (idle-delay insertion and Monte Carlo
  Pauli twirling).
- **Boundary-aware circuit mapping** — a constrained layout-search
  algorithm that fixes communication ancillas to physical ports and maps
  the remaining qubits within their own partition, so a standard
  transpiler cannot silently violate distributed boundaries.
- **Reference noise parameters** — a small, citable module of published
  T1/T2/link-duration/link-fidelity values from real distributed quantum
  hardware experiments, ready to plug into the noise models above.

## Installation

```bash
git clone https://github.com/your-org/qdice-emulator.git
cd qdice-emulator
pip install -e ".[dev,plotting,notebooks]"
```

Or, minimally:

```bash
pip install -e .
```

## Repository layout

```
qdice/
  qpu_backend.py            QPUBackend representation + QPU slicing + QPU stitching
  noise_models.py           Hamiltonian-mediated and entanglement-mediated link noise
  mapping.py                Boundary-aware distributed circuit mapping algorithm
  reference_parameters.py   Published noise parameters from real DQC hardware experiments
  utils.py                  Small shared helpers (summaries, graph drawing)
examples/
  demo.py                   End-to-end walkthrough of the full pipeline
notebooks/
  results_demo.ipynb        Notebook demoing the same *classes* of experiments used
                             in the paper's Results section, using published reference
                             noise parameters as calibration inputs
tests/                      pytest unit tests for every module
build_notebook.py           Script that (re)generates results_demo.ipynb
.github/workflows/tests.yml CI: runs the test suite on Python 3.10-3.12
pyproject.toml              Package metadata and dependencies
```

## Methodology → code map

| Concept | Implementation |
|---|---|
| QPU backend abstraction | `QPUBackend` dataclass (qubit count, coupling map, per-qubit calibration data) |
| QPU slicing (hardware path) | `slice_qpu()` — removes coupling-map edges that cross a user-defined partition boundary, keeping only user-designated nonlocal edges |
| QPU stitching (simulation path) | `stitch_qpus()` — combines independently defined QPU topologies into one backend by adding new edges between designated communication ancillas |
| Hamiltonian-mediated link noise | `hamiltonian_link_kraus()` — amplitude-damping/phase-damping Kraus channels from T1, T2, and link duration; `hardware_delay_duration()` for the hardware-mode idle-delay analogue |
| Entanglement-mediated link noise | `entanglement_link_kraus()` — two-qubit depolarizing channel from a link error probability `p`; `sample_hardware_pauli_twirl()` for the hardware-mode Monte Carlo Pauli-twirl analogue |
| Assembling a noise-aware backend | `build_density_matrix_noise_model()` — attaches the above channels to specific nonlocal gate instances in a Qiskit Aer `NoiseModel` |
| Boundary-aware circuit mapping | `distributed_aware_mapping()` — fixes communication ancillas to physical ports, randomly assigns remaining qubits within their own partition's subgraph, and transpiles against a partition-restricted coupling map, keeping the lowest-depth valid result across N iterations |
| Published reference noise parameters | `get_reference_link_params()` / `list_references()` — look up T1/T2/link-duration/link-error-probability values transcribed from specific published DQC experiments |

## Quick start

```python
from qdice import QPUBackend, slice_qpu, stitch_qpus, distributed_aware_mapping
from qdice.noise_models import HamiltonianLinkParams, build_density_matrix_noise_model
from qdice.reference_parameters import get_reference_link_params

# Describe your own monolithic device or custom topology as a QPUBackend,
# then slice or stitch it into a distributed system.
monolithic = QPUBackend(num_qubits=8, coupling_map=[(i, i + 1) for i in range(7)])
distributed = slice_qpu(
    monolithic,
    partitions={"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]},
    nonlocal_edges=[(3, 4)],
)

# Configure a nonlocal link noise model from a published reference.
link_params = get_reference_link_params("Chou_et_al")
noise_model = build_density_matrix_noise_model(
    hamiltonian_edges=[(edge, link_params) for edge in distributed.nonlocal_edges],
    nonlocal_gate_name="cx",
)

# Map your own circuit onto the distributed backend and simulate it with
# `distributed_aware_mapping()` and Qiskit Aer.
```

## Examples

```bash
python examples/demo.py
```

Runs a minimal, end-to-end walkthrough: slice a backend, stitch a backend,
build both noise-model types, boundary-aware map a toy distributed
circuit, and simulate it with Qiskit Aer.

## Notebooks

`notebooks/results_demo.ipynb` mirrors the structure of the paper's Results
section — gate teleportation, GHZ generation, distributed Grover's search,
distributed Deutsch-Jozsa, and a novel "ring-of-rings" architecture sweep —
using **published noise parameters** from `qdice.reference_parameters`
(sourced from Chou et al., Main et al., Iuliano et al., Pompili et al., Yan
et al., Tsujimoto et al., and IBM Marrakesh device calibration data) as
calibration inputs to this reimplementation's own circuits and noise
models.

**Configuring a noise model with a reference's published parameters is not
the same as reproducing that reference's reported outcome.** This
reimplementation's circuits, protocols, and noise-injection points are a
simplified stand-in for the actual experimental setups, so every
fidelity/success-rate figure the notebook produces should be read as this
codebase's own simulated output under a realistic noise regime — not as a
validation against, or reproduction of, the originally published results.


## Testing

```bash
pytest -v
```

Tests cover QPU slicing/stitching correctness, noise-channel derivations
(trace preservation, known amplitude-damping/dephasing probabilities),
boundary-aware mapping (ancilla placement, partition containment, and
correct failure when no valid routing exists), and reference-parameter
unit conversion and lookup.

## Notes and limitations

- The mapping algorithm is a constrained random search, not a guaranteed
  optimal layout selector — this mirrors the described approach, which
  trades global optimality for a search space that is dramatically reduced
  relative to unconstrained monolithic layout selection.
- Basis-gate handling, delay-instruction insertion into an actual hardware
  pulse schedule, and vendor-specific backend ingestion are left as
  extension points; this codebase focuses on the compiler/noise-modeling
  logic itself rather than vendor SDK integration.

## License
[MIT](LICENSE)
