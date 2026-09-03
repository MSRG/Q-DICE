"""
qdice.reference_parameters
============================

Convenience module bundling nonlocal-link parameters extracted from
specific experimental references, converted into the
`HamiltonianLinkParams` / `EntanglementLinkParams` objects used by
`qdice.noise_models`.

These entries let a user configure a Q-DICE-style noise model with
physically grounded parameters instead of re-deriving them by hand.
**Using a reference's parameters to configure a noise model is not the
same as reproducing that reference's reported experimental outcome** (a
measured process fidelity, state fidelity, or algorithm success rate) --
the outcome of any simulation run with these parameters still depends
entirely on the specific circuit, backend topology, and protocol being
emulated, none of which this module dictates. Nothing here stores or
reproduces a reported outcome; it only stores the physical link
parameters (T1, T2, link duration, or link error probability) that were
used to derive those outcomes in the source material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from .noise_models import EntanglementLinkParams, HamiltonianLinkParams

# ----------------------------------------------------------------------
# Raw reference data (SI units: seconds for all time quantities)
# ----------------------------------------------------------------------
REFERENCE_NOISE_PARAMETERS: Dict[str, Dict] = {
    "Chou_et_al": {
        "link_type": "Hamiltonian-Mediated",
        "T1": 50e-6,     # 50 microseconds
        "T2": 30e-6,     # 30 microseconds
        "t_link": 2e-6,  # 2 microseconds
    },
    "Main_et_al": {
        "link_type": "Entanglement-Mediated",
        "p": 0.14,       # derived from an 86% reported process fidelity
        "t_link": None,  # full teleportation-protocol duration; not fixed here
    },
    "Iuliano_et_al": {
        "link_type": "Entanglement-Mediated",
        "p": 0.235,      # derived from a 76.5% reported link fidelity
    },
    "Pompili_et_al": {
        "link_type": "Entanglement-Mediated",
        "p": 0.20,       # per-link depolarizing error probability
    },
    "Yan_et_al": {
        "link_type": "Hamiltonian-Mediated",
        "T1": 80e-6,     # 80 microseconds
        "T2": 50e-6,     # 50 microseconds
        "t_link": 1.5e-6,  # 1.5 microseconds
    },
    "Tsujimoto_et_al": {
        "link_type": "Entanglement-Mediated",
        "p": 0.16,       # derived from an 84% reported swapping fidelity
    },
    "IBM_Marrakesh_Calibration": {
        "link_type": "Hamiltonian-Mediated",
        "T1": 179.71e-6,   # 179.71 microseconds
        "T2": 88.38e-6,    # 88.38 microseconds
        "t_link": 70e-9,   # 70 nanoseconds
    },
}


def get_reference_link_params(
    name: str, t_link_ns: Optional[float] = None
) -> Union[HamiltonianLinkParams, EntanglementLinkParams]:
    """
    Build a `HamiltonianLinkParams` or `EntanglementLinkParams` instance
    from a named entry in `REFERENCE_NOISE_PARAMETERS`.

    Parameters
    ----------
    name:
        Key into `REFERENCE_NOISE_PARAMETERS` (e.g. "Chou_et_al").
    t_link_ns:
        Override for the nonlocal gate/link duration, in nanoseconds. Some
        entries (e.g. "Main_et_al") do not fix a link duration in the
        source material because it depends on the full teleportation
        protocol timing; for those, `t_link_ns` must be supplied
        explicitly if you want a nonzero value. Note that link duration
        does not affect the entanglement-mediated (depolarizing) channel
        itself, only bookkeeping such as hardware-mode delay insertion.

    Returns
    -------
    A `HamiltonianLinkParams` instance for Hamiltonian-mediated entries, or
    an `EntanglementLinkParams` instance for entanglement-mediated entries.
    """
    if name not in REFERENCE_NOISE_PARAMETERS:
        raise KeyError(
            f"Unknown reference '{name}'. Available: "
            f"{sorted(REFERENCE_NOISE_PARAMETERS)}"
        )
    entry = REFERENCE_NOISE_PARAMETERS[name]

    if entry["link_type"] == "Hamiltonian-Mediated":
        t_link_s = entry["t_link"] if t_link_ns is None else t_link_ns * 1e-9
        return HamiltonianLinkParams(
            t1_us=entry["T1"] * 1e6,
            t2_us=entry["T2"] * 1e6,
            t_link_ns=t_link_s * 1e9,
        )

    if entry["link_type"] == "Entanglement-Mediated":
        raw_t_link = entry.get("t_link")
        resolved_t_link_ns = (
            t_link_ns if t_link_ns is not None
            else (raw_t_link * 1e9 if raw_t_link is not None else 0.0)
        )
        return EntanglementLinkParams(
            link_error_probability=entry["p"],
            t_link_ns=resolved_t_link_ns,
        )

    raise ValueError(f"Unrecognized link_type for reference '{name}': {entry['link_type']}")


def list_references(link_type: Optional[str] = None) -> Dict[str, Dict]:
    """
    Return the subset of `REFERENCE_NOISE_PARAMETERS` matching a given
    `link_type` ("Hamiltonian-Mediated" or "Entanglement-Mediated"), or all
    entries if `link_type` is None.
    """
    if link_type is None:
        return dict(REFERENCE_NOISE_PARAMETERS)
    return {
        name: entry
        for name, entry in REFERENCE_NOISE_PARAMETERS.items()
        if entry["link_type"] == link_type
    }
