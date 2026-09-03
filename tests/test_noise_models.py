"""Tests for qdice.noise_models: Hamiltonian- and entanglement-mediated link noise."""

import numpy as np
import pytest

from qdice.noise_models import (
    HamiltonianLinkParams,
    EntanglementLinkParams,
    hamiltonian_link_kraus,
    entanglement_link_kraus,
    build_density_matrix_noise_model,
    hardware_delay_duration,
    sample_hardware_pauli_twirl,
)


def test_hamiltonian_link_kraus_reduces_to_identity_at_zero_link_time():
    params = HamiltonianLinkParams(t1_us=50.0, t2_us=30.0, t_link_ns=0.0)
    amp_damp, phase_damp = hamiltonian_link_kraus(params)
    # With t_link = 0, gamma = lambda = 0, so both channels should act as
    # the identity on any input state.
    identity = np.eye(2)
    np.testing.assert_allclose(amp_damp.data[0], identity, atol=1e-10)
    np.testing.assert_allclose(phase_damp.data[0], identity, atol=1e-10)


def test_hamiltonian_link_kraus_matches_known_derivation():
    # Values reported for the Chou et al. Hamiltonian-mediated reference:
    # T1 = 50us, T2 = 30us, t_link = 2us -> gamma ~= 0.039, lambda ~= 0.065.
    params = HamiltonianLinkParams(t1_us=50.0, t2_us=30.0, t_link_ns=2000.0)
    amp_damp, phase_damp = hamiltonian_link_kraus(params)

    gamma = 1 - np.exp(-2.0 / 50.0)
    lam = 1 - np.exp(-2.0 / 30.0)
    assert gamma == pytest.approx(0.039, abs=1e-3)
    assert lam == pytest.approx(0.065, abs=1e-3)

    # E1 (the off-diagonal, "decay" Kraus operator) should have
    # |E1[0,1]| = sqrt(gamma). It is identified as the operator with a
    # nonzero off-diagonal entry (E0 is diagonal for amplitude damping).
    e1_amp = [op for op in amp_damp.data if abs(op[0, 1]) > 1e-9][0]
    assert abs(e1_amp[0, 1]) == pytest.approx(np.sqrt(gamma), abs=1e-6)


def test_entanglement_link_kraus_is_trace_preserving():
    params = EntanglementLinkParams(link_error_probability=0.14, t_link_ns=0.0)
    channel = entanglement_link_kraus(params)
    total = sum(op.conj().T @ op for op in channel.data)
    np.testing.assert_allclose(total, np.eye(4), atol=1e-8)


def test_build_density_matrix_noise_model_attaches_expected_edges():
    h_params = HamiltonianLinkParams(t1_us=60.0, t2_us=40.0, t_link_ns=100.0)
    e_params = EntanglementLinkParams(link_error_probability=0.1, t_link_ns=0.0)
    noise_model = build_density_matrix_noise_model(
        hamiltonian_edges=[((0, 1), h_params)],
        entanglement_edges=[((2, 3), e_params)],
        nonlocal_gate_name="cx",
    )
    noisy_instructions = noise_model.noise_instructions
    assert "cx" in noisy_instructions


def test_hardware_delay_duration_is_nonnegative_and_correct():
    assert hardware_delay_duration(t_link_ns=500, t_local_gate_ns=200) == 300
    # Should never go negative even if the local gate already exceeds the
    # requested link time.
    assert hardware_delay_duration(t_link_ns=100, t_local_gate_ns=500) == 0


def test_sample_hardware_pauli_twirl_respects_probability_bounds():
    import random

    rng = random.Random(0)
    # p = 0 should never inject an error.
    assert all(sample_hardware_pauli_twirl(0.0, rng) is None for _ in range(50))

    rng = random.Random(0)
    # p = 1 should always inject a Pauli from {X, Y, Z}.
    outcomes = {sample_hardware_pauli_twirl(1.0, rng) for _ in range(50)}
    assert outcomes.issubset({"X", "Y", "Z"})
    assert None not in outcomes
