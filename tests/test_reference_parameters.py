"""Tests for qdice.reference_parameters: unit conversion and lookup correctness."""

import pytest

from qdice.noise_models import EntanglementLinkParams, HamiltonianLinkParams
from qdice.reference_parameters import (
    REFERENCE_NOISE_PARAMETERS,
    get_reference_link_params,
    list_references,
)


def test_hamiltonian_reference_converts_seconds_to_us_and_ns_correctly():
    params = get_reference_link_params("Chou_et_al")
    assert isinstance(params, HamiltonianLinkParams)
    # 50e-6 s -> 50.0 us, 30e-6 s -> 30.0 us, 2e-6 s -> 2000.0 ns
    assert params.t1_us == pytest.approx(50.0)
    assert params.t2_us == pytest.approx(30.0)
    assert params.t_link_ns == pytest.approx(2000.0)


def test_entanglement_reference_carries_error_probability():
    params = get_reference_link_params("Pompili_et_al")
    assert isinstance(params, EntanglementLinkParams)
    assert params.link_error_probability == pytest.approx(0.20)


def test_entanglement_reference_with_unspecified_link_time_defaults_to_zero():
    # Main_et_al does not fix a numeric t_link in the source material.
    params = get_reference_link_params("Main_et_al")
    assert params.link_error_probability == pytest.approx(0.14)
    assert params.t_link_ns == pytest.approx(0.0)


def test_t_link_override_is_respected():
    params = get_reference_link_params("Main_et_al", t_link_ns=900.0)
    assert params.t_link_ns == pytest.approx(900.0)


def test_unknown_reference_raises_key_error():
    with pytest.raises(KeyError):
        get_reference_link_params("Not_A_Real_Reference")


def test_list_references_filters_by_link_type():
    hamiltonian_refs = list_references("Hamiltonian-Mediated")
    entanglement_refs = list_references("Entanglement-Mediated")

    assert set(hamiltonian_refs) == {"Chou_et_al", "Yan_et_al", "IBM_Marrakesh_Calibration"}
    assert set(entanglement_refs) == {
        "Main_et_al", "Iuliano_et_al", "Pompili_et_al", "Tsujimoto_et_al",
    }
    assert set(hamiltonian_refs) | set(entanglement_refs) == set(REFERENCE_NOISE_PARAMETERS)


def test_list_references_without_filter_returns_everything():
    assert set(list_references()) == set(REFERENCE_NOISE_PARAMETERS)
