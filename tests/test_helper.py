import numpy as np
import pytest
from numba import njit

from LayerOpt.core import MultilayerStructure

# -------------------------------------------------------------------
# Mock permittivity model for predictable, lossless behavior
# -------------------------------------------------------------------

class MockPermittivityModel:
    def __init__(self):
        self.p = None
        self.p_extra = None

    @staticmethod
    @njit(cache=True)
    def _compute_real_permittivity(x, f, p, p_extra):
        # Constant real part -> nondispersive dielectric
        return 1.0

    @staticmethod
    @njit(cache=True)
    def _compute_imaginary_permittivity(x, f, p, p_extra):
        # No losses
        return 0.0


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def eps_model():
    return MockPermittivityModel()

@pytest.fixture
def basic_structure():
    # Air on both sides, 3 layers
    return MultilayerStructure(t=1e-3, n_lay=3, impedance_out=1)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_impedance_formula():
    """Ensure get_k0 matches theoretical formula."""
    f = np.array([1, 10, 100])
    result = MultilayerStructure.get_k0(f)
    expected = 2 * np.pi * f * 1e9 / 299792458
    assert np.allclose(result, expected)


def test_free_space_transmission(basic_structure, eps_model):
    """If all layers are vacuum, R=0, T=1, A=0."""
    freq_vec = np.linspace(1, 10, 5)
    x = np.zeros(basic_structure.n_lay)

    R, T, A = basic_structure.get_wave_properties(x, freq_vec, eps_model)

    assert np.allclose(R, 0, atol=1e-6)
    assert np.allclose(T, 1, atol=1e-6)
    assert np.allclose(A, 0, atol=1e-6)


def test_perfect_reflector(eps_model):
    """If outgoing impedance = 0 (PEC), reflection is total."""
    struct = MultilayerStructure(t=1e-3, n_lay=1, impedance_out=0)
    freq_vec = np.array([1, 5, 10])
    x = np.ones(struct.n_lay)

    R, T, A = struct.get_wave_properties(x, freq_vec, eps_model)

    assert np.allclose(R, 1, atol=1e-6)
    assert np.allclose(T, 0, atol=1e-6)
    assert np.allclose(A, 0, atol=1e-6)


def test_energy_conservation_lossless(basic_structure, eps_model):
    """For lossless layers, R + T = 1, A = 0."""
    freq_vec = np.linspace(1, 10, 10)
    x = np.linspace(0, 10, basic_structure.n_lay)

    R, T, A = basic_structure.get_wave_properties(x, freq_vec, eps_model)

    assert np.allclose(R + T, 1, atol=1e-6)
    assert np.allclose(A, 0, atol=1e-6)


def test_serial_parallel_equivalence(eps_model):
    """Serial and parallel njitted versions must yield identical results."""
    struct = MultilayerStructure(t=1e-3, n_lay=2, impedance_out=1)
    freq_vec = np.linspace(1, 5, 50)
    x = np.ones(struct.n_lay) * 5

    serial = struct._get_waves_serial(
        x, freq_vec, struct.n_lay, struct.t, struct.get_k0,
        None, None, struct.impedance_out,
        eps_model._compute_real_permittivity, eps_model._compute_imaginary_permittivity
    )

    parallel = struct._get_waves_parallel(
        x, freq_vec, struct.n_lay, struct.t, struct.get_k0,
        None, None, struct.impedance_out,
        eps_model._compute_real_permittivity, eps_model._compute_imaginary_permittivity
    )

    assert np.allclose(serial, parallel, atol=1e-12)


def test_single_frequency_shape(basic_structure, eps_model):
    """Ensure single-frequency input still returns correct array shapes."""
    freq_vec = np.array([5.0])
    x = np.zeros(basic_structure.n_lay)
    R, T, A = basic_structure.get_wave_properties(x, freq_vec, eps_model)

    assert R.shape == (1,)
    assert T.shape == (1,)
    assert A.shape == (1,)


def test_zero_layers_handling(eps_model):
    """Zero-layer case should either return trivial results or raise ValueError."""
    struct = MultilayerStructure(t=1e-3, n_lay=0)
    freq_vec = np.array([5])
    x = np.array([])

    try:
        R, T, A = struct.get_wave_properties(x, freq_vec, eps_model)
        # If it runs, check it yields the correct trivial case
        assert np.allclose(R, 0)
        assert np.allclose(T, 1)
        assert np.allclose(A, 0)
    except ValueError:
        # Acceptable behavior: explicit error
        pass