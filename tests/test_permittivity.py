import numpy as np
import pytest
from LayerOpt.Permittivity import (
    PermittivityModel,
    SimpleLinearPermittivityModel,
    LinearPermittivityModel,
    SimpleQuadraticPermittivityModel,
    QuadraticPermittivityModel,
    PiecewisePermittivityModel,
    LinearSplinePermittivityModel,
    SimpleQuadraticSplinePermittivityModel,
    QuadraticSplinePermittivityModel
)

# ------------------------------------------------------
# Base class tests
# ------------------------------------------------------

def test_base_class_instantiation_raises():
    with pytest.raises(NotImplementedError):
        PermittivityModel().get_default_params()

def test_base_class_abstract_methods_raise():
    class Dummy(PermittivityModel):
        def get_default_params(self): return [1]
    d = Dummy([1])
    with pytest.raises(NotImplementedError):
        d._compute_real_permittivity(0, 1, [1], [])
    with pytest.raises(NotImplementedError):
        d._compute_imaginary_permittivity(0, 1, [1], [])

# ------------------------------------------------------
# Generic evaluation tests
# ------------------------------------------------------

@pytest.mark.parametrize("model_cls", [
    SimpleLinearPermittivityModel,
    LinearPermittivityModel,
    SimpleQuadraticPermittivityModel,
    QuadraticPermittivityModel,
    PiecewisePermittivityModel
])
def test_model_evaluate_returns_complex_array(model_cls):
    model = model_cls()
    wt = np.linspace(0, 10, 5)
    f = 5.0
    eps = model.evaluate_permittivity(wt, f)
    assert isinstance(eps, np.ndarray)
    assert eps.dtype == np.complex128
    assert eps.shape == wt.shape
    assert np.all(np.isfinite(eps))

# ------------------------------------------------------
# Linear and quadratic model behavior
# ------------------------------------------------------

def test_simple_linear_model_behavior():
    model = SimpleLinearPermittivityModel()
    wt = np.array([0, 1, 2], dtype=np.float64)
    f = 0
    p = model.get_default_params()
    real_expected = p[0] + p[1]*wt
    imag_expected = p[2] + p[3]*wt
    eps = model.evaluate_permittivity(wt, f)
    np.testing.assert_allclose(eps.real, real_expected, rtol=1e-12)
    np.testing.assert_allclose(eps.imag, imag_expected, rtol=1e-12)

def test_linear_model_frequency_dependence_nonzero_param():
    """Ensure LinearPermittivityModel depends on frequency when p[2] != 0."""
    p = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    model = LinearPermittivityModel(params=p)
    wt = np.array([0.0, 1.0])
    f1, f2 = 1.0, 2.0
    eps1 = model.evaluate_permittivity(wt, f1)
    eps2 = model.evaluate_permittivity(wt, f2)
    assert not np.allclose(eps1, eps2), "Permittivity should vary with frequency when freq term nonzero"

def test_linear_model_reduces_to_simple_when_freq_param_zero():
    """Check that LinearPermittivityModel reduces to SimpleLinearPermittivityModel when freq term is zero."""
    p_linear = np.array([1.0, 2.0, 0.0, -1.0, -2.0, 0.0])
    model_linear = LinearPermittivityModel(params=p_linear)
    model_simple = SimpleLinearPermittivityModel(params=[1.0, 2.0, -1.0, -2.0])
    wt = np.linspace(0, 10, 5)
    f = 5.0
    eps_linear = model_linear.evaluate_permittivity(wt, f)
    eps_simple = model_simple.evaluate_permittivity(wt, f)
    np.testing.assert_allclose(
        eps_linear, eps_simple, rtol=1e-12,
        err_msg="Linear model with zero frequency term should match SimpleLinear model"
    )

def test_quadratic_model_reduces_to_simple_quadratic_when_freq_param_zero():
    """Check that QuadraticPermittivityModel reduces to SimpleQuadraticPermittivityModel when freq term is zero."""
    p_quad = np.array([2, 1, 0, 5, -0.1, -4, 0, -1])
    p_simple = np.array([2, 1, 5, -0.1, -4, -1])
    model_quad = QuadraticPermittivityModel(params=p_quad)
    model_simple = SimpleQuadraticPermittivityModel(params=p_simple)
    wt = np.linspace(0, 10, 5)
    f = 7.5
    eps_quad = model_quad.evaluate_permittivity(wt, f)
    eps_simple = model_simple.evaluate_permittivity(wt, f)
    np.testing.assert_allclose(
        eps_quad, eps_simple, rtol=1e-12,
        err_msg="Quadratic model with zero frequency term should match SimpleQuadratic model"
    )

def test_frequency_dependence_disappears_for_zero_params():
    """If frequency coefficients are zero, ε(f1) == ε(f2)."""
    model = LinearPermittivityModel(params=[1, 2, 0, -1, -2, 0])
    wt = np.array([0.0, 1.0, 2.0])
    f1, f2 = 10.0, 20.0
    eps1 = model.evaluate_permittivity(wt, f1)
    eps2 = model.evaluate_permittivity(wt, f2)
    np.testing.assert_allclose(eps1, eps2, rtol=1e-12)

# ------------------------------------------------------
# Spline model checks
# ------------------------------------------------------

@pytest.mark.parametrize("model_cls", [
    LinearSplinePermittivityModel,
    SimpleQuadraticSplinePermittivityModel,
    QuadraticSplinePermittivityModel
])
def test_spline_models_initialize_and_evaluate(model_cls):
    nodes = np.array([1, 2, 4])
    model = model_cls(nodes)
    wt = np.linspace(0, 5, 5)
    f = 3.0
    eps = model.evaluate_permittivity(wt, f)
    assert isinstance(eps, np.ndarray)
    assert eps.shape == wt.shape
    assert np.all(np.isfinite(eps))
