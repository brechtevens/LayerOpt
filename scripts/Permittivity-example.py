"""
Example: Using and extending PermittivityModel classes

This script demonstrates:
1. How to use the built-in permittivity models.
2. How to create a custom permittivity model.

Note:
-----
In this framework, **the imaginary part of permittivity (ε″) is negative
for a lossy material**. This follows the convention where the time-harmonic
field varies as exp(+iωt). With this definition, a negative ε″ corresponds
to energy dissipation (loss), while a positive ε″ would imply amplification.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from LayerOpt.Permittivity import (
    SimpleLinearPermittivityModel,
    SimpleQuadraticPermittivityModel,
    PermittivityModel
)

# ---------------------------------------------------------------------
# 1. Using a built-in permittivity model
# ---------------------------------------------------------------------
# Simple linear relation: ε' = p0 + p1*x, ε'' = p2 + p3*x
simple_model = SimpleLinearPermittivityModel(params=np.array([2.0, 0.1, -0.3, -0.2]))

# Example input: material composition (wt%) and frequency (GHz)
x = np.linspace(0, 10, 100)  # finer grid for smooth plots
f = 10  # GHz

eps_simple = simple_model.evaluate_permittivity(x, f)

# ---------------------------------------------------------------------
# 2. Using a frequency-dependent quadratic model
# ---------------------------------------------------------------------
quad_model = SimpleQuadraticPermittivityModel(params=np.array([2.0, 0.1, 0.05, -0.3, -0.2, -0.05]))
eps_quad = quad_model.evaluate_permittivity(x, f)

# ---------------------------------------------------------------------
# 3. Defining a custom permittivity model
# ---------------------------------------------------------------------
class SqrtPermittivityModel(PermittivityModel):
    """
        Example of a user-defined permittivity model.

        Convention reminder:
        --------------------
        ε = ε' + iε″, with ε″ < 0 for lossy materials.

        Always use
        
        @staticmethod
        @njit(cache=True)
        
        to ensure compatibility with
        Numba-accelerated calculations in MultilayerStructure.
    """

    def get_default_params(self):
        # Example parameters: ε' = a + b*x + c*f, ε'' = d + e*x + g*f
        return np.array([2.0, 0.5, 0.001, -0.3, -1, -0.001])

    def get_default_bounds(self):
        # Positive bounds for ε' coefficients, negative for ε'' coefficients
        return [*[(1e-6, np.inf)]*3, *[(-np.inf, -1e-6)]*3]

    @staticmethod
    @njit(cache=True)
    def _compute_real_permittivity(x, f, p, p_extra):
        return p[0] + p[1]*np.sqrt(x) + p[2]*f

    @staticmethod
    @njit(cache=True)
    def _compute_imaginary_permittivity(x, f, p, p_extra):
        return p[3] + p[4]*np.sqrt(x) + p[5]*f


# Instantiate and evaluate
custom_model = SqrtPermittivityModel()
eps_custom = custom_model.evaluate_permittivity(x, f)

# ---------------------------------------------------------------------
# 4. Plotting the permittivity
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
# Plot real parts ε'
plt.plot(x, eps_simple.real, label="Linear model")
plt.plot(x, eps_quad.real, label="Quadratic model")
plt.plot(x, eps_custom.real, label="Custom Sqrt model")
plt.ylabel("ε'")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
# Plot imaginary parts ε''
plt.plot(x, eps_simple.imag, label="Linear model")
plt.plot(x, eps_quad.imag, label="Quadratic model")
plt.plot(x, eps_custom.imag, label="Custom Sqrt model")

plt.xlabel("Material composition x (wt%)")
plt.ylabel("ε''")
plt.legend()
plt.grid(True)
plt.show()
