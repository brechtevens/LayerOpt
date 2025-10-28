"""
Example: Using the MultilayerStructure class

This script demonstrates how to:
1. Define a multilayer dielectric structure.
2. Use a simple linear permittivity model.
3. Compute and visualize reflectance (R), transmittance (T), absorption (A),
   absorption-to-reflectance ratio (A/R), and shielding effectiveness (SET).
"""

import numpy as np
import matplotlib.pyplot as plt
from LayerOpt.core import MultilayerStructure
from LayerOpt.Permittivity import SimpleLinearPermittivityModel

# ---------------------------------------------------------------------
# 1. Define the multilayer structure
# ---------------------------------------------------------------------
# Five layers, each 5 mm thick, free-space impedance on the output side
structure = MultilayerStructure(t=5e-3, n_lay=5, impedance_out=1)

print(f"Total thickness: {structure.total_t:.3e} m")

# ---------------------------------------------------------------------
# 2. Define the permittivity model
# ---------------------------------------------------------------------
# Simple linear model: ε' = p0 + p1*wt, ε'' = p2 + p3*wt
eps_model = SimpleLinearPermittivityModel(params=[2, 0.05, -0.25, -0.05])

# Material composition per layer (weight %)
x = np.array([0, 2, 4, 6, 8])

# Frequency vector in GHz
freq_vec = np.linspace(1, 20, 200)

# ---------------------------------------------------------------------
# 3. Compute desired electromagnetic properties
# ---------------------------------------------------------------------
# Electromagnetic properties for all frequencies in freq_vec
E = structure.get_waves(x, freq_vec, eps_model)
S11, S22 = structure.get_S_parameters(x, freq_vec, eps_model)
R, T, A = structure.get_wave_properties(x, freq_vec, eps_model)
AR, SET = structure.get_AR_SET(x, freq_vec, eps_model)

# Worst-case (minimum) A/R and shielding effectiveness (SET) over all frequencies in freq_vec
min_AR, min_SET = structure.get_worst_case_AR_SET(x, freq_vec, eps_model)

print(f"Minimum A/R:  {min_AR:.3f}")
print(f"Minimum SET:  {min_SET:.2f} dB")

# ---------------------------------------------------------------------
# 4. Plots included in package
# ---------------------------------------------------------------------

structure.visualize_behavior(x, freq_vec, eps_model)

# ---------------------------------------------------------------------
# 4. Other figures
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.plot(freq_vec, R, marker='.', linestyle="", label="Reflectance (R)")
plt.plot(freq_vec, T, marker='.', linestyle="", label="Transmittance (T)")
plt.plot(freq_vec, A, marker='.', linestyle="", label="Absorption (A)")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Coefficient")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_vec, AR, marker='.', linestyle="")
plt.xlabel("Frequency [GHz]")
plt.ylabel("A/R")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.semilogy(freq_vec, SET, marker='.', linestyle="")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Shielding Effectiveness [dB]")
plt.grid(True)

plt.tight_layout()
plt.show()
