"""
Example: Optimizing multilayer dielectric slab structures with MultilayerOptimizer

This script demonstrates how to:

1. Optimize the material parameters of a multilayer dielectric slab structure with a given permittivity model. The goal is to minimize R/A (or equivalently to maximize A/R), while satisfying a shielding effectiveness (SE) constraint.  
2. Visualize the optimization results.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from LayerOpt.Permittivity import SimpleLinearPermittivityModel
from LayerOpt.core import MultilayerStructure, MultilayerOptimizer

# ---------------------------------------------------------------------
# 1. Define a simple multilayer structure
# ---------------------------------------------------------------------
# Example: 5 layers, each 5 mm thick, outgoing impedance = 1 (free space)
n_layers = 5
MS = MultilayerStructure(t=5e-3, n_lay=n_layers, impedance_out=1)

# Initial guess for material parameters
x0 = np.linspace(0.2, 0.8, n_layers)

# ---------------------------------------------------------------------
# 2. Define a permittivity model
# ---------------------------------------------------------------------
eps_model = SimpleLinearPermittivityModel()

# ---------------------------------------------------------------------
# 3. Create the optimizer
# ---------------------------------------------------------------------
optimizer = MultilayerOptimizer(
    MS,
    eps_model,
    freq_optimize=np.array([8, 10, 12]),  # Optimize at 8, 10 and 12 GHz,
    SE_min=30,
    lb=0,
    ub=1,
    material_list=np.array([0, 0.125, 0.25, 0.5, 0.75, 1]),
    n_discr=3
)

# ---------------------------------------------------------------------
# 4. Initial guess and cost evaluation
# ---------------------------------------------------------------------
initial_cost = optimizer.eval_cost(x0)
print(f"Initial cost (max R/A): {initial_cost:.4f}")

# ---------------------------------------------------------------------
# 5. Optimize multilayer parameters
# ---------------------------------------------------------------------
solution = optimizer.optimize(x0)

# ---------------------------------------------------------------------
# 6. Visualize the optimization results
# ---------------------------------------------------------------------
# Print the optimization results
print(solution)

# Visualize the optimized multilayer dielectric slab structure
optimizer.visualize_optimization(solution)
plt.show(block=True)

# Visualize the behavior of the optimized multilayer dielectric slab structure
freq_vec = np.linspace(8, 12, 200)  # Example frequency range [GHz]
MS.visualize_behavior(solution, freq_vec, eps_model)
plt.show(block=True)
