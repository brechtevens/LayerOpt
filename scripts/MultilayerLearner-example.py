"""
Example: Learning permittivity model parameters with MultilayerLearner

This script demonstrates how to:

1. Learn permittivity model parameters from experimental data using MultilayerLearner with a user-defined cost function.
2. Visualize and compare predicted versus measured transmittance (T), reflectance (R), and absorption (A).
"""

import os
import time
import matplotlib.pyplot as plt

from LayerOpt.Experiment import ExperimentData
from LayerOpt.Permittivity import SimpleQuadraticPermittivityModel
from LayerOpt.core import MultilayerLearner

# ---------------------------------------------------------------------
# 1. Load experimental data
# ---------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), '../data/')
exp_data = ExperimentData(directory=data_dir, reverse=False)
experiment_names = list(exp_data.get_names())
print("Loaded experiments:", experiment_names)

# ---------------------------------------------------------------------
# 2. Initialize a simple quadratic permittivity model
# ---------------------------------------------------------------------
eps_model = SimpleQuadraticPermittivityModel()
print("Default model parameters:", eps_model.get_default_params())

# ---------------------------------------------------------------------
# 3. Create the learner
# ---------------------------------------------------------------------
learner = MultilayerLearner(measurements=exp_data, eps_model=eps_model)

# ---------------------------------------------------------------------
# 4. Define a cost function and evaluate the initial cost
# ---------------------------------------------------------------------
def custom_cost(p):
    errors = 0
    for experiment_name in exp_data.get_names():
        wt = exp_data.get_material_parameters(experiment_name)
        freq_range = exp_data.get_X(experiment_name)
        eps_model.p = p
        R_pred, T_pred, A_pred = exp_data.structure[experiment_name].get_wave_properties(wt, freq_range, eps_model)
        R_meas = exp_data.get_R(experiment_name)
        # sum of squared errors only for T
        errors += ((R_pred - R_meas)**2).sum()/exp_data.n
    return errors

cost_fun = custom_cost
p0 = eps_model.get_default_params()
initial_cost = cost_fun(p0)
print("Initial cost:", initial_cost)

# ---------------------------------------------------------------------
# 5. Optimize permittivity parameters
# ---------------------------------------------------------------------
optimal_params, optimal_cost = learner.optimize(p0, cost_fun)
print("Optimized parameters:", optimal_params)
print("Optimized cost:", optimal_cost)

# ---------------------------------------------------------------------
# 6. Compare measured vs predicted coefficients
# ---------------------------------------------------------------------
num_experiments = len(exp_data.get_names())
plt.figure(figsize=(6, 4 * num_experiments))

for j, experiment_name in enumerate(exp_data.get_names()):
    plt.subplot(num_experiments, 1, j+1)
    wt = exp_data.get_material_parameters(experiment_name)
    freq_range = exp_data.get_X(experiment_name)

    # Predicted coefficients
    R_pred, T_pred, A_pred = exp_data.structure[experiment_name].get_wave_properties(
        wt, freq_range, eps_model
    )

    # Measured coefficients
    T_meas = exp_data.get_T(experiment_name)
    R_meas = exp_data.get_R(experiment_name)
    A_meas = exp_data.get_A(experiment_name)

    # Plot T
    plt.plot(freq_range, T_meas, 'o', color='tab:blue', label='T measured')
    plt.plot(freq_range, T_pred, '-', color='tab:blue', label='T predicted')

    # Plot R
    plt.plot(freq_range, R_meas, 'o', color='tab:orange', label='R measured')
    plt.plot(freq_range, R_pred, '-', color='tab:orange', label='R predicted')

    # # Plot A
    plt.plot(freq_range, A_meas, 'o', color='tab:green', label='A measured')
    plt.plot(freq_range, A_pred, '-', color='tab:green', label='A predicted')

    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Coefficient")
    plt.title(f"{experiment_name}")
    plt.grid(True)
    plt.legend()

plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.subplots_adjust(wspace=20)
plt.show()



