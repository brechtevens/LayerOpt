"""
Example: Using ExperimentData class

This script demonstrates how to:

1. Load experimental multilayer measurement data.
2. Access S-parameters (S11, S21), transmittance (T), reflectance (R), and absorptance (A).
3. Plot A/R versus shielding effectiveness (SET) for different experiments.

Note:
-----
This example assumes the folder `../data/` contains:
- `overview_weights.csv` defining material parameters and optional metadata
- CSV files with S-parameter measurements for each experiment
"""

import os
import matplotlib.pyplot as plt
from LayerOpt.Experiment import ExperimentData 

# ---------------------------------------------------------------------
# 1. Load experiment data
# ---------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), '../data/')
exp_data = ExperimentData(directory=data_dir, reverse=False)

print("Loaded experiments:", list(exp_data.get_names()))

# ---------------------------------------------------------------------
# 2. Accessing coefficients for one experiment
# ---------------------------------------------------------------------
experiment1 = list(exp_data.get_names())[0]  # pick the first experiment

S11 = exp_data.get_S11(experiment1)
S21 = exp_data.get_S21(experiment1)
T = exp_data.get_T(experiment1)
R = exp_data.get_R(experiment1)
A = exp_data.get_A(experiment1)

# ---------------------------------------------------------------------
# 3. Compute worst-case A/R and SET for one experiment
# ---------------------------------------------------------------------
min_AR, min_SET = exp_data.get_worst_case_AR_SET(experiment1)
print(f"\nWorst-case values for {experiment1}:")
print("Minimum A/R:", min_AR)
print("Minimum SET [dB]:", min_SET)

# ---------------------------------------------------------------------
# 4. Plot A/R versus SET for multiple experiments
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
for experiment_name in exp_data.get_names():
    exp_data.plot_behavior(experiment_name)
plt.show()

# ---------------------------------------------------------------------
# 5. Find which experiments are unique and which are duplicate
# ---------------------------------------------------------------------
duplicates = exp_data.get_duplicate_experiments()
unique_experiments = exp_data.get_unique_experiments()

if duplicates:
    print("\nDuplicate experiments detected:")
    for params, names in duplicates.items():
        print(f"Material parameters {params}: {names}")
else:
    print("\nNo duplicate experiments found.")

print("\nUnique experiments by material parameters:")
for params, names in unique_experiments.items():
    print(f"Material parameters {params}: {names}")

