# test_permittivity_example.py
import subprocess
import os
import sys

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

def test_experiment_script():
    result = subprocess.run(
        [sys.executable, "scripts/Experiment-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayerlearner_script():
    result = subprocess.run(
        [sys.executable, "scripts/MultilayerLearner-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayeroptimizer_script():
    result = subprocess.run(
        [sys.executable, "scripts/MultilayerOptimizer-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayerstructure_script():
    result = subprocess.run(
        [sys.executable, "scripts/MultilayerStructure-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_permittivity_script():
    result = subprocess.run(
        [sys.executable, "scripts/Permittivity-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"
