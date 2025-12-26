# test_permittivity_example.py
import subprocess
import os

env = os.environ.copy()
env["NUMBA_DISABLE_JITCACHE"] = "1"
env["MPLBACKEND"] = "Agg"

def test_experiment_script():
    result = subprocess.run(
        ["python", "scripts/Experiment-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayerlearner_script():
    result = subprocess.run(
        ["python", "scripts/MultilayerLearner-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayeroptimizer_script():
    result = subprocess.run(
        ["python", "scripts/MultilayerOptimizer-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_multilayerstructure_script():
    result = subprocess.run(
        ["python", "scripts/MultilayerStructure-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"

def test_permittivity_script():
    result = subprocess.run(
        ["python", "scripts/Permittivity-example.py"],  # adjust path if needed
        capture_output=True,
        env=env
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr.decode()}"
