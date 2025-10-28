import os
import shutil
import numpy as np
import pandas as pd
import pytest
from LayerOpt.core import ExperimentData  # adjust import if core.py moves

# Path to the test data folder
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

@pytest.fixture
def experiment_data():
    """Fixture to load experiment data with reverse enabled."""
    return ExperimentData(directory=TEST_DATA_DIR, reverse=False)

def test_load_weights(experiment_data):
    """Test that weights are loaded correctly from CSV."""
    wt = experiment_data.wt
    assert isinstance(wt, dict)
    # Check that at least one experiment is loaded
    assert len(wt) > 0
    # Check that the weights are numpy arrays
    for w in wt.values():
        assert isinstance(w, np.ndarray)
        assert w.ndim == 1

def test_load_experiments(experiment_data):
    """Test that X and Y dictionaries are populated."""
    X = experiment_data.X
    Y = experiment_data.Y
    wt = experiment_data.wt
    assert isinstance(X, dict) and isinstance(Y, dict)
    # There should be entries corresponding to weights
    assert set(wt.keys()).issubset(X.keys())
    assert set(wt.keys()).issubset(Y.keys())
    # Check shapes
    for name in X:
        assert isinstance(X[name], np.ndarray)
        assert isinstance(Y[name], np.ndarray)
        assert Y[name].shape[0] in [2, 4]  # S11/S21 or all 4 if reverse included
        assert Y[name].shape[1] == len(X[name])

def test_reverse_data(experiment_data):
    """Test that reverse experiment data is created and weights reversed."""
    wt = experiment_data.wt
    for name in list(wt.keys()):
        if name.endswith("_reverse"):
            original_name = name.replace("_reverse", "")
            np.testing.assert_allclose(wt[name], np.flip(wt[original_name]))

def test_S_T_R_A_methods(experiment_data):
    """Test S11, S21, T, R, A methods for reasonable output shapes."""
    for name in experiment_data.get_names():
        S11 = experiment_data.get_S11(name)
        S21 = experiment_data.get_S21(name)
        T = experiment_data.get_T(name)
        R = experiment_data.get_R(name)
        A = experiment_data.get_A(name)

        # All should have the same length as X
        n = len(experiment_data.get_X(name))
        assert S11.shape[0] == n
        assert S21.shape[0] == n
        assert T.shape[0] == n
        assert R.shape[0] == n
        assert A.shape[0] == n
        # T + R + A should be ~1 (numerical tolerance)
        np.testing.assert_allclose(T + R + A, np.ones(n), rtol=1e-10)

def test_get_duplicate_experiments(experiment_data):
    """Test that duplicate detection returns correct dictionary."""
    duplicates = experiment_data.get_duplicate_experiments()
    assert isinstance(duplicates, dict)
    # If duplicates exist, each key is a tuple of weights, values are lists of names
    for weights, names in duplicates.items():
        assert isinstance(weights, tuple)
        assert isinstance(names, list)
        assert len(names) > 1

def test_get_AR_SET_and_worst_case(experiment_data):
    """Test AR/SET calculations."""
    for name in experiment_data.get_names():
        AR, SET = experiment_data.get_AR_SET(name)
        min_AR, min_SET = experiment_data.get_worst_case_AR_SET(name)
        assert np.allclose(AR, np.array(AR))
        assert np.allclose(SET, np.array(SET))
        assert min_AR == np.min(AR)
        assert min_SET == np.min(SET)

def test_optional_fields_loaded(experiment_data):
    """Test that layer_thickness and impedance_out are loaded correctly."""
    # Make sure the dictionaries exist
    for attr in ['layer_thickness', 'impedance_out']:
        attr_dict = getattr(experiment_data, attr)
        assert isinstance(attr_dict, dict)
        # At least one experiment should be loaded
        assert len(attr_dict) > 0

    # Example checks based on the sample overview.csv file
    # 000wt_1
    assert experiment_data.layer_thickness['000wt_1'] == 0.005
    assert experiment_data.impedance_out['000wt_1'] == 1  # default if missing

    # 050wt_1
    assert experiment_data.layer_thickness['050wt_1'] == 0.004
    assert experiment_data.impedance_out['050wt_1'] == 0.5

    # 100wt_1
    assert experiment_data.layer_thickness['100wt_1'] == 0.005  # default if missing
    assert experiment_data.impedance_out['100wt_1'] == 0

BROKEN_DIR = os.path.join(os.path.dirname(__file__), 'broken_data')
MISSING_DIR = os.path.join(os.path.dirname(__file__), 'missing_data')

# --- Test for NotImplementedError when reverse=True with impedance_out != 1 ---
def test_reverse_with_invalid_impedance():
    # Use the normal TEST_DATA_DIR but set reverse=True
    # 000wt_1 has impedance_out = 0.5 -> should trigger
    with pytest.raises(NotImplementedError):
        ExperimentData(directory=TEST_DATA_DIR, reverse=True)

# --- Test for unsupported CSV columns ---
def test_unsupported_csv_columns():
    os.makedirs(BROKEN_DIR, exist_ok=True)

    # Create overview_weights.csv for completeness
    df = pd.DataFrame({
        'experiment_name': ['bad_csv'],
        'parameters': ['[1,2,3]'],
        'layer_thickness': [0.001],
        'impedance_out': [1]
    })
    df.to_csv(os.path.join(BROKEN_DIR, 'overview_weights.csv'), index=False)

    # Create bad CSV with unsupported columns
    bad_csv_path = os.path.join(BROKEN_DIR, 'bad_csv.csv')
    with open(bad_csv_path, 'w') as f:
        f.write("# Version 1.00\n")
        f.write("#\n")
        f.write("freq[Hz]\tbroken\tang:Trc1_S11\tdb:Trc2_S12\tang:Trc2_S12\tdb:Trc3_S21\tang:Trc3_S21\tdb:Trc4_S22\tang:Trc4_S22\n")
        f.write("1000000\t0\t0\t0\t0\t0\t0\t0\t0\n")
        f.write("2000000\t0\t0\t0\t0\t0\t0\t0\t0\n")

    # --- Test that loading this CSV raises NotImplementedError ---
    with pytest.raises(NotImplementedError):
        ed = ExperimentData(directory=BROKEN_DIR)
        ed.load_experiment('bad_csv.csv')

    # Cleanup
    shutil.rmtree(BROKEN_DIR)

# --- Test for FileNotFoundError when overview_weights.csv is missing ---
def test_missing_overview_csv():
    os.makedirs(MISSING_DIR, exist_ok=True)
    try:
        with pytest.raises(FileNotFoundError):
            ExperimentData(directory=MISSING_DIR)
    finally:
        shutil.rmtree(MISSING_DIR)