import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ── additional functions ──────────────────────────────────────────────────────────────

def run_scale(scale_mock):
    with open("scripts/scale.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": scale_mock})


# ── tests ──────────────────────────────────────────────────────────────────────

def test_output_files_exist(scale_mock):
    """All output exists after scaling."""
    mock, _ = scale_mock
    run_scale(mock)

    assert Path(mock.output.scaler).exists(),           f"{mock.output.scaler} is missing"
    assert Path(mock.output.X_train_scaled).exists(),   f"{mock.output.X_train_scaled} is missing"
    assert Path(mock.output.X_test_scaled).exists(),    f"{mock.output.X_test_scaled} is missing"


def test_scaler_mean_matches_train_only(scale_mock):
    """Check for data leakage from scaling."""
    mock, _ = scale_mock
    run_scale(mock)

    X_train = pd.read_csv(mock.input.X_train, index_col=0)
    scaler = joblib.load(mock.output.scaler)

    expected_mean = X_train.mean().values
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6,
            err_msg="scaler.mean_ does not match computed mean of X_train")

    expected_std = X_train.std(ddof=0).values
    np.testing.assert_allclose(scaler.scale_, expected_std, rtol=1e-6,
            err_msg="scaler.scale_ does not match computed std of X_train")

    
def test_scaler_unaffected_by_test_data(scale_mock):
    """Scaler stats should be identical regardless of X_test content."""
    mock, tmp_path = scale_mock
    
    X_test_path = mock.input.X_test
    sabotaged = pd.read_csv(mock.input.X_test, index_col=0) * 1000
    sabotaged.to_csv(X_test_path)

    run_scale(mock)
    scaler = joblib.load(mock.output.scaler)
    
    X_train = pd.read_csv(mock.input.X_train, index_col=0)
    expected_mean = X_train.mean().values
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6,
        err_msg="scaler.mean_ changed even though only X_test was modified — possible leakage")


def test_X_test_scaled_uses_train_statistics(scale_mock):
    """X_test should be transformed using scaler fit on X_train, not its own stats."""
    mock, _ = scale_mock
    run_scale(mock)

    X_train = pd.read_csv(mock.input.X_train, index_col=0)
    X_test = pd.read_csv(mock.input.X_test, index_col=0)
    X_test_scaled = pd.read_csv(mock.output.X_test_scaled, index_col=0)

    expected = (X_test - X_train.mean().values) / X_train.std(ddof=0).values
    np.testing.assert_allclose(
        X_test_scaled.values, expected.values, rtol=1e-6,
        err_msg="X_test_scaled was not computed using X_train statistics"
    )