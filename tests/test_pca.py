import pandas as pd
import numpy as np
import pytest, joblib
from pathlib import Path

# ── additional function ──────────────────────────────────────────────────────────────

def run_pca(pca_mock):
    with open("scripts/pca.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": pca_mock})

def _apply_variance_filter(X):
    variance = X.var()
    high_var_genes = variance[variance > variance.median()].index
    return X[high_var_genes]

# ── tests ──────────────────────────────────────────────────────────────────────

def test_output_files_exist(pca_mock):
    """All output exists after PCA."""
    mock, _ = pca_mock
    run_pca(mock)

    assert Path(mock.output.X_train_pca).exists(),  f"{mock.output.X_train_pca} is missing"
    assert Path(mock.output.X_test_pca).exists(),   f"{mock.output.X_test_pca} is missing"
    assert Path(mock.output.pca).exists(),          f"{mock.output.pca} is missing"
    assert Path(mock.output.plot).exists(),          f"{mock.output.plot} is missing"
    assert Path(mock.output.variance_plot).exists(),          f"{mock.output.variance_plot} is missing"


def test_pca_shape(pca_mock):
    """Check if PCA has right shape."""
    mock, _ = pca_mock
    run_pca(mock)

    X_train_pca = pd.read_csv(mock.output.X_train_pca, index_col=0)
    X_test_pca = pd.read_csv(mock.output.X_test_pca, index_col=0)
    X_train_len = len(pd.read_csv(mock.input.X_train_scaled, index_col=0))
    X_test_len = len(pd.read_csv(mock.input.X_test_scaled, index_col=0))

    assert X_train_pca.shape == (X_train_len, mock.params.n_components), \
        f"X_train_pca shape {X_train_pca.shape} does not match expected ({X_train_len}, {mock.params.n_components})"
    assert X_test_pca.shape == (X_test_len, mock.params.n_components), \
        f"X_test_pca shape {X_test_pca.shape} does not match expected ({X_test_len}, {mock.params.n_components})"


def test_pca_mean_matches_train_only(pca_mock):
    """Check for data leakage from principal component analysis."""
    mock, _ = pca_mock
    run_pca(mock)

    X_train_scaled = pd.read_csv(mock.input.X_train_scaled, index_col=0)
    pca = joblib.load(mock.output.pca)

    X_train_filtered = _apply_variance_filter(X_train_scaled)

    expected_mean = X_train_filtered.mean().values
    np.testing.assert_allclose(pca.mean_, expected_mean, rtol=1e-6,
            err_msg="pca.mean_ does not match computed mean of X_train_scaled")


def test_pca_unaffected_by_test_data(pca_mock):
    """PCA stats should be identical regardless of X_test content."""
    mock, _ = pca_mock
    
    X_test_scaled_path = mock.input.X_test_scaled
    sabotaged = pd.read_csv(mock.input.X_test_scaled, index_col=0) * 1000
    sabotaged.to_csv(X_test_scaled_path)

    run_pca(mock)
    pca = joblib.load(mock.output.pca)
    
    X_train_scaled = pd.read_csv(mock.input.X_train_scaled, index_col=0)

    X_train_filtered = _apply_variance_filter(X_train_scaled)

    from sklearn.decomposition import PCA
    reference_pca = PCA(n_components=mock.params.n_components)
    reference_pca.fit(X_train_filtered)
    
    np.testing.assert_allclose(
        pca.components_, reference_pca.components_, rtol=1e-6,
        err_msg="pca.components_ changed even though only X_test was modified — possible leakage")


def test_pca_explained_variance(pca_mock):
    """Explained variance ratio is valid and sorted descending."""
    mock, _ = pca_mock
    run_pca(mock)

    pca = joblib.load(mock.output.pca)
    ratios = pca.explained_variance_ratio_

    # all values between 0 and 1
    assert ((ratios >= 0) & (ratios <= 1)).all(), \
        "explained_variance_ratio_ contains values outside [0,1]"

    # sum less than 1
    assert ratios.sum() <= 1 + 1e-6, \
        f"sum of explained_variance_ratio_ exceeds 1: {ratios.sum()}"

    # sorted descending
    assert (np.diff(ratios) <= 1e-10).all(), \
        "explained_variance_ratio_ is not sorted in descending order"


def test_X_test_pca_uses_train_statistics(pca_mock):
    """X_test should be transformed using PCA fit on X_train, not its own stats."""
    mock, _ = pca_mock
    run_pca(mock)

    pca = joblib.load(mock.output.pca)
    X_train_scaled = pd.read_csv(mock.input.X_train_scaled, index_col=0)
    X_test_scaled = pd.read_csv(mock.input.X_test_scaled, index_col=0)
    X_test_pca = pd.read_csv(mock.output.X_test_pca, index_col=0)

    X_train_filtered = _apply_variance_filter(X_train_scaled)
    X_test_filtered = X_test_scaled[X_train_filtered.columns]
    
    expected = pca.transform(X_test_filtered)
    np.testing.assert_allclose(
        X_test_pca.values, expected, rtol=1e-6,
        err_msg="X_test_pca does not match pca.transform(X_test_scaled), possible leakage"
    )
