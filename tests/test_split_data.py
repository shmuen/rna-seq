import pandas as pd
import pytest
from pathlib import Path


# ── additional function ──────────────────────────────────────────────────────────────

def run_split(split_mock):
    with open("scripts/split_data.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": split_mock})


# ── tests ──────────────────────────────────────────────────────────────────────

def test_output_files_exist(split_mock):
    """All output exists after split."""
    mock, _ = split_mock
    run_split(mock)

    assert Path(mock.output.X_train).exists(),      f"{mock.output.X_train} is missing"
    assert Path(mock.output.X_test).exists(),      f"{mock.output.X_test} is missing"
    assert Path(mock.output.y_train).exists(),      f"{mock.output.y_train} is missing"
    assert Path(mock.output.y_test).exists(),      f"{mock.output.y_test} is missing"


def test_sizes(split_mock):
    """Train/test sizes match defined test_size."""
    mock, _ = split_mock
    run_split(mock)

    size = mock.params.test_size
    l_X_train = len(pd.read_csv(mock.output.X_train, index_col = 0))
    l_X_test = len(pd.read_csv(mock.output.X_test, index_col = 0))
    l_y_train = len(pd.read_csv(mock.output.y_train, index_col = 0))
    l_y_test = len(pd.read_csv(mock.output.y_test, index_col = 0))
    
    total = l_X_train + l_X_test

    assert l_X_test == pytest.approx(total * size, abs=1), "X_test does not have defined size"
    assert l_X_train == l_y_train, "sizes of X_train and y_train do not match"
    assert l_X_test == l_y_test, "sizes of X_test and y_test do not match"


def test_no_leakage(split_mock):
    """Test for data leakage."""
    mock, _ = split_mock
    run_split(mock)
    
    X_train = pd.read_csv(mock.output.X_train, index_col = 0)
    X_test = pd.read_csv(mock.output.X_test, index_col = 0)
                         
    assert len(set(X_train.index) & set(X_test.index)) == 0, \
        "overlapping samples found between X_train and X_test"


def test_no_data_loss(split_mock):
    """All samples are preserved across the split (none lost)."""
    mock, _ = split_mock
    run_split(mock)

    l_X_train = len(pd.read_csv(mock.output.X_train, index_col = 0))
    l_X_test = len(pd.read_csv(mock.output.X_test, index_col = 0))
   
    total = len(pd.read_csv(mock.input.data, index_col=0))

    assert l_X_train + l_X_test == total, "data loss"


def test_reproducibility(split_mock):
    """Split is reproducible given the same seed."""
    mock, _ = split_mock
    run_split(mock)
    X_train1 = pd.read_csv(mock.output.X_train, index_col = 0)

    run_split(mock)
    X_train2 = pd.read_csv(mock.output.X_train, index_col = 0)
 
    assert X_train1.equals(X_train2), "split is not reproducible"


def test_stratification(split_mock):
    """Class distribution is preserved between train and test sets."""
    mock, _ = split_mock
    run_split(mock)
    
    y_train = pd.read_csv(mock.output.y_train, index_col=0)
    y_test = pd.read_csv(mock.output.y_test, index_col=0)
    
    train_dist = y_train["Class"].value_counts(normalize=True)
    test_dist = y_test["Class"].value_counts(normalize=True)
    
    for cls in train_dist.index:
        assert train_dist[cls] == pytest.approx(test_dist[cls], abs=0.1), "classes are not stratified"