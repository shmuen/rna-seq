import pandas as pd
from pathlib import Path

# ── additional function ──────────────────────────────────────────────────────────────

def run_compare(compare_mock):
    with open("scripts/compare_models.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": compare_mock})

# ── tests ──────────────────────────────────────────────────────────────

def test_output_files_exist(compare_mock):
    """All output files have to exist after script execution."""
    mock, _ = compare_mock
    run_compare(mock)

    assert Path(mock.output.comparison ).exists(),      f"{mock.output.comparison} is missing"
    assert Path(mock.output.plot).exists(),  f"{mock.output.plot} is missing"
    assert Path(mock.output.heat).exists(),  f"{mock.output.heat} is missing"


def test_comparison_columns_and_rows_exist(compare_mock):
    """Check if all columns and rows of comparison file exist."""
    mock, _ = compare_mock
    run_compare(mock)

    comparison = pd.read_csv(mock.output.comparison, index_col=0)
    columns = ["Accuracy","Kappa","MCC","F1","train time","AUC", "N. CV F1"]

    expected_models = {"model1", "model2", "model3"}
    assert set(comparison.index) == expected_models, f"expected models {expected_models}, got {set(comparison.index)}"

    assert comparison.shape == (3,7), "comparison file does not have right shape"
    
    columns_comparison = list(comparison)
    assert columns_comparison == columns, "column names of summary file do not match"

    for col in columns_comparison:
        assert pd.api.types.is_numeric_dtype(comparison[col]), f"{col} is not numeric"


def test_comparison_values_range(compare_mock):
    """Comparison values are within defined range."""
    mock, _ = compare_mock
    run_compare(mock)
    
    comparison = pd.read_csv(mock.output.comparison, index_col=0)

    assert comparison[["Accuracy","F1","AUC", "N. CV F1"]].apply(
        lambda col: col.between(0,1)
    ).all().all(), "values in result table are not as expected (between 0 and 1)"
    
    assert comparison[["Kappa","MCC"]].apply(
        lambda col: col.between(-1,1) 
    ).all().all(), "kappa or mcc are not between -1 and 1"

    assert (comparison["train time"] > 0).all(), "train time is not positive"


def test_models_accuracy_is_decreasing(compare_mock):
    """Check if the accuracy of the models decreases for test models."""
    mock, _ = compare_mock
    run_compare(mock)

    comparison = pd.read_csv(mock.output.comparison, index_col=0)
    columns = ["Accuracy","Kappa","MCC","F1","AUC", "N. CV F1"]

    for metric in columns:
        assert comparison[metric].iloc[0] >= comparison[metric].iloc[1], f"{metric} is not decreasing for models"
        assert comparison[metric].iloc[1] >= comparison[metric].iloc[2], f"{metric} is not decreasing for models"

