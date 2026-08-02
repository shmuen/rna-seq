import pandas as pd
import json, pytest
from pathlib import Path
from conftest import TEST_GRIDS

# ── additional function ──────────────────────────────────────────────────────────────

def run_nested_cv(nested_cv_mock):
    with open("scripts/nested_cv.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": nested_cv_mock})

# ── tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(TEST_GRIDS.keys()))
def test_output_files_exist(nested_cv_mock_factory, model_name):
    """nested_cv.csv exists after nested cross validation."""
    mock = nested_cv_mock_factory(model_name, param_grid=TEST_GRIDS[model_name])
    run_nested_cv(mock)

    assert Path(mock.output.nested_cv_score).exists(), f"{mock.output.nested_cv_score} is missing"


@pytest.mark.parametrize("model_name,param_grid", list(TEST_GRIDS.items()))
def test_nested_cv_score_is_valid(nested_cv_mock_factory, model_name, param_grid):
    """Nested CV score (mean/std) is a valid, plausible f1_macro value."""
    mock = nested_cv_mock_factory(model_name, param_grid=param_grid)
    run_nested_cv(mock)

    result = pd.read_csv(mock.output.nested_cv_score)

    assert list(result.columns) == ["mean", "std"], "unexpected columns in nested_cv_score"
    assert result["mean"].between(0, 1).all(), f"mean f1_macro out of [0,1] for {model_name}"
    assert (result["std"] >= 0).all(), f"std should be non-negative for {model_name}"


def test_nested_crossvalidation_reproducible(nested_cv_mock_factory):
    """Same seed produces same nested crossvalidation."""
    param_grid = TEST_GRIDS["randomforest"]

    mock1 = nested_cv_mock_factory("randomforest", param_grid=param_grid)
    run_nested_cv(mock1)
    result1 = pd.read_csv(mock1.output.nested_cv_score)

    mock2 = nested_cv_mock_factory("randomforest", param_grid=param_grid)
    run_nested_cv(mock2)
    result2 = pd.read_csv(mock2.output.nested_cv_score)

    pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-6)