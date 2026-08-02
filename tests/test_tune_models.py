import json, pytest
from pathlib import Path
from conftest import TEST_GRIDS

# ── additional function ──────────────────────────────────────────────────────────────

def run_tune(tune_mock):
    with open("scripts/tune_models.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": tune_mock})

# ── tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(TEST_GRIDS.keys()))
def test_output_files_exist(tune_mock_factory, model_name):
    """best_params.json exists after tuning."""
    mock = tune_mock_factory(model_name, param_grid=TEST_GRIDS[model_name])
    run_tune(mock)

    assert Path(mock.output.best_params).exists(), f"{mock.output.best_params} is missing"


def test_tune_uses_default_when_disabled(tune_mock_factory):
    """When tuning is disabled, default_grid is used directly as best_params."""
    default_grid = {"n_estimators": 50}
    mock = tune_mock_factory("randomforest", default_grid=default_grid, tune=False)

    run_tune(mock)

    with open(mock.output.best_params) as f:
        best_params = json.load(f)

    assert best_params == default_grid, \
        f"expected default_grid {default_grid} to be used, got {best_params}"


@pytest.mark.parametrize("model_name,param_grid", list(TEST_GRIDS.items()))
def test_tune_produces_valid_params(tune_mock_factory, model_name, param_grid):
    """Tuning produces valid best_params for each model type."""
    mock = tune_mock_factory(model_name, param_grid=param_grid)
    run_tune(mock)

    with open(mock.output.best_params) as f:
        best_params = json.load(f)

    for key, allowed_values in param_grid.items():
        assert key in best_params, f"missing param {key} for {model_name}"
        assert best_params[key] in allowed_values, \
            f"{key}={best_params[key]} not in allowed grid {allowed_values} for {model_name}"

def test_tune_reproducible(tune_mock_factory):
    """Same seed produces same best_params."""
    param_grid = TEST_GRIDS["randomforest"]

    mock1 = tune_mock_factory("randomforest", param_grid=param_grid)
    run_tune(mock1)
    with open(mock1.output.best_params) as f:
        result1 = json.load(f)

    mock2 = tune_mock_factory("randomforest", param_grid=param_grid)
    run_tune(mock2)
    with open(mock2.output.best_params) as f:
        result2 = json.load(f)

    assert result1 == result2, "tuning is not reproducible with the same seed"