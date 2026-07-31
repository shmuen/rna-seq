import pandas as pd
from pathlib import Path
from conftest import CLASS_NAMES

# ── additional function ──────────────────────────────────────────────────────────────

def run_evaluate(evaluate_mock):
    with open("scripts/evaluate_metrics.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": evaluate_mock})

# ── tests ──────────────────────────────────────────────────────────────

def test_output_files_exist(evaluate_mock):
    """All output files have to exist after script execution."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    assert Path(mock.output.report ).exists(),      f"{mock.output.report} is missing"
    assert Path(mock.output.summary).exists(),  f"{mock.output.summary} is missing"
    assert Path(mock.output.cm).exists(),  f"{mock.output.cm} is missing"
    assert Path(mock.output.roc).exists(),  f"{mock.output.roc} is missing"
    assert Path(mock.output.calibration).exists(),  f"{mock.output.calibration} is missing"


def test_report_columns_exist(evaluate_mock):
    """Report file has all defined columns with numerical values."""
    mock, _ = evaluate_mock
    run_evaluate(mock)
    
    report = pd.read_csv(mock.output.report, index_col=0)
    expected_rows = len(CLASS_NAMES) + 2 #classes + macro avg and weighted avg

    assert report.shape == (expected_rows,4), "report file does not have right shape"

    columns_report = list(report)
    assert columns_report == ["precision", "recall", "f1-score","support"], "column names of report file do not match"

    for col in columns_report:
        assert pd.api.types.is_numeric_dtype(report[col]), f"{col} is not numeric"

def test_summary_columns_exist(evaluate_mock):
    """Summary file has all defined columns with numerical values."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    summary = pd.read_csv(mock.output.summary, index_col=0)

    assert summary.shape == (1,3), "summary file does not have right shape"

    columns_summary = list(summary)
    assert columns_summary == ["accuracy", "kappa", "mcc"], "column names of summary file do not match"

    for col in columns_summary:
        assert pd.api.types.is_numeric_dtype(summary[col]), f"{col} is not numeric"


def test_report_values_range(evaluate_mock):
    """Report and summary file values are within defined range."""
    mock, _ = evaluate_mock
    run_evaluate(mock)
    
    report = pd.read_csv(mock.output.report, index_col=0)
    summary = pd.read_csv(mock.output.summary, index_col=0)

    assert report[["precision", "recall", "f1-score"]].apply(
        lambda col: col.between(0,1)
    ).all().all(), "values in result table are not as expected (between 0 and 1)"
    
    assert summary["accuracy"].between(0,1).all(), "accuracy value not as expected (between 0 and 1)"

    assert (report["support"] % 1 == 0).all(), "values in support column are not all integers"

    assert summary[["kappa","mcc"]].apply(
        lambda col: col.between(-1,1) 
    ).all().all(), "kappa or mcc are not between -1 and 1"


def test_cm_columns_exist(evaluate_mock):
    """Confusion matrix has all correct shape with numerical values."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    cm = pd.read_csv(mock.output.cm, header=0, index_col=0)

    expected_rows = len(CLASS_NAMES) 
    assert cm.shape == (expected_rows,expected_rows), "confusion matrix does not have correct size"

    assert cm.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all(), "confusion matrix contains non-numerical values"


def test_cm_values_range(evaluate_mock):
    """Confusion matrix entries are within defined range."""
    mock, _ = evaluate_mock
    run_evaluate(mock)
        
    cm = pd.read_csv(mock.output.cm, header=0, index_col=0)

    assert (cm.values >=0).all(), 'confusion matrix contains negative values'


def test_cm_sum_matches_samples(evaluate_mock):
    """Sum of confusion matrix entries matches number of test samples."""
    mock, _ = evaluate_mock
    run_evaluate(mock)
        
    y_test = pd.read_csv(mock.input.y_test, index_col=0)
    cm = pd.read_csv(mock.output.cm, header=0, index_col=0)

    assert cm.values.sum() == len(y_test), f"sum of confusion matrix ({cm.values.sum()}) does not match sample count ({len(y_test)})"


def test_roc_all_classes_exist(evaluate_mock):
    """Roc matrix has all classes with at least 2 entries."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    roc = pd.read_csv(mock.output.roc)

    assert roc[["fpr","tpr","auc"]].apply(lambda col: pd.api.types.is_numeric_dtype(col)).all(), "roc matrix contains non-numerical values"

    assert set(roc["class"]) == set(range(len(CLASS_NAMES))), "not all classes have roc curves"

    assert roc.groupby("class").size().min() >= 2, "roc curve does not have at least 2 points"


def test_roc_values_range(evaluate_mock):
    """All roc values are between 0 and 1."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    roc = pd.read_csv(mock.output.roc)

    assert roc[["fpr","tpr","auc"]].apply(
        lambda col: col.between(0,1) 
    ).all().all(), "fpr, tpr or auc are not between 0 and 1"


def test_roc_monotonic(evaluate_mock):
    """Roc curves are monotonic increasing."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    roc = pd.read_csv(mock.output.roc)
    for cls, group in roc.groupby("class"):
        # fpr/tpr are sorted by threshold in roc_curve() by construction
        assert group["fpr"].is_monotonic_increasing, f"fpr not monotonic for class {cls}"
        assert group["tpr"].is_monotonic_increasing, f"tpr not monotonic for class {cls}"


def test_calibration_all_casses_exist(evaluate_mock):
    """Calibration matrix has all classes with at least 2 entries."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    cal = pd.read_csv(mock.output.calibration)
    # only mean_predicted_value is guaranteed monotonic (bins sorted by probability);
    # fraction_of_positives is NOT tested here, as it reflects model calibration quality
 
    assert cal[["fraction_of_positives","mean_predicted_value"]].apply(lambda col: pd.api.types.is_numeric_dtype(col)).all(), "calibration matrix contains non-numerical values"

    assert set(cal["class"]) == set(range(len(CLASS_NAMES))), "not all classes have calibration curves"

    assert cal.groupby("class").size().min() >= 2, "calibration curve does not have at least 2 points"


def test_calibration_values_range(evaluate_mock):
    """All calibration values are between 0 and 1."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    cal = pd.read_csv(mock.output.calibration)

    assert cal[["fraction_of_positives","mean_predicted_value"]].apply(
        lambda col: col.between(0,1) 
    ).all().all(), "fraction of positives or mean predicted value are not between 0 and 1"


def test_mean_predicted_value_calibration_monotonic(evaluate_mock):
    """Mean predicted value of calibration is monotonic increasing."""
    mock, _ = evaluate_mock
    run_evaluate(mock)

    cal = pd.read_csv(mock.output.calibration)
    for cls, group in cal.groupby("class"):
        assert group["mean_predicted_value"].is_monotonic_increasing, f"mean predicted value not monotonic for class {cls}"

