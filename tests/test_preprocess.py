import pandas as pd
import numpy as np
import json, re
from pathlib import Path

# ── additional functions ────────────────────────────────────────────

def run_preprocess(preprocess_mock):
    with open("scripts/preprocess.py", "r") as f:
        code = f.read()
    exec(code, {"snakemake": preprocess_mock})


# ── tests ───────────────────────────────────────────────────────────

def test_output_files_exist(preprocess_mock):
    """Both output files have to exist after script execution."""
    mock, _ = preprocess_mock

    run_preprocess(mock)

    assert Path(mock.output.data).exists(),      f"{mock.output.data} is missing"
    assert Path(mock.output.label_mapping).exists(),  f"{mock.output.label_mapping} is missing"


def test_log1p_transformation(preprocess_mock):
    """All values have to be >= 0 and change the raw data."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    result = pd.read_csv(mock.output.data, index_col=0)
    gene_cols = [c for c in result.columns if c != "Class"]

    # No value can be negative after log1p transformation
    assert (result[gene_cols] >= 0).all().all(), "negative values after log1p transformation"

    # Values can not be the same as raw input data
    rna_raw = pd.read_csv(mock.input.data, index_col=0)
    assert not np.allclose(result[gene_cols].values, rna_raw.values), \
        "log1p did not change the raw data"


def test_label_encoding_is_numeric(preprocess_mock):
    """Class column has to consist of integers."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    result = pd.read_csv(mock.output.data, index_col=0)

    assert pd.api.types.is_integer_dtype(result["Class"]), \
        "class column is not numeric (Label Encoding failed)"


def test_label_mapping_contains_all_classes(preprocess_mock):
    """label_mapping.json has to contain all 5 cancer classes with unique codes."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    with open(mock.output.label_mapping) as f:
        mapping = json.load(f)

    expected_classes = {"BRCA", "COAD", "KIRC", "LUAD", "PRAD"}
    found_classes    = set(mapping.values())

    assert found_classes == expected_classes, \
        f"missing classes in mapping: {expected_classes - found_classes}"
    
    # Keys must be unique (no two classes sharing the same code)
    keys = list(mapping.keys())
    assert len(keys) == len(set(keys)), "duplicate keys in label mapping"

    # Codes must be exactly 0..n-1 (contiguous, no gaps/duplicates)
    codes = sorted(int(k) for k in keys)
    assert codes == list(range(len(expected_classes))), \
        f"label codes are not contiguous 0..{len(expected_classes)-1}: got {codes}"


def test_no_data_loss(preprocess_mock):
    """All samples do exist (no row drops)."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    result = pd.read_csv(mock.output.data, index_col=0)
    rna_raw = pd.read_csv(mock.input.data, index_col=0)

    assert len(result) == len(rna_raw), \
        f"samples missing: before {len(rna_raw)}, after {len(result)}"


def test_no_missing_values_in_output(preprocess_mock):
    """Output must not contain NaN."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    result = pd.read_csv(mock.output.data, index_col=0)
    missing = result.isnull().sum().sum()

    assert missing == 0, f"output contains {missing} missing values"


def test_log_file_is_written(preprocess_mock):
    """Log file has to contain shape, missing and duplicates info."""
    mock, _ = preprocess_mock
    run_preprocess(mock)

    log_content = Path(mock.log[0]).read_text().lower()

    assert re.search(r"shape", log_content), "log does not contain shape info"
    assert re.search(r"missing", log_content), "log does not contain missing info"
    assert re.search(r"duplicate", log_content), "log does not contain duplicate info"