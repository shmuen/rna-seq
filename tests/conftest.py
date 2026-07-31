import pandas as pd
import numpy as np
import pytest
import pickle, json
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier

CLASS_NAMES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

# ── basic data ─────────────────────────────────────────────────────────────────

def _make_gene_df(n_samples, n_genes=10, distribution="uniform"):
    """Generate synthetic gene expression data with given distribution"""
    if distribution == "uniform":
        data = np.random.randint(0, 1000, size=(n_samples, n_genes))
    elif distribution == "normal":
        data = np.random.normal(0, 1, size=(n_samples, n_genes))
    return pd.DataFrame(data, columns=[f"gene_{i}" for i in range(n_genes)])


@pytest.fixture
def raw_data(tmp_path):
    """raw data: separate RNA and label files (for preprocess)."""
    np.random.seed(42)
    rna = _make_gene_df(25)

    labels = pd.DataFrame({"Class": CLASS_NAMES * 5})

    rna_path   = tmp_path / "rna_seq.csv"
    label_path = tmp_path / "labels.csv"
    rna.to_csv(rna_path)
    labels.to_csv(label_path)
    return rna_path, label_path


@pytest.fixture
def split_data(tmp_path):
    """split data: split X and Y data for scaling."""
    np.random.seed(42)
    X_train = _make_gene_df(40)
    X_test = _make_gene_df(10)
    
    X_train_path = tmp_path / "X_train.csv"
    X_test_path  = tmp_path / "X_test.csv"
    X_train.to_csv(X_train_path)
    X_test.to_csv(X_test_path)
    return X_train_path, X_test_path


@pytest.fixture
def pca_data(tmp_path):
    """PCA input data: normally-distributed (pre-scaled) features, 
    plus y_train and label_mapping needed by pca.py."""
    np.random.seed(42)
    X_train_scaled = _make_gene_df(40, distribution="normal")
    X_test_scaled = _make_gene_df(10, distribution="normal")

    y_train = pd.DataFrame({"Class": np.random.choice(range(len(CLASS_NAMES)), size=40)})

    label_mapping = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    
    X_train_scaled_path = tmp_path / "X_train_scaled.csv"
    X_test_scaled_path  = tmp_path / "X_test_scaled.csv"
    y_train_path  = tmp_path / "y_train.csv"
    mapping_path = tmp_path / "label_mapping.json"
    X_train_scaled.to_csv(X_train_scaled_path)
    X_test_scaled.to_csv(X_test_scaled_path)
    y_train.to_csv(y_train_path)
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f)
    return X_train_scaled_path, X_test_scaled_path, y_train_path, mapping_path


@pytest.fixture
def combined_data(tmp_path):
    """Combined data after preprocessing (for split and evaluate)."""
    np.random.seed(42)
    rna = _make_gene_df(50)
    df = rna.copy()
    df["Class"] = CLASS_NAMES * 10

    path = tmp_path / "combined.csv"
    df.to_csv(path)
    return path


# ── Snakemake mocks ────────────────────────────────────────────────────────────

@pytest.fixture
def preprocess_mock(raw_data, tmp_path):
    """Snakemake-Mock for preprocess.py"""
    rna_path, label_path = raw_data

    mock = MagicMock()
    mock.input.data           = str(rna_path)
    mock.input.labels         = str(label_path)
    mock.output.data          = str(tmp_path / "data_clean.csv")
    mock.output.label_mapping = str(tmp_path / "label_mapping.json")
    mock.log                  = [str(tmp_path / "preprocess.log")]
    return mock, tmp_path


@pytest.fixture
def split_mock(combined_data, tmp_path):
    """Snakemake-Mock for split_data.py"""
    mock = MagicMock()
    mock.input.data       = str(combined_data)
    mock.output.X_train   = str(tmp_path / "X_train.csv")
    mock.output.X_test    = str(tmp_path / "X_test.csv")
    mock.output.y_train   = str(tmp_path / "y_train.csv")
    mock.output.y_test    = str(tmp_path / "y_test.csv")
    mock.params.seed      = 42
    mock.params.test_size = 0.2
    mock.log              = [str(tmp_path / "split.log")]
    return mock, tmp_path


@pytest.fixture
def scale_mock(split_data, tmp_path):
    """Snakemake-Mock for scale.py"""
    X_train_path, X_test_path = split_data

    mock = MagicMock()
    mock.input.X_train          = str(X_train_path)
    mock.input.X_test           = str(X_test_path)
    mock.output.scaler          = str(tmp_path / "scaler.pkl")
    mock.output.X_train_scaled  = str(tmp_path / "X_train_scaled.csv")    
    mock.output.X_test_scaled   = str(tmp_path / "X_test_scaled.csv")
    mock.log                    = [str(tmp_path / "scale.log")]
    return mock, tmp_path


@pytest.fixture
def pca_mock(pca_data, tmp_path):
    """Snakemake-Mock for pca.py"""

    X_train_scaled_path, X_test_scaled_path, y_train_path, mapping_path = pca_data

    mock = MagicMock()
    mock.input.X_train_scaled   = str(X_train_scaled_path)
    mock.input.X_test_scaled    = str(X_test_scaled_path)
    mock.input.y_train          = str(y_train_path)
    mock.input.label_mapping    = str(mapping_path)
    mock.params.n_components    = 5
    mock.output.X_train_pca     = str(tmp_path / "X_train_pca.csv")    
    mock.output.X_test_pca      = str(tmp_path / "X_test_pca.csv")
    mock.output.pca             = str(tmp_path / "pca.pkl")
    mock.output.plot        = str(tmp_path / "plot.png")
    mock.output.variance_plot   = str(tmp_path / "variance_plot.png")
    mock.log                    = [str(tmp_path / "pca.log")]
    return mock, tmp_path


@pytest.fixture
def evaluate_mock(combined_data, tmp_path):
    """Snakemake-Mock for evaluate_metrics.py"""

    df = pd.read_csv(combined_data, index_col=0)
    X = df.drop(columns=["Class"]).values
    y = np.array([CLASS_NAMES.index(name) for name in df["Class"].values])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    model_path   = tmp_path / "model.pkl"
    y_test_path  = tmp_path / "y_test.csv"
    y_pred_path  = tmp_path / "y_pred.csv"
    y_proba_path = tmp_path / "y_proba.csv"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    pd.DataFrame({"Class": y_test}).to_csv(y_test_path)
    pd.DataFrame({"y_pred":  y_pred}).to_csv(y_pred_path)
    pd.DataFrame(y_proba, columns=CLASS_NAMES).to_csv(y_proba_path)

    mock = MagicMock()
    mock.input.y_test   = str(y_test_path)
    mock.input.y_pred   = str(y_pred_path)
    mock.input.y_proba  = str(y_proba_path)
    mock.input.model    = str(model_path)
    mock.output.report      = str(tmp_path / "report.csv")
    mock.output.summary     = str(tmp_path / "summary.csv")
    mock.output.cm          = str(tmp_path / "cm.csv")
    mock.output.roc         = str(tmp_path / "roc.csv")
    mock.output.calibration = str(tmp_path / "calibration.csv")
    mock.log = [str(tmp_path / "evaluate.log")]
    return mock, tmp_path