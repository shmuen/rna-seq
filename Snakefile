configfile: "config/config.yaml"

rule all:
    input:
        "results/pca/pca_plot.png"

rule preprocess:
    input:
        data = config["data_path"],
        labels = config["labels_path"]
    output:
        data = "results/preprocessed/data_clean.csv",
        label_mapping = "results/preprocessed/label_mapping.json"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/preprocess.py"

rule split_data:
    input:
        data = "results/preprocessed/data_clean.csv"
    output:
        X_train = "results/split/X_train.csv",
        X_test = "results/split/X_test.csv",
        y_train = "results/split/y_train.csv",
        y_test = "results/split/y_test.csv"
    params:
        test_size = config["split"]["test_size"],
        seed = config["split"]["seed"]
    conda:
        "envs/ml.yaml"
    script:
        "scripts/split_data.py"

rule scale:
    input:
        data = "results/split/X_train.csv"
    output:
        scaler = "results/scale/scaler.pkl",
        scaled = "results/scale/scaled.csv"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/scale.py"

rule pca:
    input:
        y_train = "results/split/y_train.csv",
        mapping = "results/preprocessed/label_mapping.json",
        scaler = "results/scale/scaler.pkl"
    output:
        components = "results/pca/components.csv",
        plot = "results/pca/pca_plot.png",
        variance_plot = "results/pca/variance_plot.png"
    conda:
        "envs/ml.yaml"
    params:
        n_components = config["pca"]["n_components"]
    script:
        "scripts/pca.py"