configfile: "config/config.yaml"

MODEL = config["model"]

rule all:
    input:
        expand("models/{model}.pkl", model = MODEL)

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
        seed = config["seed"]
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
        scaler = "results/scale/scaler.pkl",
        y_train = "results/split/y_train.csv",
        mapping = "results/preprocessed/label_mapping.json"
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

rule train_models:
    input:
        X_train = "results/pca/components.csv",
        y_train = "results/split/y_train.csv"
    output:
        model = "results/models/{model}.pkl"
    params:
        seed = config["seed"]
    conda:
        "envs/ml.yaml"
    script:
        "scripts/models/train_{wildcards.model}.py"
