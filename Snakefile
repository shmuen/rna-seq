configfile: "config/config.yaml"

MODEL = config["model"]

rule all:
    input:
        expand("results/validation/{model}_report.csv", model = MODEL)
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
        X_train = "results/split/X_train.csv",
        X_test = "results/split/X_test.csv"
    output:
        scaler = "results/scale/scaler.pkl",
        X_train_scaled = "results/scale/X_train_scaled.csv",
        X_test_scaled = "results/scale/X_test_scaled.csv"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/scale.py"

rule pca:
    input:
        X_train_scaled = "results/scale/X_train_scaled.csv",
        X_test_scaled = "results/scale/X_test_scaled.csv",
        y_train = "results/split/y_train.csv",
        mapping = "results/preprocessed/label_mapping.json"
    output:
        X_train_pca = "results/pca/X_train_pca.csv",
        X_test_pca = "results/pca/X_test_pca.csv",
        pca = "results/pca/pca.pkl",
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
        X_train_pca = "results/pca/X_train_pca.csv",
        y_train = "results/split/y_train.csv"
    output:
        model = "results/models/{model}.pkl"
    params:
        seed = config["seed"]
    conda:
        "envs/ml.yaml"
    script:
        "scripts/models/train_{wildcards.model}.py"

rule validate_model:
    input:
        X_test_pca = "results/pca/X_test_pca.csv",
        y_test = "results/split/y_test.csv",
        model = "results/models/{model}.pkl",
        mapping = "results/preprocessed/label_mapping.json"
    output:
        report = "results/validation/{model}_report.csv",
        cm = "results/validation/{model}_cm.csv",
        cm_plot = "results/validation/{model}_cm_plot.png",

    params:
        n_components = config["pca"]["n_components"]
    conda:
        "envs/ml.yaml"
    script:
        "scripts/validate_model.py"