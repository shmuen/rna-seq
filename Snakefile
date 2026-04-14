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

rule pca:
    input:
        data = "results/preprocessed/data_clean.csv",
        mapping = "results/preprocessed/label_mapping.json"
    output:
        components = "results/pca/components.csv",
        plot = "results/pca/pca_plot.png"
    conda:
        "envs/ml.yaml"
    params:
        n_components = config["pca"]["n_components"]
    script:
        "scripts/pca.py"