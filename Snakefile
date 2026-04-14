configfile: "config/config.yaml"

rule all:
    input:
        "results/processed/data_clean.csv",
        "results/processed/label_mapping.json"

rule preprocess:
    input:
        data = config["data_path"],
        labels = config["labels_path"]
    output:
        data = "results/processed/data_clean.csv",
        label_mapping = "results/processed/label_mapping.json"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/preprocess.py"