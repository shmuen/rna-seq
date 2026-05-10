configfile: "config/config.yaml"

MODEL = config["model"]

rule all:
    input:
        expand("results/plots/{model}_cm_plot.png", model = MODEL),
        expand("results/plots/{model}_roc_plot.png", model = MODEL),
        expand("results/plots/{model}_calibration_curve.png", model = MODEL),
        "results/comparison/model_comparison.csv",
        "results/comparison/barplot.png",
        "results/comparison/heatmap.png"
 
rule preprocess:
    input:
        data = config["data_path"],
        labels = config["labels_path"]
    output:
        data = "results/preprocessed/data_clean.csv",
        label_mapping = "results/preprocessed/label_mapping.json"
    log:
        "logs/preprocess.log"
    benchmark:
        "benchmarks/preprocess.txt" 
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
    log:
        "logs/split_data.log"
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
    log:
        "logs/scale.log"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/scale.py"

rule pca:
    input:
        X_train_scaled = "results/scale/X_train_scaled.csv",
        X_test_scaled = "results/scale/X_test_scaled.csv",
        y_train = "results/split/y_train.csv",
        label_mapping = "results/preprocessed/label_mapping.json"
    output:
        X_train_pca = "results/pca/X_train_pca.csv",
        X_test_pca = "results/pca/X_test_pca.csv",
        pca = "results/pca/pca.pkl",
        plot = "results/pca/pca_plot.png",
        variance_plot = "results/pca/variance_plot.png"
    params:
        n_components = config["pca"]["n_components"]
    log:
        "logs/pca.log"
    benchmark:
        "benchmarks/pca.txt" 
    conda:
        "envs/ml.yaml"
    script:
        "scripts/pca.py"

rule tune_models:
    input:
        X_train = "results/split/X_train.csv",
        y_train = "results/split/y_train.csv"
    output:
        best_params = "results/tune/best_params_{model}.json"
    params:
        tune = config["tuning"]["enabled"],
        param_grid = lambda wildcards: config["tuning"]["param_grids"][wildcards.model],
        default_grid = lambda wildcards: config["tuning"]["default_grids"][wildcards.model],
        n_components = config["pca"]["n_components"],
        seed = config["seed"]
    threads: 4
    log:
        "logs/tune_{model}.log"
    benchmark:
        "benchmarks/tune_{model}.txt"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/tune_models.py"

rule train_models:
    input:
        X_train_pca = "results/pca/X_train_pca.csv",
        y_train = "results/split/y_train.csv",
        best_params = "results/tune/best_params_{model}.json"
    output:
        model = "results/models/{model}.pkl"
    params:
        seed = config["seed"]
    log:
        "logs/train_model_{model}.log"
    threads: 4
    resources:
        mem_mb = 4000
    benchmark:
        "benchmarks/train_{model}.txt" 
    conda:
        "envs/ml.yaml"
    script:
        "scripts/models/train_model.py"

rule predict:
    input:
        X_test_pca = "results/pca/X_test_pca.csv",
        model = "results/models/{model}.pkl"
    output:
        y_pred = "results/prediction/y_pred_{model}.csv",
        y_proba = "results/prediction/y_proba_{model}.csv"
    log:
        "logs/predict_{model}.log"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/predict.py"

rule evaluate_metrics:
    input:
        y_test = "results/split/y_test.csv",
        y_pred = "results/prediction/y_pred_{model}.csv",
        y_proba = "results/prediction/y_proba_{model}.csv",
        model = "results/models/{model}.pkl"
    output:
        report = "results/metrics/{model}_report.csv",
        cm = "results/metrics/{model}_cm.csv",
        roc = "results/metrics/{model}_roc.csv",
        calibration = "results/metrics/{model}_calibration.csv"
    log:
        "logs/evaluate_metrics_{model}.log"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/evaluate_metrics.py"

rule nested_cv:
    input:
        X_train = "results/split/X_train.csv",
        y_train = "results/split/y_train.csv"
    output:
        nested_cv_score = "results/metrics/{model}_nested_cv.csv"
    params:
        tune = config["tuning"]["enabled"],
        param_grid = lambda wildcards: config["tuning"]["param_grids"][wildcards.model],
        n_components = config["pca"]["n_components"],
        seed = config["seed"]
    threads: 6
    log:
        "logs/nested_cv_{model}.log"
    benchmark:
        "benchmarks/nested_cv_{model}.txt"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/nested_cv.py"

rule plot_metrics:
    input:
        cm = "results/metrics/{model}_cm.csv",
        roc = "results/metrics/{model}_roc.csv",
        y_proba = "results/prediction/y_proba_{model}.csv",
        calibration = "results/metrics/{model}_calibration.csv",
        label_mapping = "results/preprocessed/label_mapping.json"
    output:
        cm_plot = "results/plots/{model}_cm_plot.png",
        roc_plot = "results/plots/{model}_roc_plot.png",
        calibration_curve = "results/plots/{model}_calibration_curve.png"
    log:
        "logs/plot_metrics_{model}.log"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/plot_metrics.py"

rule compare_models:
    input:
        reports = expand("results/metrics/{model}_report.csv", model=MODEL),
        bench = expand("benchmarks/train_{model}.txt", model=MODEL),
        roc = expand("results/metrics/{model}_roc.csv", model=MODEL),
        nested_cv = expand("results/metrics/{model}_nested_cv.csv", model=MODEL) \
            if config.get("run_nested_cv", False) else [] 
    output:
        summary = "results/comparison/model_comparison.csv",
        plot = "results/comparison/barplot.png",
        heat = "results/comparison/heatmap.png",
    log:
        "logs/compare_models.log"
    conda:
        "envs/ml.yaml"
    script:
        "scripts/compare_models.py"