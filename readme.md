# Cancer Type Classification from Gene Expression

## Overview
This project demonstrates a fully reproducible RNA-seq analysis pipeline using Snakemake for cancer type classification based on gene expression data.
- automated preprocessing, PCA and model training pipeline
- training and evaluation of multiple machine learning models
- automated and reproducible pipeline execution
- optional hyperparameter tuning via GridSearchCV

## Background
Cancer is one of the leading causes of death worldwide, with 10 million deaths in 2022 according to the WHO. Among the most common types of cancer are lung, breast, colon and rectum and prostate cancers [1]. 
Gene expression profiles can be analysed to highlight similarities and differences between cancer types. Machine learning approaches can leverage these expression patterns to classify tumour types, potentially enabling faster and more accurate diagnosis.

## Dataset
The data is part of the RNA-Seq (HiSeq) PANCAN dataset, a random extraction of gene expression profiles from 881 patients maintained by the Cancer Genome Atlas (TCGA) pan-cancer analysis project, downloaded from [kaggle](https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016). The original dataset is hosted at [Synapse](https://www.synapse.org/#!Synapse:syn4301332). Each patient is assigned one of five tumour types based on gene expression levels across 20,531 genes:

| Label | Cancer Type |
|-------|-------------|
| BRCA  | Breast invasive carcinoma |
| KIRC  | Kidney renal clear cell carcinoma |
| COAD  | Colon adenocarcinoma |
| LUAD  | Lung adenocarcinoma |
| PRAD  | Prostate adenocarcinoma |

These cancer types are known to exhibit distinct gene expression signatures, making them a suitable benchmark for classification tasks.

## Workflow
This workflow implements a fully reproducible pipeline using Snakemake as workflow management system, with all dependencies managed via conda.

```
rna-seq/
├── Snakefile
├── config/     # pipeline configuration
├── data/       # raw input data (not tracked)
├── envs/       # conda environments
├── scripts/    # Python scripts per rule
├── logs/       # Snakemake logs (not tracked)
├── benchmarks/ # Snakemake benchmarks (not tracked)
└── results/    # all pipeline outputs (not tracked)
```

After preprocessing and dimensionality reduction via Principal Component Analysis (PCA), four classification models are trained and evaluated:

| Model | Library |
|-------|---------|
| Random Forest | scikit-learn |
| Support Vector Machine (SVM) | scikit-learn |
| Logistic Regression | scikit-learn |
| XGBoost | xgboost |

The models are compared across the following metrics using a One-vs-Rest strategy:
- Accuracy
- Macro F1
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Macro Area under the Curve (AUC)

### Rule graph of Snakemake workflow
![DAG](plots/rulegraph.png)

## Results
After hyperparameter tuning, all models performed well, with an accuracy of 0.97 or above and Macro AUC scores of 0.998 or above across all models. SVM achieved perfect scores across all metrics, while Logistic Regression and XGBoost showed slightly lower but strong performance. Random Forest performed comparably but slightly below the other models. The high performance across all models suggests that the five cancer types are well-separable in gene expression space, which is consistent with the clear cluster structure visible in the PCA plot. The strong scores likely reflect the separability of the PCA-reduced data rather than overfitting, given the small number of components. As this is a single dataset without external validation, results should be interpreted with caution.

### PCA of Gene Expression Profiles
![PCA](plots/pca_plot.png)

Despite PC1 and PC2 explaining only 10.0% and 8.1% of the total variance respectively, the five cancer types form clearly distinct clusters, indicating strong biological signal in the data.

### Model Performance Comparison
![Heatmap](plots/heatmap.png)

## Limitations
- **Single dataset without external validation:** All models are trained and evaluated on one dataset. Performance may not generalise to other cohorts or sequencing protocols.
- **Simplified problem setting:** The selected cancer types are known to be well-separated in gene expression space, making this a relatively easy classification task compared to real-world scenarios.
- **Hyperparameter tuning:** Scaler and PCA were fitted on the entire training set prior to cross-validation, which introduces a small optimistic bias in hyperparameter tuning.
- **PCA information loss:** Dimensionality reduction to 50 components retains only a fraction of total variance, potentially discarding relevant signal.

## Usage
**Requirements:** conda and snakemake ([installation guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)) must be installed.

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/shmuen/rna-seq
cd rna-seq
```

The workflow manages all dependencies automatically via conda. Run with:
```bash
snakemake --cores N --use-conda
```
Replace `N` with the number of cores to use.

Hyperparameter tuning is optional and has to be enabled in the config file.

## References
[1] Ferlay J, Ervik M, Lam F, Colombet M, Mery L, Piñeros M, et al. Global Cancer Observatory: Cancer Today. Lyon: International Agency for Research on Cancer; 2022 (https://gco.iarc.fr/today, accessed April 2026). 