# A Leakage-Aware Benchmark of AQ-10 Autistic Spectrum Disorder Screening
Leakage-aware benchmark of 11 machine-learning models on the UCI AQ-10 **Child + Adolescent** autism screening datasets (N = 392, ages 4–16).

## Key finding
In this dataset the label is the questionnaire's own scoring rule:

- `result` = A1 + … + A10 (true for 100% of rows)
- `Class/ASD` = YES ⇔ `result` > 6 (true for 100% of rows)

So the "100% accuracy" reported in many papers measures how well a model copies the AQ-10 score. It does **not** show that the model can detect autism.

## Overview of the experiments for Ablation Study

| Information available | E1 | E2 | E3 | E4 | E5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| All AQ-10 items | Yes | Yes | No | Subset | Yes/No |
| Total score | Yes | No | No | No | No |
| Demographics | Yes | Yes | Yes | No | Yes |
| Repeated cross-validation | Yes | Yes | Yes | No | No |
| Nested cross-validation | No | No | No | Yes | No |
| Separate age cohorts | No | No | No | No | Yes |

E1 = naive (leaky) · E2 = realistic · E3 = demographics only · E4 = reduced questionnaire · E5 = cross-cohort (train on one age group, test on the other)

## Results (5 × 10-fold CV)
| Setting | Best model | Accuracy |
| --- | --- | --- |
| E1: items + demographics + `result` (leaky) | 8 of 11 models | 1.000 |
| E2: items + demographics | AdaBoost / LR | 1.000 / 0.998 |
| E3: demographics only | SVM | 0.609 (near chance) |

- Logistic regression gives all 10 items equal weight and puts the cut-off at 6.5, which is exactly the AQ-10 rule.
- With 9 items accuracy is 97.7%; with 5 items it drops to 87.7%.
- An SVM trained on adolescents scores only 63.5% on children. Its ranking is still good (AUC 0.97), so the problem is the decision threshold.

## Repository
```
├── ASD_Screening_AQ10.ipynb   # full pipeline (Colab-ready)
├── Autism_Combined_Cleaned.csv
├── figures/                    # 9 figures (300 dpi)
├── tables/                     # result tables (CSV)
└── paper/main.tex              # IEEE conference paper (Overleaf)
```

## Run
**Google Colab:** open the notebook, choose Runtime → Run all, and upload the CSV when asked.
**Locally:**

```bash
pip install scikit-learn xgboost lightgbm shap seaborn pandas
jupyter notebook ASD_Screening_AQ10.ipynb
```

A full run takes about 3 minutes on a CPU.

## Models
LR · SVM · k-NN · Naive Bayes · Decision Tree · Random Forest · AdaBoost · Gradient Boosting · XGBoost · LightGBM · MLP

## Data
F. Thabtah, *Autistic Spectrum Disorder Screening Data for Children / Adolescent*, UCI ML Repository, 2017. <https://archive.ics.uci.edu/dataset/419>

## Paper

Read the paper on Overleaf: <https://www.overleaf.com/project/6b0908fccf620310d8a0ca5a>

## Disclaimer
This is research code. It is not a diagnostic tool. An AQ-10 score is a screening result, not a clinical diagnosis.
