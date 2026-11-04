# Cardiovascular-Disease-Prediction-Cost-Sensitive-with-Threshold-ML

# Cardiovascular Disease Prediction: Cost-Sensitive Screening with Threshold Optimization

**Author:** Md. Anwarul Karim, MS in Data Science and Analytics, East West University

## Overview

This project builds a screening model for self-reported cardiovascular disease (CVD) from a large health survey (N = 308,754 after cleaning; 8.1% positive). With a class imbalance this strong, a model that always predicts "no disease" is about 92% accurate and finds nobody. So the project is built around **recall, PR-AUC, and choosing a decision threshold**, not accuracy.

**Headline result (held-out test set, N = 61,751; 4,993 positives):** a calibrated gradient-boosting model reaches **ROC-AUC 0.836, PR-AUC 0.300 (baseline 0.081), recall 80.2%, specificity 72.5%** at a threshold of about 0.087.

## Data

- File: `CVD_cleaned.csv`, 308,854 rows and 19 columns
- Source: a Kaggle-cleaned version of the CDC Behavioral Risk Factor Surveillance System (BRFSS) survey. Add the exact dataset link and credit CDC here.
- Target: `Heart_Disease` (Yes/No), 8.1% positive
- Note: BRFSS is a **self-reported telephone survey**, not clinical records. The target means "ever told by a professional that you had CHD or MI", so the model detects existing disease. It does not predict future events.

## Method

1. **Cleaning.** Dropped `Height_(cm)` and `Weight_(kg)` because BMI already captures them. Kept only biologically plausible BMI values (12 to 75), which removes 100 rows. A blind 1.5×IQR filter would have removed 122,077 rows (39.5%) and capped BMI at about 43, discarding the high-risk patients with severe obesity.
2. **Encoding.** `General_Health`, `Checkup`, and `Age_Category` use explicit ordinal mappings, because alphabetical label encoding scrambles their order. Diabetes (4 states) is one-hot encoded. Binary variables are coded 0/1.
3. **No leakage.** Scaling and encoding sit inside a scikit-learn `ColumnTransformer` and `Pipeline`, fitted on the training fold only. The split is a stratified 80/20 with `random_state=42`.
4. **Models.** Balanced Logistic Regression, a pruned Decision Tree, a regularized Random Forest (`max_depth=10`, `min_samples_leaf=25`), and a cost-sensitive HistGradientBoosting model.
5. **Calibration and threshold.** Isotonic calibration (`CalibratedClassifierCV`) on the HistGradientBoosting model, then a threshold chosen to reach 80% recall.

## Exploratory findings

I plotted within-group prevalence rates, P(CVD | feature), not raw counts. Raw counts mislead on imbalanced data because healthy people are the majority.

| Factor | CVD prevalence |
|---|---|
| General health: Excellent → Very Good → Good → Fair → Poor | 2.0% → 4.4% → 9.1% → 19.0% → 31.8% |
| Age: under 40 / 55–59 / 80+ | under 1.5% / 7.1% / 21.8% |
| Diabetes: yes vs no | 20.9% vs 6.1% (about 3.4×) |
| Arthritis: yes vs no | 14.1% vs 5.2% |
| Smoking history: yes vs no | 11.6% vs 5.7% |
| Exercise: no vs yes | 13.0% vs 6.7% |

## Results

Test set: N = 61,751, 4,993 positive cases.

| Model | Threshold | ROC-AUC | PR-AUC | Recall | Specificity | Precision | F1 | F2 | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.50 | 0.8331 | 0.2995 | 0.7901 | 0.7310 | 0.2053 | 0.3260 | 0.5034 | 0.7358 |
| Decision Tree (pruned, balanced) | 0.50 | 0.8247 | 0.2871 | 0.7951 | 0.7108 | 0.1947 | 0.3129 | 0.4918 | 0.7176 |
| Random Forest (regularized, balanced) | 0.50 | 0.8336 | 0.2955 | 0.7901 | 0.7286 | 0.2039 | 0.3241 | 0.5016 | 0.7336 |
| HistGradientBoosting (cost-sensitive) | 0.50 | 0.8353 | 0.3005 | 0.8077 | 0.7203 | 0.2026 | 0.3239 | 0.5056 | 0.7274 |
| Calibrated HistGradientBoosting, τ\* | 0.0874 | 0.8355 | 0.3003 | 0.8019 | 0.7254 | 0.2044 | 0.3258 | 0.5060 | 0.7316 |

**How to read this**

- The four models are within about 0.01 ROC-AUC of each other, so the differences are small. A simple, interpretable logistic regression is almost as good as boosting.
- At 80% recall, precision is about 20%. The model finds most cases, but about four in five flagged people do not have CVD. That only makes sense as a cheap first-stage screen followed by a confirmatory test.
- Accuracy falls from about 92% to about 73%. That is a deliberate trade for higher recall.
- A permutation-importance plot in the notebook shows which features drive the predictions.

## Limitations

- The decision threshold was chosen on the test set. A stricter design would pick it on a separate validation split. Recall of about 80% is therefore an estimate, not a guarantee, and will vary on new data.
- The data are self-reported, and BRFSS survey weights were not used, so prevalence figures describe this sample and not the whole population.
- There is no external validation. Self-rated general health is a strong predictor and may act as a proxy for existing illness.
- 80 duplicate rows were left in the data and may appear in both the train and test sets.
- The model is a research and portfolio project, not a clinical tool.

## Reproduce

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter notebook Cardiovascular_Disease_Prediction.ipynb
```

`requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter` (pin the versions you used).

## Repository structure

```
├── Cardiovascular_Disease_Prediction.ipynb
├── CVD_cleaned.csv
├── requirements.txt
└── README.md
```
