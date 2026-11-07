"""
Cardiovascular Disease Prediction: six algorithms, one honest evaluation
Author: Md. Anwarul Karim | MS in Data Science and Analytics

Run:   python cvd_project.py            (CSV must sit next to this file)
Needs: pandas numpy matplotlib seaborn scikit-learn>=1.2
Output: figures/*.png, figures/*.csv and figures/results.json

v2 additions are marked with the comment  "# NEW (v2)"

Ten steps
 1 Load            6 Leak-free preprocessing
 2 Clean           7 Train six algorithms (5-fold out-of-fold scores)
 3 EDA             8 Evaluate on the test set (confusion matrix, ROC, PR, matched recall)
 4 Encode          9 Calibrate the best model and choose the threshold
 5 Split          10 Final test report (CI, importance, decision)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # remove this line to see plots in Jupyter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_validate
from sklearn.isotonic import IsotonicRegression   # NEW (v2)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss)

warnings.filterwarnings("ignore")
SEED = 42
TARGET_RECALL = 0.80
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "CVD_cleaned.csv"
KNN_SAMPLE = None            # None = KNN uses the full training set (set e.g. 60_000 to speed up)
TUNE_SAMPLE = 60_000         # NEW (v2): rows used for the quick tuning investigations
KNN_TUNE_SAMPLE = 100_000    # NEW (v2): rows used to compare k values for KNN
import json, time            # NEW (v2)
T0 = time.time()             # NEW (v2)
RESULTS = {}                 # NEW (v2): numbers saved for the report
N_BOOT = 500
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ----------------------------------------------------------------------------
# STEP 1 - LOAD
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"[1] Raw shape: {df.shape}")

# ----------------------------------------------------------------------------
# STEP 2 - CLEAN (remove only impossible values, no blind outlier deletion)
# ----------------------------------------------------------------------------
df = df.drop(columns=["Weight_(kg)", "Height_(cm)"])        # BMI already carries both
df["Diabetes"] = df["Diabetes"].map({
    "No": "No",
    "No, pre-diabetes or borderline diabetes": "Pre-Diabetes",
    "Yes, but female told only during pregnancy": "Gestational Diabetes",
    "Yes": "Yes"})
df = df[(df["BMI"] >= 12.0) & (df["BMI"] <= 75.0)].copy()
print(f"[2] Cleaned shape: {df.shape} | CVD prevalence: "
      f"{(df['Heart_Disease'] == 'Yes').mean() * 100:.2f}% | duplicates kept: {df.duplicated().sum()}")

# ----------------------------------------------------------------------------
# STEP 3 - EDA: prevalence RATES, P(CVD | group), never raw counts
# ----------------------------------------------------------------------------
age_order = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
             "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
health_order = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
checkup_order = ["Never", "5 or more years ago", "Within the past 5 years",
                 "Within the past 2 years", "Within the past year"]


def prevalence(data, feature, order=None):
    rate = data["Heart_Disease"].eq("Yes").groupby(data[feature]).mean() * 100
    return rate.reindex(order) if order is not None else rate.sort_values()


fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for a, (col, order, ttl, colr) in zip(ax.ravel(), [
        ("Age_Category", age_order, "by age", "#1d3557"),
        ("General_Health", health_order, "by self-rated health", "#e63946"),
        ("Checkup", checkup_order, "by last checkup", "#457b9d"),
        ("Exercise", None, "by exercise", "#2a9d8f")]):
    r = prevalence(df, col, order)
    sns.barplot(x=r.index.astype(str), y=r.values, ax=a, color=colr)
    a.set_title(f"CVD prevalence {ttl}", fontweight="bold")
    a.set_ylabel("CVD prevalence (%)")
    a.tick_params(axis="x", rotation=35)
    for i, v in enumerate(r.values):
        a.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("figures/eda_prevalence.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# STEP 4 - ENCODE (true clinical order for ordinal features)
# ----------------------------------------------------------------------------
binary = ["Exercise", "Skin_Cancer", "Other_Cancer", "Depression", "Arthritis",
          "Smoking_History", "Heart_Disease"]
mappings = {
    "General_Health": {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4},
    "Checkup": {c: i for i, c in enumerate(checkup_order)},
    "Age_Category": {a: i for i, a in enumerate(age_order)},
    "Sex": {"Female": 0, "Male": 1},
    **{c: {"No": 0, "Yes": 1} for c in binary},
}
enc = df.copy()
for col, mp in mappings.items():
    enc[col] = enc[col].map(mp)
assert enc.isna().sum().sum() == 0, "Unmapped category found"

# ----------------------------------------------------------------------------
# STEP 5 - STRATIFIED SPLIT (test set is locked away until step 8)
# ----------------------------------------------------------------------------
X = enc.drop(columns="Heart_Disease")
y = enc["Heart_Disease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
print(f"[5] Train: {len(X_train):,} | Test: {len(X_test):,} ({int(y_test.sum()):,} positives)")
RESULTS["split"] = {"n_train": len(X_train), "n_test": len(X_test), "test_pos": int(y_test.sum()),
                    "train_pos": int(y_train.sum()), "prevalence": float(y.mean())}   # NEW (v2)
RESULTS["clean"] = {"clean_rows": len(df), "clean_prevalence": float((df["Heart_Disease"] == "Yes").mean())}   # NEW (v2)

# ----------------------------------------------------------------------------
# STEP 6 - LEAK-FREE PREPROCESSING (scaler lives inside the Pipeline)
# ----------------------------------------------------------------------------
num_cols = ["BMI", "Alcohol_Consumption", "Fruit_Consumption",
            "Green_Vegetables_Consumption", "FriedPotato_Consumption"]
nom_cols = ["Diabetes"]
ord_cols = [c for c in X.columns if c not in num_cols + nom_cols]


def make_preprocessor(scale_everything=False):
    """KNN is distance-based, so ordinal columns must be scaled too."""
    num = num_cols + ord_cols if scale_everything else num_cols
    ordc = [] if scale_everything else ord_cols
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("nom", OneHotEncoder(drop="first", sparse_output=False), nom_cols),
        ("ord", "passthrough", ordc)])


# ----------------------------------------------------------------------------
# NEW (v2) STEP 7a - BRIEF TUNING INVESTIGATIONS (training data only, test set untouched)
# ----------------------------------------------------------------------------
def stratified_sample(n):
    if len(X_train) <= n:
        return X_train, y_train
    Xs_, _, ys_, _ = train_test_split(X_train, y_train, train_size=n,
                                      random_state=SEED, stratify=y_train)
    return Xs_, ys_


def cv_scores(estimator, Xs_, ys_, folds, train_score=False):
    r = cross_validate(estimator, Xs_, ys_,
                       cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
                       scoring={"roc": "roc_auc", "pr": "average_precision"},
                       return_train_score=train_score, n_jobs=1)
    out = {"cv_roc": r["test_roc"].mean(), "cv_pr": r["test_pr"].mean()}
    if train_score:
        out["train_roc"] = r["train_roc"].mean()
    return out


# (a) KNN: compare several k values with 5-fold cross-validation on training rows
Xk, yk = stratified_sample(KNN_TUNE_SAMPLE)
knn_rows = []
for k in [5, 15, 31, 51, 101, 201, 401, 801]:   # NEW (v2): grid reaches past sqrt(n)/2 on purpose
    pipe_k = Pipeline([("prep", make_preprocessor(scale_everything=True)),
                       ("clf", KNeighborsClassifier(n_neighbors=k, n_jobs=-1))])
    knn_rows.append({"k": k, **cv_scores(pipe_k, Xk, yk, 5)})
    print(f"[7a] KNN k={k:<4d} CV ROC-AUC={knn_rows[-1]['cv_roc']:.4f} "
          f"PR-AUC={knn_rows[-1]['cv_pr']:.4f}  ({time.time() - T0:.0f}s)", flush=True)
knn_tbl = pd.DataFrame(knn_rows).set_index("k")
BEST_K = int(knn_tbl["cv_pr"].idxmax())            # chosen on training data only
knn_tbl.round(4).to_csv("figures/tuning_knn_k.csv")
RESULTS["knn_k"] = {"grid": knn_tbl.round(4).reset_index().to_dict("records"), "best_k": BEST_K}

# (b) Decision tree: does tree depth (complexity) matter?
Xs, ys = stratified_sample(TUNE_SAMPLE)
dt_rows = []
for d in [2, 4, 6, 8, 12, 16]:
    pipe_d = Pipeline([("prep", make_preprocessor()),
                       ("clf", DecisionTreeClassifier(max_depth=d, min_samples_leaf=50,
                                                      class_weight="balanced", random_state=SEED))])
    dt_rows.append({"max_depth": d, **cv_scores(pipe_d, Xs, ys, 3, train_score=True)})
    print(f"[7a] DT depth={d:<3d} CV ROC-AUC={dt_rows[-1]['cv_roc']:.4f} "
          f"train ROC-AUC={dt_rows[-1]['train_roc']:.4f}", flush=True)
dt_tbl = pd.DataFrame(dt_rows).set_index("max_depth")
dt_tbl.round(4).to_csv("figures/tuning_dt_depth.csv")
RESULTS["dt_depth"] = dt_tbl.round(4).reset_index().to_dict("records")

# (c) Random forest: does the number of trees matter?
rf_rows = []
for n_trees in [10, 25, 50, 100, 200]:
    pipe_r = Pipeline([("prep", make_preprocessor()),
                       ("clf", RandomForestClassifier(n_estimators=n_trees, max_depth=10,
                                                      min_samples_leaf=25, class_weight="balanced",
                                                      n_jobs=-1, random_state=SEED))])
    rf_rows.append({"n_trees": n_trees, **cv_scores(pipe_r, Xs, ys, 3)})
    print(f"[7a] RF trees={n_trees:<4d} CV ROC-AUC={rf_rows[-1]['cv_roc']:.4f} "
          f"PR-AUC={rf_rows[-1]['cv_pr']:.4f}  ({time.time() - T0:.0f}s)", flush=True)
rf_tbl = pd.DataFrame(rf_rows).set_index("n_trees")
rf_tbl.round(4).to_csv("figures/tuning_rf_trees.csv")
RESULTS["rf_trees"] = rf_tbl.round(4).reset_index().to_dict("records")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(knn_tbl.index, knn_tbl["cv_pr"], marker="o", color="#1d3557")
ax[0].set(title=f"KNN: k vs CV PR-AUC (best k = {BEST_K})", xlabel="k (neighbours)",
          ylabel="CV PR-AUC")
ax[0].set_xscale("log")
ax[1].plot(dt_tbl.index, dt_tbl["cv_roc"], marker="o", color="#2a9d8f", label="CV (unseen folds)")
ax[1].plot(dt_tbl.index, dt_tbl["train_roc"], marker="s", ls="--", color="#e63946", label="Training")
ax[1].set(title="Decision tree: depth vs ROC-AUC", xlabel="max_depth", ylabel="ROC-AUC")
ax[1].legend()
ax[2].plot(rf_tbl.index, rf_tbl["cv_roc"], marker="o", color="#457b9d")
ax[2].set(title="Random forest: trees vs CV ROC-AUC", xlabel="n_estimators", ylabel="CV ROC-AUC")
plt.tight_layout()
plt.savefig("figures/tuning_investigations.png", bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------------
# STEP 7 - SIX ALGORITHMS, ONE SET OF RULES
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000,
                                   random_state=SEED))]),
    "Naive Bayes": Pipeline([
        ("prep", make_preprocessor()),
        ("clf", GaussianNB())]),
    "KNN": Pipeline([
        ("prep", make_preprocessor(scale_everything=True)),
        ("clf", KNeighborsClassifier(n_neighbors=BEST_K, n_jobs=-1))]),   # NEW (v2): k chosen by CV
    "Decision Tree": Pipeline([
        ("prep", make_preprocessor()),
        ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                       class_weight="balanced", random_state=SEED))]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=25,
                                       max_features="sqrt", class_weight="balanced",
                                       n_jobs=-1, random_state=SEED))]),
    "AdaBoost": Pipeline([
        ("prep", make_preprocessor()),
        ("clf", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                   n_estimators=200, learning_rate=0.5, random_state=SEED))]),
}


def train_view(name):
    """Training rows used by each model (KNN may use a stratified sample for speed)."""
    if name == "KNN" and KNN_SAMPLE and len(X_train) > KNN_SAMPLE:
        Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=KNN_SAMPLE,
                                        random_state=SEED, stratify=y_train)
        return Xs, ys
    return X_train, y_train


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_rows, oof_probs = [], {}
for name, pipe in models.items():
    Xt, yt = train_view(name)
    oof = cross_val_predict(pipe, Xt, yt, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    oof_probs[name] = oof
    oof_rows.append({"Model": name,
                     "OOF ROC-AUC": roc_auc_score(yt, oof),
                     "OOF PR-AUC": average_precision_score(yt, oof)})
    print(f"[7] {name:<20s} OOF ROC-AUC={oof_rows[-1]['OOF ROC-AUC']:.4f} "
          f"PR-AUC={oof_rows[-1]['OOF PR-AUC']:.4f}  ({time.time() - T0:.0f}s)", flush=True)
oof_table = pd.DataFrame(oof_rows).set_index("Model")
oof_table.round(4).to_csv("figures/oof_table.csv")

# ----------------------------------------------------------------------------
# STEP 8 - EVALUATE ON THE TEST SET (default threshold 0.5): metrics, CM, ROC, PR
# ----------------------------------------------------------------------------
def evaluate(name, y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"Model": name, "Threshold": round(float(thr), 4),
            "ROC-AUC": roc_auc_score(y_true, prob),
            "PR-AUC": average_precision_score(y_true, prob),
            "Recall": recall_score(y_true, pred), "Specificity": tn / (tn + fp),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "F1": f1_score(y_true, pred), "F2": fbeta_score(y_true, pred, beta=2.0),
            "Accuracy": accuracy_score(y_true, pred),
            "Brier": brier_score_loss(y_true, prob),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


probs, records = {}, []
for name, pipe in models.items():
    Xt, yt = train_view(name)
    pipe.fit(Xt, yt)
    probs[name] = pipe.predict_proba(X_test)[:, 1]
    records.append(evaluate(name, y_test, probs[name], 0.50))
scoreboard = pd.DataFrame(records).set_index("Model")
print("\n[8] TEST SET AT THRESHOLD 0.50")
print(scoreboard.drop(columns=["TN", "FP", "FN", "TP"]).round(4).to_string())
scoreboard.round(4).to_csv("figures/scoreboard_at_0.5.csv")

# 8a - confusion matrices for all six
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for a, name in zip(axes.ravel(), models):
    cm = confusion_matrix(y_test, (probs[name] >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Pred: No", "Pred: CVD"], yticklabels=["Actual: No", "Actual: CVD"])
    a.set_title(f"{name}\nrecall {recall_score(y_test, probs[name] >= 0.5) * 100:.1f}%",
                fontweight="bold")
plt.tight_layout()
plt.savefig("figures/confusion_matrices_at_0.5.png", bbox_inches="tight")
plt.close()

# 8b - ROC and precision-recall curves for all six
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for name, p in probs.items():
    f, t, _ = roc_curve(y_test, p)
    ax[0].plot(f, t, lw=2, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax[1].plot(rc, pr, lw=2, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set(title="ROC curves (AUC)", xlabel="False positive rate", ylabel="True positive rate")
ax[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y_test.mean():.3f})")
ax[1].axvline(TARGET_RECALL, color="red", ls=":", lw=1.5, label="Target recall 80%")
ax[1].set(title="Precision-Recall curves (PR-AUC)", xlabel="Recall", ylabel="Precision")
ax[0].legend(loc="lower right", fontsize=8)
ax[1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", bbox_inches="tight")
plt.close()

# 8c - fair comparison: every model gets its OWN threshold for 80% recall,
#      chosen on training out-of-fold scores only (models without class_weight
#      barely flag anyone at 0.5, so 0.5 is not a fair yardstick for them)
def tau_for_recall(y_true, prob, target=TARGET_RECALL):
    _, rec_, thr_ = precision_recall_curve(y_true, prob)
    return thr_[np.where(rec_[:-1] >= target)[0][-1]]


matched = []
for name in models:
    _, yt = train_view(name)
    t = tau_for_recall(yt, oof_probs[name])
    matched.append(evaluate(name, y_test, probs[name], t))
matched = pd.DataFrame(matched).set_index("Model")
print("\n[8c] TEST SET AT EACH MODEL'S OWN 80%-RECALL THRESHOLD (from training OOF)")
print(matched.drop(columns=["TN", "FP", "FN", "TP"]).round(4).to_string())
matched.round(4).to_csv("figures/scoreboard_matched_recall.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for a, name in zip(axes.ravel(), models):
    cm = confusion_matrix(y_test, (probs[name] >= matched.loc[name, "Threshold"]).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Pred: No", "Pred: CVD"], yticklabels=["Actual: No", "Actual: CVD"])
    a.set_title(f"{name}\ntau={matched.loc[name, 'Threshold']:.3f}, "
                f"recall {matched.loc[name, 'Recall'] * 100:.1f}%", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/confusion_matrices_matched_recall.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# STEP 9 - CALIBRATE THE BEST MODEL AND CHOOSE THE THRESHOLD (training data only)
# NEW (v2): calibration now re-uses the out-of-fold scores from step 7 (cheap and leak-free):
#   1) isotonic regression is fitted on out-of-fold scores -> calibrated probabilities
#   2) honest calibrated scores for the threshold come from a nested 5-fold isotonic fit
#   3) tau* = highest threshold that still reaches 80% recall on those training scores
# ----------------------------------------------------------------------------
best = oof_table["OOF PR-AUC"].idxmax()           # picked WITHOUT looking at the test set
print(f"\n[9] Best model by training out-of-fold PR-AUC: {best}", flush=True)
_, yb = train_view(best)
oof_raw = oof_probs[best]

oof_cal = np.zeros_like(oof_raw)                  # NEW (v2): nested isotonic on out-of-fold scores
for tr, va in skf.split(oof_raw.reshape(-1, 1), yb):
    iso_cv = IsotonicRegression(out_of_bounds="clip").fit(oof_raw[tr], np.asarray(yb)[tr])
    oof_cal[va] = iso_cv.predict(oof_raw[va])

iso = IsotonicRegression(out_of_bounds="clip").fit(oof_raw, np.asarray(yb))   # final calibrator
tau = tau_for_recall(yb, oof_cal)
print(f"[9] tau* (80% recall on training OOF, calibrated scale) = {tau:.4f}", flush=True)

p_raw = probs[best]                                # base model already fitted in step 8
p_final = iso.predict(p_raw)                       # calibrated test probabilities

fig, ax = plt.subplots(figsize=(6.5, 6))
for lab, p in [(f"{best} (uncalibrated)", p_raw), (f"{best} (isotonic calibrated)", p_final)]:
    frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", label=f"{lab}, Brier={brier_score_loss(y_test, p):.3f}")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set(title="Calibration (reliability) curve", xlabel="Mean predicted probability",
       ylabel="Observed CVD fraction")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/calibration_curve.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# STEP 10 - FINAL TEST REPORT (test set used once, at the pre-chosen tau*)
# ----------------------------------------------------------------------------
final = evaluate(f"{best} + calibration", y_test, p_final, tau)
final["Brier_uncalibrated"] = brier_score_loss(y_test, p_raw)
pd.DataFrame([final]).round(4).to_csv("figures/final_result.csv", index=False)
print("\n[10] FINAL RESULT", flush=True)
for k_, v_ in final.items():
    print(f"   {k_}: {v_ if isinstance(v_, str) else round(float(v_), 4)}")

rng = np.random.default_rng(SEED)
yt_np = y_test.to_numpy()
boot = {"ROC-AUC": [], "PR-AUC": [], "Recall": [], "Specificity": [], "Precision": []}
for _ in range(N_BOOT):
    i = rng.integers(0, len(yt_np), len(yt_np))
    ybt, pbt = yt_np[i], p_final[i]
    pred = pbt >= tau
    boot["ROC-AUC"].append(roc_auc_score(ybt, pbt))
    boot["PR-AUC"].append(average_precision_score(ybt, pbt))
    boot["Recall"].append(pred[ybt == 1].mean())
    boot["Specificity"].append((~pred[ybt == 0]).mean())
    boot["Precision"].append(ybt[pred].mean())
ci = {}
print("\nBootstrap 95% CI")
for k_, v_ in boot.items():
    lo, hi = np.percentile(v_, [2.5, 97.5])
    ci[k_] = [float(np.mean(v_)), float(lo), float(hi)]
    print(f"   {k_:12s} {np.mean(v_):.4f} [{lo:.4f}, {hi:.4f}]")

cm = confusion_matrix(y_test, (p_final >= tau).astype(int))
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred: No CVD", "Pred: CVD"], yticklabels=["Actual: No CVD", "Actual: CVD"])
plt.title(f"Final model at tau*={tau:.4f}", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/final_confusion_matrix.png", bbox_inches="tight")
plt.close()

# NEW (v2): importance of the uncalibrated base pipeline (isotonic is monotone, so ROC-AUC is unchanged)
perm = permutation_importance(models[best], X_test, y_test, n_repeats=5, random_state=SEED,
                              scoring="roc_auc", n_jobs=1)
imp = pd.DataFrame({"Feature": X.columns, "Mean": perm.importances_mean,
                    "Std": perm.importances_std}).sort_values("Mean")
plt.figure(figsize=(9, 6))
plt.barh(imp["Feature"], imp["Mean"], xerr=imp["Std"], color="#1d3557", alpha=0.9)
plt.title("Permutation importance (drop in test ROC-AUC)", fontweight="bold")
plt.xlabel("Decrease in ROC-AUC when the feature is shuffled")
plt.tight_layout()
plt.savefig("figures/permutation_importance.png", bbox_inches="tight")
plt.close()
print("\nTop features:")
print(imp.sort_values("Mean", ascending=False).head(8).round(4).to_string(index=False))

# NEW (v2): save everything the report needs
RESULTS["best_model"] = best
RESULTS["tau"] = float(tau)
RESULTS["final"] = {k_: (v_ if isinstance(v_, str) else float(v_)) for k_, v_ in final.items()}
RESULTS["bootstrap_ci"] = ci
RESULTS["importance"] = imp.sort_values("Mean", ascending=False).round(4).to_dict("records")
RESULTS["always_no_accuracy"] = float(1 - y_test.mean())
RESULTS["runtime_seconds"] = round(time.time() - T0)
with open("figures/results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=1, default=float)
print(f"\nDone in {time.time() - T0:.0f}s. Figures and tables are in ./figures", flush=True)
