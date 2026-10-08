"""
train_model.py
==============
Loads online_shoppers_intention.csv, preprocesses the data,
trains a Random Forest classifier to predict Revenue (purchase),
and saves the model + preprocessor artifacts into ./models/.

Run:
    python train_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join("data", "online_shoppers_intention.csv")
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── 2. Encode categorical columns ────────────────────────────────────────────
CAT_COLS = ["Month", "VisitorType"]
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Boolean → int
df["Weekend"] = df["Weekend"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).astype(int)
df["Revenue"]  = df["Revenue"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).astype(int)

# ── 3. Feature / target split ────────────────────────────────────────────────
FEATURE_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
X = df[FEATURE_COLS]
y = df["Revenue"]

print(f"  Class distribution:\n{y.value_counts()}")

# ── 4. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train models & pick best ──────────────────────────────────────────────
print("\nTraining models …")
candidates = {
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, clf in candidates.items():
    if name == "LogisticRegression":
        clf.fit(X_train_sc, y_train)
        preds = clf.predict(X_test_sc)
        proba = clf.predict_proba(X_test_sc)[:, 1]
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    results[name] = {"model": clf, "acc": acc, "auc": auc}
    print(f"  {name:<22}  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

# Best by AUC
best_name = max(results, key=lambda n: results[n]["auc"])
best      = results[best_name]
print(f"\nBest model: {best_name}  (AUC={best['auc']:.4f})")

# ── 7. Detailed evaluation of best model ─────────────────────────────────────
if best_name == "LogisticRegression":
    y_pred = best["model"].predict(X_test_sc)
else:
    y_pred = best["model"].predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importances (only for tree models)
if hasattr(best["model"], "feature_importances_"):
    fi = pd.Series(best["model"].feature_importances_, index=FEATURE_COLS)
    print("\nTop-10 Features:\n", fi.nlargest(10).to_string())

# ── 8. Save artifacts ────────────────────────────────────────────────────────
joblib.dump(best["model"],  os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(scaler,         os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(encoders,       os.path.join(MODELS_DIR, "encoders.pkl"))
joblib.dump(FEATURE_COLS,   os.path.join(MODELS_DIR, "feature_cols.pkl"))

# Save model metadata for the frontend
metadata = {
    "model_name": best_name,
    "accuracy":   round(best["acc"], 4),
    "roc_auc":    round(best["auc"], 4),
    "n_features": len(FEATURE_COLS),
    "train_size": len(X_train),
    "test_size":  len(X_test),
}
joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

print("\nArtifacts saved to ./models/")
print("  model.pkl | scaler.pkl | encoders.pkl | feature_cols.pkl | metadata.pkl")
print("\nDone! ✓")
