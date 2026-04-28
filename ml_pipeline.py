"""
ml_pipeline.py
==============
Traditional Machine-Learning pipeline on the CLaMP dataset.

Phases implemented:
    * Load and inspect the data
    * Clean (duplicates, missing values, non-numeric columns)
    * Encode labels, stratified train/test split, standard scaling
    * Train Random Forest, SVM (RBF), Gradient Boosting
    * Train a Voting Ensemble (soft voting) combining the three
    * Feature-importance plot
    * Evaluate every model with the unified evaluation module
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split
from sklearn.ensemble          import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm               import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fatima_evaluation import evaluate_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_PATH    = os.path.join(
    PROJECT_ROOT,
    "databases",
    "Classification of Malwares (CLaMP)",
    "ClaMP_Integrated-5184.csv",
)
RESULTS_DIR  = os.path.abspath(os.path.join(HERE, "..", "results"))
CM_DIR       = os.path.join(RESULTS_DIR, "confusion_matrices")
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR,      exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 - Load and inspect
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"[1] Loaded CLaMP : shape={df.shape}")
    print(f"    columns      : {len(df.columns)}")
    print(f"    class counts : {df['class'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 - Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    df = df.dropna()
    # 'fileinfo' is a free-text column - drop it.
    if "fileinfo" in df.columns:
        df = df.drop(columns=["fileinfo"])
    # Ensure every feature is numeric; drop any that still aren't.
    non_numeric = [c for c in df.columns
                   if c != "class" and not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        df = df.drop(columns=non_numeric)
    print(f"[2] Cleaned data : dropped {before - df.shape[0]} rows, "
          f"shape now {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 3-5 - Encode, split, scale
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["class"]).values
    y = df["class"].values

    le = LabelEncoder()
    y  = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]
    print(f"[3] Labels encoded -> classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"[4] Stratified split : train={X_train.shape[0]}, "
          f"test={X_test.shape[0]}")

    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f"[5] Features scaled  : mean~0, std~1")

    feature_names = [c for c in df.columns if c != "class"]
    return (X_train_s, X_test_s, y_train, y_test,
            class_names, feature_names, scaler, le)


# ---------------------------------------------------------------------------
# Steps 6-8 - Train the three base ML models
# ---------------------------------------------------------------------------
def train_models(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=RANDOM_STATE,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
    )

    print("[6] Training Random Forest ...");    rf.fit(X_train,  y_train)
    print("[7] Training SVM (RBF)    ...");     svm.fit(X_train, y_train)
    print("[8] Training Gradient Boosting ..."); gb.fit(X_train,  y_train)

    return rf, svm, gb


# ---------------------------------------------------------------------------
# Voting Ensemble (soft voting)
# ---------------------------------------------------------------------------
def train_voting_ensemble(rf, svm, gb, X_train, y_train):
    voter = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )
    print("[+] Training Voting Ensemble (soft) ...")
    voter.fit(X_train, y_train)
    return voter


# ---------------------------------------------------------------------------
# Step 9 - Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(rf, feature_names, save_path, top_n=20):
    imp = pd.Series(rf.feature_importances_, index=feature_names) \
            .sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    imp[::-1].plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[9] Feature importance plot -> {save_path}")
    return imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    (X_train, X_test, y_train, y_test,
     class_names, feature_names, scaler, le) = prepare_features(df)

    rf, svm, gb = train_models(X_train, y_train)
    voter       = train_voting_ensemble(rf, svm, gb, X_train, y_train)

    # Persist artefacts
    joblib.dump(rf,     os.path.join(RESULTS_DIR, "rf_model.pkl"))
    joblib.dump(svm,    os.path.join(RESULTS_DIR, "svm_model.pkl"))
    joblib.dump(gb,     os.path.join(RESULTS_DIR, "gb_model.pkl"))
    joblib.dump(voter,  os.path.join(RESULTS_DIR, "voting_model.pkl"))
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(RESULTS_DIR, "label_encoder.pkl"))

    # Feature importance
    plot_feature_importance(
        rf, feature_names,
        os.path.join(RESULTS_DIR, "feature_importance.png"),
    )

    # Evaluate every model
    rows = []
    for name, model in [
        ("Random Forest",     rf),
        ("SVM",               svm),
        ("Gradient Boosting", gb),
        ("Voting Ensemble",   voter),
    ]:
        preds = model.predict(X_test)
        r = evaluate_model(y_test, preds, class_names, name, CM_DIR,
                           print_report=False)
        r.pop("misclassifications", None)
        rows.append(r)

    df_out = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df_out.to_csv(os.path.join(RESULTS_DIR, "ml_results.csv"), index=False)
    print("\nML pipeline summary:")
    print(df_out[["model", "accuracy", "precision", "recall", "f1"]]
          .to_string(index=False))

    # Persist processed arrays for PCA pipeline
    np.savez(
        os.path.join(RESULTS_DIR, "clamp_processed.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=np.array(feature_names),
        class_names=np.array(class_names),
    )
    print("\nSaved processed arrays -> clamp_processed.npz")


if __name__ == "__main__":
    main()
