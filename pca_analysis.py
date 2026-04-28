"""
pca_analysis.py
===============
Phase: Principal Component Analysis on the CLaMP features.

    Step 10 - Choose the number of components to keep.
    Step 11 - Apply PCA at 30, 50, 100 components and retrain
              Random Forest on the reduced space.  Compare
              accuracy and F1 against the no-PCA baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition   import PCA
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

HERE         = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.abspath(os.path.join(HERE, "..", "results"))
RANDOM_STATE = 42


def load_processed():
    path = os.path.join(RESULTS_DIR, "clamp_processed.npz")
    z = np.load(path, allow_pickle=True)
    return (z["X_train"], z["X_test"], z["y_train"], z["y_test"],
            list(z["class_names"]))


# ---------------------------------------------------------------------------
# Step 10 - Explained-variance curve
# ---------------------------------------------------------------------------
def explained_variance_curve(X_train):
    pca = PCA().fit(X_train)
    cum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(cum) + 1), cum, marker="o", markersize=3)
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.axhline(0.99, color="green", linestyle="--", label="99% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA - Explained Variance vs Components")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pca_explained_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    idx95 = int(np.searchsorted(cum, 0.95) + 1)
    idx99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"[10] 95% variance captured by {idx95} components, "
          f"99% by {idx99}")
    return cum


# ---------------------------------------------------------------------------
# Step 11 - Baseline + PCA variants
# ---------------------------------------------------------------------------
def evaluate_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return (
        accuracy_score(y_test, preds),
        f1_score(y_test, preds, average="weighted", zero_division=0),
    )


def run_pca_experiments(X_train, X_test, y_train, y_test):
    results = []

    # Baseline (no PCA)
    acc, f1 = evaluate_rf(X_train, X_test, y_train, y_test)
    results.append({"variant": "No PCA",
                    "n_components": X_train.shape[1],
                    "accuracy": acc, "f1": f1})
    print(f"Baseline (no PCA, {X_train.shape[1]} feats): "
          f"acc={acc:.4f}, f1={f1:.4f}")

    for k in (30, 50, 100):
        k_eff = min(k, X_train.shape[1])
        pca   = PCA(n_components=k_eff, random_state=RANDOM_STATE)
        Xtr   = pca.fit_transform(X_train)
        Xte   = pca.transform(X_test)
        acc, f1 = evaluate_rf(Xtr, Xte, y_train, y_test)
        results.append({"variant": f"PCA-{k_eff}",
                        "n_components": k_eff,
                        "accuracy": acc, "f1": f1})
        print(f"PCA-{k_eff}: acc={acc:.4f}, f1={f1:.4f}")

    return pd.DataFrame(results)


def main():
    X_train, X_test, y_train, y_test, _ = load_processed()
    explained_variance_curve(X_train)

    df = run_pca_experiments(X_train, X_test, y_train, y_test)
    out = os.path.join(RESULTS_DIR, "pca_comparison.csv")
    df.to_csv(out, index=False)
    print(f"\nPCA comparison saved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
