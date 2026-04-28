"""
generate_comparison.py
======================
Assemble the final comparison artefacts:

    * comparison_table.csv  - ranked metrics across every model variant
      (ML models, Voting Ensemble, PCA-30/50/100 on Random Forest, CNN)
    * model_comparison_chart.png - grouped bar chart of the same.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))


def load_ml_results():
    return pd.read_csv(os.path.join(RESULTS_DIR, "ml_results.csv"))


def load_cnn_results():
    path = os.path.join(RESULTS_DIR, "cnn_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_pca_results():
    """Turn pca_comparison.csv into rows with accuracy & F1 per variant."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "pca_comparison.csv"))
    rows = []
    for _, r in df.iterrows():
        name = "Random Forest (no PCA)" if r["variant"] == "No PCA" \
               else f"Random Forest + {r['variant']}"
        rows.append({
            "model":     name,
            "accuracy":  r["accuracy"],
            "precision": np.nan,
            "recall":    np.nan,
            "f1":        r["f1"],
        })
    return pd.DataFrame(rows)


def main():
    frames = [load_ml_results()]

    pca = load_pca_results()
    # Keep only the PCA rows, not the baseline (already in ml_results)
    pca = pca[pca["model"] != "Random Forest (no PCA)"]
    frames.append(pca)

    cnn = load_cnn_results()
    if cnn is not None:
        frames.append(cnn)

    df = pd.concat(frames, ignore_index=True)
    for c in ("accuracy", "precision", "recall", "f1"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df[["model", "accuracy", "precision", "recall", "f1"]].to_csv(
        out_csv, index=False)
    print(f"Saved -> {out_csv}")
    print(df[["model", "accuracy", "precision", "recall", "f1"]]
          .to_string(index=False))

    # --- Bar chart --------------------------------------------------------
    metrics  = ["accuracy", "precision", "recall", "f1"]
    plot_df  = df.dropna(subset=metrics, how="all").copy()
    models   = plot_df["model"].tolist()
    x        = np.arange(len(models))
    width    = 0.2

    plt.figure(figsize=(max(10, len(models) * 1.1), 6))
    for i, m in enumerate(metrics):
        vals = plot_df[m].fillna(0).values
        plt.bar(x + (i - 1.5) * width, vals, width, label=m.capitalize())

    plt.xticks(x, models, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison - All Variants")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
