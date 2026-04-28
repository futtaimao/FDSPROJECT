"""
Master driver - runs every phase in order:

    1. ML pipeline    (RF, SVM, GB, Voting Ensemble)
    2. PCA experiments
    3. CNN pipeline
    4. Final comparison table and chart
"""

import os
import sys
import runpy

HERE    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
sys.path.insert(0, SCRIPTS)


def run(name):
    print("\n" + "=" * 70)
    print(f">>> {name}")
    print("=" * 70)
    runpy.run_path(os.path.join(SCRIPTS, name), run_name="__main__")


if __name__ == "__main__":
    run("ml_pipeline.py")
    run("pca_analysis.py")
    run("cnn_pipeline.py")
    run("generate_comparison.py")
    print("\nAll pipelines complete. Results in implementation/results/")
