# FDSPROJECT
End-to-end implementation of the Malware Classification project, combining
traditional machine learning on the CLaMP dataset with a CNN on the Malimg
dataset, plus a soft-voting ensemble and a PCA study.
## Requirements

* Python 3.11+
* scikit-learn 1.3
* PyTorch 2.7 with CUDA 11.8 (CPU also works, slower)
* pandas, numpy, matplotlib, seaborn, joblib, python-docx

Install with:

pip install scikit-learn pandas numpy matplotlib seaborn joblib python-docx torch torchvision
## Final Results

| Rank | Model                    | Accuracy | F1     |
|------|--------------------------|----------|--------|
| 1    | Random Forest            | 0.9904   | 0.9904 |
| 2    | Gradient Boosting        | 0.9808   | 0.9808 |
| 3    | CNN (Malimg)             | 0.9781   | 0.9731 |
| 4    | Voting Ensemble          | 0.9779   | 0.9779 |
| 5    | Random Forest + PCA-30   | 0.9703   | 0.9702 |
| 6    | Random Forest + PCA-67   | 0.9693   | 0.9693 |
| 7    | Random Forest + PCA-50   | 0.9683   | 0.9683 |
| 8    | SVM                      | 0.9559   | 0.9558 |
