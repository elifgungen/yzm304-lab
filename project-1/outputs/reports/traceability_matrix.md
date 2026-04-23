# Izlenebilirlik Matrisi

| PDF Gereksinimi | Kanit | Durum |
| --- | --- | --- |
| Ikili siniflandirma verisi | `https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv` ve `data/raw/heart_failure_clinical_records_dataset.csv` | Tamam |
| Veri analizi ve on isleme | `outputs/figures/class_distribution.png`, `src/dataset.py` | Tamam |
| Laboratuvar temel modeli | `baseline_raw`, `baseline_scaled`, `src/numpy_mlp.py` | Tamam |
| Overfitting / underfitting incelemesi | `outputs/figures/numpy_learning_curves.png` | Tamam |
| Cok katmanli model / veri miktari / regülarizasyon | `deep_scaled*`, `outputs/figures/data_fraction_comparison.png` | Tamam |
| Class tabanli temiz yapi | `src/numpy_mlp.py`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |
| Model secimi | `outputs/tables/model_selection.csv`, `outputs/reports/selected_model_report.md` | Tamam |
| sklearn ve PyTorch tekrar yazimi | `src/sklearn_backend.py`, `src/pytorch_backend.py`, `outputs/tables/backend_comparison_metrics.csv` | Tamam |
| Confusion matrix ve temel metrikler | `outputs/figures/confusion_matrix_selected_*`, `outputs/tables/*.csv` | Tamam |
| Ayni split / agirlik / SGD | `data/splits/split_manifest.json`, `data/weights/*.npz`, backend modulleri | Tamam |
| BCE loss | `src/numpy_mlp.py`, `src/pytorch_backend.py`, `src/metrics.py` | Tamam |
| Uygun repo hiyerarsisi | `src/`, `data/`, `outputs/`, `tests/` | Tamam |
| IMRAD README | `README.md` | Tamam |