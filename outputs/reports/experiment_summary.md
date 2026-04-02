# Deney Ozeti

## Genel Sonuc
Secilen model `wide_scaled` oldu. Validation balanced accuracy `0.8601`, test accuracy `0.8167` ve test ROC-AUC `0.8549` olarak olculdu.

## NumPy Deneyleri
| run_name              | architecture   | use_scaler   |   train_fraction |   val_accuracy |   val_balanced_accuracy |   test_accuracy |   test_f1 |   test_roc_auc |
|:----------------------|:---------------|:-------------|-----------------:|---------------:|------------------------:|----------------:|----------:|---------------:|
| wide_scaled           | 12-16-1        | True         |             1    |       0.866667 |                0.860077 |        0.816667 |  0.666667 |       0.854942 |
| baseline_scaled       | 12-8-1         | True         |             1    |       0.833333 |                0.821566 |        0.816667 |  0.666667 |       0.870347 |
| deep_scaled           | 12-24-12-1     | True         |             1    |       0.833333 |                0.807445 |        0.833333 |  0.666667 |       0.851091 |
| deep_scaled_l2        | 12-24-12-1     | True         |             1    |       0.833333 |                0.807445 |        0.833333 |  0.666667 |       0.851091 |
| deep_scaled_l2_data75 | 12-24-12-1     | True         |             0.75 |       0.8      |                0.783055 |        0.816667 |  0.666667 |       0.866496 |
| deep_scaled_l2_data50 | 12-24-12-1     | True         |             0.5  |       0.783333 |                0.728498 |        0.8      |  0.647059 |       0.820282 |
| baseline_raw          | 12-8-1         | False        |             1    |       0.683333 |                0.5      |        0.683333 |  0        |       0.5      |

## Backend Karsilastirmasi
| run_name        | backend   |   test_accuracy |   test_balanced_accuracy |   test_f1 |   test_roc_auc |
|:----------------|:----------|----------------:|-------------------------:|----------:|---------------:|
| baseline_scaled | numpy     |        0.816667 |                 0.752888 |  0.666667 |       0.870347 |
| baseline_scaled | sklearn   |        0.816667 |                 0.752888 |  0.666667 |       0.870347 |
| baseline_scaled | pytorch   |        0.816667 |                 0.752888 |  0.666667 |       0.870347 |
| wide_scaled     | numpy     |        0.816667 |                 0.752888 |  0.666667 |       0.854942 |
| wide_scaled     | sklearn   |        0.816667 |                 0.752888 |  0.666667 |       0.854942 |
| wide_scaled     | pytorch   |        0.816667 |                 0.752888 |  0.666667 |       0.854942 |
| deep_scaled_l2  | numpy     |        0.833333 |                 0.750963 |  0.666667 |       0.851091 |
| deep_scaled_l2  | sklearn   |        0.833333 |                 0.750963 |  0.666667 |       0.851091 |
| deep_scaled_l2  | pytorch   |        0.833333 |                 0.750963 |  0.666667 |       0.852375 |