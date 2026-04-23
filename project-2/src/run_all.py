import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import FEATURES_DIR, FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, TABLES_DIR, ensure_output_dirs
from src.data import load_digits_splits
from src.metrics import classification_metrics, save_confusion_matrix, save_learning_curve
from src.models import AlexNetSmallCNN, ImprovedLeNetCNN, LeNetLikeCNN
from src.reporting import write_readme
from src.training import extract_features, set_seed, train_model


def main() -> None:
    ensure_output_dirs()
    set_seed()
    data = load_digits_splits()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_builders = [
        ("LeNetLikeCNN", LeNetLikeCNN),
        ("ImprovedLeNetCNN", ImprovedLeNetCNN),
        ("AlexNetSmallCNN", AlexNetSmallCNN),
    ]

    rows: list[dict] = []
    trained_models: dict[str, torch.nn.Module] = {}
    for model_name, builder in model_builders:
        result = train_model(builder(n_classes=data.n_classes), data, model_name, device)
        metrics = classification_metrics(data.y_test, result["test_pred"])
        row = {
            "model": model_name,
            **metrics,
            "test_loss": result["test_loss"],
            "best_val_accuracy": result["best_val_accuracy"],
        }
        rows.append(row)
        trained_models[model_name] = result["model"]
        save_confusion_matrix(
            data.y_test,
            result["test_pred"],
            f"{model_name} confusion matrix",
            FIGURES_DIR / f"confusion_matrix_{model_name}.png",
        )
        save_learning_curve(
            result["history"],
            f"{model_name} learning curve",
            FIGURES_DIR / f"learning_curve_{model_name}.png",
        )

    feature_model_name = "AlexNetSmallCNN"
    feature_model = trained_models[feature_model_name]
    x_train_full = np.concatenate([data.x_train, data.x_val], axis=0)
    y_train_full = np.concatenate([data.y_train, data.y_val], axis=0)
    train_features = extract_features(feature_model, x_train_full, device)
    test_features = extract_features(feature_model, data.x_test, device)

    np.save(FEATURES_DIR / "train_features.npy", train_features)
    np.save(FEATURES_DIR / "train_labels.npy", y_train_full)
    np.save(FEATURES_DIR / "test_features.npy", test_features)
    np.save(FEATURES_DIR / "test_labels.npy", data.y_test)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    svm = SVC(kernel="rbf", C=10.0, gamma="scale")
    svm.fit(train_features_scaled, y_train_full)
    hybrid_pred = svm.predict(test_features_scaled)
    hybrid_metrics = classification_metrics(data.y_test, hybrid_pred)
    rows.append(
        {
            "model": "Hybrid_AlexNetFeatures_SVM",
            **hybrid_metrics,
            "test_loss": np.nan,
            "best_val_accuracy": np.nan,
        }
    )
    save_confusion_matrix(
        data.y_test,
        hybrid_pred,
        "Hybrid AlexNet features + SVM confusion matrix",
        FIGURES_DIR / "confusion_matrix_Hybrid_AlexNetFeatures_SVM.png",
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)

    hybrid_summary = {
        "train_features_shape": list(train_features.shape),
        "test_features_shape": list(test_features.shape),
        "train_labels_shape": list(y_train_full.shape),
        "test_labels_shape": list(data.y_test.shape),
        "classification_report": classification_report(data.y_test, hybrid_pred, output_dict=True),
    }
    data_summary = {
        "total_samples": int(len(data.y_train) + len(data.y_val) + len(data.y_test)),
        "train_samples": int(len(data.y_train)),
        "val_samples": int(len(data.y_val)),
        "test_samples": int(len(data.y_test)),
        "n_classes": int(data.n_classes),
    }
    summary = {
        "device": str(device),
        "data": data_summary,
        "metrics": metrics_df.to_dict(orient="records"),
        "hybrid": hybrid_summary,
    }
    (REPORTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(PROJECT_ROOT, metrics_df, hybrid_summary, data_summary)

    print(metrics_df.to_string(index=False))
    print(f"Feature files written to {FEATURES_DIR}")


if __name__ == "__main__":
    main()
