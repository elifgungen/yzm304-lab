from __future__ import annotations

import pandas as pd

from src.config import (
    BACKEND_COMPARISON_RUNS,
    EXPECTED_OUTPUTS,
    FEATURE_COLUMNS,
    FIGURE_DIR,
    GLOBAL_SEED,
    NUMPY_EXPERIMENTS,
    OUTPUT_DIR,
    REPORT_DIR,
    TABLE_DIR,
)
from src.dataset import export_dataset_artifacts, prepare_features
from src.metrics import confusion_values
from src.numpy_mlp import NumpyMLP
from src.pytorch_backend import fit_torch_model
from src.reporting import (
    build_experiment_summary,
    build_run_summary,
    build_selected_model_report,
    build_traceability_matrix,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_data_fraction_comparison,
    plot_learning_curves,
    save_dataframe,
    save_json,
    save_markdown,
)
from src.sklearn_backend import SklearnMLPAdapter
from src.weights import ensure_weight_bundle


def _metrics_row(
    *,
    run_name: str,
    backend: str,
    architecture: str,
    hidden_activation: str,
    learning_rate: float,
    l2_lambda: float,
    use_scaler: bool,
    train_fraction: float,
    epochs: int,
    parameter_count: int,
    train_loss: float,
    val_loss: float,
    test_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    weight_norm: float,
    notes: str,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "backend": backend,
        "architecture": architecture,
        "hidden_activation": hidden_activation,
        "learning_rate": learning_rate,
        "l2_lambda": l2_lambda,
        "use_scaler": use_scaler,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "n_steps": epochs,
        "parameter_count": parameter_count,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "val_precision": val_metrics["precision"],
        "test_precision": test_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "val_recall": val_metrics["recall"],
        "test_recall": test_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_f1": val_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "train_specificity": train_metrics["specificity"],
        "val_specificity": val_metrics["specificity"],
        "test_specificity": test_metrics["specificity"],
        "train_roc_auc": train_metrics["roc_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "weight_l2_norm": weight_norm,
        "notes": notes,
    }


def run_numpy_experiments(bundle) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    history_map: dict[str, pd.DataFrame] = {}
    run_cache: dict[str, dict[str, object]] = {}
    for spec in NUMPY_EXPERIMENTS:
        prepared = prepare_features(
            bundle,
            FEATURE_COLUMNS,
            use_scaler=spec.use_scaler,
            train_fraction=spec.train_fraction,
        )
        initial_weights, initial_biases = ensure_weight_bundle(
            spec.architecture,
            spec.hidden_activation,
            seed=GLOBAL_SEED,
        )
        model = NumpyMLP(
            initial_weights,
            initial_biases,
            hidden_activation=spec.hidden_activation,
            learning_rate=spec.learning_rate,
            l2_lambda=spec.l2_lambda,
        )
        result = model.fit(
            prepared.X_train.to_numpy(dtype=float),
            prepared.y_train.to_numpy(dtype=int),
            prepared.X_val.to_numpy(dtype=float),
            prepared.y_val.to_numpy(dtype=int),
            prepared.X_test.to_numpy(dtype=float),
            prepared.y_test.to_numpy(dtype=int),
            epochs=spec.epochs,
        )
        history_map[spec.name] = result.history
        metric_rows.append(
            _metrics_row(
                run_name=spec.name,
                backend="numpy",
                architecture=spec.architecture_label,
                hidden_activation=spec.hidden_activation,
                learning_rate=spec.learning_rate,
                l2_lambda=spec.l2_lambda,
                use_scaler=spec.use_scaler,
                train_fraction=spec.train_fraction,
                epochs=spec.epochs,
                parameter_count=spec.parameter_count,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
                test_loss=result.test_loss,
                train_metrics=result.train_metrics,
                val_metrics=result.val_metrics,
                test_metrics=result.test_metrics,
                weight_norm=result.weight_norm,
                notes=spec.notes,
            )
        )
        run_cache[spec.name] = {"spec": spec, "prepared": prepared, "result": result}
    metrics_frame = pd.DataFrame(metric_rows)
    save_dataframe(metrics_frame, TABLE_DIR / "numpy_experiment_metrics.csv")
    return metrics_frame, history_map, run_cache


def build_model_selection(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    selection_frame = metrics_frame.sort_values(
        ["val_balanced_accuracy", "n_steps", "parameter_count", "val_roc_auc"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    selection_frame["selection_rank"] = selection_frame.index + 1
    save_dataframe(selection_frame, TABLE_DIR / "model_selection.csv")
    return selection_frame


def build_backend_comparison(run_cache: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    backend_rows: list[dict[str, object]] = []
    backend_cache: dict[str, dict[str, object]] = {}
    for spec in BACKEND_COMPARISON_RUNS:
        cached = run_cache[spec.name]
        prepared = cached["prepared"]
        numpy_result = cached["result"]

        backend_rows.append(
            _metrics_row(
                run_name=spec.name,
                backend="numpy",
                architecture=spec.architecture_label,
                hidden_activation=spec.hidden_activation,
                learning_rate=spec.learning_rate,
                l2_lambda=spec.l2_lambda,
                use_scaler=spec.use_scaler,
                train_fraction=spec.train_fraction,
                epochs=spec.epochs,
                parameter_count=spec.parameter_count,
                train_loss=numpy_result.train_loss,
                val_loss=numpy_result.val_loss,
                test_loss=numpy_result.test_loss,
                train_metrics=numpy_result.train_metrics,
                val_metrics=numpy_result.val_metrics,
                test_metrics=numpy_result.test_metrics,
                weight_norm=numpy_result.weight_norm,
                notes="NumPy referans egitimi.",
            )
        )
        backend_cache[f"{spec.name}:numpy"] = {
            "spec": spec,
            "prepared": prepared,
            "result": numpy_result,
        }

        weights, biases = ensure_weight_bundle(spec.architecture, spec.hidden_activation, seed=GLOBAL_SEED)
        X_train = prepared.X_train.to_numpy(dtype=float)
        y_train = prepared.y_train.to_numpy(dtype=int)
        X_val = prepared.X_val.to_numpy(dtype=float)
        y_val = prepared.y_val.to_numpy(dtype=int)
        X_test = prepared.X_test.to_numpy(dtype=float)
        y_test = prepared.y_test.to_numpy(dtype=int)

        sklearn_result = SklearnMLPAdapter(
            spec.architecture,
            hidden_activation=spec.hidden_activation,
            learning_rate=spec.learning_rate,
            l2_lambda=spec.l2_lambda,
        ).fit(
            weights,
            biases,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            epochs=spec.epochs,
        )
        backend_rows.append(
            _metrics_row(
                run_name=spec.name,
                backend="sklearn",
                architecture=spec.architecture_label,
                hidden_activation=spec.hidden_activation,
                learning_rate=spec.learning_rate,
                l2_lambda=spec.l2_lambda,
                use_scaler=spec.use_scaler,
                train_fraction=spec.train_fraction,
                epochs=spec.epochs,
                parameter_count=spec.parameter_count,
                train_loss=sklearn_result.train_loss,
                val_loss=sklearn_result.val_loss,
                test_loss=sklearn_result.test_loss,
                train_metrics=sklearn_result.train_metrics,
                val_metrics=sklearn_result.val_metrics,
                test_metrics=sklearn_result.test_metrics,
                weight_norm=sklearn_result.weight_norm,
                notes="Ayni agirliklarla sklearn MLPClassifier tekrari.",
            )
        )
        backend_cache[f"{spec.name}:sklearn"] = {
            "spec": spec,
            "prepared": prepared,
            "result": sklearn_result,
        }

        torch_result = fit_torch_model(
            spec.architecture,
            hidden_activation=spec.hidden_activation,
            learning_rate=spec.learning_rate,
            l2_lambda=spec.l2_lambda,
            weights=weights,
            biases=biases,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=spec.epochs,
        )
        backend_rows.append(
            _metrics_row(
                run_name=spec.name,
                backend="pytorch",
                architecture=spec.architecture_label,
                hidden_activation=spec.hidden_activation,
                learning_rate=spec.learning_rate,
                l2_lambda=spec.l2_lambda,
                use_scaler=spec.use_scaler,
                train_fraction=spec.train_fraction,
                epochs=spec.epochs,
                parameter_count=spec.parameter_count,
                train_loss=torch_result.train_loss,
                val_loss=torch_result.val_loss,
                test_loss=torch_result.test_loss,
                train_metrics=torch_result.train_metrics,
                val_metrics=torch_result.val_metrics,
                test_metrics=torch_result.test_metrics,
                weight_norm=torch_result.weight_norm,
                notes="Ayni agirliklarla PyTorch tekrari.",
            )
        )
        backend_cache[f"{spec.name}:pytorch"] = {
            "spec": spec,
            "prepared": prepared,
            "result": torch_result,
        }
    backend_frame = pd.DataFrame(backend_rows)
    save_dataframe(backend_frame, TABLE_DIR / "backend_comparison_metrics.csv")
    return backend_frame, backend_cache


def ensure_selected_backend_runs(
    selected_spec,
    run_cache: dict[str, dict[str, object]],
    backend_cache: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    selected_runs: dict[str, dict[str, object]] = {}
    prepared = run_cache[selected_spec.name]["prepared"]
    selected_runs["numpy"] = run_cache[selected_spec.name]

    for backend_name in ("sklearn", "pytorch"):
        cache_key = f"{selected_spec.name}:{backend_name}"
        if cache_key in backend_cache:
            selected_runs[backend_name] = backend_cache[cache_key]
            continue
        weights, biases = ensure_weight_bundle(
            selected_spec.architecture,
            selected_spec.hidden_activation,
            seed=GLOBAL_SEED,
        )
        X_train = prepared.X_train.to_numpy(dtype=float)
        y_train = prepared.y_train.to_numpy(dtype=int)
        X_val = prepared.X_val.to_numpy(dtype=float)
        y_val = prepared.y_val.to_numpy(dtype=int)
        X_test = prepared.X_test.to_numpy(dtype=float)
        y_test = prepared.y_test.to_numpy(dtype=int)
        if backend_name == "sklearn":
            result = SklearnMLPAdapter(
                selected_spec.architecture,
                hidden_activation=selected_spec.hidden_activation,
                learning_rate=selected_spec.learning_rate,
                l2_lambda=selected_spec.l2_lambda,
            ).fit(
                weights,
                biases,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                epochs=selected_spec.epochs,
            )
        else:
            result = fit_torch_model(
                selected_spec.architecture,
                hidden_activation=selected_spec.hidden_activation,
                learning_rate=selected_spec.learning_rate,
                l2_lambda=selected_spec.l2_lambda,
                weights=weights,
                biases=biases,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                epochs=selected_spec.epochs,
            )
        selected_runs[backend_name] = {
            "spec": selected_spec,
            "prepared": prepared,
            "result": result,
        }
    return selected_runs


def build_reference_runs(
    specs,
    run_cache: dict[str, dict[str, object]],
    backend_cache: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    reference_runs: dict[str, dict[str, dict[str, object]]] = {}
    for spec in specs:
        spec_runs = ensure_selected_backend_runs(spec, run_cache, backend_cache)
        reference_runs[spec.name] = spec_runs
    return reference_runs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = export_dataset_artifacts()
    plot_class_distribution(bundle.dataframe, FIGURE_DIR / "class_distribution.png")

    metrics_frame, history_map, run_cache = run_numpy_experiments(bundle)
    plot_learning_curves(
        history_map,
        FIGURE_DIR / "numpy_learning_curves.png",
        ["baseline_raw", "baseline_scaled", "wide_scaled", "deep_scaled_l2"],
    )
    plot_data_fraction_comparison(metrics_frame, FIGURE_DIR / "data_fraction_comparison.png")

    selection_frame = build_model_selection(metrics_frame)
    selected_row = selection_frame.iloc[0]
    selected_spec = next(spec for spec in NUMPY_EXPERIMENTS if spec.name == selected_row["run_name"])

    backend_frame, backend_cache = build_backend_comparison(run_cache)
    selected_backend_runs = ensure_selected_backend_runs(selected_spec, run_cache, backend_cache)
    reference_runs = build_reference_runs(BACKEND_COMPARISON_RUNS, run_cache, backend_cache)

    labels = bundle.target_names
    for backend_name, payload in selected_backend_runs.items():
        prepared = payload["prepared"]
        result = payload["result"]
        confusion = confusion_values(
            prepared.y_test.to_numpy(dtype=int),
            result.test_predictions,
        )
        plot_confusion_matrix(
            confusion,
            labels,
            FIGURE_DIR / f"confusion_matrix_selected_{backend_name}.png",
            title=f"Selected model - {backend_name}",
        )
    for spec_name, backend_runs in reference_runs.items():
        for backend_name, payload in backend_runs.items():
            prepared = payload["prepared"]
            result = payload["result"]
            confusion = confusion_values(
                prepared.y_test.to_numpy(dtype=int),
                result.test_predictions,
            )
            plot_confusion_matrix(
                confusion,
                labels,
                FIGURE_DIR / f"confusion_matrix_{spec_name}_{backend_name}.png",
                title=f"{spec_name} - {backend_name}",
            )

    save_markdown(
        build_selected_model_report(selected_row, metrics_frame, backend_frame),
        REPORT_DIR / "selected_model_report.md",
    )
    save_markdown(
        build_experiment_summary(metrics_frame, backend_frame, selected_row),
        REPORT_DIR / "experiment_summary.md",
    )
    save_markdown(build_traceability_matrix(), REPORT_DIR / "traceability_matrix.md")
    save_json(
        build_run_summary(metrics_frame, backend_frame, selected_row),
        REPORT_DIR / "run_summary.json",
    )
    save_json(
        {
            "selected_model": selected_row.to_dict(),
            "numpy_runs": metrics_frame.to_dict(orient="records"),
            "backend_runs": backend_frame.to_dict(orient="records"),
        },
        REPORT_DIR / "experiment_summary.json",
    )

    missing = [str(path) for path in EXPECTED_OUTPUTS if not path.exists()]
    if missing:
        raise RuntimeError(f"Eksik cikti olustu: {missing}")

    print(f"Secilen model: {selected_row['run_name']}")
    print(f"Validation balanced accuracy: {selected_row['val_balanced_accuracy']:.4f}")
    print(f"Test accuracy: {selected_row['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
