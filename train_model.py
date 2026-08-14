from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-train the insurance approval model and export reproducible artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional decision threshold override. If omitted, the script uses "
            "configuration/parameter/ElasticNet_youden_threshold.json."
        ),
    )
    return parser.parse_args()


def build_woe_features(
    x_frame: pd.DataFrame, y_series: pd.Series
) -> tuple[pd.DataFrame, dict[str, OptimalBinning], dict[str, OptimalBinning]]:
    fitted_binning_num: dict[str, OptimalBinning] = {}
    fitted_binning_cat: dict[str, OptimalBinning] = {}

    numerical_features = x_frame.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x_frame.select_dtypes(include=["object", "category"]).columns.tolist()

    x_woe_num = pd.DataFrame(index=x_frame.index)
    x_woe_cat = pd.DataFrame(index=x_frame.index)

    for feature in numerical_features:
        optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        optb.fit(x_frame[feature].fillna(-999).values, y_series.values)
        fitted_binning_num[feature] = optb
        x_woe_num[f"{feature}_woe"] = optb.transform(
            x_frame[feature].fillna(-999).values, metric="woe"
        )

    for feature in categorical_features:
        optb = OptimalBinning(name=feature, dtype="categorical", solver="cp")
        optb.fit(x_frame[feature].astype(str).fillna("missing").values, y_series.values)
        fitted_binning_cat[feature] = optb
        x_woe_cat[f"{feature}_woe"] = optb.transform(
            x_frame[feature].astype(str).fillna("missing").values, metric="woe"
        )

    x_train_woe = pd.concat([x_woe_num, x_woe_cat], axis=1)
    return x_train_woe, fitted_binning_num, fitted_binning_cat


def transform_woe(
    x_frame: pd.DataFrame,
    binning_num: dict[str, OptimalBinning],
    binning_cat: dict[str, OptimalBinning],
) -> pd.DataFrame:
    x_num = pd.DataFrame(index=x_frame.index)
    x_cat = pd.DataFrame(index=x_frame.index)

    for feature, optb in binning_num.items():
        x_num[f"{feature}_woe"] = optb.transform(
            x_frame[feature].fillna(-999).values, metric="woe"
        )

    for feature, optb in binning_cat.items():
        x_cat[f"{feature}_woe"] = optb.transform(
            x_frame[feature].astype(str).fillna("missing").values, metric="woe"
        )

    return pd.concat([x_num, x_cat], axis=1)


def build_woe_summary(
    numerical_features: list[str],
    fitted_binning_num: dict[str, OptimalBinning],
    fitted_binning_cat: dict[str, OptimalBinning],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_binnings = {**fitted_binning_num, **fitted_binning_cat}

    for feature, optb in all_binnings.items():
        binning_table = optb.binning_table.build().copy()
        bin_labels = binning_table["Bin"].astype(str)
        regular_bins = binning_table.loc[~bin_labels.isin(["Special", "Missing", "Totals"])]
        rows.append(
            {
                "feature": feature,
                "feature_type": "numerical" if feature in numerical_features else "categorical",
                "n_regular_bins": int(len(regular_bins)),
                "iv": float(binning_table["IV"].iloc[-1]),
            }
        )

    return pd.DataFrame(rows).sort_values(["iv", "n_regular_bins"], ascending=[False, False])


def classification_metrics(
    y_true: pd.Series, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_dir = project_root / "configuration"
    parameter_dir = config_dir / "parameter"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    train_path = project_root / "data" / "insurance_train.csv"
    test_path = project_root / "data" / "insurance_test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    target_col = "claim_status"

    x_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    numerical_features = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    x_train_woe, fitted_binning_num, fitted_binning_cat = build_woe_features(x_train, y_train)
    woe_summary = build_woe_summary(numerical_features, fitted_binning_num, fitted_binning_cat)

    features_to_drop = ["support_interactions", "channel", "person_age"]
    iv_series = pd.Series(
        {
            feat: b.binning_table.build()["IV"].iloc[-1]
            for feat, b in {**fitted_binning_num, **fitted_binning_cat}.items()
        }
    )
    selected_features = iv_series[iv_series >= 0.02].index.tolist()
    selected_features_filtered = [f for f in selected_features if f not in features_to_drop]
    selected_features_woe = [f"{feat}_woe" for feat in selected_features_filtered]
    x_train_woe = x_train_woe[selected_features_woe]

    feature_selection_summary = woe_summary.copy()
    feature_selection_summary["passed_iv_threshold"] = feature_selection_summary["feature"].isin(
        selected_features
    )
    feature_selection_summary["manually_removed"] = feature_selection_summary["feature"].isin(
        features_to_drop
    )
    feature_selection_summary["kept_in_final_model"] = feature_selection_summary["feature"].isin(
        selected_features_filtered
    )
    feature_selection_summary["status"] = np.select(
        [
            feature_selection_summary["kept_in_final_model"],
            feature_selection_summary["manually_removed"],
            feature_selection_summary["passed_iv_threshold"],
        ],
        [
            "kept in final model",
            "removed after review",
            "passed threshold but excluded",
        ],
        default="below IV threshold",
    )

    x_test_woe = transform_woe(df_test, fitted_binning_num, fitted_binning_cat)
    x_test_woe = x_test_woe[selected_features_woe]

    with open(parameter_dir / "ElasticNet_best_params.json", "r", encoding="utf-8") as file:
        best_params = json.load(file)

    if args.threshold is None:
        with open(
            parameter_dir / "ElasticNet_youden_threshold.json", "r", encoding="utf-8"
        ) as file:
            threshold = float(json.load(file)["youden_threshold"])
        threshold_source = "artifact_json"
    else:
        threshold = float(args.threshold)
        threshold_source = "cli_override"

    pipeline = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42)),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.set_params(**best_params)
    pipeline.fit(x_train_woe, y_train)

    joblib.dump(pipeline, config_dir / "classification_model.pkl")

    train_scores = pipeline.predict_proba(x_train_woe)[:, 1]
    train_metrics = classification_metrics(y_train, train_scores, threshold)

    test_scores = pipeline.predict_proba(x_test_woe)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)
    prediction_ids = (
        df_test["observation_id"] if "observation_id" in df_test.columns else df_test.index + 1
    )
    predictions = pd.DataFrame(
        {
            "observation_id": prediction_ids,
            "predicted_claim_status": test_predictions,
            "predicted_probability": test_scores,
        }
    )
    predictions.to_csv(output_dir / "insurance_predictions.csv", index=False)

    woe_summary.to_csv(output_dir / "woe_feature_summary.csv", index=False)
    feature_selection_summary.to_csv(output_dir / "feature_selection_summary.csv", index=False)

    run_summary = {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "artifacts": {
            "model": str((config_dir / "classification_model.pkl").relative_to(project_root)),
            "predictions": str((output_dir / "insurance_predictions.csv").relative_to(project_root)),
            "woe_summary": str((output_dir / "woe_feature_summary.csv").relative_to(project_root)),
            "feature_selection_summary": str(
                (output_dir / "feature_selection_summary.csv").relative_to(project_root)
            ),
        },
    }
    with open(output_dir / "training_run_summary.json", "w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)

    print("Training complete.")
    print(f"Threshold used: {threshold} ({threshold_source})")
    print(f"Saved predictions to: {output_dir / 'insurance_predictions.csv'}")
    print(f"Saved run summary to: {output_dir / 'training_run_summary.json'}")


if __name__ == "__main__":
    main()
