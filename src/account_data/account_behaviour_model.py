import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classic ML models on the random labeled dataset.")
    parser.add_argument(
        "--data-csv",
        default="./src/features_labeled.csv",
        help="Path to the labeled CSV file (Author, label, features).",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="The binary label column. Expected values: B/H.",
    )
    parser.add_argument(
        "--author-col",
        default="author",
        help="Author column to exclude from model features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the labeled data to keep for the holdout test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the test split/cross-validation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for model comparison.",
    )
    parser.add_argument(
        "--predict-csv",
        default="./src/features_ml_labeled.csv",
        help="Path to save the full feature dataset with a predicted label column.",
    )
    parser.add_argument(
        "--model-output",
        default="./src/best_account_behaviour_model.joblib",
        help="Path to save the best trained model pipeline for future predictions.",
    )
    parser.add_argument(
        "--predict-full",
        action="store_true",
        help="After training, predict labels for every row in ./src/features.csv and save them to --predict-csv.",
    )
    parser.add_argument(
        "--feature-importance-output",
        default="./src/account_feature_importance_by_group.csv",
        help="Path to save grouped permutation feature importance results.",
    )
    parser.add_argument(
        "--individual-feature-importance-output",
        default="./src/account_feature_importance.csv",
        help="Path to save individual permutation feature importance results.",
    )
    parser.add_argument(
        "--feature-importance-plot-output",
        default="./src/account_top_15_feature_importance.png",
        help="Path to save the top-15 feature importance plot.",
    )
    return parser.parse_args()


def load_dataset(data_csv: str, label_col: str, author_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_csv}")
    if author_col in df.columns:
        df = df.drop(columns=[author_col])

    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    valid_labels = {"B", "H"}
    invalid = sorted(df.loc[~df[label_col].isin(valid_labels), label_col].unique().tolist())
    if invalid:
        raise ValueError(f"Unexpected label values found: {invalid}. Expected only B and H.")

    y = df[label_col].map({"B": 1, "H": 0}).astype(int)
    X = df.drop(columns=[label_col]).select_dtypes(include=["number"]).copy()
    X = X.drop(columns=X.columns[X.isna().all()].tolist(), errors="ignore")
    if X.empty:
        raise ValueError("No numeric feature columns were found in the labeled CSV file.")

    return X, y


def compute_sample_weights(y: pd.Series) -> pd.Series:
    counts = y.value_counts(normalize=False)
    n_total = len(y)
    weights = pd.Series(1.0, index=y.index)
    for label, count in counts.items():
        weights.loc[y == label] = n_total / (len(counts) * count)
    return weights


def build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def make_pipeline(model: object) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def feature_groups(feature_columns: List[str]) -> Dict[str, List[str]]:
    """Assign every model feature to its literature-defined feature group."""
    santia_features = {"average_response_time", "maximum_daily_comments"}
    chu_features = {"ff_ratio"}
    cdfa_features = {
        column for column in feature_columns if "cdfa" in column.lower()
    }

    unknown_santia = santia_features - set(feature_columns)
    unknown_chu = chu_features - set(feature_columns)
    if unknown_santia or unknown_chu:
        missing = sorted(unknown_santia | unknown_chu)
        raise ValueError(f"Required group features are missing from the dataset: {missing}")

    assigned = santia_features | chu_features | cdfa_features
    kim_features = set(feature_columns) - assigned

    groups = {
        "Santia et al.": [
            column for column in feature_columns if column in santia_features
        ],
        "Chu et al.": [
            column for column in feature_columns if column in chu_features
        ],
        "Kim et al.": [
            column for column in feature_columns if column in kim_features
        ],
        "CDFA": [
            column for column in feature_columns if column in cdfa_features
        ],
    }
    if set().union(*groups.values()) != set(feature_columns):
        raise RuntimeError("Feature group assignment did not cover every model feature.")
    return groups


def grouped_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: Dict[str, List[str]],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure F1 decrease after jointly permuting each feature group."""
    baseline_f1 = f1_score(y_test, model.predict(X_test), zero_division=0)
    rng = np.random.default_rng(random_state)
    rows = []

    for group_name, columns in groups.items():
        importances = []
        for _ in range(n_repeats):
            shuffled = X_test.copy()
            permutation = rng.permutation(len(X_test))
            shuffled.loc[:, columns] = X_test.iloc[permutation][columns].to_numpy()
            shuffled_f1 = f1_score(
                y_test, model.predict(shuffled), zero_division=0
            )
            importances.append(baseline_f1 - shuffled_f1)

        rows.append(
            {
                "Group": group_name,
                "Features": ", ".join(columns),
                "Feature Count": len(columns),
                "Importance": float(pd.Series(importances).mean()),
                "Importance Std": float(pd.Series(importances).std(ddof=0)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def individual_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure F1 decrease after permuting each feature independently."""
    baseline_f1 = f1_score(y_test, model.predict(X_test), zero_division=0)
    rng = np.random.default_rng(random_state)
    rows = []

    for feature in X_test.columns:
        importances = []
        for _ in range(n_repeats):
            shuffled = X_test.copy()
            permutation = rng.permutation(len(X_test))
            shuffled[feature] = X_test.iloc[permutation][feature].to_numpy()
            shuffled_f1 = f1_score(
                y_test, model.predict(shuffled), zero_division=0
            )
            importances.append(baseline_f1 - shuffled_f1)

        rows.append(
            {
                "Feature": feature,
                "Importance": float(np.mean(importances)),
                "Importance Std": float(np.std(importances)),
            }
        )

    importance_df = pd.DataFrame(rows)
    importance_df["Nonnegative Importance"] = importance_df["Importance"].clip(lower=0)
    total_importance = importance_df["Nonnegative Importance"].sum()
    if total_importance > 0:
        importance_df["Importance %"] = (
            importance_df["Nonnegative Importance"] / total_importance * 100
        )
    else:
        importance_df["Importance %"] = 0.0

    return (
        importance_df.drop(columns=["Nonnegative Importance"])
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def save_top_feature_importance_plot(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: str,
    top_n: int = 15,
) -> None:
    """Save a horizontal bar chart for the top individual features."""
    top_features = importance_df.head(top_n).sort_values("Importance %")
    figure_height = max(6, 0.4 * len(top_features) + 1.5)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    axis.barh(top_features["Feature"], top_features["Importance %"], color="#4472C4")
    axis.set_title(f"Top {len(top_features)} Feature Importance ({model_name})")
    axis.set_xlabel("Importance (%)")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_best_model(
    model_name: str,
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str,
    grouped_importance: pd.DataFrame,
    individual_importance: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> None:
    pipeline = make_pipeline(estimator)
    pipeline.fit(X, y, model__sample_weight=compute_sample_weights(y))

    artifact = {
        "model_name": model_name,
        "model": pipeline,
        "feature_columns": list(X.columns),
        "label_mapping": {1: "B", 0: "H"},
        "training_rows": len(X),
        "feature_groups": groups,
        "grouped_feature_importance": grouped_importance.to_dict(orient="records"),
        "feature_importance": individual_importance.to_dict(orient="records"),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"Saved best model artifact to: {out_path}")


def evaluate_model(model_name: str, estimator: object, X: pd.DataFrame, y: pd.Series, seed: int, cv_folds: int) -> Dict[str, float]:
    pipeline = make_pipeline(estimator)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    sample_weight = compute_sample_weights(y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        "test_pr_auc": float(average_precision_score(y_test, y_score)),
    }
    return metrics


def predict_full_features(best_model_name: str, model_dict: Dict[str, object], X: pd.DataFrame, y: pd.Series, output_path: str) -> None:
    full_df = pd.read_csv("./src/features.csv")
    feature_columns = list(X.columns)
    missing_cols = [col for col in feature_columns if col not in full_df.columns]
    if missing_cols:
        raise ValueError(f"features.csv is missing required model features: {missing_cols}")

    pred_df = full_df[feature_columns].copy()
    pred_df = pred_df.apply(pd.to_numeric, errors="coerce")
    model = model_dict[best_model_name]
    pipeline = make_pipeline(model)
    pipeline.fit(X, y, model__sample_weight=compute_sample_weights(y))
    predictions = pipeline.predict(pred_df)

    out_df = full_df.copy()
    out_df["label"] = pd.Series(predictions).map({1: "B", 0: "H"})

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Predicted labels saved to: {out_path}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    X, y = load_dataset(args.data_csv, args.label_col, args.author_col)
    print(f"Loaded {len(X)} samples with {len(X.columns)} features and {y.value_counts().to_dict()} class counts.")

    model_dict = build_models()
    results: List[Dict[str, float]] = []

    for model_name, estimator in model_dict.items():
        metrics = evaluate_model(model_name, estimator, X, y, args.seed, args.cv_folds)
        results.append(metrics)
        print(f"\n=== {model_name} ===")
        print(f"CV F1 mean: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test precision: {metrics['test_precision']:.4f}")
        print(f"Test recall: {metrics['test_recall']:.4f}")
        print(f"Test F1: {metrics['test_f1']:.4f}")
        print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"Test PR-AUC: {metrics['test_pr_auc']:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        ["test_accuracy", "cv_f1_mean"], ascending=[False, False]
    )
    print("\n=== Ranking by test accuracy ===")
    print(
        results_df[
            [
                "model",
                "cv_f1_mean",
                "cv_f1_std",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc",
            ]
        ].to_string(index=False)
    )

    best_model_name = results_df.iloc[0]["model"]
    groups = feature_groups(list(X.columns))
    importance_model = make_pipeline(model_dict[best_model_name])
    X_importance_train, X_importance_test, y_importance_train, y_importance_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    importance_model.fit(
        X_importance_train,
        y_importance_train,
        model__sample_weight=compute_sample_weights(y_importance_train),
    )
    grouped_importance = grouped_permutation_importance(
        importance_model,
        X_importance_test,
        y_importance_test,
        groups,
        random_state=args.seed,
    )
    grouped_importance.to_csv(args.feature_importance_output, index=False)
    print("\n=== Grouped feature importance for best model ===")
    print(f"Best model: {best_model_name}")
    print(grouped_importance.to_string(index=False))
    print(f"Grouped feature importance saved to: {args.feature_importance_output}")

    individual_importance = individual_permutation_importance(
        importance_model,
        X_importance_test,
        y_importance_test,
        random_state=args.seed,
    )
    individual_importance.to_csv(args.individual_feature_importance_output, index=False)
    save_top_feature_importance_plot(
        individual_importance,
        best_model_name,
        args.feature_importance_plot_output,
    )
    print("\n=== Individual feature importance for best model ===")
    print(individual_importance.head(15).to_string(index=False))
    print(
        f"Individual feature importance saved to: "
        f"{args.individual_feature_importance_output}"
    )
    print(f"Top-15 feature importance plot saved to: {args.feature_importance_plot_output}")

    save_best_model(
        best_model_name,
        model_dict[best_model_name],
        X,
        y,
        args.model_output,
        grouped_importance,
        individual_importance,
        groups,
    )

    if args.predict_full:
        predict_full_features(best_model_name, model_dict, X, y, args.predict_csv)


if __name__ == "__main__":
    main()
