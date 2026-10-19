import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==============================================================================
# 1. LOAD DATA & CLEANING
# ==============================================================================
CSV_FILE_PATH = "src/labeled_data/labeled_data.csv"  # <-- Update with your actual CSV path

print(f"Loading dataset from: {CSV_FILE_PATH}")
df = pd.read_csv(CSV_FILE_PATH)

# Drop 'Unable to evaluate' / 'U' as done in the paper
df = df[
    ~df["label"].isin(["Unable to evaluate", "U", "unable to evaluate"])
].copy()

# Binary Target Mapping (0: Not Deserving, 1: Deserving / Partially Deserving)
label_mapping = {
    "Not deserving of reward": 0,
    "N": 0,
    "Partially deserving of reward": 1,
    "P": 1,
    "Deserving of reward": 1,
    "D": 1,
}
df["target"] = df["label"].map(label_mapping)

# Drop non-feature columns
non_feature_cols = ["author", "title", "label", "target"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols].values
y = df["target"].values

print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")

# ==============================================================================
# 2. STRATIFIED TRAIN / VALIDATION / TEST SPLIT (70% / 15% / 15%)
# ==============================================================================
# Step A: 15% Test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Step B: 70% Train, 15% Validation (0.15 / 0.85 ≈ 0.17647)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
)

print("\n" + "=" * 65)
print("SPLIT SUMMARY:")
print(f"  Training Set   : {X_train.shape[0]:>5} rows (70.0%)")
print(f"  Validation Set : {X_val.shape[0]:>5} rows (15.0%)")
print(f"  Test Set       : {X_test.shape[0]:>5} rows (15.0%)")
print("=" * 65)

# ==============================================================================
# 3. FEATURE STANDARDIZATION (NO DATA LEAKAGE)
# ==============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 4. DEFINE ALL 5 CLASSIFIERS
# ==============================================================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    ),
    "Support Vector Machine (SVM)": SVC(
        kernel="rbf", C=1.0, probability=True, random_state=42
    ),
    "Logistic Regression (LR)": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Multilayer Perceptron (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    ),
}

# ==============================================================================
# 5. TRAIN & EVALUATE ALL MODELS
# ==============================================================================
results_summary = []
best_test_f1 = -1.0
best_model_name = None
best_model_obj = None

for name, model in models.items():
    print("\n" + "#" * 65)
    print(f"TRAINING: {name.upper()}")
    print("#" * 65)

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    # Evaluate on Validation & Test
    splits = [
        ("Val", X_val_scaled, y_val),
        ("Test", X_test_scaled, y_test),
    ]

    for split_name, X_data, y_data in splits:
        y_pred = model.predict(X_data)
        y_prob = model.predict_proba(X_data)[:, 1]

        acc = accuracy_score(y_data, y_pred)
        f1 = f1_score(y_data, y_pred)
        prec = precision_score(y_data, y_pred)
        rec = recall_score(y_data, y_pred)
        auc = roc_auc_score(y_data, y_prob)
        cm = confusion_matrix(y_data, y_pred)

        results_summary.append(
            {
                "Model": name,
                "Split": split_name,
                "Accuracy": acc,
                "F1-Score": f1,
                "Precision": prec,
                "Recall": rec,
                "ROC-AUC": auc,
            }
        )

        print(f"\n--- {name} [{split_name} Set Results] ---")
        print(
            f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}"
        )
        print(f"Confusion Matrix:\n{cm}")

        if split_name == "Test":
            print("\nClassification Report (Test Set):")
            print(
                classification_report(
                    y_data,
                    y_pred,
                    target_names=["Not Deserving (0)", "Deserving (1)"],
                )
            )

            # Track best model on Test F1
            if f1 > best_test_f1:
                best_test_f1 = f1
                best_model_name = name
                best_model_obj = model

# ==============================================================================
# 6. RESULTS COMPARISON TABLE
# ==============================================================================
results_df = pd.DataFrame(results_summary)

print("\n" + "=" * 80)
print("FINAL BENCHMARK COMPARISON TABLE (ALL 5 MODELS):")
print("=" * 80)
# Format numeric columns to 4 decimals
formatted_df = results_df.copy()
for col in ["Accuracy", "F1-Score", "Precision", "Recall", "ROC-AUC"]:
    formatted_df[col] = formatted_df[col].apply(lambda v: f"{v:.4f}")

print(formatted_df.to_string(index=False))
print("=" * 80)

# ==============================================================================
# 7. GENERATE FEATURE IMPORTANCE FOR THE BEST PERFORMING MODEL
# ==============================================================================
if best_model_obj is None or best_model_name is None:
    raise RuntimeError("No best model was selected; feature importance cannot be generated.")

importance_result = permutation_importance(
    best_model_obj,
    X_test_scaled,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

feature_importance_df = (
    pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": importance_result.importances_mean,
            "Importance Std": importance_result.importances_std,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

feature_importance_output = "content_quality_feature_importance.csv"
feature_importance_df.to_csv(feature_importance_output, index=False)

print("\n" + "=" * 80)
print(f"FEATURE IMPORTANCE FOR BEST MODEL: {best_model_name}")
print("=" * 80)
print(feature_importance_df.to_string(index=False))
print(f"Saved feature importance to '{feature_importance_output}'")
print("=" * 80)

# ==============================================================================
# 8. SAVE THE BEST PERFORMING MODEL
# ==============================================================================
print(
    f"\nBest Overall Model on Test Set: '{best_model_name}' (Test F1: {best_test_f1:.4f})"
)

package = {
    "scaler": scaler,
    "model_name": best_model_name,
    "model": best_model_obj,
    "feature_names": feature_cols,
    "feature_importance": feature_importance_df.to_dict(orient="records"),
    "label_mapping": label_mapping,
}

joblib.dump(package, "best_reward_classifier.joblib")
print("Successfully saved best model and scaler to 'best_reward_classifier.joblib'")