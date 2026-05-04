"""
main.py -- BRFSS CVD Risk Prediction Pipeline Orchestrator
===========================================================
Runs in sequence:
  1. PySpark preprocessing pipeline  -> data/new_brfss.csv
  2. Model training (LightGBM, Random Forest, SVM) on new_brfss.csv
  3. Saves best model + all artifacts to models/
  4. Launches FastAPI backend (uvicorn) on port 8000

Usage:
    python main.py                   # full pipeline + start API
    python main.py --skip-spark      # skip PySpark step (use existing new_brfss.csv)
    python main.py --skip-train      # skip training (use existing saved model)
    python main.py --api-only        # only start the API server
"""

import os
import sys
import json
import argparse
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    recall_score, f1_score, average_precision_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
SRC_DIR     = os.path.join(ROOT_DIR, "src")

RAW_CSV     = os.path.join(DATA_DIR, "brfss_data.csv")
SAMPLE_CSV  = os.path.join(DATA_DIR, "sample_brfss_data.csv")
CLEAN_CSV   = os.path.join(DATA_DIR, "new_brfss.csv")

# Fallback to sample data if full dataset is missing
if not os.path.exists(RAW_CSV) and os.path.exists(SAMPLE_CSV):
    print(f"\n[INFO] Full dataset not found. Using sample dataset for demonstration.")
    RAW_CSV = SAMPLE_CSV


TARGET_COL  = "CVDINFR4"

# Artifacts saved to models/
BEST_MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH       = os.path.join(MODELS_DIR, "imputer.pkl")
FEATURES_PATH      = os.path.join(MODELS_DIR, "features.json")
MODEL_INFO_PATH    = os.path.join(MODELS_DIR, "model_info.json")
COMPARISON_PATH    = os.path.join(MODELS_DIR, "model_comparison.csv")

# Feature columns from new_brfss.csv  (all columns except target)
# These are set dynamically after loading the CSV


# ==============================================================================
# STEP 1 -- PySpark Preprocessing
# ==============================================================================

def run_spark_pipeline():
    print("\n" + "="*60)
    print("  STEP 1: PySpark Preprocessing Pipeline")
    print("="*60)

    # Import here so --skip-spark avoids PySpark boot-up cost
    sys.path.insert(0, SRC_DIR)
    from src.data_processing import BRFSSDataProcessor

    processor = BRFSSDataProcessor()
    try:
        processor.run_full_pipeline(
            input_path=RAW_CSV,
            output_path=CLEAN_CSV,
            target_col=TARGET_COL,
            majority_ratio=3.0,
        )
    finally:
        processor.stop()

    print(f"\n  [OK] Clean dataset written to: {CLEAN_CSV}")


# ==============================================================================
# STEP 2 -- Model Training
# ==============================================================================

def run_training():
    print("\n" + "="*60)
    print("  STEP 2: Model Training & Evaluation")
    print("="*60)

    # ── Load ───────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading clean dataset...")
    df = pd.read_csv(CLEAN_CSV)
    print(f"      Shape: {df.shape}  |  Target: {TARGET_COL}")

    # Separate features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    feature_names = X.columns.tolist()
    print(f"      Features: {len(feature_names)}  |  Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")

    # ── Split ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting 80/20 (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Impute + Scale ─────────────────────────────────────────────────────────
    print("\n[3/5] Imputing (median) and scaling...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    # ── SMOTE ──────────────────────────────────────────────────────────────────
    print("\n[4/5] Applying SMOTE on training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
    print(f"      After SMOTE: {X_train_res.shape[0]:,} samples "
          f"(pos={y_train_res.sum():,}, neg={(y_train_res==0).sum():,})")

    # ── Train & Evaluate ───────────────────────────────────────────────────────
    print("\n[5/5] Training models...")
    models = {
        "LightGBM":     lgb.LGBMClassifier(
                            random_state=42, n_jobs=-1, verbose=-1,
                            n_estimators=300, learning_rate=0.05,
                            num_leaves=63, scale_pos_weight=5.0
                        ),
        "RandomForest": RandomForestClassifier(
                            n_estimators=100, max_depth=15,
                            random_state=42, n_jobs=-1,
                        ),
        "SVM":          LinearSVC(random_state=42, dual=False, max_iter=2000),
    }

    results = []
    trained_models = {}

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("BRFSS_CVD_Risk")

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_sc)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_sc)[:, 1]
        else:
            y_scores = model.decision_function(X_test_sc)

        pr_auc = average_precision_score(y_test, y_scores)

        results.append({
            "Model": name, "Accuracy": round(acc, 4),
            "Recall": round(rec, 4), "F1-Score": round(f1, 4),
            "PR-AUC": round(pr_auc, 4)
        })
        trained_models[name] = model

        print(f"      Accuracy={acc:.4f}  Recall={rec:.4f}  "
              f"F1={f1:.4f}  PR-AUC={pr_auc:.4f}")

        _save_confusion_matrix(name, y_test, y_pred)
        _save_feature_importance(name, model, feature_names)

        # ── MLflow Logging ───────────────────────────────────────────────────
        try:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "accuracy": acc,
                    "recall": rec,
                    "f1_score": f1,
                    "pr_auc": pr_auc
                })
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"      [MLflow] Could not log run: {e}")

    # ── Pick Best Model (by Recall) ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(COMPARISON_PATH, index=False)
    print(f"\n  Model comparison saved to: {COMPARISON_PATH}")
    print(results_df.to_string(index=False))

    best_row  = results_df.loc[results_df["Recall"].idxmax()]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    print(f"\n  Best model (Recall): {best_name}")

    # ── Save Artifacts ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(imputer,    IMPUTER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    model_info = {
        "best_model_name": best_name,
        "target_col": TARGET_COL,
        "feature_count": len(feature_names),
        "test_metrics": best_row.to_dict(),
        "all_results": results,
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n  [OK] Saved: {BEST_MODEL_PATH}")
    print(f"  [OK] Saved: {SCALER_PATH}")
    print(f"  [OK] Saved: {IMPUTER_PATH}")
    print(f"  [OK] Saved: {FEATURES_PATH}")
    print(f"  [OK] Saved: {MODEL_INFO_PATH}")


def _save_confusion_matrix(name: str, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No CVD", "CVD"], yticklabels=["No CVD", "CVD"])
    plt.title(f"{name} -- Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"{name.lower()}_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"      Saved: {path}")


def _save_feature_importance(name: str, model, feature_names: list):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title(f"{name} -- Top 15 Feature Importances")
    plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"{name.lower()}_feature_importance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"      Saved: {path}")


# ==============================================================================
# STEP 3 -- Launch FastAPI
# ==============================================================================

def run_api():
    import uvicorn
    print("\n" + "="*60)
    print("  STEP 3: Starting FastAPI Server")
    print("="*60)
    print("  URL: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")

    # api.py lives in src/
    sys.path.insert(0, SRC_DIR)
    uvicorn.run(
        "api:app",
        host="localhost",
        port=8000,
        reload=False,
        app_dir=SRC_DIR,
    )


# ==============================================================================
# CLI Entry-Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BRFSS CVD Risk Prediction Pipeline"
    )
    parser.add_argument(
        "--skip-spark", action="store_true",
        help="Skip PySpark preprocessing (use existing new_brfss.csv)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model training (use existing saved model)"
    )
    parser.add_argument(
        "--api-only", action="store_true",
        help="Skip preprocessing and training, only start FastAPI"
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Run preprocessing and training without starting the API"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.api_only:
        run_api()
        sys.exit(0)

    if not args.skip_spark:
        if not os.path.exists(CLEAN_CSV):
            print(f"\n  new_brfss.csv not found -- running PySpark pipeline...")
            run_spark_pipeline()
        else:
            print(f"\n  [OK] Using existing: {CLEAN_CSV}  (pass --skip-spark to suppress this message)")
            run_spark_pipeline()  # always refresh if not explicitly skipped
    else:
        if not os.path.exists(CLEAN_CSV):
            print(f"ERROR: {CLEAN_CSV} not found. Run without --skip-spark first.")
            sys.exit(1)
        print(f"  [SKIP] Spark step -- using: {CLEAN_CSV}")

    if not args.skip_train:
        run_training()
    else:
        if not os.path.exists(BEST_MODEL_PATH):
            print(f"ERROR: No saved model at {BEST_MODEL_PATH}. Run without --skip-train first.")
            sys.exit(1)
        print(f"  [SKIP] Training step -- using existing model: {BEST_MODEL_PATH}")

    if not args.no_api:
        run_api()
    else:
        print("\n  [SKIP] API not started (--no-api flag set).")
        print("  Run  python main.py --api-only  to start the API.")
