import os
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Paths
RAW_DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MLflow settings
EXPERIMENT_NAME = "credit-risk-scoring"
MODEL_NAME = "credit-risk-xgboost"

# XGBoost parameters 
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 10,  # handles class imbalance
    "early_stopping_rounds": 20,
    "random_state": 42
}

def load_data():
    df = pd.read_csv(RAW_DATA_DIR / "cs-training.csv", index_col=0)
    print(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess(df):
    # Fill missing values
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Cap outliers
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(0, 1)
    df["DebtRatio"] = df["DebtRatio"].clip(0, 1)

    # Feature engineering
    df["TotalLate"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (
        df["NumberOfDependents"] + 1
    )

    return df

def compute_ks(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    df["cum_bad"] = (df["y_true"] == 1).cumsum() / (df["y_true"] == 1).sum()
    df["cum_good"] = (df["y_true"] == 0).cumsum() / (df["y_true"] == 0).sum()
    ks = (df["cum_bad"] - df["cum_good"]).abs().max()
    return round(ks, 4)

def compute_gini(auc):
    return round(2 * auc - 1, 4)

def train():
    # Load and preprocess
    df = load_data()
    df = preprocess(df)

    # Features and target
    TARGET = "SeriousDlqin2yrs"
    FEATURES = [c for c in df.columns if c != TARGET]

    X = df[FEATURES]
    y = df[TARGET]

    # Train / validation / out-of-time split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-baseline") as run:

        # Log parameters
        mlflow.log_params(PARAMS)

        # Train model
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        # Evaluate on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_pred_proba), 4)
        ks = compute_ks(y_test.values, y_pred_proba)
        gini = compute_gini(auc)

        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("ks_statistic", ks)
        mlflow.log_metric("gini", gini)

        print(f"\nModel Performance:")
        print(f"  AUC:           {auc}")
        print(f"  KS-Statistic:  {ks}")
        print(f"  Gini:          {gini}")

        # Log model to MLflow registry
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME
        )

        # Save model locally
        joblib.dump(model, MODEL_DIR / "model.joblib")

        # Save baseline scores and metrics for monitoring
        baseline_scores = model.predict_proba(X_train)[:, 1]
        np.save(MODEL_DIR / "baseline_scores.npy", baseline_scores)
        import json
        with open(MODEL_DIR / "baseline_metrics.json", "w") as f:
            json.dump({"auc": auc}, f)
        print("Baseline scores and metrics saved for monitoring")
        print(f"\nModel saved to {MODEL_DIR / 'model.joblib'}")
        print(f"MLflow Run ID: {run.info.run_id}")

    return model, X_test, y_test

if __name__ == "__main__":
    train()
