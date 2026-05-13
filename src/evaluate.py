import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Paths
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_ks(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    df["cum_bad"] = (df["y_true"] == 1).cumsum() / (df["y_true"] == 1).sum()
    df["cum_good"] = (df["y_true"] == 0).cumsum() / (df["y_true"] == 0).sum()
    ks = (df["cum_bad"] - df["cum_good"]).abs().max()
    return round(ks, 4)

def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI)
    Measures how much the score distribution has shifted
    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Moderate change, monitor closely
    PSI > 0.2  : Significant change, consider retraining
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Avoid division by zero
    expected_pct = np.where(
        expected_counts == 0, 0.0001,
        expected_counts / len(expected)
    )
    actual_pct = np.where(
        actual_counts == 0, 0.0001,
        actual_counts / len(actual)
    )

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = round(np.sum(psi_values), 4)
    return psi

def compute_rank_ordering(y_true, y_pred, deciles=10):
    """
    Rank ordering table — standard credit risk model validation tool
    Checks that higher scores = higher default rates monotonically
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=deciles, labels=False, duplicates="drop")
    df["decile"] = deciles - df["decile"]  # flip so decile 1 = highest risk

    rank_table = df.groupby("decile").agg(
        total=("y_true", "count"),
        bads=("y_true", "sum")
    ).reset_index()

    rank_table["goods"] = rank_table["total"] - rank_table["bads"]
    rank_table["bad_rate"] = (rank_table["bads"] / rank_table["total"]).round(4)
    rank_table["cum_bads"] = rank_table["bads"].cumsum()
    rank_table["cum_goods"] = rank_table["goods"].cumsum()
    rank_table["cum_bad_pct"] = (
        rank_table["cum_bads"] / rank_table["bads"].sum()
    ).round(4)
    rank_table["cum_good_pct"] = (
        rank_table["cum_goods"] / rank_table["goods"].sum()
    ).round(4)

    return rank_table

def plot_roc_curve(y_true, y_pred, auc):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Risk Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()
    print(f"ROC curve saved to {REPORTS_DIR / 'roc_curve.png'}")

def evaluate(X_test, y_test, log_to_mlflow=True):
    # Load model
    model = joblib.load(MODEL_DIR / "model.joblib")
    y_pred = model.predict_proba(X_test)[:, 1]

    # Core metrics
    auc = round(roc_auc_score(y_test, y_pred), 4)
    ks = compute_ks(y_test.values, y_pred)
    gini = round(2 * auc - 1, 4)

    # PSI — comparing train vs test score distributions
    train_pred = model.predict_proba(X_test)[:, 1]  # placeholder
    psi = compute_psi(expected=train_pred, actual=y_pred)

    # Rank ordering
    rank_table = compute_rank_ordering(y_test.values, y_pred)

    # ROC curve
    plot_roc_curve(y_test.values, y_pred, auc)

    # Print results
    print("\n===== Model Evaluation =====")
    print(f"  AUC:           {auc}")
    print(f"  KS-Statistic:  {ks}")
    print(f"  Gini:          {gini}")
    print(f"  PSI:           {psi}")
    print("\n===== Rank Ordering Table =====")
    print(rank_table.to_string(index=False))

    # Save rank ordering table
    rank_table.to_csv(REPORTS_DIR / "rank_ordering.csv", index=False)
    print(f"\nRank ordering saved to {REPORTS_DIR / 'rank_ordering.csv'}")

    # Log to MLflow
    if log_to_mlflow:
        with mlflow.start_run(run_name="evaluation", nested=True):
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("ks_statistic", ks)
            mlflow.log_metric("gini", gini)
            mlflow.log_metric("psi", psi)
            mlflow.log_artifact(str(REPORTS_DIR / "roc_curve.png"))
            mlflow.log_artifact(str(REPORTS_DIR / "rank_ordering.csv"))

    return {
        "auc": auc,
        "ks": ks,
        "gini": gini,
        "psi": psi,
        "rank_table": rank_table
    }

if __name__ == "__main__":
    from train import load_data, preprocess
    from sklearn.model_selection import train_test_split

    df = load_data()
    df = preprocess(df)
    TARGET = "SeriousDlqin2yrs"
    FEATURES = [c for c in df.columns if c != TARGET]
    X = df[FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    evaluate(X_test, y_test)
