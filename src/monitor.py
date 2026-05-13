import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import joblib
import json
from datetime import datetime
from evaluate import compute_psi

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("credit-risk-scoring")

# Paths
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MONITOR_DIR = Path("reports/monitoring")
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# PSI Thresholds — standard credit risk monitoring rules
PSI_THRESHOLDS = {
    "green": 0.1,   # No action needed
    "amber": 0.2,   # Monitor closely
    "red": 0.2      # Trigger retraining
}

# AUC degradation threshold
AUC_DEGRADATION_THRESHOLD = 0.03  # retrain if AUC drops more than 3 points

def load_baseline_scores():
    """
    Load baseline score distribution from training
    This is the reference distribution PSI compares against
    """
    baseline_path = MODEL_DIR / "baseline_scores.npy"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline scores not found at {baseline_path}. "
            "Run train.py first to generate baseline scores."
        )
    return np.load(baseline_path)

def load_baseline_metrics():
    """
    Load baseline AUC from training run
    """
    metrics_path = MODEL_DIR / "baseline_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found at {metrics_path}. "
            "Run train.py first."
        )
    with open(metrics_path, "r") as f:
        return json.load(f)

def interpret_psi(psi):
    """
    Standard credit risk PSI interpretation
    """
    if psi < PSI_THRESHOLDS["green"]:
        return "GREEN", "No significant change — model stable"
    elif psi < PSI_THRESHOLDS["amber"]:
        return "AMBER", "Moderate shift detected — monitor closely"
    else:
        return "RED", "Significant shift detected — retraining recommended"

def check_psi_drift(current_scores, baseline_scores):
    """
    Compute PSI between baseline and current score distributions
    """
    psi = compute_psi(expected=baseline_scores, actual=current_scores)
    status, message = interpret_psi(psi)
    return psi, status, message

def check_auc_degradation(current_auc, baseline_auc):
    """
    Check if AUC has degraded beyond acceptable threshold
    """
    degradation = baseline_auc - current_auc
    if degradation > AUC_DEGRADATION_THRESHOLD:
        return True, round(degradation, 4)
    return False, round(degradation, 4)

def trigger_retraining(reason):
    """
    Retraining trigger — in production this would kick off
    a GitHub Actions workflow or Airflow DAG
    For now logs the trigger event with timestamp
    """
    trigger_event = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "action": "retraining_triggered",
        "status": "pending"
    }

    trigger_path = MONITOR_DIR / "retraining_trigger.json"
    with open(trigger_path, "w") as f:
        json.dump(trigger_event, f, indent=2)

    print(f"\n!!! RETRAINING TRIGGERED !!!")
    print(f"Reason: {reason}")
    print(f"Trigger event logged to {trigger_path}")
    return trigger_event

def save_monitoring_report(report):
    """
    Save monitoring report with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = MONITOR_DIR / f"monitoring_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMonitoring report saved to {report_path}")
    return report_path

def monitor(X_current, y_current):
    """
    Main monitoring function
    Checks PSI drift and AUC degradation
    Triggers retraining if thresholds breached
    """
    print("\n===== Model Monitoring =====")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load model and baseline
    model = joblib.load(MODEL_DIR / "model.joblib")
    baseline_scores = load_baseline_scores()
    baseline_metrics = load_baseline_metrics()
    baseline_auc = baseline_metrics["auc"]

    # Score current data
    current_scores = model.predict_proba(X_current)[:, 1]

    # 1. PSI Check
    psi, psi_status, psi_message = check_psi_drift(current_scores, baseline_scores)
    print(f"\nPSI Check:")
    print(f"  PSI:     {psi}")
    print(f"  Status:  {psi_status}")
    print(f"  Message: {psi_message}")

    # 2. AUC Degradation Check
    from sklearn.metrics import roc_auc_score
    current_auc = round(roc_auc_score(y_current, current_scores), 4)
    retrain_needed, auc_drop = check_auc_degradation(current_auc, baseline_auc)
    print(f"\nAUC Degradation Check:")
    print(f"  Baseline AUC:  {baseline_auc}")
    print(f"  Current AUC:   {current_auc}")
    print(f"  AUC Drop:      {auc_drop}")
    print(f"  Retrain:       {'YES' if retrain_needed else 'NO'}")

    # 3. Retraining Decision
    retraining_triggered = False
    trigger_event = None

    if psi_status == "RED":
        trigger_event = trigger_retraining(
            reason=f"PSI threshold breached: PSI={psi} > {PSI_THRESHOLDS['red']}"
        )
        retraining_triggered = True

    elif retrain_needed:
        trigger_event = trigger_retraining(
            reason=f"AUC degradation exceeded threshold: drop={auc_drop} > {AUC_DEGRADATION_THRESHOLD}"
        )
        retraining_triggered = True

    # 4. Build monitoring report
    report = {
        "timestamp": datetime.now().isoformat(),
        "psi": psi,
        "psi_status": psi_status,
        "psi_message": psi_message,
        "baseline_auc": baseline_auc,
        "current_auc": current_auc,
        "auc_drop": auc_drop,
        "retraining_triggered": retraining_triggered,
        "trigger_event": trigger_event
    }

    # 5. Save report
    report_path = save_monitoring_report(report)

    # 6. Log to MLflow
    with mlflow.start_run(run_name="monitoring"):
        mlflow.log_metric("psi", psi)
        mlflow.log_metric("current_auc", current_auc)
        mlflow.log_metric("auc_drop", auc_drop)
        mlflow.log_param("psi_status", psi_status)
        mlflow.log_param("retraining_triggered", retraining_triggered)
        mlflow.log_artifact(str(report_path))

    print("\n===== Monitoring Complete =====")
    return report

if __name__ == "__main__":
    from train import load_data, preprocess
    from sklearn.model_selection import train_test_split

    # Simulate current production data
    # In production this would be recent scoring data
    df = load_data()
    df = preprocess(df)
    TARGET = "SeriousDlqin2yrs"
    FEATURES = [c for c in df.columns if c != TARGET]
    X = df[FEATURES]
    y = df[TARGET]
    _, X_current, _, y_current = train_test_split(
        X, y, test_size=0.2, random_state=99, stratify=y
    )
    monitor(X_current, y_current)
