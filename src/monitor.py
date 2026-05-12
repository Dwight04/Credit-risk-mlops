import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import joblib
import json
from datetime import datetime
from evaluate import compute_psi

# Paths
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MONITOR_DIR = Path("reports/monitoring")
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# PSI Thresholds — standard credit risk monitoring rules
PSI_THRESHOLDS = {
