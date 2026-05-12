# Credit Risk MLOps Pipeline

An end-to-end machine learning pipeline for credit risk scoring, built with production MLOps practices including experiment tracking, model versioning, drift monitoring, automated retraining, and REST API deployment.

## Business Context
Credit risk models require rigorous development, monitoring, and governance to remain accurate over time. This project simulates a production-grade credit scoring pipeline — from model training to deployment — applying the same practices used in regulated financial services environments.

## Dataset
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle competition dataset for predicting probability of financial distress within two years.

## Project Structure
credit-risk-mlops/
├── data/                        # Downloaded via Kaggle API
├── src/
│   ├── train.py                 # XGBoost training + MLflow tracking
│   ├── evaluate.py              # AUC, KS-statistic, PSI, rank ordering
│   └── monitor.py               # PSI drift detection + retraining trigger
├── api/
│   └── main.py                  # FastAPI REST endpoint
├── .github/
│   └── workflows/
│       └── ci_cd.yml            # GitHub Actions CI/CD pipeline
├── dvc.yaml                     # DVC pipeline versioning
└── requirements.txt
## MLOps Components

| Component | Tool |
|---|---|
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Drift Detection | PSI (Population Stability Index) |
| REST API | FastAPI |
| CI/CD Pipeline | GitHub Actions |
| Data Versioning | Kaggle API + DVC |

## Model Performance
| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC | — | — |
| KS-Statistic | — | — |
| Gini | — | — |

*(Populated after training)*

## Key Features
- Automated PSI-based drift detection with retraining trigger
- MLflow experiment tracking and model versioning
- REST API for real-time scoring
- GitHub Actions pipeline for continuous integration and deployment
- Modular, reproducible pipeline following financial services model development standards

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-risk-mlops.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
python src/download_data.py

# 4. Train model
python src/train.py

# 5. Run API
uvicorn api.main:app --reload
```

## Author
Monica | Senior Data Scientist | Credit & Fraud Risk Modeling
