# Summary - 

An MLOps learning project built around credit risk scoring — predicting the probability that a borrower will default within 2 years (using the classic "Give Me Some Credit" Kaggle dataset).

The focus here is MLOps tooling and workflow, not model development. The credit risk model itself is intentionally simple; it exists to give the pipeline something real to train, serve, monitor, and retrain. This project explores:
- Automated experiment tracking and model registry with MLflow
- Production-style model serving via FastAPI
- Statistical drift detection (PSI) and automated retraining triggers
- Credit-scoring-specific validation practices (KS-statistic, Gini, rank-ordering/decile tables) — loosely inspired by conventions used in regulated financial services settings

⚠️ Disclaimer: This is a personal learning project, not a production system. It has not been validated for regulatory, fair-lending, or real-world lending use.

# Credit Risk MLOps Pipeline

An end-to-end machine learning pipeline for credit risk scoring, built with production MLOps practices including experiment tracking, model versioning, drift monitoring, automated retraining, and REST API deployment.

## Business Context
 This project explores MLOps concepts through a credit scoring use case, focusing on deployment over training — loosely modeled on practices used in regulated financial services settings.

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
