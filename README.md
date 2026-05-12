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
