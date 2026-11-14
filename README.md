# Cyber Attack Classification and Alerting Using Machine Learning

MLOps-1 (25CSH-320) mini-project. Python, Scikit-learn, Flask, Streamlit.
Results are **PENDING**: every number must come from running the training command on the real CICIDS2017 dataset.

## 1. Problem statement
Given a network-flow record, decide whether it is BENIGN or one of seven attack categories
(DDoS, DoS, PortScan, Bot, Brute Force, Web Attack, Other), report the confidence, and raise an alert when it is not benign.

## 2. Dataset
CICIDS2017 (Kaggle mirror: <https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset>), eight CSV files, about 2.8 million labelled flows, about 78 numeric features plus `Label`.
The 15 raw labels are grouped into 8 categories in `src/utils.py::normalize_label`. The data is not included in the repository.

## 3. Syllabus mapping
| Syllabus topic | Where |
|---|---|
| Supervised learning, features, labels | all of `src/` |
| Decision tree, k-NN, SVM, random forest | `src/model_training.py` |
| Data cleaning, missing values, outliers, normalisation, feature engineering | `src/data_preprocessing.py`, `src/feature_engineering.py` |
| Train / validation / test split | `split_data` in `src/model_training.py` (70/15/15, stratified) |
| Cross-validation, GridSearchCV | `src/model_training.py` |
| Accuracy, precision, recall, F1, confusion matrix, ROC | `src/model_evaluation.py` |
| Overfitting, underfitting, regularisation | train-validation gap table, `depth_curve`, `max_depth`, `min_samples_leaf`, SVM `C` |
| REST API with Flask | `api.py` |
| Docker | `Dockerfile` |
| CI/CD with GitHub Actions | `.github/workflows/ci.yml` |
| Model versioning, DVC | `models/versions/`, `dvc.yaml` |
| Monitoring and logging, Prometheus | `/metrics` endpoint and logging in `api.py` |
| Ethics: fairness, privacy, security | Section 8 and the dashboard page "About and Ethics" |

Linear regression is not used: it predicts continuous numbers, while attack labels are unordered categories.

## 4. Project structure
```
api.py                  Flask REST API
app.py                  Streamlit dashboard
src/                    utils, data_preprocessing, feature_engineering,
                        model_training, model_evaluation, model_prediction
tests/                  pytest tests
notebooks/              exploratory data analysis
data/ models/ reports/  generated files
Dockerfile  dvc.yaml  .github/workflows/ci.yml
```

## 5. Setup
Python 3.10+. Run every command from the project root.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Download the dataset and copy all `*.csv` files into `data/raw/`.

## 6. Run
```bash
python -m src.data_preprocessing --max-per-class 20000
python -m src.model_training                     # add --quick for a fast check
python -m src.model_evaluation
python -m pytest
```
Options for training: `--cv-folds 3`, `--slow-model-rows 15000`, `--n-jobs -1`, `--seed 42`.

**Dashboard**
```bash
streamlit run app.py
```
Pages: Overview, Attack Classification, Model Performance, Dataset Analysis, About and Ethics.

**API**
```bash
python api.py
```
| Endpoint | Purpose |
|---|---|
| `GET /health` | status and whether the model is loaded |
| `GET /model-info` | model name, version, classes, test metrics |
| `GET /features` | required feature names |
| `POST /predict` | body `{"records": [{feature: value, ...}]}` (1 to 1000 records) |
| `GET /metrics` | Prometheus metrics (`predictions_total`, `predict_requests_total`) |

Errors are JSON: `422` invalid input, `503` model not trained, `500` unexpected error.

**Docker**
```bash
docker build -t cyber-attack-api .
docker run -p 8000:8000 cyber-attack-api
```
Train the model first so `models/best_model.joblib` is copied into the image.

**DVC**
```bash
dvc init
dvc repro
```

## 7. Method
1. Clean and merge the CSV files (names, labels, infinite values, duplicates, per-class cap of 20 000 rows).
2. Stratified 70/15/15 split before any learning step.
3. Pipeline per model: median imputation, percentile outlier clipping (0.5 and 99.5), min-max scaling for SVM and k-NN, classifier. Everything is fitted on training data only.
4. Tune each model with GridSearchCV (stratified k-fold, macro F1) on the training split.
5. Select the best model by validation macro F1; report on the untouched test set.
6. Save the pipeline with joblib plus a versioned copy.

SVM and k-NN are tuned on a stratified subset (default 15 000 rows) for speed. SVM confidence is a softmax of its decision scores, not a calibrated probability.

Alert rule: `IF predicted_class != BENIGN: attack alert (severity from category) ELSE: normal traffic`.
Severity: DDoS Critical; DoS, Bot, Brute Force, Web Attack High; PortScan, Other Medium.

## 8. Ethics and limitations
* **Fairness and bias:** the model only knows the 2017 lab traffic; rare classes have few samples and unreliable scores. A human should review alerts.
* **Privacy:** flow records reveal communication patterns. Uploads are processed in memory; only an aggregate summary of the latest upload is saved.
* **Security:** evasion, data poisoning and over-trust in automated decisions. Mitigations: input validation, a 1000-record request limit, versioned models.
* One dataset, class capping changes class proportions, offline analysis only, no live capture or blocking.

## 9. Results
**PENDING.** After training, copy the numbers from `reports/metrics/model_comparison.csv` into this table.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Decision Tree | PENDING | PENDING | PENDING | PENDING |
| Random Forest | PENDING | PENDING | PENDING | PENDING |
| Support Vector Machine | PENDING | PENDING | PENDING | PENDING |
| K-Nearest Neighbours | PENDING | PENDING | PENDING | PENDING |

## 10. References
1. Sharafaldin, Lashkari, Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP, 2018.
2. Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study," IEEE S&P Workshops, 2021.
3. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011.
