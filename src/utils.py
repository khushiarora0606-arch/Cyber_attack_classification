import json
import logging
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

PROCESSED_FILE = PROCESSED_DIR / "cicids2017_clean.csv.gz"
DEMO_SAMPLE_FILE = PROCESSED_DIR / "demo_sample_heldout.csv"
PREPROCESSING_REPORT = METRICS_DIR / "preprocessing_report.json"
EVALUATION_RESULTS = METRICS_DIR / "evaluation_results.json"
LATEST_UPLOAD_SUMMARY = METRICS_DIR / "latest_upload_summary.json"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

LABEL_COLUMN = "Label"
BENIGN = "BENIGN"
CLASS_ORDER = ["BENIGN", "DDoS", "DoS", "PortScan", "Bot", "Brute Force", "Web Attack", "Other"]
SEVERITY = {"BENIGN": "None", "DDoS": "Critical", "DoS": "High", "Bot": "High",
            "Brute Force": "High", "Web Attack": "High", "PortScan": "Medium", "Other": "Medium"}
SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}

DOWNLOAD_HELP = ("No CSV files found in '{path}'.\n"
                 "1. Download the dataset from https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset\n"
                 "2. Unzip it and copy all the *.csv files into '{path}'.\n"
                 "3. Run: python -m src.data_preprocessing")


class ProjectError(Exception):
    pass


class DataNotFoundError(ProjectError):
    pass


class DataValidationError(ProjectError):
    pass


def normalize_label(raw, strict=False):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip().lower()
    if not text or text == "nan":
        return None
    if text.startswith("dos"):
        return "DoS"
    if text in ("ftp-patator", "ssh-patator"):
        return "Brute Force"
    if "web attack" in text:
        return "Web Attack"
    if text in ("heartbleed", "infiltration"):
        return "Other"
    for name in CLASS_ORDER:
        if text == name.lower():
            return name
    return None if strict else "Other"


def order_classes(classes):
    return sorted(classes, key=lambda c: CLASS_ORDER.index(c) if c in CLASS_ORDER else len(CLASS_ORDER))


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def clean_column_names(columns):
    return [re.sub(r"\s+", " ", str(c)).strip() for c in columns]
