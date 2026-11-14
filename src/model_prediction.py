from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import clean_columns
from src.feature_engineering import add_engineered_features
from src.model_evaluation import get_scores
from src.utils import BENIGN, MODEL_FILE, SEVERITY, SEVERITY_RANK, ProjectError, normalize_label

LOW_CONFIDENCE = 0.60
REQUIRED_BUNDLE_KEYS = {"pipeline", "classes", "input_columns", "feature_columns", "model_version"}


class ModelNotFoundError(ProjectError):
    pass


class InputValidationError(ProjectError):
    pass


def load_model_bundle(path=MODEL_FILE):
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"Trained model '{path.name}' not found. Train it first with: python -m src.model_training")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelNotFoundError(f"Could not load '{path.name}' ({exc}). Re-train with the same scikit-learn version "
                                 "that you use to run the app.") from exc
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ModelNotFoundError(f"Model file is missing entries: {sorted(missing)}. Re-train the model.")
    return bundle


def alert_for(predicted_class):
    if predicted_class == BENIGN:
        return {"is_attack": False, "severity": "None", "status": "Normal traffic"}
    return {"is_attack": True, "severity": SEVERITY.get(predicted_class, "Medium"),
            "status": f"ATTACK ALERT: {predicted_class}"}


def validate_traffic_frame(df, input_columns):
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise InputValidationError("The file contains no rows. Upload a CSV with network traffic records.")
    df = clean_columns(df)
    actual = next((df[c] for c in df.columns if c.lower() == "label"), None)
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise InputValidationError(f"{len(missing)} required column(s) are missing: {shown}. "
                                   "Upload a CICIDS2017-style CSV that has the same flow-feature columns.")
    X = df[input_columns].apply(pd.to_numeric, errors="coerce")
    inf_cells = int(np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().all(axis=None):
        raise InputValidationError("None of the feature columns contain numeric values.")
    info = {"missing_cells": int(X.isna().sum().sum()), "infinite_cells": inf_cells, "rows": len(X),
            "unrecognised_labels": []}
    if actual is not None:
        mapped = actual.map(lambda v: normalize_label(v, strict=True))
        info["unrecognised_labels"] = actual[mapped.isna()].astype(str).str.strip().unique().tolist()[:10]
        actual = mapped
    return X, actual, info


def predict_frame(bundle, df):
    X, actual, info = validate_traffic_frame(df, bundle["input_columns"])
    X = add_engineered_features(X)[bundle["feature_columns"]]
    pipeline = bundle["pipeline"]
    classes = list(pipeline.classes_)
    labels = pipeline.predict(X)
    scores = get_scores(pipeline, X)
    confidence = scores[np.arange(len(labels)), [classes.index(lab) for lab in labels]]
    results = pd.DataFrame({"Record": np.arange(1, len(labels) + 1), "Predicted Class": labels,
                            "Confidence": np.round(confidence, 4)})
    results["Alert"] = np.where(results["Predicted Class"] == BENIGN, "Normal", "ATTACK")
    results["Severity"] = results["Predicted Class"].map(lambda c: alert_for(c)["severity"])
    results["Low Confidence"] = results["Confidence"] < LOW_CONFIDENCE
    if actual is not None:
        results["Actual Label"] = actual.fillna("Unrecognised").to_numpy()
    return results, info


def summarise_predictions(results):
    total = len(results)
    attacks = results[results["Alert"] == "ATTACK"]
    n_attacks = len(attacks)
    by_class = {k: int(v) for k, v in results["Predicted Class"].value_counts().items()}
    top_severity = max(attacks["Severity"].unique(), key=SEVERITY_RANK.get, default="None")
    attack_counts = {k: v for k, v in by_class.items() if k != BENIGN}
    summary = {
        "total_records": total, "benign_records": total - n_attacks, "attack_records": n_attacks,
        "attack_percentage": round(100 * n_attacks / total, 2) if total else 0.0,
        "by_class": by_class, "highest_severity": top_severity,
        "dominant_attack": max(attack_counts, key=attack_counts.get) if attack_counts else None,
        "mean_confidence": round(float(results["Confidence"].mean()), 4) if total else 0.0,
        "attack_mean_confidence": round(float(attacks["Confidence"].mean()), 4) if n_attacks else None,
        "low_confidence_records": int(results["Low Confidence"].sum()),
    }
    if "Actual Label" in results:
        known = results["Actual Label"] != "Unrecognised"
        if known.any():
            summary["agreement_with_uploaded_labels"] = round(
                float((results.loc[known, "Actual Label"] == results.loc[known, "Predicted Class"]).mean()), 4)
    summary["alert"] = build_alert(summary)
    return summary


def build_alert(summary):
    if summary["attack_records"] == 0:
        return {"level": "normal", "severity": "None", "title": "Normal traffic status",
                "message": "No record was classified as an attack."}
    return {"level": "attack", "severity": summary["highest_severity"],
            "title": f"Attack alert: {summary['attack_records']:,} suspicious record(s)",
            "message": (f"Most frequent attack category: {summary['dominant_attack']}. "
                        f"Highest severity: {summary['highest_severity']}. "
                        f"{summary['attack_percentage']}% of the uploaded records are suspicious.")}
