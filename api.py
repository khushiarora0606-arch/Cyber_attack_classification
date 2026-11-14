import logging

import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from src.model_prediction import ModelNotFoundError, load_model_bundle, predict_frame, summarise_predictions
from src.utils import MODEL_FILE, ProjectError

API_VERSION = "1.0.0"
MAX_RECORDS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("api")

app = Flask(__name__)
_bundle = None
PREDICTIONS = Counter("predictions_total", "Predicted records by class", ["predicted_class"])
REQUESTS = Counter("predict_requests_total", "Prediction requests by status", ["status"])


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_model_bundle(MODEL_FILE)
    return _bundle


def error(message, status):
    return jsonify({"error": message}), status


@app.errorhandler(ProjectError)
def project_error(exc):
    return error(str(exc), 503 if isinstance(exc, ModelNotFoundError) else 422)


@app.errorhandler(Exception)
def unexpected_error(exc):
    log.exception("Unexpected error")
    return error(f"Unexpected server error: {type(exc).__name__}", 500)


@app.get("/")
def root():
    return jsonify({"service": "Cyber Attack Classification API", "api_version": API_VERSION,
                    "endpoints": ["/health", "/model-info", "/features", "/predict", "/metrics"]})


@app.get("/health")
def health():
    try:
        return jsonify({"status": "ok", "model_loaded": True, "model_version": get_bundle()["model_version"]})
    except ModelNotFoundError as exc:
        return jsonify({"status": "degraded", "model_loaded": False, "detail": str(exc)})


@app.get("/model-info")
def model_info():
    b = get_bundle()
    return jsonify({"model_name": b["model_name"], "model_version": b["model_version"],
                    "trained_at": b.get("trained_at"), "classes": b["classes"],
                    "n_input_features": len(b["input_columns"]), "confidence_type": b.get("confidence_type"),
                    "test_metrics": b.get("test_metrics"), "api_version": API_VERSION})


@app.get("/features")
def features():
    cols = get_bundle()["input_columns"]
    return jsonify({"count": len(cols), "columns": cols})


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict():
    bundle = get_bundle()
    body = request.get_json(silent=True)
    records = body.get("records") if isinstance(body, dict) else None
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS or not all(isinstance(r, dict) for r in records):
        REQUESTS.labels("invalid").inc()
        return error(f"'records' must be a list of 1 to {MAX_RECORDS} objects (feature name -> number).", 422)
    try:
        results, info = predict_frame(bundle, pd.DataFrame(records))
    except ProjectError:
        REQUESTS.labels("invalid").inc()
        raise
    summary = summarise_predictions(results)
    for cls, n in summary["by_class"].items():
        PREDICTIONS.labels(cls).inc(n)
    REQUESTS.labels("ok").inc()
    log.info("Predicted %d records, %d attacks", summary["total_records"], summary["attack_records"])
    preds = [{"record": int(r["Record"]), "predicted_class": r["Predicted Class"], "confidence": float(r["Confidence"]),
              "is_attack": r["Alert"] == "ATTACK", "severity": r["Severity"], "low_confidence": bool(r["Low Confidence"])}
             for r in results.to_dict(orient="records")]
    return jsonify({"model_name": bundle["model_name"], "model_version": bundle["model_version"],
                    "n_records": len(preds), "data_quality": info, "summary": summary, "predictions": preds})


if __name__ == "__main__":
    app.run(port=8000)
