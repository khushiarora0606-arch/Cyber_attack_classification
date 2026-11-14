import pandas as pd
import pytest

import api as api_module
from src.model_prediction import (InputValidationError, ModelNotFoundError, alert_for, load_model_bundle,
                                  predict_frame, summarise_predictions)
from tests.conftest import BASE_COLUMNS, make_test_traffic


def test_training_creates_all_artifacts(trained):
    root = trained["root"]
    for p in ["models/best_model.joblib", "models/model_metadata.json", "metrics/evaluation_results.json",
              "metrics/model_comparison.csv", "figures/model_comparison.png", "demo.csv"]:
        assert (root / p).exists(), p
    assert list((root / "models" / "versions").glob("*.joblib"))
    res = trained["results"]
    assert set(res["models"]) == {"Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbours"}
    assert res["best_model"] in res["models"]
    for r in res["models"].values():
        assert all(0 <= r["test"][m] <= 1 for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro"))


def test_best_model_chosen_on_validation_not_test(trained):
    res = trained["results"]
    best_val = max(r["validation"]["f1_macro"] for r in res["models"].values())
    assert res["models"][res["best_model"]]["validation"]["f1_macro"] == pytest.approx(best_val)


def test_model_save_load_and_predict(trained):
    bundle = load_model_bundle(trained["model"])
    df = make_test_traffic(20, seed=99)
    results, _ = predict_frame(bundle, df)
    assert len(results) == len(df)
    assert results["Confidence"].between(0, 1).all()
    assert set(results["Predicted Class"]) <= set(bundle["classes"])
    summary = summarise_predictions(results)
    assert summary["total_records"] == summary["benign_records"] + summary["attack_records"]
    assert summary["alert"]["level"] in ("attack", "normal")


def test_prediction_works_without_label_column_and_with_missing_values(trained):
    bundle = load_model_bundle(trained["model"])
    df = make_test_traffic(5, seed=5, with_label=False)
    df.iloc[0, 3] = float("nan")
    results, info = predict_frame(bundle, df)
    assert "Actual Label" not in results and info["missing_cells"] == 1


def test_prediction_errors(trained):
    bundle = load_model_bundle(trained["model"])
    with pytest.raises(InputValidationError):
        predict_frame(bundle, pd.DataFrame())
    with pytest.raises(InputValidationError):
        predict_frame(bundle, make_test_traffic(5).drop(columns=BASE_COLUMNS[:3]))


def test_missing_and_corrupt_model_file(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.model_training"):
        load_model_bundle(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    with pytest.raises(ModelNotFoundError):
        load_model_bundle(bad)


def test_alert_rule():
    assert alert_for("BENIGN") == {"is_attack": False, "severity": "None", "status": "Normal traffic"}
    assert alert_for("DDoS")["is_attack"] and alert_for("DDoS")["severity"] == "Critical"
    assert alert_for("PortScan")["severity"] == "Medium"


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setattr(api_module, "MODEL_FILE", trained["model"])
    monkeypatch.setattr(api_module, "_bundle", None)
    return api_module.app.test_client()


def test_api_health_info_features_metrics(client):
    assert client.get("/health").get_json()["model_loaded"] is True
    info = client.get("/model-info").get_json()
    assert "model_version" in info and "classes" in info
    assert client.get("/features").get_json()["count"] == len(BASE_COLUMNS)
    assert client.get("/metrics").status_code == 200


def test_api_predict_success(client):
    records = make_test_traffic(3, seed=3, with_label=False).head(4).to_dict(orient="records")
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_records"] == 4 and len(body["predictions"]) == 4
    assert {"predicted_class", "confidence", "is_attack", "severity"} <= set(body["predictions"][0])
    assert "model_version" in body


def test_api_validation_errors(client):
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"records": [{"Flow Duration": 1}]}).status_code == 422
    assert client.post("/predict", data="not json", content_type="application/json").status_code == 422
    assert client.post("/predict", json={"records": [{"a": 1}] * 1001}).status_code == 422


def test_api_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "MODEL_FILE", tmp_path / "none.joblib")
    monkeypatch.setattr(api_module, "_bundle", None)
    c = api_module.app.test_client()
    assert c.get("/health").get_json()["model_loaded"] is False
    assert c.post("/predict", json={"records": [{"a": 1}]}).status_code == 503
