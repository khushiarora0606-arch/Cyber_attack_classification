import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (cap_per_class, clean_columns, find_csv_files, find_label_column,
                                    load_raw_datasets, prepare_frame)
from src.feature_engineering import OutlierClipper, add_engineered_features, build_preprocessor
from src.model_prediction import InputValidationError, validate_traffic_frame
from src.utils import DataNotFoundError, DataValidationError, normalize_label
from tests.conftest import BASE_COLUMNS, make_test_traffic


@pytest.mark.parametrize("raw,expected", [
    ("BENIGN", "BENIGN"), ("DDoS", "DDoS"), ("DoS Hulk", "DoS"), ("DoS slowloris", "DoS"),
    ("PortScan", "PortScan"), ("Bot", "Bot"), ("FTP-Patator", "Brute Force"),
    ("SSH-Patator", "Brute Force"), ("Web Attack \x96 XSS", "Web Attack"),
    ("Web Attack � Sql Injection", "Web Attack"), ("Heartbleed", "Other"),
    ("Infiltration", "Other"), ("  benign ", "BENIGN"),
])
def test_label_mapping(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_and_empty_labels():
    assert normalize_label("SomethingNew") == "Other"
    assert normalize_label("SomethingNew", strict=True) is None
    assert normalize_label(None) is None and normalize_label(float("nan")) is None
    assert normalize_label("   ") is None


def test_clean_columns_strips_and_drops_duplicate_copy():
    df = pd.DataFrame({" Label": ["a"], " Fwd Header Length": [1], " Fwd Header Length.1": [1]})
    assert list(clean_columns(df).columns) == ["Label", "Fwd Header Length"]


def test_find_label_column_missing_raises():
    with pytest.raises(DataValidationError):
        find_label_column(pd.DataFrame({"a": [1]}))


def test_prepare_frame_handles_inf_text_and_missing_label():
    df = pd.DataFrame({" Label": ["BENIGN", "DDoS", None, "DoS Hulk"], "Flow Bytes/s": [1.0, np.inf, 3.0, "bad"]})
    out, stats = prepare_frame(df)
    assert stats["rows_without_label"] == 1 and stats["inf_replaced"] == 1
    assert len(out) == 3 and out["Flow Bytes/s"].isna().sum() == 2
    assert set(out["Label"]) == {"BENIGN", "DDoS", "DoS"}


def test_cap_per_class():
    df = pd.DataFrame({"Label": ["A"] * 50 + ["B"] * 5, "x": range(55)})
    assert cap_per_class(df, 10)["Label"].value_counts().to_dict() == {"A": 10, "B": 5}


def test_missing_raw_folder_gives_helpful_error(tmp_path):
    with pytest.raises(DataNotFoundError, match="kaggle"):
        find_csv_files(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataNotFoundError):
        find_csv_files(tmp_path / "empty")


def test_load_multiple_csv_files_and_duplicates(tmp_path):
    df = make_test_traffic(30).rename(columns={"Label": " Label"})
    df.iloc[:100].to_csv(tmp_path / "a.csv", index=False)
    pd.concat([df.iloc[:20], df.iloc[100:]]).to_csv(tmp_path / "b.csv", index=False)
    out, report = load_raw_datasets(tmp_path, max_per_class=None)
    assert len(report["files"]) == 2 and report["rows_read"] == 100 + 20 + len(df) - 100
    assert len(out) == len(df)
    assert report["duplicates_removed_global"] + report["duplicates_removed_in_chunks"] == 20


def test_engineered_features_added_and_finite():
    df = pd.DataFrame({"Total Fwd Packets": [10.0], "Total Backward Packets": [0.0],
                       "Total Length of Fwd Packets": [100.0], "Total Length of Bwd Packets": [0.0]})
    out = add_engineered_features(df)
    assert out.loc[0, "Packet Ratio"] == 10.0 and np.isfinite(out.loc[0, "Bytes Per Packet"])


def test_outlier_clipper_uses_only_training_statistics():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    out = OutlierClipper(0.0, 0.9).fit(train).transform(np.array([[1000.0], [-50.0]]))
    assert out[0, 0] == pytest.approx(np.quantile(train, 0.9)) and out[1, 0] == 0.0


def test_preprocessor_handles_nan_and_scales():
    X = np.random.default_rng(1).normal(size=(100, 4))
    X[0, 0] = np.nan
    out = build_preprocessor(scale=True).fit_transform(X)
    assert not np.isnan(out).any() and out.shape[1] == 4
    assert out.min() >= 0 and out.max() <= 1


def test_validation_rejects_empty_and_missing_columns(traffic):
    with pytest.raises(InputValidationError, match="no rows"):
        validate_traffic_frame(pd.DataFrame(), BASE_COLUMNS)
    with pytest.raises(InputValidationError, match="no rows"):
        validate_traffic_frame(traffic.iloc[0:0], BASE_COLUMNS)
    with pytest.raises(InputValidationError, match="missing"):
        validate_traffic_frame(traffic.drop(columns=["Flow Duration"]), BASE_COLUMNS)


def test_validation_rejects_non_numeric_data(traffic):
    bad = traffic.copy()
    bad[BASE_COLUMNS] = "text"
    with pytest.raises(InputValidationError, match="numeric"):
        validate_traffic_frame(bad, BASE_COLUMNS)


def test_validation_counts_missing_cells_and_flags_unknown_labels(traffic):
    df = traffic.head(10).copy()
    df.loc[0, "Flow Duration"] = np.nan
    df.loc[1, "Flow Duration"] = np.inf
    df.loc[2, "Label"] = "TotallyNewAttack"
    X, actual, info = validate_traffic_frame(df, BASE_COLUMNS)
    assert info["missing_cells"] == 2 and info["infinite_cells"] == 1
    assert info["unrecognised_labels"] == ["TotallyNewAttack"] and actual.isna().sum() == 1
