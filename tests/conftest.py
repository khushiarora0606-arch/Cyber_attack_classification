import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_COLUMNS = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
                "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Total", "Packet Length Mean",
                "SYN Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", "Active Mean"]
RAW_LABELS = ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "Bot", "FTP-Patator", "Web Attack \x96 Brute Force", "Infiltration"]


def make_test_traffic(n_per_class=120, seed=0, with_label=True):
    rng = np.random.default_rng(seed)
    frames = []
    for k, label in enumerate(RAW_LABELS):
        centre = rng.normal(0, 1, size=len(BASE_COLUMNS)) * 3 + k * 2
        part = pd.DataFrame(np.abs(rng.normal(centre, 1.5, size=(n_per_class, len(BASE_COLUMNS)))) * 10, columns=BASE_COLUMNS)
        part["Label"] = label
        frames.append(part)
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df if with_label else df.drop(columns=["Label"])


@pytest.fixture
def traffic():
    return make_test_traffic()


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    from src.data_preprocessing import preprocess
    from src.model_training import train_all

    root = tmp_path_factory.mktemp("project")
    raw = root / "raw"
    raw.mkdir()
    df = make_test_traffic(n_per_class=150)
    half = len(df) // 2
    for i, part in enumerate([df.iloc[:half], df.iloc[half:]]):
        part.rename(columns={"Label": " Label"}).to_csv(raw / f"part{i}.csv", index=False)
    processed = root / "clean.csv.gz"
    preprocess(raw, processed, report_path=root / "prep.json", max_per_class=500)
    res = train_all(processed, models_dir=root / "models", metrics_dir=root / "metrics", figures_dir=root / "figures",
                    demo_sample_file=root / "demo.csv", quick=True, cv_folds=2, slow_model_rows=600, n_jobs=1)
    return {"root": root, "results": res, "model": root / "models" / "best_model.joblib"}
