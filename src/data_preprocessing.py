import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (DOWNLOAD_HELP, LABEL_COLUMN, PREPROCESSING_REPORT, PROCESSED_FILE, RAW_DIR,
                       DataNotFoundError, DataValidationError, clean_column_names, get_logger,
                       normalize_label, save_json)

logger = get_logger(__name__)


def find_csv_files(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv")) if raw_dir.exists() else []
    if not files:
        raise DataNotFoundError(DOWNLOAD_HELP.format(path=raw_dir))
    return files


def clean_columns(df):
    df = df.copy()
    df.columns = clean_column_names(df.columns)
    drop = [c for c in df.columns if c.endswith(".1") and c[:-2] in df.columns]
    return df.drop(columns=drop).loc[:, lambda d: ~d.columns.duplicated()]


def find_label_column(df):
    for col in df.columns:
        if str(col).strip().lower() == "label":
            return col
    raise DataValidationError("No 'Label' column found. CICIDS2017 files must contain a 'Label' column. "
                              f"Columns seen: {list(df.columns)[:8]}...")


def prepare_frame(df):
    df = clean_columns(df)
    df = df.rename(columns={find_label_column(df): LABEL_COLUMN})
    stats = {"rows_in": len(df)}
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(normalize_label)
    no_label = df[LABEL_COLUMN].isna()
    stats["rows_without_label"] = int(no_label.sum())
    df = df[~no_label].copy()
    features = [c for c in df.columns if c != LABEL_COLUMN]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    stats["inf_replaced"] = int(np.isinf(df[features].to_numpy()).sum())
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df, stats


def cap_per_class(df, cap, seed=42):
    if not cap or cap <= 0 or df.empty:
        return df
    return df.sample(frac=1, random_state=seed).groupby(LABEL_COLUMN, sort=False).head(cap)


def load_raw_datasets(raw_dir=RAW_DIR, max_per_class=20000, chunksize=200_000, seed=42):
    files = find_csv_files(raw_dir)
    report = {"files": [], "rows_read": 0, "rows_without_label": 0, "inf_replaced": 0,
              "duplicates_removed_in_chunks": 0, "duplicates_removed_global": 0,
              "class_counts_after_dedup_before_cap": {}, "max_per_class": max_per_class}
    parts = []
    for path in files:
        file_rows = 0
        try:
            for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding_errors="replace"):
                chunk, stats = prepare_frame(chunk)
                file_rows += stats["rows_in"]
                for key in ("rows_without_label", "inf_replaced"):
                    report[key] += stats[key]
                report["rows_read"] += stats["rows_in"]
                before = len(chunk)
                chunk = chunk.drop_duplicates()
                report["duplicates_removed_in_chunks"] += before - len(chunk)
                counts = report["class_counts_after_dedup_before_cap"]
                for lab, n in chunk[LABEL_COLUMN].value_counts().items():
                    counts[lab] = counts.get(lab, 0) + int(n)
                parts.append(cap_per_class(chunk, max_per_class, seed))
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataValidationError(f"Could not parse '{path.name}': {exc}") from exc
        report["files"].append({"name": path.name, "rows": file_rows})
        logger.info("%-60s %9d rows", path.name, file_rows)
    if not parts:
        raise DataValidationError("All CSV files were empty.")
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed_global"] = before - len(df)
    df = cap_per_class(df, max_per_class, seed).reset_index(drop=True)
    report["rows_final"] = len(df)
    report["n_features"] = df.shape[1] - 1
    report["class_counts_final"] = df[LABEL_COLUMN].value_counts().to_dict()
    report["missing_values_per_column"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    return df, report


def preprocess(raw_dir=RAW_DIR, output=PROCESSED_FILE, report_path=PREPROCESSING_REPORT,
               max_per_class=20000, seed=42):
    df, report = load_raw_datasets(raw_dir, max_per_class=max_per_class, seed=seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    save_json(report, report_path)
    logger.info("Saved %d rows x %d columns -> %s", df.shape[0], df.shape[1], output)
    logger.info("Final class counts: %s", report["class_counts_final"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_FILE)
    parser.add_argument("--max-per-class", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        preprocess(args.raw_dir, args.output, max_per_class=args.max_per_class, seed=args.seed)
    except (DataNotFoundError, DataValidationError) as exc:
        raise SystemExit(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
