import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import find_label_column
from src.feature_engineering import add_engineered_features, build_preprocessor
from src.model_evaluation import (compute_metrics, confusion, generate_report_figures, get_scores,
                                  per_class_report, results_table, roc_data)
from src.utils import (DEMO_SAMPLE_FILE, EVALUATION_RESULTS, FIGURES_DIR, LABEL_COLUMN, METRICS_DIR,
                       MODEL_FILE, MODEL_METADATA_FILE, MODELS_DIR, PROCESSED_FILE, DataNotFoundError,
                       DataValidationError, get_logger, order_classes, save_json)

logger = get_logger(__name__)

MODEL_VERSION = "1.0.0"
MIN_CLASS_SIZE = 20


def load_processed(path=PROCESSED_FILE):
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Processed dataset '{path}' not found. Run first: python -m src.data_preprocessing")
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError("The processed dataset is empty.")
    df = df.rename(columns={find_label_column(df): LABEL_COLUMN})
    counts = df[LABEL_COLUMN].value_counts()
    if len(counts) < 2:
        raise DataValidationError("Need at least two classes to train a classifier.")
    rare = counts[counts < MIN_CLASS_SIZE]
    if len(rare):
        raise DataValidationError(f"Classes with fewer than {MIN_CLASS_SIZE} rows cannot be split reliably: "
                                  f"{rare.to_dict()}. Check the label mapping in src/utils.py.")
    return df


def split_data(df, seed=42):
    X, y = df.drop(columns=[LABEL_COLUMN]), df[LABEL_COLUMN]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_subsample(X, y, max_rows, seed=42):
    if max_rows is None or len(X) <= max_rows:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X_sub, y_sub


def build_models(quick=False, seed=42):
    def pipe(scale, clf):
        return Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", clf)])

    if quick:
        grids = [{"clf__max_depth": [5, 12]}, {"clf__n_estimators": [50], "clf__max_depth": [12]},
                 {"clf__C": [1]}, {"clf__n_neighbors": [5]}]
    else:
        grids = [{"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 5, 20]},
                 {"clf__n_estimators": [100, 200], "clf__max_depth": [20, 30], "clf__min_samples_leaf": [1, 3]},
                 {"clf__C": [0.5, 2, 8]},
                 {"clf__n_neighbors": [3, 5, 11], "clf__weights": ["uniform", "distance"]}]
    return {
        "Decision Tree": (pipe(False, DecisionTreeClassifier(class_weight="balanced", random_state=seed)), grids[0], False),
        "Random Forest": (pipe(False, RandomForestClassifier(class_weight="balanced", random_state=seed)), grids[1], False),
        "Support Vector Machine": (pipe(True, SVC(class_weight="balanced", random_state=seed)), grids[2], True),
        "K-Nearest Neighbours": (pipe(True, KNeighborsClassifier()), grids[3], True),
    }


def depth_curve(X_train, y_train, X_val, y_val, max_rows=20000, seed=42):
    Xs, ys = stratified_subsample(X_train, y_train, max_rows, seed)
    prep = build_preprocessor(scale=False).fit(Xs)
    Xs_t, Xv_t = prep.transform(Xs), prep.transform(X_val)
    depths = [1, 2, 3, 5, 8, 12, 16, 20, 25]
    train_f1, val_f1 = [], []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=seed).fit(Xs_t, ys)
        train_f1.append(f1_score(ys, tree.predict(Xs_t), average="macro", zero_division=0))
        val_f1.append(f1_score(y_val, tree.predict(Xv_t), average="macro", zero_division=0))
    return {"depths": depths, "train_f1": train_f1, "val_f1": val_f1}


def train_all(processed_file=PROCESSED_FILE, models_dir=MODELS_DIR, metrics_dir=METRICS_DIR,
              figures_dir=FIGURES_DIR, demo_sample_file=DEMO_SAMPLE_FILE, quick=False, cv_folds=3,
              slow_model_rows=15000, n_jobs=-1, seed=42):
    models_dir, metrics_dir = Path(models_dir), Path(metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed(processed_file)
    base_columns = [c for c in df.columns if c != LABEL_COLUMN]
    df_fe = pd.concat([add_engineered_features(df[base_columns]), df[LABEL_COLUMN]], axis=1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_fe, seed)
    feature_columns = list(X_train.columns)
    classes = order_classes(sorted(y_train.unique()))
    logger.info("Rows: train=%d validation=%d test=%d | features=%d | classes=%s",
                len(X_train), len(X_val), len(X_test), len(feature_columns), classes)

    X_train_eval, y_train_eval = stratified_subsample(X_train, y_train, 20000, seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results, fitted = {}, {}

    for name, (pipeline, grid, slow) in build_models(quick, seed).items():
        X_fit, y_fit = stratified_subsample(X_train, y_train, slow_model_rows, seed) if slow else (X_train, y_train)
        logger.info("Tuning %-24s on %d training rows", name, len(X_fit))
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs)
        start = time.time()
        search.fit(X_fit, y_fit)
        seconds = time.time() - start
        best = fitted[name] = search.best_estimator_
        i = search.best_index_
        train_m = compute_metrics(y_train_eval, best.predict(X_train_eval))
        val_m = compute_metrics(y_val, best.predict(X_val))
        y_pred = best.predict(X_test)
        results[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_macro_mean": float(search.cv_results_["mean_test_score"][i]),
            "cv_f1_macro_std": float(search.cv_results_["std_test_score"][i]),
            "cv_folds": cv_folds, "fit_seconds": round(seconds, 1), "n_train_rows_used": len(X_fit),
            "train": train_m, "validation": val_m, "test": compute_metrics(y_test, y_pred),
            "overfit_gap_f1": float(train_m["f1_macro"] - val_m["f1_macro"]),
            "test_confusion_matrix": confusion(y_test, y_pred, classes),
            "test_report": per_class_report(y_test, y_pred, classes),
            "roc": roc_data(y_test, get_scores(best, X_test), classes),
        }
        logger.info("CV F1 %.4f | val F1 %.4f | test F1 %.4f | %.1fs", results[name]["cv_f1_macro_mean"],
                    val_m["f1_macro"], results[name]["test"]["f1_macro"], seconds)

    best_name = max(results, key=lambda n: (round(results[n]["validation"]["f1_macro"], 6), -results[n]["overfit_gap_f1"]))
    logger.info("Best model (validation macro-F1): %s", best_name)

    version = f"{MODEL_VERSION}+{datetime.now().strftime('%Y%m%d%H%M%S')}"
    trained_at = datetime.now().isoformat(timespec="seconds")
    confidence_type = "probability" if hasattr(fitted[best_name], "predict_proba") else "softmax_of_decision_function"
    summary = {"model_version": version, "best_model": best_name, "trained_at": trained_at, "classes": classes,
               "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
               "n_input_columns": len(base_columns), "n_model_features": len(feature_columns),
               "class_distribution": df[LABEL_COLUMN].value_counts().to_dict(),
               "confidence_type": confidence_type, "selection_metric": "validation macro-F1",
               "best_params": results[best_name]["best_params"], "test": results[best_name]["test"]}
    all_results = {**summary, "models": results, "depth_curve": depth_curve(X_train, y_train, X_val, y_val, seed=seed)}

    bundle = {"pipeline": fitted[best_name], "model_name": best_name, "model_version": version, "classes": classes,
              "input_columns": base_columns, "feature_columns": feature_columns, "confidence_type": confidence_type,
              "trained_at": trained_at, "test_metrics": results[best_name]["test"]}
    model_path = models_dir / MODEL_FILE.name
    joblib.dump(bundle, model_path, compress=3)
    (models_dir / "versions").mkdir(exist_ok=True)
    joblib.dump(bundle, models_dir / "versions" / f"model_v{version.replace('+', '_')}.joblib", compress=3)
    save_json(summary, models_dir / MODEL_METADATA_FILE.name)
    save_json(all_results, metrics_dir / EVALUATION_RESULTS.name)
    results_table(all_results).to_csv(metrics_dir / "model_comparison.csv", index=False)
    generate_report_figures(all_results, figures_dir)

    demo_idx = stratified_subsample(X_test, y_test, min(300, len(X_test)), seed)[0].index
    Path(demo_sample_file).parent.mkdir(parents=True, exist_ok=True)
    df.loc[demo_idx].sample(frac=1, random_state=seed).to_csv(demo_sample_file, index=False)
    logger.info("Saved model -> %s", model_path)
    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=PROCESSED_FILE)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--slow-model-rows", type=int, default=15000)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    try:
        res = train_all(a.processed, quick=a.quick, cv_folds=a.cv_folds, slow_model_rows=a.slow_model_rows,
                        n_jobs=a.n_jobs, seed=a.seed)
    except (DataNotFoundError, DataValidationError) as exc:
        raise SystemExit(f"\nERROR: {exc}")
    table = results_table(res)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    print("\nTest-set comparison:\n", table.round(4).to_string(index=False))
    print(f"\nBest model: {res['best_model']}  (version {res['model_version']})")


if __name__ == "__main__":
    main()
