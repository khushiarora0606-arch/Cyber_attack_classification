from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.preprocessing import label_binarize

from src.utils import EVALUATION_RESULTS, FIGURES_DIR, get_logger, load_json

logger = get_logger(__name__)

DARK_RC = {"figure.facecolor": "#0F1A2D", "axes.facecolor": "#0F1A2D", "savefig.facecolor": "#0F1A2D",
           "axes.edgecolor": "#2B3A55", "axes.labelcolor": "#C9D4E5", "text.color": "#C9D4E5",
           "xtick.color": "#9FB0C8", "ytick.color": "#9FB0C8", "grid.color": "#1E2B44",
           "legend.facecolor": "#0F1A2D", "legend.edgecolor": "#2B3A55"}
PALETTE = ["#5B8DEF", "#F2B84B", "#46C2A5", "#EF6B6B", "#A78BFA", "#F28DB2", "#7DD3FC", "#9CA3AF"]


def _rc(dark):
    return plt.rc_context(DARK_RC if dark else {})


def compute_metrics(y_true, y_pred):
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return {k: float(v) for k, v in out.items()}


def per_class_report(y_true, y_pred, classes):
    return classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)


def confusion(y_true, y_pred, classes):
    return confusion_matrix(y_true, y_pred, labels=classes)


def get_scores(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    d = pipeline.decision_function(X)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def roc_data(y_true, scores, classes, max_points=200):
    y_bin = label_binarize(np.asarray(y_true), classes=classes)
    result = {}
    for i, cls in enumerate(classes):
        positives = int(y_bin[:, i].sum())
        if positives in (0, len(y_bin)):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        if len(fpr) > max_points:
            keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
            fpr, tpr = fpr[keep], tpr[keep]
        result[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
    return result


def plot_confusion_matrix(cm, classes, title="Confusion matrix", dark=False, normalize=True):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    with _rc(dark):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(shown, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues", xticklabels=classes,
                    yticklabels=classes, cbar=False, ax=ax, linewidths=0.5,
                    linecolor="#0F1A2D" if dark else "white")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(title + (" (row-normalised)" if normalize else ""))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
    return fig


def plot_roc_curves(roc, title="ROC curves (one-vs-rest)", dark=False):
    with _rc(dark):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for color, (cls, d) in zip(PALETTE * 2, roc.items()):
            ax.plot(d["fpr"], d["tpr"], color=color, lw=2, label=f"{cls} (AUC {d['auc']:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#6B7C99", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
    return fig


def plot_model_comparison(table, dark=False):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    with _rc(dark):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(table))
        for i, (m, lab) in enumerate(zip(metrics, labels)):
            ax.bar(x + (i - 1.5) * 0.2, table[m], 0.2, label=lab, color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels(table["model"], rotation=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (test set)")
        ax.set_title("Model comparison on the held-out test set")
        ax.legend(ncol=4, fontsize=8, loc="lower center")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
    return fig


def plot_class_distribution(counts, title="Class distribution", dark=False, log=False):
    items = sorted(counts.items(), key=lambda kv: kv[1])
    with _rc(dark):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.barh([k for k, _ in items], [v for _, v in items],
                color=[PALETTE[2] if k == "BENIGN" else PALETTE[0] for k, _ in items])
        if log:
            ax.set_xscale("log")
        for i, (_, v) in enumerate(items):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8)
        ax.set_xlabel("Records" + (" (log scale)" if log else ""))
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
    return fig


def plot_depth_curve(depths, train_scores, val_scores, dark=False):
    with _rc(dark):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.plot(depths, train_scores, marker="o", color=PALETTE[0], label="Training F1 (macro)")
        ax.plot(depths, val_scores, marker="o", color=PALETTE[1], label="Validation F1 (macro)")
        ax.set_xlabel("Decision tree max_depth")
        ax.set_ylabel("Macro F1")
        ax.set_title("Underfitting vs overfitting (Decision Tree)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
    return fig


def results_table(results):
    rows = []
    for name, r in results["models"].items():
        row = {"model": name, "cv_f1_macro": r["cv_f1_macro_mean"], "fit_seconds": r["fit_seconds"]}
        for split in ("train", "validation", "test"):
            for k, v in r[split].items():
                row[k if split == "test" else f"{split}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def generate_report_figures(results, figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(figures_dir / name)

    classes = results["classes"]
    save(plot_model_comparison(results_table(results)), "model_comparison.png")
    for name, r in results["models"].items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        save(plot_confusion_matrix(r["test_confusion_matrix"], classes, f"{name} - test set"), f"confusion_matrix_{slug}.png")
        if r.get("roc"):
            save(plot_roc_curves(r["roc"], f"{name} - ROC (test set)"), f"roc_{slug}.png")
    if results.get("class_distribution"):
        save(plot_class_distribution(results["class_distribution"], "Class distribution (modelling data)"), "class_distribution.png")
    dc = results.get("depth_curve")
    if dc:
        save(plot_depth_curve(dc["depths"], dc["train_f1"], dc["val_f1"]), "depth_curve.png")
    return saved


def main():
    results = load_json(EVALUATION_RESULTS)
    if results is None:
        raise SystemExit(f"ERROR: {EVALUATION_RESULTS} not found. Train first: python -m src.model_training")
    saved = generate_report_figures(results)
    logger.info("Re-created %d figures in %s", len(saved), FIGURES_DIR)
    print(results_table(results)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
