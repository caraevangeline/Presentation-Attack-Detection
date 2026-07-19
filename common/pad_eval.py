"""Model-agnostic PAD evaluation utilities: dataset loading against the Real/Spoof folder layout,
and metrics/plots given y_true + y_score.

Computes ROC-AUC, PR-AUC, EER, and APCER/BPCER/ACER/precision/recall/F1 (at both a configured threshold
and the EER threshold), plus confusion matrices, rendered as a metrics table image and ROC/PR curve plots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def rel_key(path, root):
    """Path relative to `root`, e.g. 'spoof/00962.png'. Used instead of bare filename since
    real/ and spoof/ can share filenames here.
    Args:
        path: full path to the file to generate a key for.
        root: common base directory that `path` is expressed relative to.
    """
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")


def list_images(input_dir):
    """List all image files directly inside `input_dir`, sorted by name.
    Args:
        input_dir: path to the folder to scan.
    """
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def gather_labeled_dataset(labeled_dir):
    """Labels are keyed by path relative to labeled_dir (e.g. 'spoof/00962.png'), not bare filename,
    since the two class folders can share filenames.
    Args:
        labeled_dir: root folder containing 'real/' and 'spoof/' subfolders.
    """
    labeled_dir = Path(labeled_dir)
    paths, labels = [], {}
    for sub, label in [("real", 0), ("spoof", 1)]:
        folder = labeled_dir / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected subfolder not found: {folder}")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(f)
                labels[rel_key(f, labeled_dir)] = label
    return paths, labels


def load_labels_csv(labels_csv):
    """Load a filename -> label CSV, accepting several common label spellings.
    Args:
        labels_csv: path to a CSV with 'filename' and 'label' columns.
    """
    label_map = {"1": 1, "attack": 1, "screen": 1, "screens": 1, "spoof": 1,
                 "0": 0, "bonafide": 0, "real": 0, "genuine": 0}
    labels = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            raw = row["label"].strip().lower()
            labels[row["filename"]] = label_map.get(raw, int(raw))
    return labels


def compute_pad_metrics(y_true, y_score, threshold):
    """Compute standard PAD evaluation metrics at both a given threshold and at the EER threshold, for comparison.

    "Attack" (label=1) is treated as the positive class throughout, per
    the APCER/BPCER convention: tp/fn are attacks, tn/fp are bonafide.

    Args:
        y_true: array of ground-truth labels (0 = real/bonafide, 1 = spoof/attack).
        y_score: array of predicted attack scores/probabilities.
        threshold: the configured decision threshold.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thr[eer_idx])

    def metrics_at(t):
        preds = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        apcer = fn / (fn + tp) if (fn + tp) else float("nan")  # attacks missed
        bpcer = fp / (fp + tn) if (fp + tn) else float("nan")  # bonafide false-rejected
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")  # == 1 - APCER
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) and not np.isnan(precision) and not np.isnan(recall)
              else float("nan"))
        return {
            "threshold": float(t), "APCER": float(apcer), "BPCER": float(bpcer),
            "ACER": float((apcer + bpcer) / 2),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    return {
        "n_bonafide": int((y_true == 0).sum()),
        "n_attack": int((y_true == 1).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "at_configured_threshold": metrics_at(threshold),
        "at_eer_threshold": metrics_at(eer_threshold),
    }


def save_metrics_table(metrics, output_dir):
    """Save a PAD metrics dict to disk as both JSON (full detail) and a
    flattened CSV (one row per metric, for easy viewing/spreadsheet use).

    Args:
        metrics: dict as returned by compute_pad_metrics().
        output_dir: folder to write the output files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "pad_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    rows = [
        ("n_bonafide", metrics["n_bonafide"]),
        ("n_attack", metrics["n_attack"]),
        ("ROC_AUC", round(metrics["roc_auc"], 4)),
        ("PR_AUC", round(metrics["pr_auc"], 4)),
        ("EER", round(metrics["eer"], 4)),
        ("EER_threshold", round(metrics["eer_threshold"], 6)),
    ]
    for tag, m in [("configured_threshold", metrics["at_configured_threshold"]),
                   ("eer_threshold", metrics["at_eer_threshold"])]:
        rows += [
            (f"threshold@{tag}", round(m["threshold"], 6)),
            (f"APCER@{tag}", round(m["APCER"], 4)),
            (f"BPCER@{tag}", round(m["BPCER"], 4)),
            (f"ACER@{tag}", round(m["ACER"], 4)),
            (f"precision@{tag}", round(m["precision"], 4)),
            (f"recall@{tag}", round(m["recall"], 4)),
            (f"f1@{tag}", round(m["f1"], 4)),
            (f"TN@{tag}", m["tn"]), (f"FP@{tag}", m["fp"]),
            (f"FN@{tag}", m["fn"]), (f"TP@{tag}", m["tp"]),
        ]

    csv_path = output_dir / "pad_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    return csv_path, output_dir / "pad_metrics.json"


def save_metrics_image(metrics, output_dir):
    """Render the metrics table + confusion matrices as a single PNG.
    Args:
        metrics: dict as returned by compute_pad_metrics().
        output_dir: folder to write the output files into.
    """
    matplotlib.use("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_m = metrics["at_configured_threshold"]
    eer_m = metrics["at_eer_threshold"]

    overall_rows = [
        ["n_bonafide", str(metrics["n_bonafide"])],
        ["n_attack", str(metrics["n_attack"])],
        ["ROC_AUC", f"{metrics['roc_auc']:.4f}"],
        ["PR_AUC", f"{metrics['pr_auc']:.4f}"],
        ["EER", f"{metrics['eer']:.4f}"],
        ["EER_threshold", f"{metrics['eer_threshold']:.6f}"],
    ]

    metric_keys = ["threshold", "APCER", "BPCER", "ACER", "precision", "recall", "f1"]
    per_threshold_rows = [
        [k, (f"{cfg_m[k]:.6f}" if k == "threshold" else f"{cfg_m[k]:.4f}"),
         (f"{eer_m[k]:.6f}" if k == "threshold" else f"{eer_m[k]:.4f}")]
        for k in metric_keys
    ]

    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 2.0, 2.2], hspace=0.6, wspace=0.4)

    ax_overall = fig.add_subplot(gs[0, :])
    ax_overall.axis("off")
    ax_overall.set_title("Overall PAD metrics", fontweight="bold", pad=30)
    t1 = ax_overall.table(cellText=overall_rows, colLabels=["metric", "value"],
                          loc="center", cellLoc="center")
    for col in range(2):
        t1[(0, col)].set_text_props(fontweight="bold")
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1, 1.4)

    ax_thr = fig.add_subplot(gs[1, :])
    ax_thr.axis("off")
    ax_thr.set_title("Metrics by operating threshold", fontweight="bold")
    t2 = ax_thr.table(cellText=per_threshold_rows,
                      colLabels=["metric", "configured_threshold", "eer_threshold"],
                      loc="center", cellLoc="center")
    for col in range(3):
        t2[(0, col)].set_text_props(fontweight="bold")
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1, 1.4)

    for i, (tag, m) in enumerate([("Configured threshold", cfg_m), ("EER threshold", eer_m)]):
        ax = fig.add_subplot(gs[2, i])
        mat = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        vmax = max(max(row) for row in mat)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: bonafide", "Pred: attack"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True: bonafide", "True: attack"])
        for r in range(2):
            for c in range(2):
                val = mat[r][c]
                color = "white" if val > vmax / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center",
                        color=color, fontweight="bold", fontsize=12)
        ax.set_title(f"Confusion matrix\n{tag} (thr={m['threshold']:.4f})", fontsize=9)

    fig.suptitle("PAD Evaluation Summary", fontsize=14, fontweight="bold")
    img_path = output_dir / "metrics_summary.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


def save_curves(y_true, y_score, metrics, output_dir):
    """Plot and save ROC and Precision-Recall curves as PNG files.
    Args:
        y_true: array of ground-truth labels (0 = real, 1 = spoof).
        y_score: array of predicted attack scores/probabilities.
        metrics: dict as returned by compute_pad_metrics().
        output_dir: folder to save the plot images into.
    """
    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    eer = metrics["eer"]
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER={eer:.3f}")
    ax.set_xlabel("False Positive Rate (BPCER)")
    ax.set_ylabel("True Positive Rate (1 - APCER)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP={metrics['pr_auc']:.3f})")
    baseline = metrics["n_attack"] / (metrics["n_attack"] + metrics["n_bonafide"])
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"baseline={baseline:.3f}")
    ax.set_xlabel("Recall (1 - APCER)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    pr_path = output_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    return roc_path, pr_path
