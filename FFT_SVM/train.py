"""Train the FFT + SVM screen-attack detector (Method 1).

Data layout (matches the exercise's supplied data):
    <data_dir>/real/*.png|jpg   -> label 0 (bona fide)
    <data_dir>/spoof/*.png|jpg  -> label 1 (screen attack)

Uses nested cross-validation: an outer StratifiedKFold gives an honest, out-of-fold performance estimate, while an inner
StratifiedKFold (via GridSearchCV) selects hyperparameters within each outer fold -- avoiding optimism from tuning and
evaluating on the same split. The final deployed model is a separate GridSearchCV refit on all available data.

Usage:
    python train.py /path/to/data --output_dir ./artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, extract_features

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001],
}


def load_dataset(data_dir):
    """Gather image paths and labels from data_dir's real/spoof subfolders.
    Args:
        data_dir: root folder containing 'real/' (label 0) and 'spoof/' (label 1) subfolders.
    """
    data_dir = Path(data_dir)
    paths, labels = [], []
    for sub, label in [("real", 0), ("spoof", 1)]:
        folder = data_dir / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected subfolder not found: {folder}")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
                labels.append(label)
    return paths, np.array(labels)


def build_features(paths):
    """Extract FFT features for a list of image paths, skipping and recording any that fail (e.g. unreadable files).
    Args:
        paths: list of image file path strings.
    """
    feats, ok_idx, failures = [], [], []
    for i, p in enumerate(paths):
        try:
            feats.append(extract_features(p))
            ok_idx.append(i)
        except Exception as e:
            failures.append((p, str(e)))
    return np.vstack(feats), np.array(ok_idx), failures


def make_pipeline():
    """Build the scaler + RBF-SVM pipeline used throughout training/eval.
    Returns:
        sklearn Pipeline: StandardScaler -> SVC(rbf, class_weight='balanced').
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ])


def nested_cv_evaluate(X, y, n_outer, n_inner, seed):
    """Run nested cross-validation: an outer loop for an honest, out-of-fold performance estimate, with hyperparameters
    selected by an inner GridSearchCV within each outer training fold.
    Args:
        X: feature matrix.
        y: label array (0=real, 1=spoof).
        n_outer: number of outer CV folds.
        n_inner: number of inner CV folds (for hyperparameter search).
        seed: random seed for both outer and inner fold splits.
    """
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof_scores = np.zeros(len(y))

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_pipeline(), PARAM_GRID, scoring="average_precision",
                          cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        oof_scores[te] = gs.predict_proba(X[te])[:, 1]
        print(f"  outer fold {fold + 1}/{n_outer}: best_params={gs.best_params_}, "
              f"inner best AP={gs.best_score_:.3f}")

    return oof_scores


def pick_threshold(y_true, scores):
    """Pick a decision threshold via Youden's J statistic (maximizes sensitivity + specificity - 1).
    Args:
        y_true: ground-truth label array (0=real, 1=spoof).
        scores: predicted attack-probability scores.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def summarize(y_true, scores, threshold):
    """Compute PAD evaluation metrics at a given threshold.
   Args:
       y_true: ground-truth label array (0=real, 1=spoof).
       scores: predicted attack-probability scores.
       threshold: decision threshold to apply.
    """
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    apcer = fn / (fn + tp) if (fn + tp) else float("nan")  # attacks missed
    bpcer = fp / (fp + tn) if (fp + tn) else float("nan")  # bonafide rejected
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "APCER_attack_miss_rate": float(apcer),
        "BPCER_bonafide_false_reject_rate": float(bpcer),
        "n_bonafide": int((y_true == 0).sum()),
        "n_attack": int((y_true == 1).sum()),
    }


def main(data_dir, output_dir, n_outer_folds, n_inner_folds, seed):
    """Train the FFT + SVM screen-attack detector: extract features, run nested CV for an honest performance estimate,
    pick a threshold, then fit and save the final deployed model.
    Args:
        data_dir: root folder with 'real/' and 'spoof/' subfolders.
        output_dir: folder to save the trained model, config, and CV metrics into.
        n_outer_folds: number of outer CV folds (performance estimate).
        n_inner_folds: number of inner CV folds (hyperparameter search), also used for the final model's GridSearchCV.
        seed: random seed for reproducible CV splits.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}")
    paths, labels = load_dataset(data_dir)
    print(f"Found {len(paths)} images ({int((labels == 0).sum())} bonafide, "
          f"{int((labels == 1).sum())} attack)")

    print("Extracting FFT features...")
    X, ok_idx, failures = build_features(paths)
    y = labels[ok_idx]
    if failures:
        print(f"WARNING: {len(failures)} images failed to load and were skipped:")
        for p, err in failures[:10]:
            print(f"  {p}: {err}")

    print(f"\nRunning nested {n_outer_folds}x{n_inner_folds} CV "
          f"for an honest out-of-fold performance estimate...")
    oof_scores = nested_cv_evaluate(X, y, n_outer_folds, n_inner_folds, seed)
    threshold = pick_threshold(y, oof_scores)
    metrics = summarize(y, oof_scores, threshold)

    print("\n--- Out-of-fold (nested CV) metrics ---")
    print(json.dumps(metrics, indent=2))

    print("\nFitting final GridSearchCV on all data for the deployed model...")
    final_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    final_gs = GridSearchCV(make_pipeline(), PARAM_GRID, scoring="average_precision",
                            cv=final_cv, n_jobs=-1)
    final_gs.fit(X, y)
    print(f"Final best_params: {final_gs.best_params_}")

    model_path = out_dir / "model.joblib"
    dump(final_gs.best_estimator_, model_path)

    config = {
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "final_best_params": final_gs.best_params_,
        "resize_dim": 384,
        "n_radial_bins": 32,
        "note": "attack_score >= threshold => predicted screen attack. "
                "Threshold picked via Youden's J on nested-CV out-of-fold "
                "scores; adjust in this file to trade off APCER vs BPCER.",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved config to {out_dir / 'config.json'}")
    print(f"Saved CV metrics to {out_dir / 'cv_metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=str(
        Path(__file__).resolve().parents[1] / "test" / "data"))
    ap.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--n_outer_folds", type=int, default=5)
    ap.add_argument("--n_inner_folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data_dir,
         args.output_dir,
         args.n_outer_folds,
         args.n_inner_folds,
         args.seed)
