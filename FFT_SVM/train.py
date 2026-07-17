"""Train the FFT + SVM screen-attack detector (Method 1).

Data layout expected (matches the exercise's supplied data):
    <data_dir>/real/*.png|jpg   -> label 0 (bona fide)
    <data_dir>/spoof/*.png|jpg    -> label 1 (screen attack)

Why nested cross-validation:
There are only small attack examples. Picking hyperparameters and reporting
performance on the same CV split would leak optimism into the reported
numbers. We use an outer StratifiedKFold purely for an honest, out-of-fold
performance estimate, and an inner StratifiedKFold (via GridSearchCV) for
hyperparameter selection inside each outer training fold. The final
deployed model is a separate GridSearchCV refit on all available data.

Why SVM over a deep net:
Small positive examples is not enough to fine-tune a CNN like EfficientNet/MiniFASNet
without either massive overfitting or needing heavy synthetic
augmentation whose realism we could not verify. A small hand-crafted
feature vector (~40 dims) + RBF-SVM is far more sample-efficient and
remains fast to train (seconds) and to run at inference (milliseconds).
"""

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
    feats, ok_idx, failures = [], [], []
    for i, p in enumerate(paths):
        try:
            feats.append(extract_features(p))
            ok_idx.append(i)
        except Exception as e:
            failures.append((p, str(e)))
    return np.vstack(feats), np.array(ok_idx), failures


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ])


def nested_cv_evaluate(X, y, n_outer, n_inner, seed):
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
    """Youden's J statistic: maximizes (sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def summarize(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    apcer = fn / (fn + tp) if (fn + tp) else float("nan")   # attacks missed
    bpcer = fp / (fp + tn) if (fp + tn) else float("nan")   # bonafide rejected
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=str(
        Path(__file__).resolve().parents[1] / "test" / "data"))
    ap.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--n_outer_folds", type=int, default=5)
    ap.add_argument("--n_inner_folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data_dir}")
    paths, labels = load_dataset(args.data_dir)
    print(f"Found {len(paths)} images ({int((labels == 0).sum())} bonafide, "
          f"{int((labels == 1).sum())} attack)")

    print("Extracting FFT features...")
    X, ok_idx, failures = build_features(paths)
    y = labels[ok_idx]
    if failures:
        print(f"WARNING: {len(failures)} images failed to load and were skipped:")
        for p, err in failures[:10]:
            print(f"  {p}: {err}")

    print(f"\nRunning nested {args.n_outer_folds}x{args.n_inner_folds} CV "
          f"for an honest out-of-fold performance estimate...")
    oof_scores = nested_cv_evaluate(X, y, args.n_outer_folds, args.n_inner_folds, args.seed)
    threshold = pick_threshold(y, oof_scores)
    metrics = summarize(y, oof_scores, threshold)

    print("\n--- Out-of-fold (nested CV) metrics ---")
    print(json.dumps(metrics, indent=2))

    print("\nFitting final GridSearchCV on all data for the deployed model...")
    final_cv = StratifiedKFold(n_splits=args.n_inner_folds, shuffle=True, random_state=args.seed)
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
    main()
