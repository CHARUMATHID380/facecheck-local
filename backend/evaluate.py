"""
Face Recognition System — Evaluation Script
=============================================
Measures system performance using a labelled test dataset.

Metrics computed
----------------
  Per-identity  : precision, recall, F1
  System-wide   : macro / weighted averages
  Rejection     : False Acceptance Rate (FAR), False Rejection Rate (FRR)
  Similarity    : mean cosine similarity for true-match pairs
  Confusion matrix (saved as PNG)
  ROC curve + AUC (saved as PNG)
  Full results saved to evaluation_results.json

Dataset layout expected
-----------------------
  test_dataset/
    ├── known/
    │     ├── alice/          ← one sub-folder per enrolled identity
    │     │     ├── img1.jpg
    │     │     └── img2.jpg
    │     └── bob/
    │           └── img1.jpg
    └── unknown/              ← images of people NOT in the enrolled DB
          ├── stranger1.jpg
          └── stranger2.jpg

Usage
-----
  # 1. Enroll the 'known' identities first:
  python evaluate.py --enroll

  # 2. Run evaluation (both known and unknown):
  python evaluate.py --evaluate

  # 3. Do both in one shot:
  python evaluate.py --enroll --evaluate

  # Override threshold:
  python evaluate.py --enroll --evaluate --threshold 0.35

  # Use a custom dataset path:
  python evaluate.py --enroll --evaluate --dataset /path/to/test_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root and imports
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from face_recognition_system import (  # noqa: E402
    COSINE_THRESHOLD,
    clear_database,
    enroll_face,
    identify_face,
    list_enrolled_faces,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = BACKEND_DIR / "test_dataset"
RESULTS_PATH = BACKEND_DIR / "evaluation_results.json"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(folder: Path) -> list[Path]:
    return [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def _banner(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Enrollment phase
# ---------------------------------------------------------------------------

def enroll_phase(known_dir: Path, *, clear_first: bool = True) -> dict[str, int]:
    """Enroll all identities found under known_dir/"""
    _banner("ENROLLMENT PHASE")

    if not known_dir.exists():
        print(f"[WARN] known/ directory not found: {known_dir}")
        print("       Create test_dataset/known/<name>/*.jpg and re-run.")
        return {}

    if clear_first:
        clear_database()
        print("  Cleared existing database.")

    stats: dict[str, int] = {}
    identity_dirs = sorted([d for d in known_dir.iterdir() if d.is_dir()])

    if not identity_dirs:
        print("[WARN] No identity sub-folders found under known/")
        return stats

    for identity_dir in identity_dirs:
        name = identity_dir.name
        images = _collect_images(identity_dir)
        enrolled = 0
        for i, img_path in enumerate(images):
            try:
                # Enroll first image as primary; subsequent ones as overwrite=False
                # (DeepFace averages aren't used here; each image = 1 enrollment slot)
                # For multi-image enrollment, we use a numbered suffix strategy.
                enroll_name = name if i == 0 else f"{name}_{i}"
                enroll_face(str(img_path), enroll_name, replace=True)
                enrolled += 1
                print(f"  ✓  Enrolled '{enroll_name}' from {img_path.name}")
            except Exception as exc:
                print(f"  ✗  Skip {img_path.name}: {exc}")
        stats[name] = enrolled

    total = sum(stats.values())
    print(f"\n  Enrolled {total} face(s) across {len(stats)} identit(ies).")
    return stats


# ---------------------------------------------------------------------------
# Evaluation phase
# ---------------------------------------------------------------------------

def evaluate_phase(
    known_dir: Path,
    unknown_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    """
    Run identification on all test images and collect metrics.

    Returns a rich results dictionary.
    """
    _banner(f"EVALUATION PHASE  (threshold = {threshold})")

    enrolled = {f["name"]: f["face_id"] for f in list_enrolled_faces()}
    if not enrolled:
        print("[ERROR] No faces enrolled. Run with --enroll first.")
        return {}

    # ── Collect test images ──────────────────────────────────────────────────

    known_probes: list[tuple[str, Path]] = []   # (true_name, img_path)
    if known_dir.exists():
        for identity_dir in sorted([d for d in known_dir.iterdir() if d.is_dir()]):
            for img in _collect_images(identity_dir):
                known_probes.append((identity_dir.name, img))

    unknown_probes: list[Path] = []
    if unknown_dir.exists():
        unknown_probes = _collect_images(unknown_dir)

    total_known = len(known_probes)
    total_unknown = len(unknown_probes)
    print(f"  Known probes  : {total_known}")
    print(f"  Unknown probes: {total_unknown}")

    if total_known == 0 and total_unknown == 0:
        print("[WARN] No test images found. Nothing to evaluate.")
        return {}

    # ── Run identification ───────────────────────────────────────────────────

    records: list[dict[str, Any]] = []  # one per probe image

    def _run(img_path: Path, true_label: str) -> dict[str, Any]:
        try:
            result = identify_face(str(img_path), threshold=threshold)
        except ValueError as exc:
            return {
                "image": str(img_path),
                "true_label": true_label,
                "predicted_label": "error",
                "identified": False,
                "similarity": 0.0,
                "distance": 1.0,
                "error": str(exc),
            }
        # Normalise predicted name: strip numbered suffix (e.g. "alice_1" → "alice")
        pred = result["name"]
        if pred != "unknown":
            pred_base = pred.rsplit("_", 1)[0] if pred[-1].isdigit() else pred
        else:
            pred_base = "unknown"

        return {
            "image": img_path.name,
            "true_label": true_label,
            "predicted_label": pred_base,
            "predicted_raw": pred,
            "identified": result["identified"],
            "similarity": result["similarity"],
            "distance": result["distance"],
        }

    print("\n  Running identification…")
    start = time.time()

    for true_name, img_path in known_probes:
        r = _run(img_path, true_name)
        records.append(r)
        icon = "✓" if r["predicted_label"] == true_name else "✗"
        print(
            f"  {icon}  [{true_name:15s}] → {r['predicted_label']:15s}  "
            f"sim={r['similarity']:5.1f}%  dist={r['distance']:.3f}"
        )

    for img_path in unknown_probes:
        r = _run(img_path, "unknown")
        records.append(r)
        icon = "✓" if r["predicted_label"] == "unknown" else "✗"
        print(
            f"  {icon}  [{'unknown':15s}] → {r['predicted_label']:15s}  "
            f"sim={r['similarity']:5.1f}%  dist={r['distance']:.3f}"
        )

    elapsed = time.time() - start
    print(f"\n  Identification complete in {elapsed:.1f}s  "
          f"({elapsed / max(len(records), 1):.2f}s / image)")

    # ── Compute metrics ──────────────────────────────────────────────────────

    metrics = _compute_metrics(records, threshold)
    _print_metrics(metrics)

    # ── Plots (optional — gracefully skipped if matplotlib not installed) ────

    try:
        _plot_confusion_matrix(records, metrics["labels"])
        _plot_roc(records, threshold)
    except ImportError:
        print("\n  [INFO] matplotlib not installed — skipping plots.")

    # ── Save results ─────────────────────────────────────────────────────────

    output: dict[str, Any] = {
        "threshold": threshold,
        "total_known_probes": total_known,
        "total_unknown_probes": total_unknown,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "records": records,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to: {RESULTS_PATH}")

    return output


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def _compute_metrics(
    records: list[dict[str, Any]],
    threshold: float = COSINE_THRESHOLD,
) -> dict[str, Any]:
    """Compute per-identity + system-wide metrics."""

    labels = sorted({r["true_label"] for r in records})

    # ── Per-identity TP/FP/FN ────────────────────────────────────────────────
    per_identity: dict[str, dict[str, int]] = {
        lbl: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for lbl in labels
    }
    for r in records:
        true = r["true_label"]
        pred = r["predicted_label"]
        if pred == "error":
            per_identity[true]["FN"] += 1
            continue
        for lbl in labels:
            is_true_pos = (true == lbl)
            is_pred_pos = (pred == lbl)
            if is_true_pos and is_pred_pos:
                per_identity[lbl]["TP"] += 1
            elif not is_true_pos and is_pred_pos:
                per_identity[lbl]["FP"] += 1
            elif is_true_pos and not is_pred_pos:
                per_identity[lbl]["FN"] += 1
            else:
                per_identity[lbl]["TN"] += 1

    per_identity_metrics: dict[str, dict[str, float]] = {}
    for lbl, counts in per_identity.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_identity_metrics[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": counts["TP"] + counts["FN"],
        }

    # ── Overall accuracy ─────────────────────────────────────────────────────
    correct = sum(1 for r in records if r["predicted_label"] == r["true_label"])
    accuracy = correct / len(records) if records else 0.0

    # ── Macro / weighted averages ─────────────────────────────────────────────
    supports = [v["support"] for v in per_identity_metrics.values()]
    total_support = sum(supports)

    macro_precision = np.mean([v["precision"] for v in per_identity_metrics.values()])
    macro_recall    = np.mean([v["recall"]    for v in per_identity_metrics.values()])
    macro_f1        = np.mean([v["f1"]        for v in per_identity_metrics.values()])

    weighted_precision = (
        sum(v["precision"] * v["support"] for v in per_identity_metrics.values())
        / total_support
        if total_support > 0
        else 0.0
    )
    weighted_recall = (
        sum(v["recall"] * v["support"] for v in per_identity_metrics.values())
        / total_support
        if total_support > 0
        else 0.0
    )
    weighted_f1 = (
        sum(v["f1"] * v["support"] for v in per_identity_metrics.values())
        / total_support
        if total_support > 0
        else 0.0
    )

    # ── FAR / FRR ─────────────────────────────────────────────────────────────
    # FAR: fraction of unknown/impostors accepted as known
    # FRR: fraction of genuine probes rejected as unknown
    unknown_records = [r for r in records if r["true_label"] == "unknown"]
    known_records   = [r for r in records if r["true_label"] != "unknown"]

    false_accepts  = sum(1 for r in unknown_records if r["predicted_label"] != "unknown")
    false_rejects  = sum(1 for r in known_records   if r["predicted_label"] == "unknown")

    far = false_accepts / len(unknown_records) if unknown_records else 0.0
    frr = false_rejects / len(known_records)   if known_records   else 0.0

    # ── Similarity stats for genuine pairs ───────────────────────────────────
    genuine_sims = [
        r["similarity"]
        for r in known_records
        if r["predicted_label"] == r["true_label"]
    ]
    impostor_sims = [
        r["similarity"]
        for r in records
        if r["predicted_label"] != r["true_label"]
    ]

    return {
        "labels": labels,
        "accuracy": round(float(accuracy), 4),
        "correct": correct,
        "total": len(records),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall":    round(float(macro_recall), 4),
        "macro_f1":        round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall":    round(float(weighted_recall), 4),
        "weighted_f1":        round(float(weighted_f1), 4),
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "genuine_mean_similarity":   round(float(np.mean(genuine_sims)) if genuine_sims else 0.0, 2),
        "genuine_std_similarity":    round(float(np.std(genuine_sims))  if genuine_sims else 0.0, 2),
        "impostor_mean_similarity":  round(float(np.mean(impostor_sims)) if impostor_sims else 0.0, 2),
        "per_identity": per_identity_metrics,
    }


def _print_metrics(m: dict[str, Any]) -> None:
    _banner("RESULTS SUMMARY")
    print(f"  Accuracy       : {m['accuracy'] * 100:.2f}%  ({m['correct']}/{m['total']})")
    print(f"  Macro P/R/F1   : {m['macro_precision']:.3f} / {m['macro_recall']:.3f} / {m['macro_f1']:.3f}")
    print(f"  Weighted P/R/F1: {m['weighted_precision']:.3f} / {m['weighted_recall']:.3f} / {m['weighted_f1']:.3f}")
    print(f"  FAR (impostor acceptance): {m['FAR'] * 100:.2f}%  ({m['false_accepts']} false accepts)")
    print(f"  FRR (genuine rejection)  : {m['FRR'] * 100:.2f}%  ({m['false_rejects']} false rejects)")
    print(f"  Genuine  similarity: {m['genuine_mean_similarity']:.1f}% ± {m['genuine_std_similarity']:.1f}%")
    print(f"  Impostor similarity: {m['impostor_mean_similarity']:.1f}%")
    print()
    print(f"  {'Identity':<20} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")
    for lbl, v in sorted(m["per_identity"].items()):
        print(
            f"  {lbl:<20} {v['precision']:>10.3f} {v['recall']:>10.3f} "
            f"{v['f1']:>8.3f} {v['support']:>9}"
        )


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    records: list[dict[str, Any]],
    labels: list[str],
) -> None:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    n = len(labels)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)

    for r in records:
        true = r["true_label"]
        pred = r["predicted_label"]
        if pred == "error":
            pred = "unknown"
        if true in label_idx and pred in label_idx:
            matrix[label_idx[true]][label_idx[pred]] += 1

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    out = BACKEND_DIR / "confusion_matrix.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to: {out}")


def _plot_roc(
    records: list[dict[str, Any]],
    threshold: float,
) -> None:
    """
    Plot a binary ROC curve treating identification as:
      positive = correctly identified as a known person
      negative = correctly rejected as unknown
    """
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    thresholds = np.linspace(0.0, 1.0, 200)
    tprs, fprs = [], []

    for t in thresholds:
        # Recompute decisions at each threshold using stored distances
        tp, fp, tn, fn = 0, 0, 0, 0
        for r in records:
            is_genuine = r["true_label"] != "unknown"
            accepted = r["distance"] <= t  # distance <= t → "known"
            if is_genuine and accepted:
                tp += 1
            elif not is_genuine and accepted:
                fp += 1
            elif is_genuine and not accepted:
                fn += 1
            else:
                tn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    auc = float(np.trapz(tprs_arr[::-1], fprs_arr[::-1]))

    # Current operating point
    op_tp, op_fp, op_tn, op_fn = 0, 0, 0, 0
    for r in records:
        is_genuine = r["true_label"] != "unknown"
        accepted = r["distance"] <= threshold
        if is_genuine and accepted:
            op_tp += 1
        elif not is_genuine and accepted:
            op_fp += 1
        elif is_genuine and not accepted:
            op_fn += 1
        else:
            op_tn += 1
    op_tpr = op_tp / (op_tp + op_fn) if (op_tp + op_fn) > 0 else 0.0
    op_fpr = op_fp / (op_fp + op_tn) if (op_fp + op_tn) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.scatter(
        [op_fpr], [op_tpr],
        color="red", zorder=5,
        label=f"Operating point (t={threshold:.2f})",
    )
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
        xlim=[0, 1],
        ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = BACKEND_DIR / "roc_curve.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved to: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Face Recognition Identification System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Enroll identities from test_dataset/known/ before evaluating.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation pass.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=COSINE_THRESHOLD,
        help=f"Cosine-distance rejection threshold (default: {COSINE_THRESHOLD}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help=f"Root path of the test dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do NOT clear the database before enrolling (append mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.enroll and not args.evaluate:
        print("Nothing to do. Pass --enroll, --evaluate, or both.")
        print("Run with --help for full usage.")
        sys.exit(0)

    dataset_root = Path(args.dataset)
    known_dir    = dataset_root / "known"
    unknown_dir  = dataset_root / "unknown"

    if args.enroll:
        enroll_phase(known_dir, clear_first=not args.no_clear)

    if args.evaluate:
        results = evaluate_phase(known_dir, unknown_dir, threshold=args.threshold)
        if results:
            m = results["metrics"]
            print(
                f"\n  ✓  Accuracy {m['accuracy'] * 100:.1f}%  |  "
                f"FAR {m['FAR'] * 100:.1f}%  |  FRR {m['FRR'] * 100:.1f}%  |  "
                f"Macro-F1 {m['macro_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
