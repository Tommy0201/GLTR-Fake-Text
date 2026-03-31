"""
classify.py — Binary human vs AI detection using GLTR statistics.

Usage:
    python classify.py --results results/results.csv
    python classify.py --results results/results.csv --threshold 0.5
"""

import argparse
import csv
import os
import sys

import numpy as np


def compute_thresholds(rows):
    """
    Derive per-feature midpoints from the actual data distribution.

    Instead of hardcoded thresholds (which break for cross-model detection,
    e.g. BERT scoring ChatGPT text), we compute the median for each class
    and use the midpoint as the decision boundary.  Direction (which way
    means AI) is inferred automatically from which class has a higher median.
    """
    human = [r for r in rows if r["true_binary"] == "human"]
    ai    = [r for r in rows if r["true_binary"] == "ai"]

    thresholds = {}
    for col in ["frac_top10", "frac_rest", "mean_rank"]:
        h_med = np.median([float(r[col]) for r in human])
        a_med = np.median([float(r[col]) for r in ai])
        thresholds[col] = {"human": h_med, "ai": a_med}

    return thresholds



def gltr_ai_score(row, thresholds):
    """
    Returns a score in [0, 1].  Higher = more likely AI-generated.

    For each feature, linearly interpolates between the human median (→ 0)
    and the AI median (→ 1).  Direction is inferred from the data, so this
    works regardless of which LM is used as the GLTR scorer.
    """
    scores = []
    for col in ["frac_top10", "frac_rest", "mean_rank"]:
        val = float(row[col])
        h   = thresholds[col]["human"]
        a   = thresholds[col]["ai"]
        if abs(a - h) < 1e-9:          # no separation — skip
            scores.append(0.5)
            continue
        s = (val - h) / (a - h)        # 0 = human median, 1 = AI median
        scores.append(float(np.clip(s, 0, 1)))

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",   default="eval/results/results.csv",
                        help="CSV produced by evaluate.py")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score >= threshold → AI  (default: 0.5)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Save text report to classify/<name>.txt")
    args = parser.parse_args()

    rows = []
    with open(args.results, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error") or not row.get("frac_top10"):
                continue
            rows.append(row)

    if not rows:
        sys.exit("No usable rows found in CSV.")

    # Collapse every non-human label to "ai"
    for r in rows:
        r["true_binary"] = "human" if r["label"] == "human" else "ai"

    has_human = any(r["true_binary"] == "human" for r in rows)
    has_ai    = any(r["true_binary"] == "ai"    for r in rows)

    if not has_human or not has_ai:
        print("ERROR: need both human and AI samples to compute accuracy.")
        print("Re-run: python evaluate.py --limit 100 --models BERT --output results.csv")
        sys.exit(1)

    thresholds = compute_thresholds(rows)
    print("\nData-adaptive thresholds (median per class):")
    for col, vals in thresholds.items():
        direction = "↑ AI" if vals["ai"] > vals["human"] else "↓ AI"
        print(f"  {col:<15}  human={vals['human']:.4f}  ai={vals['ai']:.4f}  {direction}")

    correct = 0
    prediction_rows = []
    print(f"\n{'GLTR':<12} {'True':<8} {'Score':>6}  {'Predicted':<14}  {'OK':>3}")
    print("-" * 52)

    for r in rows:
        score      = gltr_ai_score(r, thresholds)
        predicted  = "ai" if score >= args.threshold else "human"
        true_label = r["true_binary"]
        ok         = "✓" if predicted == true_label else "✗"
        if predicted == true_label:
            correct += 1
        print(f"{r['gltr_model']:<12} {true_label:<8} {score:>6.3f}  {predicted:<14}  {ok:>3}")
        prediction_rows.append({
            "gltr_model": r["gltr_model"],
            "source":     r.get("source", ""),
            "true_label": true_label,
            "score":      round(score, 4),
            "predicted":  predicted,
            "correct":    predicted == true_label,
        })

    total    = len(rows)
    n_human  = sum(1 for r in rows if r["true_binary"] == "human")
    n_ai     = total - n_human
    accuracy = correct / total

    print(f"\n{'='*52}")
    print(f"Samples   : {total}  ({n_human} human, {n_ai} AI)")
    print(f"Correct   : {correct}")
    print(f"Accuracy  : {accuracy:.1%}")
    print(f"Threshold : {args.threshold}")

    if args.output:
        stem = args.output.rsplit(".", 1)[0]

        txt_dest = os.path.join("eval", "classify", stem + ".txt")
        with open(txt_dest, "w") as f:
            f.write("Data-adaptive thresholds (median per class):\n")
            for col, vals in thresholds.items():
                direction = "↑ AI" if vals["ai"] > vals["human"] else "↓ AI"
                f.write(f"  {col:<15}  human={vals['human']:.4f}  ai={vals['ai']:.4f}  {direction}\n")

            f.write(f"\n{'GLTR':<12} {'True':<8} {'Score':>6}  {'Predicted':<14}  {'OK':>3}\n")
            f.write("-" * 52 + "\n")
            for pr in prediction_rows:
                ok = "✓" if pr["correct"] else "✗"
                f.write(f"{pr['gltr_model']:<12} {pr['true_label']:<8} {pr['score']:>6.3f}  {pr['predicted']:<14}  {ok:>3}\n")

            f.write(f"\n{'='*52}\n")
            f.write(f"Samples   : {total}  ({n_human} human, {n_ai} AI)\n")
            f.write(f"Correct   : {correct}\n")
            f.write(f"Accuracy  : {accuracy:.1%}\n")
            f.write(f"Threshold : {args.threshold}\n")
        print(f"Saved text report  → {txt_dest}")


if __name__ == "__main__":
    main()
