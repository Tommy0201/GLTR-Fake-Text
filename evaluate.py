"""
evaluate.py — Run GLTR models on text datasets and compute scores.

Supported datasets (--dataset flag):
  cheat  : https://huggingface.co/datasets/acmc/cheat
             columns: text, model (null=human)
  hc3    : https://huggingface.co/datasets/Hello-SimpleAI/HC3
             columns: id, question, human_answers, chatgpt_answers, source

Usage:
    python evaluate.py --dataset cheat --limit 100 --models BERT --output results.csv
    python evaluate.py --dataset hc3   --limit 100 --models BERT --output hc3BERT.csv
    python evaluate.py --dataset hc3   --limit 200 --models BERT RoBERTa --output hc3All.csv
"""

import argparse
import csv
import time

import numpy as np


# ---------------------------------------------------------------------------
# GLTR scoring
# ---------------------------------------------------------------------------

def gltr_score(real_topk):
    ranks = np.array([r for r, _ in real_topk], dtype=float)
    probs = np.array([p for _, p in real_topk], dtype=float)
    n = len(ranks)
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return {
        "frac_top10":   float(np.sum(ranks < 10) / n),
        "frac_top100":  float(np.sum((ranks >= 10)  & (ranks < 100))  / n),
        "frac_top1000": float(np.sum((ranks >= 100) & (ranks < 1000)) / n),
        "frac_rest":    float(np.sum(ranks >= 1000) / n),
        "mean_rank":    float(np.mean(ranks)),
        "mean_prob":    float(np.mean(probs)),
        "perplexity":   float(np.exp(-np.mean(log_probs))),
    }


# ---------------------------------------------------------------------------
# Dataset loading — acmc/cheat
# ---------------------------------------------------------------------------

def load_cheat(limit=None):
    """
    Load acmc/cheat from HuggingFace.
    label = "human" when model column is null/empty, "ai" otherwise.
    Samples evenly across both classes when --limit is set.
    """
    from datasets import load_dataset

    print("Downloading acmc/cheat ...")
    ds = load_dataset("acmc/cheat", split="train")

    human_rows, ai_rows = [], []
    for i, row in enumerate(ds):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ai_model = (row.get("model") or "").strip()
        label = "ai" if ai_model else "human"
        entry = {
            "text":     text,
            "label":    label,
            "source":   row.get("id", f"item_{i}"),
            "ai_model": ai_model,
        }
        if label == "human":
            human_rows.append(entry)
        else:
            ai_rows.append(entry)

    if limit:
        half = limit // 2
        human_rows = human_rows[:half]
        ai_rows    = ai_rows[:limit - half]

    texts = human_rows + ai_rows
    print(f"Loaded {len(texts)} samples — {len(human_rows)} human, {len(ai_rows)} AI")
    return texts


# ---------------------------------------------------------------------------
# Dataset loading — Hello-SimpleAI/HC3
# ---------------------------------------------------------------------------

def load_hc3(limit=None):
    """
    Load Hello-SimpleAI/HC3 from HuggingFace.
    Each row contains a list of human_answers and chatgpt_answers.
    We expand them into individual entries, then sample evenly.
    """
    from datasets import load_dataset

    print("Downloading Hello-SimpleAI/HC3 ...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
        split="train",
    )

    human_rows, ai_rows = [], []
    for i, row in enumerate(ds):
        qid = row.get("id", f"item_{i}")
        for j, text in enumerate((row.get("human_answers") or [])):
            text = text.strip()
            if text:
                human_rows.append({
                    "text":     text,
                    "label":    "human",
                    "source":   f"{qid}_h{j}",
                    "ai_model": "",
                })
        for j, text in enumerate((row.get("chatgpt_answers") or [])):
            text = text.strip()
            if text:
                ai_rows.append({
                    "text":     text,
                    "label":    "ai",
                    "source":   f"{qid}_a{j}",
                    "ai_model": "chatgpt",
                })

    if limit:
        half = limit // 2
        human_rows = human_rows[:half]
        ai_rows    = ai_rows[:limit - half]

    texts = human_rows + ai_rows
    print(f"Loaded {len(texts)} samples — {len(human_rows)} human, {len(ai_rows)} AI")
    return texts


# ---------------------------------------------------------------------------
# Dataset loading — SJTU-CL/ArguGPT
# ---------------------------------------------------------------------------

def load_argugpt(limit=None, splits=("train", "dev", "test")):
    """
    Load SJTU-CL/ArguGPT from HuggingFace.
    NOTE: This dataset contains only machine-generated essays (no human text).
    All rows are labelled "ai".  Run classify.py only after combining with
    human data from another source (e.g. cheat or hc3).
    Columns used: id, text, model
    """
    from datasets import load_dataset

    print("Downloading SJTU-CL/ArguGPT ...")
    ai_rows = []
    for split in splits:
        ds = load_dataset(
            "csv",
            data_files=f"hf://datasets/SJTU-CL/ArguGPT/machine-{split}.csv",
            split="train",
        )
        for i, row in enumerate(ds):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            ai_rows.append({
                "text":     text,
                "label":    "ai",
                "source":   row.get("id", f"{split}_{i}"),
                "ai_model": row.get("model", ""),
            })

    if limit:
        ai_rows = ai_rows[:limit]

    print(f"Loaded {len(ai_rows)} AI samples from ArguGPT (no human data in this dataset)")
    return ai_rows


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(texts, model_names, topk=40):
    from backend.api import AVAILABLE_MODELS

    results = []

    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            print(f"[warn] '{model_name}' not registered, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Loading GLTR model: {model_name}")
        lm = AVAILABLE_MODELS[model_name]()

        for entry in texts:
            text   = entry["text"]
            label  = entry["label"]
            source = str(entry["source"])

            print(f"  [{model_name}] {source[:40]:<40} ({label}) ...", end=" ", flush=True)
            t0 = time.time()
            try:
                payload  = lm.check_probabilities(text, topk=topk)
                elapsed  = time.time() - t0
                scores   = gltr_score(payload["real_topk"])
                n_tokens = len(payload["real_topk"])

                results.append({
                    "gltr_model": model_name,
                    "source":     source,
                    "label":      label,
                    "ai_model":   entry.get("ai_model", ""),
                    "n_tokens":   n_tokens,
                    "elapsed_s":  round(elapsed, 3),
                    **{k: round(v, 4) for k, v in scores.items()},
                })
                print(f"done  top10={scores['frac_top10']:.2%}  "
                      f"mean_rank={scores['mean_rank']:.0f}  "
                      f"ppl={scores['perplexity']:.1f}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {e}")
                results.append({
                    "gltr_model": model_name, "source": source,
                    "label": label, "n_tokens": 0,
                    "elapsed_s": round(elapsed, 3), "error": str(e),
                })

    return results


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(results, path):
    if not results:
        print("No results to save.")
        return
    dest_path = "eval/results/" + path
    fieldnames = list(dict.fromkeys(k for row in results for k in row))
    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows → {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    from collections import defaultdict
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results.")
        return

    grouped = defaultdict(list)
    for r in valid:
        grouped[(r["gltr_model"], r["label"])].append(r)

    print(f"\n{'='*60}")
    print("SUMMARY  (mean per GLTR model × label)")
    print(f"{'='*60}")
    print(f"  {'GLTR':<12} {'Label':<8} {'N':>4}  {'top10':>7}  {'mean_rank':>10}  {'perplexity':>12}")
    print("  " + "-" * 58)
    for (gltr, label), rows in sorted(grouped.items()):
        print(f"  {gltr:<12} {label:<8} {len(rows):>4}  "
              f"{np.mean([r['frac_top10'] for r in rows]):>7.2%}  "
              f"{np.mean([r['mean_rank']  for r in rows]):>10.1f}  "
              f"{np.mean([r['perplexity'] for r in rows]):>12.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GLTR on text datasets")
    parser.add_argument("--dataset", choices=["cheat", "hc3", "argugpt"], default="cheat",
                        help="Dataset to use: 'cheat', 'hc3', or 'argugpt'")
    parser.add_argument("--limit",   type=int, default=100,
                        help="Total samples to evaluate, split evenly human/AI (default: 100)")
    parser.add_argument("--models",  nargs="+", default=["BERT"],
                        help="GLTR models to run (default: BERT)")
    parser.add_argument("--topk",    type=int, default=40)
    parser.add_argument("--output",  type=str, default=None,
                        help="Save results to this CSV path inside eval/results/")
    args = parser.parse_args()

    if args.dataset == "hc3":
        texts = load_hc3(limit=args.limit)
    elif args.dataset == "argugpt":
        texts = load_argugpt(limit=args.limit)
    else:
        texts = load_cheat(limit=args.limit)

    results = evaluate(texts, args.models, topk=args.topk)
    print_summary(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
