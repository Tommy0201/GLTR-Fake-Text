"""Evaluate trained RoBERTa models on their respective test sets.

Usage:
    python evaluate_roberta.py --model models/roberta/roberta_no_essay --test-file train-dataset-text/Essay_LLMs_processed.csv
    python evaluate_roberta.py --model models/roberta/roberta_no_reuters --test-file train-dataset-text/Reuters_LLMs_processed.csv
    python evaluate_roberta.py --model models/roberta/roberta_no_wp --test-file train-dataset-text/WP_LLMs_processed.csv
    python evaluate_roberta.py --all  # Evaluate all odd-one-out folds
"""

import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_csv_data(path: Path):
    """Load preprocessed CSV with columns: label, text, source."""
    texts = []
    labels = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()

            if text and label:
                texts.append(text)
                labels.append(int(label))

    if not texts:
        raise ValueError(f"No valid text data found in {path}")

    return texts, labels


def evaluate_model(model_path, test_file, batch_size=16, max_length=512, device="cpu"):
    """Evaluate a trained RoBERTa model on a test file."""
    model_path = Path(model_path)
    test_file = Path(test_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    print(f"\nLoading model from: {model_path}")
    print(f"Loading test data from: {test_file}")

    # Load test data
    test_texts, test_labels = load_csv_data(test_file)
    print(f"Loaded {len(test_texts)} test samples")

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Create test dataset and loader
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    all_predictions = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')

    results = {
        "model_path": str(model_path),
        "test_file": str(test_file),
        "num_samples": len(test_labels),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return results


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*70)
    print(f"Model:       {results['model_path']}")
    print(f"Test File:   {results['test_file']}")
    print(f"Samples:     {results['num_samples']}")
    print("="*70)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("="*70)


def save_results_csv(results_list, output_file):
    """Save evaluation results to CSV."""
    if not results_list:
        print("No results to save.")
        return

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results_list[0].keys())
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RoBERTa models")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all odd-one-out folds"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/roberta_results.csv",
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for evaluation"
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    results_list = []

    if args.all:
        # Evaluate all odd-one-out folds
        folds = [
            ("models/roberta/roberta_no_essay", "train-dataset-text/Essay_LLMs_processed.csv"),
            ("models/roberta/roberta_no_reuters", "train-dataset-text/Reuters_LLMs_processed.csv"),
            ("models/roberta/roberta_no_wp", "train-dataset-text/WP_LLMs_processed.csv"),
            ("models/roberta/roberta_no_cheat", "train-dataset-text/cheat_processed.csv"),
            ("models/roberta/roberta_no_hc3", "train-dataset-text/hc3_processed.csv"),
        ]

        for model_path, test_file in folds:
            try:
                results = evaluate_model(model_path, test_file, args.batch_size, device=device)
                print_results(results)
                results_list.append(results)
            except FileNotFoundError as e:
                print(f"[warn] Skipping: {e}")

        if results_list:
            save_results_csv(results_list, args.output)
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            for r in results_list:
                fold_name = Path(r['model_path']).name
                print(f"{fold_name:<30} Acc: {r['accuracy']:.4f}  F1: {r['f1']:.4f}  Prec: {r['precision']:.4f}  Rec: {r['recall']:.4f}")

    else:
        # Evaluate single model
        if not args.model or not args.test_file:
            parser.error("Either use --all or provide both --model and --test-file")

        results = evaluate_model(args.model, args.test_file, args.batch_size, device=device)
        print_results(results)
        results_list.append(results)
        save_results_csv(results_list, args.output)


if __name__ == "__main__":
    main()
    main()
