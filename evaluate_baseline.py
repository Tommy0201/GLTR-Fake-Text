#!/usr/bin/env python3
"""
Standalone script to evaluate baseline RoBERTa model performance on the test data.
This shows how well the pre-trained model performs without fine-tuning.
"""

import csv
from pathlib import Path

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
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


def combine_csv_files(file_paths):
    """Combine multiple CSV files into a single dataset."""
    all_texts = []
    all_labels = []

    for file_path in file_paths:
        texts, labels = load_csv_data(file_path)
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"Loaded {len(texts)} samples from {file_path.name}")

    print(f"Combined total: {len(all_texts)} samples")
    return all_texts, all_labels


def evaluate_baseline_model(test_dataset, device="cpu"):
    """
    Evaluate the baseline (pre-trained, non-fine-tuned) RoBERTa model on the test set.
    """
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL (Pre-trained RoBERTa, no fine-tuning)")
    print("="*60)

    # Load baseline model (not fine-tuned)
    baseline_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    baseline_model.to(device)
    baseline_model.eval()

    # Create data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_predictions = []
    all_labels = []

    print("Running baseline evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = baseline_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Human', 'AI']))

    return {
        'baseline_accuracy': accuracy,
        'baseline_precision': precision,
        'baseline_recall': recall,
        'baseline_f1': f1
    }


def main():
    # Load the same data as your training script
    file_paths = [
        Path("train-dataset-text/cheat_processed.csv"),
        Path("train-dataset-text/hc3_processed.csv"),
        Path("train-dataset-text/Essay_LLMs_processed.csv"),
        Path("train-dataset-text/Reuters_LLMs_processed.csv"),
        Path("train-dataset-text/WP_LLMs_processed.csv")
    ]

    print("Loading combined dataset...")
    all_texts, all_labels = combine_csv_files(file_paths)

    # Split the same way as your training script (80-20 split)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    print(f"Test set: {len(test_texts)} samples")

    # Load tokenizer and create test dataset
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=512)

    # Evaluate baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    baseline_results = evaluate_baseline_model(test_dataset, device)

    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE")
    print("="*60)
    print("This shows how well the pre-trained RoBERTa model performs")
    print("without any fine-tuning on your specific task.")


if __name__ == "__main__":
    main()