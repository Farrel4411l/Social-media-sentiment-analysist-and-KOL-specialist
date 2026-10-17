# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-18

"""
Eksperimen Lanjutan: IndoBERT Fine-Tuning dengan Label Smoothing (epsilon = 0.1) & Unweighted Cross-Entropy.
Label smoothing memitigasi overconfidence model pada label bintang 3 yang berisik (label noise / subjective ambiguity)
dengan meratakan distribusi target (0.933 untuk kelas target, 0.033 untuk kelas lainnya).
Akselerasi: NVIDIA GeForce RTX 4070 Laptop GPU (PyTorch CUDA 12.1 + FP16).
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

set_seed(42)

os.makedirs("data/results", exist_ok=True)
os.makedirs("paper_assets", exist_ok=True)
output_dir = "models/indobert_smoothed_finetuned"
os.makedirs(output_dir, exist_ok=True)

log_file = "data/results/training_indobert_smoothed.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("IndoBERTSmoothed")


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_per, r_per, f_per, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_class_0": f_per[0],
        "f1_class_1": f_per[1],
        "f1_class_2": f_per[2]
    }


def main():
    logger.info("=" * 80)
    logger.info("INDOBERT LABEL SMOOTHING (EPSILON=0.1) & UNWEIGHTED LOSS EXPERIMENT")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Hardware Accelerator: {device} ({gpu_name})")

    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    logger.info(f"Partisi Data: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    model_name = "indobenchmark/indobert-base-p1"
    logger.info(f"Memuat Pretrained Tokenizer & Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    logger.info("Membangun PyTorch Datasets (max_length=256)...")
    train_dataset = ReviewDataset(train_df["cleaned_text"].fillna(""), train_df["label"].values, tokenizer, max_len=256)
    val_dataset = ReviewDataset(val_df["cleaned_text"].fillna(""), val_df["label"].values, tokenizer, max_len=256)
    test_dataset = ReviewDataset(test_df["cleaned_text"].fillna(""), test_df["label"].values, tokenizer, max_len=256)

    # TrainingArguments with label_smoothing_factor = 0.1
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,  # Label smoothing regularization
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=42,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    logger.info("\n" + "-"*70)
    logger.info("MEMULAI PELATIHAN INDOBERT DENGAN LABEL SMOOTHING (EPSILON=0.1)...")
    logger.info("-"*70)
    t_start = time.time()
    trainer.train()
    training_time = time.time() - t_start
    logger.info(f"Pelatihan selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit).")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("\n" + "-"*70)
    logger.info("EVALUASI TEST SET INDOBERT LABEL SMOOTHING (N = 1.278)...")
    logger.info("-"*70)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_test = test_df["label"].values

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    y_prob = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    acc = accuracy_score(y_test, y_pred)
    p_per, r_per, f_per, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = np.nan

    metrics_smoothed = {
        "Model": "IndoBERT (Label Smoothing eps=0.1)",
        "Accuracy": acc,
        "Precision_Macro": p_macro,
        "Recall_Macro": r_macro,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "ROC_AUC_OVR": roc_auc,
        "Precision_Class_0": p_per[0],
        "Recall_Class_0": r_per[0],
        "F1_Class_0": f_per[0],
        "Precision_Class_1": p_per[1],
        "Recall_Class_1": r_per[1],
        "F1_Class_1": f_per[1],
        "Precision_Class_2": p_per[2],
        "Recall_Class_2": r_per[2],
        "F1_Class_2": f_per[2],
        "Training_Time_Sec": training_time
    }

    pd.DataFrame([metrics_smoothed]).to_csv("data/results/metrics_indobert_smoothed.csv", index=False)
    logger.info("Saved: data/results/metrics_indobert_smoothed.csv")

    # Save test predictions for ensemble later
    np.save("data/results/test_probs_indobert_smoothed.npy", y_prob)

    logger.info("\n" + "="*80)
    logger.info(f"HASIL TEST SET INDOBERT SMOOTHED: Accuracy = {acc*100:.2f}%, Macro-F1 = {f1_macro*100:.2f}%, F1 Netral = {f_per[1]*100:.2f}%")
    logger.info(f"Target 80% Tercapai: {'YA (BERHASIL)' if acc >= 0.80 else 'TIDAK (BELUM TERCAPAI)'}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
