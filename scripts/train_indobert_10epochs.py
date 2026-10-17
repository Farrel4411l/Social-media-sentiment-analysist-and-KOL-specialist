# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-18

"""
Eksperimen Tambahan 10 Epochs IndoBERT Fine-Tuning:
Uji hipotesis peningkatan akurasi menuju 80%+ pada korpus multi-vendor ulasan e-commerce (12.777 data).
Protokol ketat:
- Tanpa data leakage: split train (10.221), val (1.278), test (1.278) terisolasi penuh.
- Audit overfitting: pencatatan train loss vs validation loss di setiap epoch (1 s.d. 10).
- Cosine learning rate schedule dengan warmup 10% dan weight decay 0.05 untuk regularisasi.
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

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
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
os.makedirs("models/indobert_10epochs", exist_ok=True)

log_file = "data/results/training_10epochs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("IndoBERT10Epochs")


class ReviewDataset(Dataset):
    """PyTorch Dataset untuk Ulasan Teks E-Commerce dengan max_length=256."""
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


class WeightedTrainer(Trainer):
    """
    Trainer dengan Weighted Cross-Entropy Loss.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weights = self.class_weights.to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Perhitungan metrik akurasi, Macro-F1, dan F1 per kelas."""
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


class DetailedEpochTrackerCallback(TrainerCallback):
    """Pencatat metrik dan loss setiap epoch untuk deteksi overfitting empiris."""
    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = metrics.get("epoch", state.epoch)
            eval_loss = metrics.get("eval_loss", None)
            eval_acc = metrics.get("eval_accuracy", None)
            eval_f1_macro = metrics.get("eval_f1_macro", None)
            eval_f1_c0 = metrics.get("eval_f1_class_0", None)
            eval_f1_c1 = metrics.get("eval_f1_class_1", None)
            eval_f1_c2 = metrics.get("eval_f1_class_2", None)

            train_loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry and "epoch" in entry and entry["epoch"] <= epoch:
                    train_loss = entry["loss"]
                    break

            entry_record = {
                "epoch": round(float(epoch), 1),
                "train_loss": train_loss if train_loss is not None else np.nan,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_acc,
                "eval_f1_macro": eval_f1_macro,
                "eval_f1_negative": eval_f1_c0,
                "eval_f1_neutral": eval_f1_c1,
                "eval_f1_positive": eval_f1_c2,
                "step": state.global_step
            }
            self.history.append(entry_record)
            logger.info(
                f"[EPOCH {int(round(epoch))}/10 EVAL] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {eval_loss:.4f} | "
                f"Val Acc: {eval_acc*100:.2f}% | "
                f"Val Macro-F1: {eval_f1_macro*100:.2f}% | "
                f"F1 Neutral: {eval_f1_c1*100:.2f}%"
            )


def plot_10epoch_convergence(history_df, output_path="paper_assets/figure3_training_curves_10epochs.png"):
    """Plot kurva loss dan akurasi 10 epoch untuk audit overfitting visual (300 DPI)."""
    plt.figure(figsize=(12, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # Subplot 1: Train vs Val Loss
    plt.subplot(1, 2, 1)
    epochs = history_df["epoch"].values
    plt.plot(epochs, history_df["train_loss"], marker="o", color="#d9534f", label="Training Loss", linewidth=2)
    plt.plot(epochs, history_df["eval_loss"], marker="s", color="#0275d8", label="Validation Loss", linewidth=2)
    plt.title("(a) Training vs. Validation Loss (Overfitting Gap)", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    # Subplot 2: Val Accuracy & Macro-F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df["eval_accuracy"] * 100, marker="^", color="#f0ad4e", label="Validation Accuracy (%)", linewidth=2)
    plt.plot(epochs, history_df["eval_f1_macro"] * 100, marker="D", color="#5cb85c", label="Validation Macro-F1 (%)", linewidth=2)
    plt.axhline(80.0, color="red", linestyle="--", linewidth=1.5, label="80% Target Threshold")
    plt.title("(b) Validation Accuracy & Macro-F1 Across 10 Epochs", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.suptitle("Figure 3-Extended: IndoBERT 10-Epoch Convergence & Overfitting Audit (RTX 4070)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def plot_confusion_matrix_10epochs(y_true, y_pred, output_path="paper_assets/figure4_confusion_matrix_10epochs.png"):
    """Plot Confusion Matrix 10-Epoch run (300 DPI)."""
    labels = ["Negatif", "Netral", "Positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

    plt.title(f"Confusion Matrix: IndoBERT 10-Epoch Run\n(Test Set, N={len(y_true):,})", fontsize=11, pad=12, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=10, labelpad=8)
    plt.ylabel("True Sentiment Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def main():
    logger.info("=" * 80)
    logger.info("INDOBERT 10-EPOCH FINE-TUNING EXPERIMENT (TARGET ACCURACY >= 80% AUDIT)")
    logger.info("=" * 80)

    # 1. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Hardware Accelerator: {device} ({gpu_name})")

    # 2. Load Strict Partitions (Zero Data Leakage)
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    logger.info(f"Partisi Data Terverifikasi: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 3. Class Weights
    class_counts = train_df["label"].value_counts().sort_index().values
    total_samples = len(train_df)
    n_classes = len(class_counts)
    class_weights = total_samples / (n_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Distribusi Kelas Latih: {class_counts}")
    logger.info(f"Class Weights (w_0, w_1, w_2): {class_weights_tensor.tolist()}")

    # 4. Tokenizer & Model
    model_name = "indobenchmark/indobert-base-p1"
    logger.info(f"Memuat Pretrained Tokenizer & Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # 5. Build PyTorch Datasets
    logger.info("Membangun PyTorch Tensor Datasets (max_length=256)...")
    train_dataset = ReviewDataset(train_df["cleaned_text"].fillna(""), train_df["label"].values, tokenizer, max_len=256)
    val_dataset = ReviewDataset(val_df["cleaned_text"].fillna(""), val_df["label"].values, tokenizer, max_len=256)
    test_dataset = ReviewDataset(test_df["cleaned_text"].fillna(""), test_df["label"].values, tokenizer, max_len=256)

    # 6. Training Arguments for 10 Epochs
    output_dir = "models/indobert_10epochs"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=42,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    epoch_tracker = DetailedEpochTrackerCallback()

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[epoch_tracker]
    )

    # 7. Execute 10-Epoch Training
    logger.info("\n" + "-"*70)
    logger.info("MEMULAI 10 EPOCH INDOBERT FINE-TUNING (COSINE SCHEDULER, WEIGHT DECAY=0.05)...")
    logger.info("-"*70)
    t_start = time.time()
    train_result = trainer.train()
    training_time = time.time() - t_start
    logger.info(f"Pelatihan 10 Epoch selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit).")

    # 8. Save Epoch History DataFrame
    history_df = pd.DataFrame(epoch_tracker.history)
    history_df.to_csv("data/results/epoch_history_10epochs.csv", index=False)
    logger.info("Saved: data/results/epoch_history_10epochs.csv")

    # 9. Plot Convergence Curves
    plot_10epoch_convergence(history_df, "paper_assets/figure3_training_curves_10epochs.png")

    # 10. Best Model Evaluation on Test Set
    logger.info("\n" + "-"*70)
    logger.info("EVALUASI TEST SET TERAKHIR (MODEL TERBAIK DARI VALIDATION SET)...")
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

    metrics_10epochs = {
        "Model": "IndoBERT (10 Epochs Experiment)",
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

    pd.DataFrame([metrics_10epochs]).to_csv("data/results/metrics_indobert_10epochs.csv", index=False)
    logger.info("Saved: data/results/metrics_indobert_10epochs.csv")

    plot_confusion_matrix_10epochs(y_test, y_pred, "paper_assets/figure4_confusion_matrix_10epochs.png")

    logger.info("\n" + "="*80)
    logger.info(f"HASIL AKHIR TEST SET: Accuracy = {acc*100:.2f}%, Macro-F1 = {f1_macro*100:.2f}%")
    logger.info(f"Target 80% Tercapai: {'YA (BERHASIL)' if acc >= 0.80 else 'TIDAK (BELUM TERCAPAI)'}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
