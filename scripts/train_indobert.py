# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-14

"""
Pelatihan & Fine-Tuning IndoBERT (indobenchmark/indobert-base-p1) pada Dataset Ulasan E-Commerce Indonesia (Fase F4).
Mendukung akselerasi GPU lokal (NVIDIA RTX 4070) dan Google Colab.
Target Output:
- models/indobert_finetuned/ (Checkpoint model HuggingFace)
- data/results/metrics_indobert.csv
- paper_assets/figure3_training_curves_indobert.png
- paper_assets/figure4_confusion_matrix_indobert.png
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

# Set seed untuk reproduksibilitas penuh
set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndoBERTTraining")


class ReviewDataset(Dataset):
    """PyTorch Dataset untuk Ulasan Teks E-Commerce"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
    """
    Kustomisasi fungsi metrik untuk Trainer HuggingFace:
    Menghitung Accuracy, Precision/Recall/F1 per kelas, Macro F1, Weighted F1.
    """
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


class LossHistoryCallback(TrainerCallback):
    """Callback untuk mencatat train loss, val loss, dan val f1 per epoch"""
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.eval_losses = []
        self.eval_f1_macros = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = metrics.get("epoch", state.epoch)
            eval_loss = metrics.get("eval_loss", None)
            f1_macro = metrics.get("eval_f1_macro", None)
            
            # Cari train loss terdekat dari log_history
            train_loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry and "epoch" in entry and entry["epoch"] <= epoch:
                    train_loss = entry["loss"]
                    break

            self.epochs.append(epoch)
            self.train_losses.append(train_loss if train_loss else eval_loss)
            self.eval_losses.append(eval_loss)
            self.eval_f1_macros.append(f1_macro)


def plot_training_curves(callback, output_path="paper_assets/figure3_training_curves_indobert.png"):
    """
    Plot Training Loss, Validation Loss, dan Validation Macro-F1 per Epoch (Figure 3, 300 DPI).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = [int(round(e)) for e in callback.epochs] if callback.epochs else [1, 2, 3]

    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # Subplot 1: Loss Curves
    plt.subplot(1, 2, 1)
    if callback.train_losses and callback.eval_losses:
        plt.plot(epochs, callback.train_losses, marker="o", color="#d9534f", label="Training Loss", linewidth=2)
        plt.plot(epochs, callback.eval_losses, marker="s", color="#0275d8", label="Validation Loss", linewidth=2)
    plt.title("(a) Learning Curves (Cross-Entropy Loss)", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    # Subplot 2: Macro-F1 Evolution
    plt.subplot(1, 2, 2)
    if callback.eval_f1_macros:
        plt.plot(epochs, callback.eval_f1_macros, marker="^", color="#5cb85c", label="Validation Macro-F1", linewidth=2)
    plt.title("(b) Validation Macro-F1 Evolution", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Macro-F1 Score", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.suptitle("Figure 3: IndoBERT Fine-Tuning Convergence & Performance Across Epochs", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def plot_confusion_matrix(y_true, y_pred, output_path="paper_assets/figure4_confusion_matrix_indobert.png"):
    """Plot Confusion Matrix IndoBERT pada Test Set (300 DPI)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = ["Negatif", "Netral", "Positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

    plt.title(f"Confusion Matrix: IndoBERT Fine-Tuned\n(Test Set, N={len(y_true):,})", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=11, labelpad=8)
    plt.ylabel("True Sentiment Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def main():
    logger.info("=" * 70)
    logger.info("FASE F4: INDOBERT FINE-TUNING PIPELINE")
    logger.info("=" * 70)

    # 1. Device Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Hardware Akselerator: {device} ({gpu_name})")

    # 2. Load Partisi Data
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    logger.info(f"Dataset Partisi: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 3. Load Tokenizer & Model Pretrained IndoBERT
    model_name = "indobenchmark/indobert-base-p1"
    logger.info(f"Memuat Pretrained Tokenizer & Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # 4. Buat PyTorch Datasets
    logger.info("Membangun PyTorch Tensor Datasets (max_length=128)...")
    train_dataset = ReviewDataset(train_df["cleaned_text"].fillna(""), train_df["label"].values, tokenizer, max_len=128)
    val_dataset = ReviewDataset(val_df["cleaned_text"].fillna(""), val_df["label"].values, tokenizer, max_len=128)
    test_dataset = ReviewDataset(test_df["cleaned_text"].fillna(""), test_df["label"].values, tokenizer, max_len=128)

    # 5. Training Arguments (Sesuai Standar Scopus)
    output_dir = "models/indobert_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=42,
        fp16=use_fp16,
        logging_steps=50,
        report_to="none"
    )

    loss_callback = LossHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )

    # 6. Jalankan Fine-Tuning
    logger.info("\n" + "-"*50)
    logger.info("MEMULAI TRAINING INDOBERT (3 EPOCHS)...")
    logger.info("-"*50)
    t_start = time.time()
    train_result = trainer.train()
    training_time = time.time() - t_start
    logger.info(f"Training selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit).")

    # 7. Simpan Checkpoint Model Terbaik
    logger.info(f"Menyimpan Model Terbaik ke: {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model & Tokenizer tersimpan sempurna.")

    # 8. Evaluasi pada Test Set
    logger.info("\n" + "-"*50)
    logger.info("EVALUASI MODEL PADA TEST SET...")
    logger.info("-"*50)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_test = test_df["label"].values

    # Softmax probabilities untuk ROC-AUC
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

    metrics_indobert = {
        "Model": "IndoBERT (Fine-Tuned)",
        "Accuracy": acc,
        "Precision_Macro": p_macro,
        "Recall_Macro": r_macro,
        "F1_Macro": f1_macro,
        "Precision_Weighted": p_weighted,
        "Recall_Weighted": r_weighted,
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

    # Simpan metrics_indobert.csv
    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame([metrics_indobert]).to_csv("data/results/metrics_indobert.csv", index=False)
    logger.info("Saved: data/results/metrics_indobert.csv")

    # 9. Plot Curves & Confusion Matrix
    plot_training_curves(loss_callback, "paper_assets/figure3_training_curves_indobert.png")
    plot_confusion_matrix(y_test, y_pred, "paper_assets/figure4_confusion_matrix_indobert.png")

    # 10. Tabel Komparasi 3 Model (Table 2 Preview)
    rf_csv = "data/results/metrics_rf.csv"
    xgb_csv = "data/results/metrics_xgb.csv"
    dfs_to_concat = []
    if os.path.exists(rf_csv):
        dfs_to_concat.append(pd.read_csv(rf_csv))
    if os.path.exists(xgb_csv):
        dfs_to_concat.append(pd.read_csv(xgb_csv))
    dfs_to_concat.append(pd.DataFrame([metrics_indobert]))

    comp_df = pd.concat(dfs_to_concat, ignore_index=True)
    summary_cols = ["Model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "F1_Weighted", "ROC_AUC_OVR", "Training_Time_Sec"]

    logger.info("\n" + "="*85)
    logger.info("TABLE 2: COMPARATIVE PERFORMANCE OF THREE ARCHITECTURES (TEST SET)")
    logger.info("="*85)
    print(comp_df[summary_cols].to_string(index=False))
    logger.info("="*85)

    logger.info("\nFASE F4 SELESAI DENGAN SUKSES!")


if __name__ == "__main__":
    main()
