# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-15

"""
Pelatihan Eksploratif IndoBERTweet (indolem/indobertweet-base-uncased).
Arsitektur: Transformer yang telah di-pretrain khusus pada 409M kata teks informal/slang Indonesia.
Konfigurasi: Weighted Cross-Entropy Loss, max_length=256, Warmup 10%, Early Stopping (Patience=2).
Akselerasi: NVIDIA GeForce RTX 4070 Laptop GPU (PyTorch CUDA 12.1 + FP16).
Output disimpan terpisah tanpa menimpa dokumentasi IndoBERT-base:
- Model Checkpoint: models/indobertweet_finetuned/
- Metrik CSV: data/results/metrics_indobertweet.csv
- Visualisasi: paper_assets/figure3_training_curves_indobertweet.png & figure4_confusion_matrix_indobertweet.png
- Tabel Komparasi Terpadu: paper_assets/table2_model_comparison_with_indobertweet.csv
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndoBERTweetTraining")


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
    Memberi penalti proporsional lebih besar pada kesalahan klasifikasi kelas minoritas/netral.
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
    """Kustomisasi metrik multi-kelas makro dan per-kelas."""
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
    """Callback pencatat riwayat loss dan F1-Score per epoch."""
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

            train_loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry and "epoch" in entry and entry["epoch"] <= epoch:
                    train_loss = entry["loss"]
                    break

            self.epochs.append(epoch)
            self.train_losses.append(train_loss if train_loss else eval_loss)
            self.eval_losses.append(eval_loss)
            self.eval_f1_macros.append(f1_macro)


def plot_training_curves(callback, output_path="paper_assets/figure3_training_curves_indobertweet.png"):
    """Plot kurva konvergensi 300 DPI."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = [int(round(e)) for e in callback.epochs] if callback.epochs else list(range(1, len(callback.eval_losses)+1))

    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    plt.subplot(1, 2, 1)
    if callback.train_losses and callback.eval_losses:
        plt.plot(epochs, callback.train_losses, marker="o", color="#d9534f", label="Training Loss", linewidth=2)
        plt.plot(epochs, callback.eval_losses, marker="s", color="#0275d8", label="Validation Loss", linewidth=2)
    plt.title("(a) IndoBERTweet Weighted Cross-Entropy Loss", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.subplot(1, 2, 2)
    if callback.eval_f1_macros:
        plt.plot(epochs, callback.eval_f1_macros, marker="^", color="#5cb85c", label="Validation Macro-F1", linewidth=2)
    plt.title("(b) Validation Macro-F1 Evolution", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Macro-F1 Score", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.suptitle("Figure 3: IndoBERTweet Convergence with Weighted Loss & Context (max_len=256)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def plot_confusion_matrix(y_true, y_pred, output_path="paper_assets/figure4_confusion_matrix_indobertweet.png"):
    """Plot Confusion Matrix 300 DPI."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = ["Negatif", "Netral", "Positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

    plt.title(f"Confusion Matrix: IndoBERTweet Fine-Tuned\n(Test Set, N={len(y_true):,})", fontsize=11, pad=12, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=10, labelpad=8)
    plt.ylabel("True Sentiment Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def main():
    logger.info("=" * 75)
    logger.info("PELATIHAN INDOBERTWEET (indolem/indobertweet-base-uncased)")
    logger.info("=" * 75)

    # 1. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Akselerator Hardware: {device} ({gpu_name})")

    # 2. Load Partisi Data
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    logger.info(f"Dataset Partisi: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # 3. Hitung Class Weights untuk Weighted Cross-Entropy Loss
    class_counts = train_df["label"].value_counts().sort_index().values
    total_samples = len(train_df)
    n_classes = len(class_counts)
    class_weights = total_samples / (n_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Distribusi Kelas Latih: {class_counts}")
    logger.info(f"Class Weights (w_0, w_1, w_2): {class_weights_tensor.tolist()}")

    # 4. Tokenizer & Model Pretrained IndoBERTweet
    model_name = "indolem/indobertweet-base-uncased"
    logger.info(f"Memuat Pretrained Tokenizer & Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # 5. Buat PyTorch Datasets dengan max_len=256
    logger.info("Membangun PyTorch Tensor Datasets dengan kapasitas konteks penuh (max_length=256)...")
    train_dataset = ReviewDataset(train_df["cleaned_text"].fillna(""), train_df["label"].values, tokenizer, max_len=256)
    val_dataset = ReviewDataset(val_df["cleaned_text"].fillna(""), val_df["label"].values, tokenizer, max_len=256)
    test_dataset = ReviewDataset(test_df["cleaned_text"].fillna(""), test_df["label"].values, tokenizer, max_len=256)

    # 6. Training Arguments
    output_dir = "models/indobertweet_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=42,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    loss_callback = LossHistoryCallback()
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback, early_stopping_callback]
    )

    # 7. Jalankan Fine-Tuning
    logger.info("\n" + "-"*60)
    logger.info("MEMULAI PELATIHAN INDOBERTWEET (5 EPOCHS, WEIGHTED LOSS, CONTEXT=256)...")
    logger.info("-"*60)
    t_start = time.time()
    train_result = trainer.train()
    training_time = time.time() - t_start
    logger.info(f"Pelatihan selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit).")

    # 8. Simpan Model Terbaik
    logger.info(f"Menyimpan Model Terbaik ke: {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model & Tokenizer IndoBERTweet tersimpan sempurna.")

    # 9. Evaluasi pada Test Set
    logger.info("\n" + "-"*60)
    logger.info("EVALUASI TEST SET (N = 1.278)...")
    logger.info("-"*60)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_test = test_df["label"].values

    # Probabilitas Softmax untuk ROC-AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_test_bin = pd.get_dummies(y_test).values
    roc_auc = roc_auc_score(y_test_bin, probs, multi_class="ovr")

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    p_per, r_per, f_per, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)

    logger.info(f"\n{'='*55}")
    logger.info(f"HASIL AKHIR EVALUASI TEST SET INDOBERTWEET:")
    logger.info(f"{'='*55}")
    logger.info(f"Accuracy         : {acc * 100:.2f}%")
    logger.info(f"Macro Precision  : {p_macro * 100:.2f}%")
    logger.info(f"Macro Recall     : {r_macro * 100:.2f}%")
    logger.info(f"Macro F1-Score   : {f1_macro * 100:.2f}%")
    logger.info(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    logger.info(f"ROC-AUC (OVR)    : {roc_auc:.4f}")
    logger.info(f"Training Time    : {training_time:.2f} s")
    logger.info(f"F1-Score Negatif : {f_per[0] * 100:.2f}%")
    logger.info(f"F1-Score Netral  : {f_per[1] * 100:.2f}%")
    logger.info(f"F1-Score Positif : {f_per[2] * 100:.2f}%")

    # 10. Simpan Metrik ke CSV Terpisah
    metrics_data = [{
        "Model": "IndoBERTweet (Fine-Tuned)",
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
    }]
    m_df = pd.DataFrame(metrics_data)
    os.makedirs("data/results", exist_ok=True)
    m_df.to_csv("data/results/metrics_indobertweet.csv", index=False)
    logger.info("Metrik tersimpan di: data/results/metrics_indobertweet.csv")

    # 11. Simpan Visualisasi Gambar
    plot_training_curves(loss_callback)
    plot_confusion_matrix(y_test, y_pred)

    # 12. Simpan Tabel Komparasi Terpadu (Preserving All Previous Documentation)
    # Baca model yang ada
    base_t2 = pd.read_csv("paper_assets/table2_model_comparison_extended.csv")
    new_row = pd.DataFrame([{
        "Model": "IndoBERTweet (Pretrained on Informal Social)",
        "Balancing_Strategy": "Class-Weighted Cross-Entropy Loss",
        "Accuracy": f"{acc * 100:.2f}%",
        "Precision (macro)": f"{p_macro * 100:.2f}%",
        "Recall (macro)": f"{r_macro * 100:.2f}%",
        "F1 (macro)": f"{f1_macro * 100:.2f}%",
        "F1 (weighted)": f"{f1_weighted * 100:.2f}%",
        "F1 (Neutral)": f"{f_per[1] * 100:.2f}%",
        "ROC-AUC (OVR)": f"{roc_auc:.4f}",
        "Training Time (s)": f"{training_time:.2f}"
    }])
    combined_t2 = pd.concat([base_t2, new_row], ignore_index=True)
    combined_t2.to_csv("paper_assets/table2_model_comparison_with_indobertweet.csv", index=False)
    combined_t2.to_csv("data/results/table2_model_comparison_with_indobertweet.csv", index=False)
    logger.info("Tabel Komparasi Terpadu Tersimpan di: paper_assets/table2_model_comparison_with_indobertweet.csv")


if __name__ == "__main__":
    main()
