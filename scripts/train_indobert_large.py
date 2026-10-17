# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-18

"""
Eksperimen Peningkatan Kapasitas Model: IndoBERT-Large Fine-Tuning.
Menguji arsitektur 335M parameter (indobenchmark/indobert-large-p1) pada GPU RTX 4070.
Protokol:
- Batch Size 8 dengan Gradient Accumulation 2 (Effective Batch Size = 16)
- Learning Rate: 1e-5, Warmup 10%, Weight Decay 0.01
- Class-Weighted Cross-Entropy Loss
- Context Length: max_length=256
- Early Stopping Patience: 2
- Akselerasi: PyTorch CUDA 12.1 + FP16
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

os.makedirs("data/results", exist_ok=True)
os.makedirs("paper_assets", exist_ok=True)
output_dir = "models/indobert_large_finetuned"
os.makedirs(output_dir, exist_ok=True)

log_file = "data/results/training_indobert_large.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("IndoBERTLargeTraining")


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


class WeightedTrainer(Trainer):
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
                f"[INDOBERT-LARGE EPOCH {int(round(epoch))} EVAL] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {eval_loss:.4f} | "
                f"Val Acc: {eval_acc*100:.2f}% | "
                f"Val Macro-F1: {eval_f1_macro*100:.2f}% | "
                f"F1 Neutral: {eval_f1_c1*100:.2f}%"
            )


def plot_curves(history_df, output_path="paper_assets/figure3_training_curves_indobert_large.png"):
    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    plt.subplot(1, 2, 1)
    epochs = history_df["epoch"].values
    plt.plot(epochs, history_df["train_loss"], marker="o", color="#d9534f", label="Training Loss", linewidth=2)
    plt.plot(epochs, history_df["eval_loss"], marker="s", color="#0275d8", label="Validation Loss", linewidth=2)
    plt.title("(a) IndoBERT-Large Cross-Entropy Loss", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df["eval_accuracy"] * 100, marker="^", color="#f0ad4e", label="Validation Accuracy (%)", linewidth=2)
    plt.plot(epochs, history_df["eval_f1_macro"] * 100, marker="D", color="#5cb85c", label="Validation Macro-F1 (%)", linewidth=2)
    plt.axhline(80.0, color="red", linestyle="--", linewidth=1.5, label="80% Target Threshold")
    plt.title("(b) Validation Accuracy & Macro-F1", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.xticks(epochs)
    plt.legend()

    plt.suptitle("Figure: IndoBERT-Large (335M Params) Convergence & Accuracy Curve", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def plot_cm(y_true, y_pred, output_path="paper_assets/figure4_confusion_matrix_indobert_large.png"):
    labels = ["Negatif", "Netral", "Positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

    plt.title(f"Confusion Matrix: IndoBERT-Large (335M)\n(Test Set, N={len(y_true):,})", fontsize=11, pad=12, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=10, labelpad=8)
    plt.ylabel("True Sentiment Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {output_path} (300 DPI)")


def main():
    logger.info("=" * 80)
    logger.info("INDOBERT-LARGE FINE-TUNING EXPERIMENT (335M PARAMS - TARGET >= 80% ACCURACY)")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Hardware Accelerator: {device} ({gpu_name})")

    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    logger.info(f"Partisi Data: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    class_counts = train_df["label"].value_counts().sort_index().values
    total_samples = len(train_df)
    n_classes = len(class_counts)
    class_weights = total_samples / (n_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    model_name = "indobenchmark/indobert-large-p1"
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
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

    epoch_tracker = DetailedEpochTrackerCallback()
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[epoch_tracker, early_stopping]
    )

    logger.info("\n" + "-"*70)
    logger.info("MEMULAI PELATIHAN INDOBERT-LARGE (BATCH=8, GRAD_ACCUM=2, LR=1e-5)...")
    logger.info("-"*70)
    t_start = time.time()
    trainer.train()
    training_time = time.time() - t_start
    logger.info(f"Pelatihan IndoBERT-Large selesai dalam {training_time:.2f} detik ({training_time/60:.2f} menit).")

    history_df = pd.DataFrame(epoch_tracker.history)
    history_df.to_csv("data/results/epoch_history_indobert_large.csv", index=False)
    logger.info("Saved: data/results/epoch_history_indobert_large.csv")

    plot_curves(history_df, "paper_assets/figure3_training_curves_indobert_large.png")

    logger.info("\n" + "-"*70)
    logger.info("EVALUASI TEST SET INDOBERT-LARGE (N = 1.278)...")
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

    metrics_large = {
        "Model": "IndoBERT-Large (335M Params)",
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

    pd.DataFrame([metrics_large]).to_csv("data/results/metrics_indobert_large.csv", index=False)
    logger.info("Saved: data/results/metrics_indobert_large.csv")

    plot_cm(y_test, y_pred, "paper_assets/figure4_confusion_matrix_indobert_large.png")

    logger.info("\n" + "="*80)
    logger.info(f"HASIL TEST SET INDOBERT-LARGE: Accuracy = {acc*100:.2f}%, Macro-F1 = {f1_macro*100:.2f}%")
    logger.info(f"Target 80% Tercapai: {'YA (BERHASIL)' if acc >= 0.80 else 'TIDAK (BELUM TERCAPAI)'}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
