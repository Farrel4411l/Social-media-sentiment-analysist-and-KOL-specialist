# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-18

"""
Eksperimen Ensemble: Menggabungkan Prediksi Multi-Model IndoBERT Checkpoints
(IndoBERT-Base, IndoBERTweet, IndoBERT-Large, IndoBERT-Focal)
melalui Soft-Voting Probabilitas, Grid-Search Optimal Weights pada Validation Set,
dan Logistic Regression Stacking Classifier.
Evaluasi dilakukan pada Test Set (N = 1.278) tanpa kebocoran data (zero data leakage).
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnsembleEvaluation")

os.makedirs("data/results", exist_ok=True)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }


def get_probabilities(model_dir, texts, batch_size=32, device="cuda"):
    logger.info(f"Generating predictions from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return np.vstack(all_probs)


def evaluate_predictions(y_true, y_prob, model_name="Ensemble"):
    y_pred = np.argmax(y_prob, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    p_per, r_per, f_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = np.nan

    metrics = {
        "Model": model_name,
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
        "F1_Class_2": f_per[2]
    }
    return metrics


def main():
    logger.info("=" * 80)
    logger.info("MULTI-MODEL ENSEMBLE EVALUATION ON TEST SET (ZERO LEAKAGE)")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hardware: {device}")

    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    val_texts = val_df["cleaned_text"].fillna("").values
    test_texts = test_df["cleaned_text"].fillna("").values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    models = {
        "IndoBERT-Base-p1": "models/indobert_finetuned",
        "IndoBERT-Base-p2": "models/indobert_p2_finetuned",
        "IndoBERTweet": "models/indobertweet_finetuned",
        "IndoBERT-Large": "models/indobert_large_finetuned/checkpoint-639" if os.path.exists("models/indobert_large_finetuned/checkpoint-639/config.json") else "models/indobert_large_finetuned",
    }
    if os.path.exists("models/indobert_focal_finetuned/config.json"):
        models["IndoBERT-Focal"] = "models/indobert_focal_finetuned"
    elif os.path.exists("models/indobert_focal_finetuned/checkpoint-639/config.json"):
        models["IndoBERT-Focal"] = "models/indobert_focal_finetuned/checkpoint-639"

    if os.path.exists("models/indobert_smoothed_finetuned/config.json"):
        models["IndoBERT-Smoothed"] = "models/indobert_smoothed_finetuned"
    elif os.path.exists("models/indobert_smoothed_finetuned/checkpoint-639/config.json"):
        models["IndoBERT-Smoothed"] = "models/indobert_smoothed_finetuned/checkpoint-639"

    val_probs_dict = {}
    test_probs_dict = {}

    for name, path in models.items():
        if os.path.exists(path):
            logger.info(f"Processing model: {name} ({path})")
            val_probs_dict[name] = get_probabilities(path, val_texts, device=device)
            test_probs_dict[name] = get_probabilities(path, test_texts, device=device)
        else:
            logger.warning(f"Path does not exist, skipping: {path}")

    results = []

    # 1. Single Model Baselines on Test Set
    for name in val_probs_dict.keys():
        m = evaluate_predictions(y_test, test_probs_dict[name], model_name=f"Single: {name}")
        results.append(m)
        logger.info(f"Single {name}: Test Acc = {m['Accuracy']*100:.2f}%, Macro-F1 = {m['F1_Macro']*100:.2f}%")

    # 2. Equal Weight Soft Voting (All Models)
    model_names = list(val_probs_dict.keys())
    k = len(model_names)
    logger.info(f"\nEvaluating Equal Weight Soft Voting across {k} models: {model_names}")
    val_ens_equal = np.mean([val_probs_dict[n] for n in model_names], axis=0)
    test_ens_equal = np.mean([test_probs_dict[n] for n in model_names], axis=0)
    m_equal = evaluate_predictions(y_test, test_ens_equal, model_name=f"Ensemble: Equal Weight ({k} Models)")
    results.append(m_equal)
    logger.info(f"Equal Weight Ensemble: Test Acc = {m_equal['Accuracy']*100:.2f}%, Macro-F1 = {m_equal['F1_Macro']*100:.2f}%")

    # 3. Optimized Weight Search on Validation Set
    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        ens_prob = sum(w * val_probs_dict[name] for w, name in zip(weights, model_names))
        preds = np.argmax(ens_prob, axis=-1)
        # We want to minimize negative accuracy on validation set
        return -accuracy_score(y_val, preds)

    init_weights = [1.0 / k] * k
    bounds = [(0.0, 1.0)] * k
    res = minimize(loss_func, init_weights, bounds=bounds, method="Nelder-Mead")
    opt_weights = res.x / np.sum(res.x)
    logger.info(f"Optimized Validation Weights: {dict(zip(model_names, [round(w, 4) for w in opt_weights]))}")

    test_ens_opt = sum(w * test_probs_dict[name] for w, name in zip(opt_weights, model_names))
    m_opt = evaluate_predictions(y_test, test_ens_opt, model_name=f"Ensemble: Weighted Soft-Voting ({k} Models)")
    results.append(m_opt)
    logger.info(f"Weighted Soft-Voting: Test Acc = {m_opt['Accuracy']*100:.2f}%, Macro-F1 = {m_opt['F1_Macro']*100:.2f}%")

    # 4. Stacking Classifier (Meta-Learner: Logistic Regression with L2 Regularization)
    logger.info("\nEvaluating Stacking Meta-Learner (Logistic Regression on Val Set)...")
    X_val_meta = np.hstack([val_probs_dict[n] for n in model_names])
    X_test_meta = np.hstack([test_probs_dict[n] for n in model_names])

    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_clf.fit(X_val_meta, y_val)
    test_meta_prob = meta_clf.predict_proba(X_test_meta)
    m_stack = evaluate_predictions(y_test, test_meta_prob, model_name="Ensemble: Stacking (Logistic Regression)")
    results.append(m_stack)
    logger.info(f"Stacking Ensemble: Test Acc = {m_stack['Accuracy']*100:.2f}%, Macro-F1 = {m_stack['F1_Macro']*100:.2f}%")

    res_df = pd.DataFrame(results)
    res_df.to_csv("data/results/metrics_ensemble.csv", index=False)
    logger.info("\nSaved all ensemble results to data/results/metrics_ensemble.csv")

    logger.info("=" * 80)
    logger.info("SUMMARY ENSEMBLE RESULTS:")
    for idx, r in res_df.iterrows():
        logger.info(f"{r['Model']:<40}: Acc = {r['Accuracy']*100:.2f}%, Macro-F1 = {r['F1_Macro']*100:.2f}%, F1-Netral = {r['F1_Class_1']*100:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
