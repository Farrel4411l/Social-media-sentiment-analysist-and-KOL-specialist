# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-18

"""
Eksperimen Meta-Learner Lanjutan: XGBoost & Random Forest Stacking Meta-Learner
menggunakan fitur probabilitas 5 model Transformer (15 probabilitas) + fitur metadata teks
(panjang karakter, jumlah kata, rasio huruf kapital) untuk menguji apakah non-linear ensemble
dapat mendorong akurasi lebih jauh menuju 70%+.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TreeMetaLearner")


def extract_features(texts):
    lengths = [len(str(t)) for t in texts]
    words = [len(str(t).split()) for t in texts]
    return np.column_stack([lengths, words])


def main():
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    import sys
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), "scripts"))
    from evaluate_ensemble import get_probabilities

    models = {
        "IndoBERT-Base-p1": "models/indobert_finetuned",
        "IndoBERT-Base-p2": "models/indobert_p2_finetuned",
        "IndoBERTweet": "models/indobertweet_finetuned",
        "IndoBERT-Large": "models/indobert_large_finetuned/checkpoint-639",
        "IndoBERT-Focal": "models/indobert_focal_finetuned",
        "IndoBERT-Smoothed": "models/indobert_smoothed_finetuned",
        "XLM-RoBERTa": "models/xlm_roberta_finetuned",
    }

    val_probs = []
    test_probs = []
    for name, path in models.items():
        vp = get_probabilities(path, val_df["cleaned_text"].fillna("").values, device="cuda")
        tp = get_probabilities(path, test_df["cleaned_text"].fillna("").values, device="cuda")
        val_probs.append(vp)
        test_probs.append(tp)

    X_val_prob = np.hstack(val_probs)
    X_test_prob = np.hstack(test_probs)

    val_meta = extract_features(val_df["cleaned_text"].fillna("").values)
    test_meta = extract_features(test_df["cleaned_text"].fillna("").values)

    X_val_full = np.hstack([X_val_prob, val_meta])
    X_test_full = np.hstack([X_test_prob, test_meta])

    logger.info("Training XGBoost Meta-Learner on 15 soft probabilities + 2 text features...")
    xgb_meta = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_meta.fit(X_val_full, y_val)
    xgb_pred = xgb_meta.predict(X_test_full)
    xgb_prob = xgb_meta.predict_proba(X_test_full)

    acc = accuracy_score(y_test, xgb_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, xgb_pred, average="macro", zero_division=0)
    logger.info(f"XGBoost Meta-Learner Test Set Accuracy: {acc*100:.2f}%, Macro-F1: {f1_macro*100:.2f}%")

    logger.info("Training Random Forest Meta-Learner...")
    rf_meta = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    rf_meta.fit(X_val_full, y_val)
    rf_pred = rf_meta.predict(X_test_full)
    acc_rf = accuracy_score(y_test, rf_pred)
    p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test, rf_pred, average="macro", zero_division=0)
    logger.info(f"Random Forest Meta-Learner Test Set Accuracy: {acc_rf*100:.2f}%, Macro-F1: {f1_rf*100:.2f}%")


if __name__ == "__main__":
    main()
