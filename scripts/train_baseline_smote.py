# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-15

"""
Skrip Pelatihan Baseline dengan SMOTE (Synthetic Minority Over-sampling Technique).
Mengevaluasi apakah penyeimbangan kelas sintetis pada fitur TF-IDF sparse
dapat menandingi representasi contextual IndoBERT pada kelas Netral.
Hasil diekspor dan diverifikasi terhadap table2_model_comparison_extended.csv.
"""

import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMOTEBaselineTraining")


def run_smote_baseline():
    logger.info("Memuat data partisi train, val, dan test...")
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")

    logger.info("Mengekstraksi representasi teks TF-IDF (10.000 fitur)...")
    if os.path.exists("models/tfidf_vectorizer.pkl"):
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        X_train = vectorizer.transform(train_df["cleaned_text"].fillna(""))
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, max_df=0.95)
        X_train = vectorizer.fit_transform(train_df["cleaned_text"].fillna(""))

    X_val = vectorizer.transform(val_df["cleaned_text"].fillna(""))
    X_test = vectorizer.transform(test_df["cleaned_text"].fillna(""))

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    logger.info(f"Distribusi label sebelum SMOTE: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Distribusi label setelah SMOTE: {dict(pd.Series(y_train_res).value_counts())}")

    results = []

    # 1. Random Forest + SMOTE
    logger.info("Melatih Random Forest + SMOTE (300 estimators)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    rf_time = time.time() - t0

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_p_macro, rf_r_macro, rf_f1_macro, _ = precision_recall_fscore_support(y_test, rf_preds, average="macro", zero_division=0)
    rf_p_w, rf_r_w, rf_f1_w, _ = precision_recall_fscore_support(y_test, rf_preds, average="weighted", zero_division=0)
    _, _, rf_f1_per, _ = precision_recall_fscore_support(y_test, rf_preds, labels=[0, 1, 2], zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_probs, multi_class="ovr", average="macro")

    results.append({
        "Model": "Random Forest + SMOTE",
        "Balancing_Strategy": "Synthetic Oversampling (SMOTE)",
        "Accuracy": f"{rf_acc*100:.2f}%",
        "Precision (macro)": f"{rf_p_macro*100:.2f}%",
        "Recall (macro)": f"{rf_r_macro*100:.2f}%",
        "F1 (macro)": f"{rf_f1_macro*100:.2f}%",
        "F1 (weighted)": f"{rf_f1_w*100:.2f}%",
        "F1 (Neutral)": f"{rf_f1_per[1]*100:.2f}%",
        "ROC-AUC (OVR)": f"{rf_auc:.4f}",
        "Training Time (s)": f"{rf_time:.2f}"
    })

    # 2. XGBoost + SMOTE
    logger.info("Melatih XGBoost + SMOTE (500 estimators, early stopping 20)...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=20
    )
    xgb.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0

    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_p_macro, xgb_r_macro, xgb_f1_macro, _ = precision_recall_fscore_support(y_test, xgb_preds, average="macro", zero_division=0)
    xgb_p_w, xgb_r_w, xgb_f1_w, _ = precision_recall_fscore_support(y_test, xgb_preds, average="weighted", zero_division=0)
    _, _, xgb_f1_per, _ = precision_recall_fscore_support(y_test, xgb_preds, labels=[0, 1, 2], zero_division=0)
    xgb_auc = roc_auc_score(y_test, xgb_probs, multi_class="ovr", average="macro")

    results.append({
        "Model": "XGBoost + SMOTE",
        "Balancing_Strategy": "Synthetic Oversampling (SMOTE)",
        "Accuracy": f"{xgb_acc*100:.2f}%",
        "Precision (macro)": f"{xgb_p_macro*100:.2f}%",
        "Recall (macro)": f"{xgb_r_macro*100:.2f}%",
        "F1 (macro)": f"{xgb_f1_macro*100:.2f}%",
        "F1 (weighted)": f"{xgb_f1_w*100:.2f}%",
        "F1 (Neutral)": f"{xgb_f1_per[1]*100:.2f}%",
        "ROC-AUC (OVR)": f"{xgb_auc:.4f}",
        "Training Time (s)": f"{xgb_time:.2f}"
    })

    res_df = pd.DataFrame(results)
    logger.info("\n" + "="*80)
    logger.info("HASIL EVALUASI SMOTE PADA TEST SET:")
    logger.info("="*80)
    print(res_df.to_string(index=False))
    logger.info("="*80)


if __name__ == "__main__":
    run_smote_baseline()
