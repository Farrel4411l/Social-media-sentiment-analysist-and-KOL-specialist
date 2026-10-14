# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-14

"""
Modul Pelatihan Model Baseline: Random Forest & XGBoost dengan TF-IDF Vectorizer (Fase F3).
Mengeksekusi eksperimen komparatif pada dataset partisi (train/val/test),
menghitung metrik akademis (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, ROC-AUC),
menyimpan artefak model (.pkl), metrik (.csv), dan visualisasi Confusion Matrix (.png 300 DPI).
"""

import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineTraining")


def load_partitions():
    """Load train, val, test datasets from data/splits/"""
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    
    logger.info(f"Loaded train set: {len(train_df):,} baris")
    logger.info(f"Loaded val set  : {len(val_df):,} baris")
    logger.info(f"Loaded test set : {len(test_df):,} baris")
    return train_df, val_df, test_df


def extract_features(train_df, val_df, test_df):
    """
    Ekstraksi Fitur TF-IDF (Unigram + Bigram, Max Features: 10,000, min_df=2, max_df=0.95).
    """
    logger.info("Membangun representasi teks TF-IDF (ngram_range=(1,2), max_features=10000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_df["cleaned_text"].fillna(""))
    X_val = vectorizer.transform(val_df["cleaned_text"].fillna(""))
    X_test = vectorizer.transform(test_df["cleaned_text"].fillna(""))
    
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    logger.info(f"Fitur TF-IDF selesai diekstrak. Dimensi fitur: {X_train.shape[1]}")
    logger.info("Saved: models/tfidf_vectorizer.pkl")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer


def evaluate_model(model_name: str, y_true, y_pred, y_prob):
    """
    Menghitung seluruh metrik komprehensif Scopus:
    Accuracy, Macro Precision/Recall/F1, Weighted F1, Per-class metrics, ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    p_per, r_per, f_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"Gagal menghitung ROC-AUC: {e}")
        roc_auc = np.nan
        
    metrics_summary = {
        "Model": model_name,
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
        "F1_Class_2": f_per[2]
    }
    
    return metrics_summary


def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str, filename: str):
    """
    Membuat visualisasi Confusion Matrix standar publikasi (300 DPI, Blues colormap).
    """
    os.makedirs("paper_assets", exist_ok=True)
    labels = ["Negatif", "Netral", "Positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    
    plt.title(f"Confusion Matrix: {model_name}\n(Test Set, N={len(y_true):,})", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=11, labelpad=8)
    plt.ylabel("True Sentiment Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    out_path = os.path.join("paper_assets", filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path} (300 DPI)")


def run_baseline_pipeline():
    logger.info("=" * 70)
    logger.info("FASE F3: BASELINE TRAINING (RANDOM FOREST & XGBOOST)")
    logger.info("=" * 70)
    
    train_df, val_df, test_df = load_partitions()
    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = extract_features(train_df, val_df, test_df)
    
    all_metrics = []
    
    # -------------------------------------------------------------
    # 1. Random Forest Classifier
    # -------------------------------------------------------------
    logger.info("\n" + "-"*50)
    logger.info("Melatih Model 1: Random Forest (300 Estimators, Class Balanced)...")
    logger.info("-"*50)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    t0_rf = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0_rf
    logger.info(f"Random Forest selesai dilatih dalam {rf_train_time:.2f} detik.")
    
    # Save Model
    joblib.dump(rf, "models/rf_tfidf.pkl")
    logger.info("Saved: models/rf_tfidf.pkl")
    
    # Inference on Test Set
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)
    
    rf_metrics = evaluate_model("Random Forest (TF-IDF)", y_test, rf_preds, rf_probs)
    rf_metrics["Training_Time_Sec"] = rf_train_time
    all_metrics.append(rf_metrics)
    
    # Save individual CSV
    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame([rf_metrics]).to_csv("data/results/metrics_rf.csv", index=False)
    logger.info("Saved: data/results/metrics_rf.csv")
    
    # Confusion Matrix
    plot_and_save_confusion_matrix(y_test, rf_preds, "Random Forest + TF-IDF", "figure4_confusion_matrix_rf.png")
    
    # -------------------------------------------------------------
    # 2. XGBoost Classifier
    # -------------------------------------------------------------
    logger.info("\n" + "-"*50)
    logger.info("Melatih Model 2: XGBoost (500 Estimators, Early Stopping 20 rounds)...")
    logger.info("-"*50)
    
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
    
    t0_xgb = time.time()
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_train_time = time.time() - t0_xgb
    best_iteration = getattr(xgb, "best_iteration", 500)
    logger.info(f"XGBoost selesai dilatih dalam {xgb_train_time:.2f} detik. Best Iteration: {best_iteration}")
    
    # Save Model
    joblib.dump(xgb, "models/xgb_tfidf.pkl")
    logger.info("Saved: models/xgb_tfidf.pkl")
    
    # Inference on Test Set
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    
    xgb_metrics = evaluate_model("XGBoost (TF-IDF)", y_test, xgb_preds, xgb_probs)
    xgb_metrics["Training_Time_Sec"] = xgb_train_time
    all_metrics.append(xgb_metrics)
    
    # Save individual CSV
    pd.DataFrame([xgb_metrics]).to_csv("data/results/metrics_xgb.csv", index=False)
    logger.info("Saved: data/results/metrics_xgb.csv")
    
    # Confusion Matrix
    plot_and_save_confusion_matrix(y_test, xgb_preds, "XGBoost + TF-IDF", "figure4_confusion_matrix_xgb.png")
    
    # -------------------------------------------------------------
    # Table 2 Preview (Comparison Table)
    # -------------------------------------------------------------
    comp_df = pd.DataFrame(all_metrics)
    preview_cols = [
        "Model", "Accuracy", "Precision_Macro", "Recall_Macro",
        "F1_Macro", "F1_Weighted", "ROC_AUC_OVR", "Training_Time_Sec"
    ]
    preview_table = comp_df[preview_cols].copy()
    
    logger.info("\n" + "="*80)
    logger.info("TABLE 2 PREVIEW: BASELINE PERFORMANCE COMPARISON (TEST SET)")
    logger.info("="*80)
    print(preview_table.to_string(index=False))
    logger.info("="*80)
    
    logger.info("\nFASE F3 SELESAI DENGAN SUKSES!")


if __name__ == "__main__":
    run_baseline_pipeline()
