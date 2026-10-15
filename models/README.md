<!--
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS
Date: 2026-10-15
-->

# Model Artifacts, Weights & Checkpoints

Direktori ini berisi model *Machine Learning* (Tree-Based + TF-IDF) dan *Deep Learning* (Transformer) yang digunakan dalam sistem **Decision Support System (DSS) PR & KOL Specialist** serta pipeline riset komparatif analisis sentimen e-commerce Indonesia (Target Scopus Q2/Q3).

---

## 1. Model yang Tersedia di Repositori

| Model / Berkas | Arsitektur | Hyperparameter Kunci | Metrik Uji (Test Set) | Lokasi Direktori |
|---|---|---|:---:|---|
| `tfidf_vectorizer.pkl` | TF-IDF (10.000 fitur) | Unigram + Bigram, min_df=2, max_df=0.95 | — | `models/tfidf_vectorizer.pkl` |
| `xgb_tfidf.pkl` | XGBoost Classifier | 500 estimators, depth=8, lr=0.1, multi:softprob | Acc: 64.08%, Macro-F1: 53.05% | `models/xgb_tfidf.pkl` |
| `rf_tfidf.pkl` | Random Forest | 300 trees, class_weight="balanced" | Acc: 62.60%, Macro-F1: 48.27% | `models/rf_tfidf.pkl` |
| `indobert_finetuned/` | IndoBERT (`indobert-base-p1`) | 110M params, max_len=256, Class-Weighted CE Loss, Warmup 10%, Early Stopping | **Acc: 65.26%, Macro-F1: 61.76%, ROC-AUC: 0.8055** (Primary Model) | `models/indobert_finetuned/` |
| `indobertweet_finetuned/` | IndoBERTweet (`indobertweet-base-uncased`) | 110M params, max_len=256, Class-Weighted CE Loss, lr=3e-5 | Acc: 63.46%, Macro-F1: 61.23%, **Netral-F1: 42.23%** (Exploratory Variant) | `models/indobertweet_finetuned/` |
| `budget_model.joblib` | Random Forest Regressor | Estimasi budget influencer/KOL berdasarkan followers, engagement, & sentimen | $R^2 \ge 0.82$ | `data/budget_model.joblib` |

---

## 2. Struktur Checkpoint Transformer (HuggingFace Format)

Folder `models/indobert_finetuned/` dan `models/indobertweet_finetuned/` masing-masing memuat:
- `config.json`: Spesifikasi arsitektur Transformer dan pemetaan label sentimen (`0: negatif`, `1: netral`, `2: positif`).
- `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`: Tokenizer kosakata bahasa Indonesia.
- `training_args.bin`: Log konfigurasi PyTorch TrainingArguments yang sepenuhnya dapat direproduksi (*reproducible*).

---

## 3. Instruksi Pelatihan & Ekstraksi Bobot Ulang

Demi mematuhi aturan batas ukuran berkas Git (100 MB), file bobot biner besar diexclude dari Git tracking:

1. **Melatih Ulang Baseline Tree-Based (TF-IDF + Random Forest + XGBoost):**
   ```bash
   python scripts/train_baseline_rf_xgb.py
   ```
2. **Melatih Ulang Baseline dengan SMOTE Oversampling:**
   ```bash
   py -3.10 scratch/test_smote.py
   ```
3. **Melatih Ulang Model Utama IndoBERT (GPU Accelerated):**
   ```bash
   py -3.10 scripts/train_indobert.py
   ```
4. **Melatih Ulang Model Varian IndoBERTweet (GPU Accelerated):**
   ```bash
   py -3.10 scripts/train_indobertweet.py
   ```
