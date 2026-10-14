# Model Artifacts & Checkpoints

Direktori ini berisi model machine learning dan deep learning yang digunakan dalam Decision Support System (DSS) dan riset komparatif sentimen e-commerce Indonesia.

## File yang Tersedia di Repositori
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer (10.000 n-gram features) terlatih untuk representasi teks baseline.
- `xgb_tfidf.pkl`: Model XGBoost Classifier terlatih (Multi:softprob, F1-macro: 49.47%).
- `budget_model.joblib` (di `/data`): Model regresi estimasi budget kampanye KOL.
- `indobert_finetuned/`:
  - `config.json`: Konfigurasi arsitektur IndoBERT Sequence Classification.
  - `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`: Tokenizer lengkap IndoBERT.
  - `training_args.bin`: Hyperparameter training reproduksibel.

## Reproduksi Model Berkas Besar (>100MB)
Demi mematuhi batas ukuran file GitHub (100MB), bobot biner besar diexclude dari Git tracking:
1. **Random Forest (`rf_tfidf.pkl` ~212 MB)**:
   Jalankan:
   ```bash
   python scripts/train_baseline_rf_xgb.py
   ```
2. **IndoBERT Weights (`model.safetensors` ~497 MB)**:
   Jalankan:
   ```bash
   python scripts/train_indobert.py
   # atau gunakan notebook:
   # notebooks/04_indobert_finetune.ipynb
   ```
