<!--
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS
Date: 2026-10-15
-->

# Panduan Reproduksibilitas Lengkap (Reproducibility Guide)
## Comparative Sentiment Analysis Pipeline (Target Publikasi Scopus Q2/Q3)

Dokumen ini memuat panduan langkah-demi-langkah yang sepenuhnya deterministik dan dapat direproduksi (*end-to-end reproducible*) untuk menjalankan seluruh pipeline penelitian: mulai dari pengumpulan data (*scraping*), pembersihan data & leksikon slang, validasi inter-annotator, pelatihan model baseline (dengan & tanpa SMOTE), *fine-tuning* Transformer (IndoBERT & IndoBERTweet), hingga pengujian signifikansi statistik (*McNemar test*) dan ekstraksi aset publikasi 300 DPI.

---

## 1. Persyaratan Sistem (*System Prerequisites*)

- **Sistem Operasi:** Windows 10/11 (PowerShell) atau Linux (Ubuntu 20.04/22.04 LTS).
- **Python Runtime:** Python 3.10 (Direkomendasikan Python 3.10 untuk kompatibilitas penuh PyTorch 2.5.1 + CUDA 12.1).
- **Hardware Acceleration (Disarankan):** NVIDIA GPU dengan VRAM $\ge 8$ GB (e.g. GeForce RTX 4070 Laptop GPU / RTX 3060/4060) atau Google Colab GPU (Tesla T4 / A100).
- **Determinisme:** Seluruh skrip mengunci nilai acak pada `seed=42` (`numpy.random`, `random`, `torch.manual_seed`, `scikit-learn`, `xgboost`, `DetectorFactory`).

---

## 2. Instalasi Dependensi Lingkungan

Jalankan instalasi pustaka Python menggunakan terminal:

```bash
# 1. Instalasi dependensi NLP, Scraping, Scikit-Learn, & Imbalanced-Learn
pip install google-play-scraper pandas numpy scikit-learn imbalanced-learn xgboost transformers==4.44.2 datasets==2.21.0 evaluate accelerate wordcloud matplotlib seaborn scipy langdetect

# 2. Instalasi PyTorch dengan akselerasi CUDA 12.1 (Jika menggunakan GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Eksekusi Pipeline Berurutan (*Step-by-Step Execution*)

### Langkah 1: Pengumpulan Korpus Mentah (Fase F1)
Mengikis 16.000 ulasan merata (skor rating 1 s.d. 5) dari 4 aplikasi e-commerce Indonesia (Tokopedia, Shopee, Lazada, Bukalapak):
```bash
python scraper/gplay_scraper.py
```
*Hasil:* 4 file CSV mentah tersimpan di `data/raw/` (~4.000 baris per aplikasi).

---

### Langkah 2: Preprocessing, Kamus Slang 150+ Kata, & Disambiguasi (Fase F2)
Melakukan deduplikasi konten, filter bahasa Indonesia, normalisasi kata slang lokal e-commerce, disambiguasi leksikal rating 3, dan partisi terstratifikasi 80/10/10:
```bash
python scripts/run_preprocessing.py
```
*Hasil yang dihasilkan:*
- `data/processed/dataset_clean.csv` (12.777 ulasan bersih, 79,86% retensi).
- `data/splits/train.csv` (10.221 baris), `val.csv` (1.278 baris), `test.csv` (1.278 baris).
- `data/manual_validation_300_dual_annotators.csv` (300 sampel validasi ganda independen).
- `paper_assets/cohen_kappa_inter_annotator_agreement.json` ($\kappa = 0,8662$).
- `paper_assets/label_consistency_audit.json` (Audit konsistensi aturan 89,39%).
- `paper_assets/table1_dataset_statistics.csv` & `figure1_data_distribution.png` (300 DPI).

---

### Langkah 3: Pelatihan Model Baseline Tree-Based (Fase F3)
Mengekstraksi representasi unigram + bigram TF-IDF (10.000 fitur), kemudian melatih Random Forest (300 trees) dan XGBoost (early stopping):
```bash
python scripts/train_baseline_rf_xgb.py
```
*Untuk menguji baseline dengan oversampling sintetis (SMOTE):*
```bash
python scripts/train_baseline_smote.py
```
*Hasil:*
- `models/rf_tfidf.pkl`, `models/xgb_tfidf.pkl`, `models/tfidf_vectorizer.pkl`.
- `data/results/metrics_rf.csv` (Acc: 62.60%, Macro-F1: 48.27%, F1-Netral: 7.31%).
- `data/results/metrics_xgb.csv` (Acc: 64.08%, Macro-F1: 53.05%, F1-Netral: 18.39%).
- Evaluasi SMOTE: RF+SMOTE (F1-Netral: 24.49%), XGB+SMOTE (F1-Netral: 26.58%, Acc: 60.80%).

---

### Langkah 4: Fine-Tuning Model Utama IndoBERT (Fase F4)
Melatih arsitektur Transformer IndoBERT (`indobenchmark/indobert-base-p1`) dengan *Class-Weighted Cross-Entropy Loss*, kapasitas konteks `max_length=256`, 5 epoch, warmup 10%, dan early stopping (`patience=2`):
```bash
py -3.10 scripts/train_indobert.py
```
*Hasil:*
- `models/indobert_finetuned/` (Model checkpoint HuggingFace).
- `data/results/metrics_indobert.csv` (**Acc: 65.26%, Macro-F1: 61.76%, F1-Netral: 40.73%, ROC-AUC: 0.8055**).
- `paper_assets/figure3_training_curves_indobert.png` (300 DPI).
- `paper_assets/figure4_confusion_matrix_indobert.png` (300 DPI).

---

### Langkah 5: Fine-Tuning Eksploratif IndoBERTweet (Fase F4b - Opsional)
Melatih model varian media sosial informal IndoBERTweet (`indolem/indobertweet-base-uncased`):
```bash
py -3.10 scripts/train_indobertweet.py
```
*Hasil:*
- `models/indobertweet_finetuned/`.
- `data/results/metrics_indobertweet.csv` (Acc: 63.46%, Macro-F1: 61.23%, **F1-Netral: 42.23%**, ROC-AUC: 0.8016).
- `paper_assets/figure3_training_curves_indobertweet.png` & `figure4_confusion_matrix_indobertweet.png`.

---

### Langkah 6: Analisis Komparatif, Studi Ablasi, & Uji McNemar (Fase F5)
Menghasilkan tabel perbandingan terpadu, uji signifikansi statistik, studi ablasi 4-tahap, dan sampel kualitatif:
```bash
py -3.10 scripts/run_comparative_analysis.py
```
*Hasil:*
- `paper_assets/table2_model_comparison.csv` & `table2_model_comparison_extended.csv`.
- `paper_assets/table3_per_class_metrics.csv` (Metrik per kelas: Negatif, Netral, Positif).
- `paper_assets/table4_ablation_study.csv` (Studi ablasi kontribusi Slang, Context 256, & Weighted Loss).
- `paper_assets/mcnemar_test.json` (Uji McNemar: 153 kesalahan dikoreksi IndoBERT, $\chi^2 = 0,6735$).
- `paper_assets/figure5_f1_comparison_barplot.png` (300 DPI).
- `paper_assets/figure2_wordcloud_per_class.png` (300 DPI).
- `paper_assets/figure4_confusion_matrix_comparison.png` (300 DPI side-by-side).
- `paper_assets/error_analysis_samples.csv` (20 analisis linguistik teks riil).

---

## 4. Pemetaan Interaktif Jupyter Notebooks (`/notebooks/`)

Setiap langkah di atas memiliki padanan notebook interaktif yang telah disinkronkan:
1. `notebooks/01_data_collection.ipynb` — Pengecekan korpus mentah 4 aplikasi Play Store dan QC awal.
2. `notebooks/02_preprocessing.ipynb` — Kamus slang 150+ kata, disambiguasi rating 3, dan Cohen's Kappa IAA.
3. `notebooks/03_baseline_rf_xgb.ipynb` — Vektorisasi TF-IDF, pelatihan Random Forest dan XGBoost.
4. `notebooks/04_indobert_finetune.ipynb` — Fine-tuning IndoBERT dengan Weighted Loss dan Early Stopping.
5. `notebooks/05_comparative_analysis.ipynb` — Visualisasi komparatif, studi ablasi Table 4, McNemar test, dan error analysis.
