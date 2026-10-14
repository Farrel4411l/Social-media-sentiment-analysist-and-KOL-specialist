<!--
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS
Date: 2026-10-14
-->

# Reproducibility Guide: Comparative Sentiment Analysis Pipeline

Panduan ini berisi instruksi terperinci untuk mereproduksi (*re-run*) seluruh tahapan riset dari *data collection*, *preprocessing*, *baseline training*, *IndoBERT fine-tuning*, hingga *statistical & error analysis*.

---

## 1. Persyaratan Sistem (*System Prerequisites*)
- **Sistem Operasi:** Windows 10/11 atau Linux (Ubuntu 22.04+)
- **Python:** Python 3.10+ (Direkomendasikan Python 3.10 untuk kompatibilitas PyTorch CUDA)
- **GPU (Opsional namun Disarankan):** NVIDIA GPU dengan VRAM $\ge 6$ GB (e.g. RTX 3060/4060/4070) atau Google Colab GPU (Tesla T4 / A100).
- **Random Seed:** Telah dikunci `seed=42` pada Numpy, Python Random, PyTorch, Scikit-Learn, dan XGBoost.

---

## 2. Instalasi Dependensi
Jalankan perintah berikut di terminal:

```bash
# Instalasi pustaka utama
pip install google-play-scraper pandas numpy scikit-learn xgboost transformers==4.44.2 datasets==2.21.0 evaluate accelerate wordcloud matplotlib seaborn scipy
```

*Catatan untuk akselerasi GPU NVIDIA (CUDA):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Eksekusi Pipeline Secara Berurutan (End-to-End)

### Langkah 1: Pengumpulan Data Mentah (Fase F1)
Mengumpulkan 16.000 ulasan merata (skor 1 s.d. 5) dari Tokopedia, Shopee, Lazada, dan Bukalapak:
```bash
python scraper/gplay_scraper.py
```
*Output:* File CSV mentah tersimpan di `data/raw/` (~4.000 ulasan per aplikasi).

---

### Langkah 2: Preprocessing, Normalisasi Slang & Split (Fase F2)
Melakukan deduplikasi, filter bahasa Indonesia, normalisasi kamus slang, dan partisi stratified 80/10/10:
```bash
python scripts/run_preprocessing.py
```
*Output:*
- `data/processed/dataset_clean.csv` (12.777 ulasan bersih)
- `data/splits/train.csv` (10.221 baris)
- `data/splits/val.csv` (1.278 baris)
- `data/splits/test.csv` (1.278 baris)
- `data/manual_validation_sample.csv` (1.278 baris sample validasi manual 10%)
- `paper_assets/table1_dataset_statistics.csv`
- `paper_assets/figure1_data_distribution.png` (300 DPI)

---

### Langkah 3: Pelatihan Model Baseline Tree-Based (Fase F3)
Mengekstraksi TF-IDF n-gram (10.000 fitur), melatih Random Forest (300 trees) dan XGBoost (early stopping):
```bash
python scripts/train_baseline_rf_xgb.py
```
*Output:*
- `models/rf_tfidf.pkl`
- `models/xgb_tfidf.pkl`
- `models/tfidf_vectorizer.pkl`
- `data/results/metrics_rf.csv`
- `data/results/metrics_xgb.csv`
- `paper_assets/figure4_confusion_matrix_rf.png`
- `paper_assets/figure4_confusion_matrix_xgb.png`

---

### Langkah 4: Fine-Tuning IndoBERT (Fase F4)
Melatih model kontekstual IndoBERT (`indobenchmark/indobert-base-p1`) sebanyak 3 epoch menggunakan mixed precision FP16:
```bash
py -3.10 scripts/train_indobert.py
```
*Output:*
- `models/indobert_finetuned/` (Model checkpoint HuggingFace)
- `data/results/metrics_indobert.csv`
- `paper_assets/figure3_training_curves_indobert.png` (300 DPI)
- `paper_assets/figure4_confusion_matrix_indobert.png` (300 DPI)

---

### Langkah 5: Analisis Komparatif, Uji McNemar & Error Analysis (Fase F5)
Menghasilkan tabel agregat perbandingan, uji signifikansi statistik, dan visualisasi side-by-side:
```bash
py -3.10 scripts/run_comparative_analysis.py
```
*Output:*
- `paper_assets/table2_model_comparison.csv`
- `paper_assets/table3_per_class_metrics.csv`
- `paper_assets/figure5_f1_comparison_barplot.png` (300 DPI)
- `paper_assets/figure2_wordcloud_per_class.png` (300 DPI)
- `paper_assets/figure4_confusion_matrix_comparison.png` (300 DPI)
- `paper_assets/mcnemar_test_result.json`
- `paper_assets/error_analysis_samples.csv` (20 sampel ulasan)

---

## 4. Struktur Interaktif Jupyter Notebooks
Setiap tahapan di atas juga dapat dijalankan secara interaktif melalui folder `/notebooks/`:
1. `notebooks/01_data_collection.ipynb` — Eksplorasi ulasan mentah dan QC awal
2. `notebooks/02_preprocessing.ipynb` — Eksplorasi kamus slang dan visualisasi distribusi
3. `notebooks/03_baseline_rf_xgb.ipynb` — Eksplorasi pohon keputusan dan TF-IDF
4. `notebooks/04_indobert_finetune.ipynb` — Eksplorasi fine-tuning IndoBERT (Colab/Local GPU auto-detect)
5. `notebooks/05_comparative_analysis.ipynb` — Visualisasi komparatif akhir, McNemar test, dan error analysis
