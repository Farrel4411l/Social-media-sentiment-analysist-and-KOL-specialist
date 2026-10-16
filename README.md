<!-- 
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS & Comparative Research Pipeline
Date: 2026-10-15
-->

# AI PR & KOL Specialist: Decision Support System & Empirical Research Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace IndoBERT](https://img.shields.io/badge/HuggingFace-IndoBERT-yellow.svg)](https://huggingface.co/indobenchmark/indobert-base-p1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Gambaran Umum (Overview)
Repositori ini merupakan implementasi terpadu dari sistem **Decision Support System (DSS) untuk Public Relations & KOL Management** serta **Pipeline Penelitian Komparatif Analisis Sentimen Ulasan E-Commerce Indonesia (Target Jurnal Terindeks Scopus)**.

Sistem ini mengintegrasikan:
1. **Interactive Dashboard (Streamlit)**: Dashboard visual interaktif untuk simulasi kalkulasi budget KOL, scraping opini publik real-time, analisis sentimen & emosi, serta pembuatan brief KOL otomatis.
2. **REST API (FastAPI)**: Endpoint terpadu untuk inferensi sentimen, estimasi budget berbasis ML, retrieval RAG psikologi marketing, dan orkestrasi brief KOL.
3. **Comparative NLP Research Pipeline**: Studi empiris berskala 12.777 ulasan bersih dari Tokopedia, Shopee, Lazada, dan Bukalapak membandingkan **IndoBERT Transformer**, **XGBoost**, dan **Random Forest**.
4. **Java OOP Simulation**: Implementasi arsitektur perbandingan *Bad Campaign* vs *Good Campaign* menggunakan prinsip Object-Oriented Programming (OOP) di Java.

---

## 🏛️ Arsitektur Sistem & Modul Utama

```
                               ┌─────────────────────────────────────────┐
                               │       Streamlit Dashboard / Frontend    │
                               │              (frontend/app.py)          │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Backend Engine (api/main.py)                      │
├─────────────────────┬──────────────────────┬────────────────────┬──────────────────────┤
│  Sentiment Engine   │   Budget Predictor   │    RAG Psychology  │    KOL Generator     │
│ (VADER, Aspect,     │   (ML Regression /   │    (ChromaDB +     │   (Gemini / SLM /    │
│  Dominant Emotion)  │    Random Forest)    │     LangChain)     │     Persuasion)      │
└──────────┬──────────┴──────────┬───────────┴─────────┬──────────┴──────────┬───────────┘
           │                     │                     │                     │
           ▼                     ▼                     ▼                     ▼
┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│  Live Web Scraper   │ │ ML Model Weights │ │ Marketing Vector │ │ Dynamic Brief Output │
│ (DuckDuckGo, GPlay) │ │ (data/models)    │ │ Knowledge Base   │ │ (Persona, Storyline) │
└─────────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────────┘
```

---

## 🔬 Riset Komparatif: IndoBERT vs Tree-Based Models (Scopus Pipeline)

Riset ini mengevaluasi keandalan model *Contextual Transformer* versus *Tree-Based Machine Learning* pada teks ulasan informal e-commerce Indonesia (Tokopedia, Shopee, Lazada, Bukalapak).

### Ringkasan Hasil Eksperimen Komparatif (Test Set, $N = 1.278$)
| Model | Representasi Fitur / Strategi | Accuracy (%) | Macro F1 (%) | F1 Netral (%) | ROC-AUC (OVR) | Training Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (1-2 gram, 10k) + Balanced | 62.60% | 48.27% | 7.31% 🚨 | 0.7430 | **7.51 s** |
| **Random Forest + SMOTE** | TF-IDF + Synthetic Oversampling | 61.97% | 53.40% | 24.49% | 0.7383 | 7.73 s |
| **XGBoost** | TF-IDF (1-2 gram, 10k) + Early Stop | 64.08% | 53.05% | 18.39% ⚠️ | 0.7540 | 52.90 s |
| **XGBoost + SMOTE** | TF-IDF + Synthetic Oversampling | 60.80% | 53.96% | 26.58% | 0.7523 | 55.84 s |
| **IndoBERT (base-p1)** | Deep Contextual (max_len=256, Weighted CE) | **65.26%** 🏆 | **61.76%** 🏆 | **40.73%** 🎯 | **0.8055** 🏆 | 607.32 s |
| **IndoBERTweet (uncased)** | Deep Contextual (max_len=256, Weighted CE) | 63.46% | 61.23% | **42.23%** 🚀 | 0.8016 | 436.06 s |

- **Temuan Kunci (Penyelesaian Neutral Collapse)**: Keunggulan signifikan IndoBERT terutama bersumber dari keberhasilan memecahkan kegagalan batas keputusan (*decision boundary*) pada kelas Netral. IndoBERT-base mencapai **F1 Netral 40.73%**, unggul masif **+22.34pp** di atas XGBoost standar (18.39%) dan **+14.15pp** di atas XGBoost dengan SMOTE (26.58%).
- **Eksplorasi Varian IndoBERTweet**: Model yang di-pretrain khusus pada media sosial informal Indonesia menghasilkan **F1 Netral tertinggi (42.23%)**, membuktikan adaptabilitas tinggi pada kosakata gaul, sementara IndoBERT-base tetap menjadi juara global (Akurasi 65.26%) pada ulasan komplain transaksional panjang.
- **Keterbatasan SMOTE pada Sparse N-Grams**: Eksperimen membuktikan bahwa oversampling sintetis (SMOTE) pada TF-IDF hanya menaikkan F1 Netral secara terbatas dan justru menurunkan akurasi total XGBoost (dari 64.08% menjadi 60.80%), memvalidasi landasan teori Cui et al. (2019) bahwa *Class-Weighted Cross-Entropy Loss* pada fitur kontekstual jauh lebih efektif dibandingkan oversampling sintetis yang rentan memicu *overfitting* batas keputusan.
- **Integritas Anotasi & Uji Reliabilitas**:
  - **Human Inter-Annotator Agreement (IAA)**: 300 sampel acak terstratifikasi diuji independen oleh 2 penilai domain menghasilkan skor **Cohen’s Kappa $\kappa = 0.8662$** (`paper_assets/cohen_kappa_inter_annotator_agreement.json`).
  - **Internal Heuristic Consistency Audit**: Verifikasi konsistensi leksikal aturan rating 3 menghasilkan skor kecocokan **89.39%** (`paper_assets/label_consistency_audit.json`).
- **Ablation Study (Table 4, `paper_assets/table4_ablation_study.csv`)**: Mengonfirmasi kontribusi bertahap dari Base IndoBERT (Macro-F1 57.97%, F1-Netral 32.40%) $\rightarrow$ +Slang & Disambiguation (Macro-F1 59.45% [+1.48%], F1-Netral 34.18%) $\rightarrow$ +Extended Context Length 256 (Macro-F1 60.50% [+2.53%], F1-Netral 35.80%) $\rightarrow$ Proposed Full IndoBERT dengan Class-Weighted CE Loss (**Akurasi 65.26%, Macro-F1 61.76% [+3.79%], F1-Netral 40.73%**).
- **Uji Signifikansi Statistik McNemar (`paper_assets/mcnemar_test.json`)**: Mengacu pada prosedur Dietterich (1998), uji McNemar pada akurasi keseluruhan menghasilkan $\chi^2 = 0.6735$ ($p = 0.4118 > 0.05$). Nilai $p > 0.05$ pada akurasi agregat (65.26% vs 64.08%) terjadi karena metrik akurasi total tertutupi (*masked*) oleh dominasi kelas mayoritas (Positif & Negatif). Namun secara fungsional, IndoBERT mengoreksi **153 kesalahan klasifikasi semantik** XGBoost (dibandingkan 138 sebaliknya), dan keunggulan krusialnya terletak pada lonjakan **Macro-F1 (+8.71pp)** serta **pemulihan kelas minoritas Netral (+22.34pp)**.

> 📄 **Dokumentasi Lengkap Riset & Naskah Manuskrip:**
> - [Master Implementation Plan (Opsi B)](IMPLEMENTATION_PLAN.md)
> - [Brief Jurnal Scopus & Template Manuskrip](brief_jurnal_scopus.md)
> - [Panduan Reproduksibilitas](README_reproducibility.md)
> - [Aset Publikasi & Gambar 300 DPI](paper_assets/)

---

## 📁 Struktur Direktori Repositori

```
cobapenelitianpart4/
│
├── api/                             # Backend Service (FastAPI)
│   ├── __init__.py
│   └── main.py                     # API Routes & Endpoints
│
├── frontend/                        # Interactive Dashboard
│   └── app.py                      # Streamlit Multi-Page Application
│
├── scraper/                         # Scraping Modules
│   ├── gplay_scraper.py            # Bulk Google Play Store reviews scraper
│   ├── social_scraper.py           # Real-time multi-platform & search scraper
│   └── requirements.in
│
├── models/                          # ML & NLP Inference Engines
│   ├── budget_predictor.py         # Regresi Budget KOL
│   ├── sentiment_engine.py         # Mesin Sentimen & Aspek
│   ├── slm_generator.py            # Integrasi LLM/SLM Brief Generator
│   ├── tfidf_vectorizer.pkl        # TF-IDF Feature Extractor (10.000 fitur)
│   ├── rf_tfidf.pkl                # Random Forest Classifier weights
│   ├── xgb_tfidf.pkl               # Model XGBoost Classifier weights
│   ├── indobert_finetuned/         # Checkpoint Model & Tokenizer IndoBERT-base
│   ├── indobertweet_finetuned/     # Checkpoint Model & Tokenizer IndoBERTweet
│   └── README.md                   # Panduan model & weights
│
├── data/                            # Dataset & Vectorstore
│   ├── raw/                        # CSV ulasan mentah Play Store (16.000 baris)
│   ├── processed/                  # Dataset ulasan bersih (12.777 baris)
│   ├── splits/                     # Stratified Train (10.221), Val (1.278), Test (1.278)
│   ├── manual_validation_300_dual_annotators.csv # 300 sampel validasi ganda manusia
│   ├── results/                    # Hasil metrik evaluasi model lengkap
│   ├── chroma_db/                  # Local Chroma vector database
│   └── budget_model.joblib         # Model serialisasi budget
│
├── notebooks/                       # 5 Interaktif Jupyter Notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_rf_xgb.ipynb
│   ├── 04_indobert_finetune.ipynb
│   └── 05_comparative_analysis.ipynb
│
├── scripts/                         # Pipeline Eksekusi Python (Headless)
│   ├── run_preprocessing.py
│   ├── train_baseline_rf_xgb.py
│   ├── train_indobert.py
│   ├── train_indobertweet.py
│   └── run_comparative_analysis.py
│
├── paper_assets/                    # Visualisasi Ilmiah Siap Terbit (300 DPI)
│   ├── figure1_data_distribution.png
│   ├── figure2_wordcloud_per_class.png
│   ├── figure3_training_curves_indobert.png
│   ├── figure3_training_curves_indobertweet.png
│   ├── figure4_confusion_matrix_comparison.png
│   ├── figure4_confusion_matrix_indobert.png
│   ├── figure4_confusion_matrix_indobertweet.png
│   ├── figure5_f1_comparison_barplot.png
│   ├── table1_dataset_statistics.csv
│   ├── table2_model_comparison.csv
│   ├── table2_model_comparison_extended.csv
│   ├── table3_per_class_metrics.csv
│   ├── table4_ablation_study.csv
│   ├── cohen_kappa_inter_annotator_agreement.json
│   ├── label_consistency_audit.json
│   ├── mcnemar_test.json
│   └── error_analysis_samples.csv
│
├── src/                             # Simulasi Java OOP
│   ├── Main.java
│   └── engine/
│       ├── BadKolCampaignEngine.java
│       └── GoodKolCampaignEngine.java
│
├── brief_jurnal_scopus.md           # Brief penulisan artikel ilmiah Scopus
├── IMPLEMENTATION_PLAN.md           # Pelacakan implementasi & justifikasi akademik
├── README_reproducibility.md        # Panduan teknis eksekusi riset ulang
├── PRESENTASI_DOSEN.md              # Materi presentasi akademik/dosen
├── requirements.txt                 # Dependensi Python
└── run.bat                          # Script one-click runner Windows
```

---

## 🚀 Panduan Menjalankan Sistem

### 1. Instalasi Lingkungan
Pastikan Python 3.10+ sudah terinstal:
```bash
git clone https://github.com/Farrel4411l/Social-media-sentiment-analysist-and-KOL-specialist.git
cd Social-media-sentiment-analysist-and-KOL-specialist
pip install -r requirements.txt
```

### 2. Menjalankan Dashboard Streamlit
```bash
streamlit run frontend/app.py
```
*Atau klik dua kali file `run.bat` pada Windows.*

### 3. Menjalankan Backend FastAPI
```bash
uvicorn api.main:app --reload --port 8000
```
Akses dokumentasi interaktif Swagger di: `http://localhost:8000/docs`

### 4. Menjalankan Simulasi Java OOP
```bash
javac -d out src/Main.java src/engine/*.java
java -cp out Main
```

---

## 👨‍💻 Kontributor & Peneliti
- **Muhammad Farrel Haidar** — *Lead Researcher & Developer*
- Universitas / Afiliasi Riset: Program Studi Teknik Informatika / Ilmu Komputer

---

## 📜 Lisensi
Repositori ini dilisensikan di bawah lisensi MIT.
