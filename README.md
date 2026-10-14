<!-- 
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS & Comparative Research Pipeline
Date: 2026-10-14
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

Riset ini mengevaluasi keandalan model *Contextual Transformer* versus *Tree-Based Machine Learning* pada teks ulasan informal bahasa Indonesia.

### Ringkasan Hasil Eksperimen Komparatif (Opsi B: Test Set, $N = 1.278$)
| Model | Representasi Fitur | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | ROC-AUC (OVR) | Training Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** (300 Trees) | TF-IDF (1-2 gram, 10k) | 62.60% | 48.27% | 56.46% | 0.7430 | **7.51 s** |
| **XGBoost** (Early Stopping) | TF-IDF (1-2 gram, 10k) | 64.08% | 53.05% | 59.96% | 0.7540 | 52.90 s |
| **IndoBERT** (`indobert-base-p1`) | Deep Contextual (max_len=256) | **65.26%** | **61.76%** | **66.00%** | **0.8055** | 607.32 s |

- **Keunggulan Signifikan IndoBERT**: Unggul **+8.71% Macro-F1** dari XGBoost dan **+13.49%** dari Random Forest, dengan ROC-AUC melampaui ambang batas **0.8055**.
- **Solusi Neutral Collapse**: Penerapan *Class-Weighted Cross-Entropy Loss* berhasil menaikkan F1-Score kelas Netral ke **37.59%** (bebas dari fenomena kolaps model pohon).
- **Integritas Ground-Truth Labeling**: Uji kesepakatan inter-annotator pada 500 sampel acak terstratifikasi menghasilkan skor **Cohen’s Kappa $\kappa = 0.8939$** (*Almost Perfect Agreement*).
- **Ablation Study (Table 4)**: Mengonfirmasi kontribusi bertahap dari Slang Expansion (+1.48%), Extended Context Length 256 (+2.53%), dan Class-Weighted Loss (+3.79% F1-Macro).

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
│   ├── tfidf_vectorizer.pkl        # TF-IDF Feature Extractor
│   ├── xgb_tfidf.pkl               # Model XGBoost Classifier
│   ├── indobert_finetuned/         # Tokenizer & Konfigurasi IndoBERT
│   └── README.md                   # Panduan model & weights
│
├── data/                            # Dataset & Vectorstore
│   ├── raw/                        # CSV ulasan mentah Play Store
│   ├── processed/                  # Dataset ulasan bersih (12.777 baris)
│   ├── splits/                     # Stratified Train, Val, Test splits
│   ├── results/                    # Hasil metrik evaluasi model
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
│   └── run_comparative_analysis.py
│
├── paper_assets/                    # Visualisasi Ilmiah Siap Terbit (300 DPI)
│   ├── figure1_data_distribution.png
│   ├── figure2_wordcloud_per_class.png
│   ├── figure3_training_curves_indobert.png
│   ├── figure4_confusion_matrix_comparison.png
│   ├── figure5_f1_comparison_barplot.png
│   ├── table1_dataset_statistics.csv
│   ├── table2_model_comparison.csv
│   ├── table3_per_class_metrics.csv
│   └── error_analysis_samples.csv
│
├── src/                             # Simulasi Java OOP
│   ├── Main.java
│   └── engine/
│       ├── BadKolCampaignEngine.java
│       └── GoodKolCampaignEngine.java
│
├── brief_jurnal_scopus.md           # Brief penulisan artikel ilmiah Scopus
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
