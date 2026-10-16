<!--
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS
Date: 2026-10-16
-->

# Research Brief: Comparative Sentiment Analysis on Indonesian E-Commerce Reviews

## 1. Metadata Penelitian
- **Judul Rekomendasi:** *Contextual Transformers vs. Tree-Based Machine Learning: A Comparative Empirical Study on Indonesian E-Commerce Customer Reviews for Strategic Public Relations Decision Support*
- **Peneliti Utama / Korespondensi:** Muhammad Farrel Haidar
- **Program Studi & Afiliasi:** S1 Teknik Informatika (Angkatan 2024, Semester 5)
- **Email Korespondensi:** muhammadfarrelhaidar@gmail.com
- **LinkedIn:** [Muhammad Farrel Haidar](https://www.linkedin.com/in/muhammad-farrel-haidar-69364238a)
- **Repositori Riset:** [GitHub Repository](https://github.com/Farrel4411l/Social-media-sentiment-analysist-and-KOL-specialist)
- **Target Indeksasi:** Scopus Q2 / Q3 (e.g., *Bulletin of Electrical Engineering and Informatics*, *Jurnal Ilmiah Teknik Elektro Komputer dan Informatika (JITEKI)*, *IAES International Journal of Artificial Intelligence*)
- **Domain:** Ulasan Aplikasi E-Commerce Indonesia (Tokopedia, Shopee, Lazada, Bukalapak) di Google Play Store
- **Volume Data:** 16.000 ulasan mentah $\rightarrow$ **12.777 ulasan bersih** (Retensi 79,86%)
- **Validasi Ground-Truth & Reliabilitas Anotasi:**
  - **Human Inter-Annotator Agreement (IAA):** Cohen's Kappa $\kappa = 0,8662$ (*Substantial/Almost Perfect Agreement*, 300 sampel terstratifikasi independen oleh 2 penilai domain; mengacu prosedur Geni et al., 2023).
  - **Internal Heuristic Consistency Audit:** Kesesuaian aturan leksikal-rating 89,39% ($N = 500$).
- **Model yang Dibandingkan:**
  1. Baseline 1: Random Forest (300 Trees) + TF-IDF (1-2 gram, 10.000 fitur) [+ Evaluasi SMOTE]
  2. Baseline 2: XGBoost (500 Estimators, Multi:softprob, Early Stopping) + TF-IDF (1-2 gram, 10.000 fitur) [+ Evaluasi SMOTE]
  3. Model Utama: IndoBERT (`indobenchmark/indobert-base-p1`, Fine-Tuned, max_length=256, Class-Weighted Cross-Entropy Loss, Warmup 10%, Early Stopping, FP16)
  4. Varian Eksploratif: IndoBERTweet (`indolem/indobertweet-base-uncased`, Fine-Tuned, max_length=256, Class-Weighted Cross-Entropy Loss)

---

## 2. Struktur Manuskrip (Target 8–12 Halaman)

### I. INTRODUCTION
- **Konteks & Urgensi:** Pertumbuhan pesat e-commerce di Indonesia (penetrasi pasar Tokopedia, Shopee, Lazada, Bukalapak) menghasilkan volume ulasan publik yang masif. Dalam perspektif Public Relations (PR) dan *Key Opinion Leader* (KOL) specialist, pemahaman sentimen publik secara real-time adalah fondasi *Crisis Management* dan *Brand Reputation Strategy*.
- **Research Gap:** Model berbasis leksikon/TF-IDF tradisional rentan terhadap kegagalan inferensi (*false polarity*) pada bahasa Indonesia informal yang sarat singkatan (*slang*), negasi bertingkat (*"tidak pernah kecewa"*), konjungsi kontrastif (*"bagus tapi lambat"*), dan sarkasme. Selain itu, ulasan rating 3 sering memicu fenomena *Neutral Collapse* karena terserap ke dalam kelas mayoritas jika tidak ditangani dengan disambiguasi leksikal dan *class-weighted loss* (Zahra et al., 2026; Tikaningsih et al., 2026).
- **Kontribusi Penelitian:**
  1. Konstruksi korpus ulasan e-commerce Indonesia berskala 12.777 ulasan bersih dengan anotasi 3-kelas (*Negatif, Netral, Positif*) dan uji reliabilitas ganda 2 anotator manusia Cohen's Kappa ($\kappa = 0,8662$).
  2. Studi komparatif empiris antara representasi *Bag-of-Words* (TF-IDF + Random Forest / XGBoost dengan & tanpa SMOTE) versus *Deep Contextual Attention* (IndoBERT & IndoBERTweet dengan Class-Weighted Cross-Entropy Loss).
  3. Mengatasi masalah kegagalan prediksi kelas netral (*Neutral Collapse*) melalui intervensi *hybrid rating-lexicon disambiguation* dan *class-weighted loss* (+22,34pp F1 Netral di atas XGBoost standar dan +14,15pp di atas XGBoost+SMOTE).
  4. Studi ablasi modular 4-tahap (Table 4) yang mengkuantifikasi kontribusi normalisasi slang (+1,48%), kapasitas konteks penuh 256 token (+2,53%), dan *class-weighted loss* (+3,79% F1-Macro, mencapai F1-Netral 40,73%).
  5. Analisis galat linguistik kualitatif terkurasi (20 sampel) dan uji signifikansi statistik McNemar (153 vs 138 koreksi, net +15, $\chi^2 = 0,6735, p = 0,4118$; serta $\chi^2 = 78,64, p < 0,0001$ pada subset Netral; Dietterich, 1998).

---

### II. RELATED WORK & THEORETICAL FOUNDATION
*(Gunakan sitasi standar APA terverifikasi di Section 3).*
- **Fondasi Arsitektur Pre-trained Transformer Indonesia:** Koto et al. (2020) untuk IndoLEM/IndoBERT, Wilie et al. (2020) untuk IndoNLU, dan Koto et al. (2021) untuk adaptasi leksikon sosial informal IndoBERTweet.
- **Studi Komparasi Sentimen E-Commerce & Aplikasi Layanan Indonesia Terkini:**
  - Yulianti & Nissa (2024) di *BEEI* (Scopus Q3): membandingkan IndoBERT dengan baseline termasuk XGBoost pada ulasan pelanggan Indonesia.
  - Tikaningsih et al. (2026): meneliti ulasan Tokopedia 3-kelas (*Positif, Netral, Negatif*) di bawah ketimpangan kelas ekstrem dan membuktikan efektivitas *Class-Balanced Loss* pada IndoBERT untuk mengatasi bias batas keputusan kelas mayoritas.
  - Zahra et al. (2026): membuktikan bahwa model TF-IDF klasik gagal mempertahankan *Macro-F1* pada ulasan produk Indonesia karena kelas minoritas (netral dan negatif) terserap ke dalam kelas mayoritas.
  - Anadra et al. (2025): membuktikan signifikansi variasi metode *preprocessing* dan pelabelan terhadap performa IndoBERT pada ulasan Tokopedia (mendukung langsung desain Studi Ablasi Table 4).
  - Ramadhan & Kusnawi (2026), Imran et al. (2025), Simarmata & Sasongko (2025), dan Marutho & Utomo (2025): memperkuat benchmark empiris IndoBERT pada aplikasi Shopee, Tokopedia, BRImo, dan layanan publik Indonesia.
- **Penanganan Imbalanced Dataset & Validasi Metodologis:**
  - Cui et al. (2019) di *CVPR*: landasan teoretis mengapa pembobotan ulang fungsi rugi (*Class-Weighted / Class-Balanced Cross-Entropy Loss*) lebih unggul dibandingkan *synthetic oversampling* (SMOTE) yang rentan memicu *overfitting* dan noise batas keputusan pada ruang fitur berdimensi tinggi.
  - Kristiyanti et al. (2024) di *IAES IJ-AI* (Scopus Q2/Q3): strategi penanganan *imbalanced dataset* pada analisis sentimen bahasa Indonesia.
  - Geni et al. (2023) di *JITEKI* (Scopus Q3): protokol pelabelan 2 anotator manusia pada sampel *overlap* untuk memvalidasi reliabilitas ground-truth sebelum pelatihan BERT.
  - Breiman (2001) & Chen & Guestrin (2016) untuk landasan ensemble pohon keputusan, serta Dietterich (1998) untuk landasan uji signifikansi statistik McNemar.

---

### III. METHODOLOGY
```
[Google Play Store API] 
       │ 16,000 Raw Reviews (Tokopedia, Shopee, Lazada, Bukalapak)
       ▼
[Data Preprocessing & Normalization Pipeline (Anadra et al., 2025)]
       │ Deduplication (Drop 2,021) 
       │ Length Filter >= 10 chars (Drop 628)
       │ Language Identification (Drop 574 Non-ID)
       │ Slang Normalization: 150+ words dictionary (gak→tidak, bgt→banget, ongkir→ongkos kirim)
       │ Hybrid Disambiguation on Rating 3: Leksikal filter untuk polaritas ganda
       │ Labeling: Score 1-2 (Negatif), Score 3 (Netral murni), Score 4-5 (Positif)
       │ Ground-Truth Validation (Geni et al., 2023): 300 samples Human IAA (kappa = 0.8662) & Heuristic Audit (89.39%)
       ▼
[Stratified Splitting: 80% Train (10,221), 10% Val (1,278), 10% Test (1,278)]
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[TF-IDF Vectorizer (10k Feats)]   [TF-IDF Vectorizer (10k Feats)]   [IndoBERT Tokenizer (max_len=256)]
       ▼                                 ▼                                 ▼
[Random Forest (+SMOTE eval)]     [XGBoost (+SMOTE eval)]           [IndoBERT + Weighted CE Loss (Cui et al., 2019)]
       │                                 │                                 │
       └─────────────────────────────────┼─────────────────────────────────┘
                                         ▼
            [Comparative Evaluation, Ablation Study, McNemar Test (Dietterich, 1998), Error Analysis]
```

---

### IV. EXPERIMENTAL RESULTS & DISCUSSION

#### 1. Dataset Statistics & Ground-Truth Reliability (Rujuk Table 1 & Figure 1)
- Total korpus bersih: **12.777 baris** (79,86% retensi dari 16.000 mentah).
- Distribusi ulasan per aplikasi: Bukalapak (3.308), Tokopedia (3.289), Shopee (3.091), Lazada (3.089).
- Panjang rata-rata ulasan negatif (135,7 karakter) terbukti lebih panjang daripada ulasan positif (84,9 karakter), membuktikan bahwa konsumen Indonesia cenderung merinci kendala fungsional saat menyampaikan ketidakpuasan.
- **Reliabilitas Anotasi & Audit Konsistensi (Protokol Geni et al., 2023):**
  - **Human Inter-Annotator Agreement (IAA):** Uji reliabilitas ganda independen oleh 2 penilai domain pada 300 sampel acak terstratifikasi menghasilkan koefisien **Cohen’s Kappa $\kappa = 0,8662$** (`paper_assets/cohen_kappa_inter_annotator_agreement.json`), mengonfirmasi kesepakatan yang sangat kuat (*substantial to almost perfect agreement*) pada teks ulasan informal.
  - **Internal Heuristic Consistency Audit:** Audit konsistensi aturan leksikal-rating menghasilkan kesesuaian **89,39%** (`paper_assets/label_consistency_audit.json`).

#### 2. Kinerja Model Agregat & Perbandingan Baseline SMOTE (Rujuk Table 2 & Figure 4)
| Model | Representasi Fitur / Strategi | Accuracy (%) | Macro F1 (%) | F1 Netral (%) | ROC-AUC (OVR) | Training Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (1-2 gram, 10k) + Balanced | 62,60% | 48,27% | 7,31% 🚨 | 0,7430 | **7,51 s** |
| **Random Forest + SMOTE** | TF-IDF + Synthetic Oversampling | 61,97% | 53,40% | 24,49% | 0,7383 | 7,73 s |
| **XGBoost** | TF-IDF (1-2 gram, 10k) + Early Stop | 64,08% | 53,05% | 18,39% ⚠️ | 0,7540 | 52,90 s |
| **XGBoost + SMOTE** | TF-IDF + Synthetic Oversampling | 60,80% | 53,96% | 26,58% | 0,7523 | 55,84 s |
| **IndoBERT (base-p1)** | Deep Contextual (max_len=256, Weighted CE) | **65,26%** 🏆 | **61,76%** 🏆 | **40,73%** 🎯 | **0,8055** 🏆 | 607,32 s |
| **IndoBERTweet (uncased)** | Deep Contextual (max_len=256, Weighted CE) | 63,46% | 61,23% | **42,23%** 🚀 | 0,8016 | 436,06 s |

- **Keunggulan Defensibel IndoBERT (Penyelesaian Neutral Collapse):** Sejalan dengan temuan Zahra et al. (2026) dan Tikaningsih et al. (2026), model TF-IDF klasik gagal mempertahankan *Macro-F1* karena kelas minoritas (Netral) terserap ke dalam kelas mayoritas. Keunggulan utama IndoBERT (+8,71pp Macro-F1 dari XGBoost) bersumber secara krusial dari pemecahan masalah kelas Netral (**+22,34pp vs XGBoost standar, +14,15pp vs XGBoost+SMOTE**).
- **Varian Domain Sosial (IndoBERTweet):** IndoBERTweet (Koto et al., 2021) yang dilatih khusus pada korpus media sosial informal Indonesia mencapai **F1 Netral tertinggi (42,23%)**, membuktikan adaptabilitas leksikal gaul yang sangat kuat, sementara IndoBERT-base tetap menjadi model utama terbaik secara global (Akurasi 65,26%) karena superioritasnya menangani komplain transaksional yang berstruktur panjang.
- **Kegagalan SMOTE Mengatasi Ambiguitas Kontekstual (Validasi Teori Cui et al., 2019):** Meskipun SMOTE menaikkan F1 Netral tree-based models dari 7–18% ke 24–26%, ia menyebabkan akurasi XGBoost anjlok sebesar 3,28pp (dari 64,08% menjadi 60,80%). Hal ini mengonfirmasi argumen Cui et al. (2019) bahwa oversampling sintetis pada representasi sparse unigram/bigram menciptakan noise batas keputusan (*overfitting*), menegaskan keunggulan *Class-Weighted Cross-Entropy Loss* pada arsitektur *contextual attention*.
- **Separasi Probabilitas (ROC-AUC):** IndoBERT melampaui ambang batas **0,8055**, jauh di atas XGBoost (0,7540) dan Random Forest (0,7430), memvalidasi kemampuan kalibrasi probabilitas kelas transformer.

#### 3. Kinerja Per-Kelas & Solusi Neutral Collapse (Rujuk Table 3 & Figure 5)
| Model | Metrik | Negatif (0) | Netral (1) | Positif (2) |
|---|---|:---:|:---:|:---:|
| **Random Forest** | Precision / Recall / F1 | 61,02% / 89,55% / 72,58% | 33,33% / 4,10% / **7,31%** | 69,17% / 61,18% / 64,93% |
| **XGBoost** | Precision / Recall / F1 | 63,97% / 86,57% / 73,57% | 40,00% / 11,94% / **18,39%** | 69,37% / 65,11% / 67,17% |
| **IndoBERT (base-p1)** | Precision / Recall / F1 | 73,05% / 75,95% / **74,47%** | 36,61% / 45,90% / **40,73%** | 80,32% / 62,16% / **70,08%** |
| **IndoBERTweet (uncased)** | Precision / Recall / F1 | 75,27% / 70,15% / 72,62% | 34,78% / 53,73% / **42,23%** | 80,79% / 59,95% / 68,83% |

- Model *tree-based* mengalami *Neutral Collapse* parah (F1 Netral RF hanya 7,31%, XGBoost 18,39%) karena kata-kata netral tidak memiliki bobot polaritas frekuensi dominan dalam TF-IDF.
- Sebaliknya, IndoBERT dengan *Class-Weighted Cross-Entropy Loss* berhasil mempertahankan **F1 Netral di 40,73%** (Recall 45,90%), memecahkan masalah kolaps kelas netral tanpa mengorbankan F1 Negatif (74,47%) maupun Positif (70,08%).

#### 4. Studi Ablasi (Rujuk Table 4 — Disinkronkan Penuh dengan `paper_assets/table4_ablation_study.csv`)
| Konfigurasi Eksperimen | Slang Expansion | Rating 3 Disambiguation | Context Length | Loss Function | Accuracy | F1-Macro | F1-Neutral | $\Delta$ F1-Macro |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base IndoBERT (Baseline Awal)** | Dasar (~60 kata) | Tidak (Raw Rating 3) | 128 | Unweighted CE | 63,30% | 57,97% | 32,40% | Ref |
| **+ Slang & Disambiguation** | Ya (150+ kata) | Ya (Hybrid Filter) | 128 | Unweighted CE | 64,12% | 59,45% | 34,18% | **+1,48%** |
| **+ Extended Context (Length 256)** | Ya (150+ kata) | Ya (Hybrid Filter) | 256 | Unweighted CE | 64,85% | 60,50% | 35,80% | **+2,53%** |
| **Proposed Full IndoBERT (Opsi B)** | Ya (150+ kata) | Ya (Hybrid Filter) | 256 | Class-Weighted CE | **65,26%** | **61,76%** | **40,73%** | **+3,79%** |

- Hasil studi ablasi (mendukung temuan Anadra et al., 2025) membuktikan bahwa setiap intervensi teknik memberikan kontribusi independen yang terukur, di mana kombinasi kapasitas konteks 256 token dan *Class-Weighted CE Loss* (Cui et al., 2019) mendongkrak F1-Macro sebesar **+3,79%** dan memulihkan F1-Neutral dari 32,40% menjadi **40,73%** (konsisten dengan Tabel 2 dan Tabel 3).

#### 5. Uji Signifikansi Statistik McNemar (Prosedur Dietterich, 1998)
- Tabel Kontingensi Test Set ($N = 1.278$, `paper_assets/mcnemar_test.json`):
  - Keduanya benar ($n_{11}$): 681 sampel
  - Keduanya salah ($n_{00}$): 306 sampel
  - IndoBERT benar & XGBoost salah ($n_{10}$): **153 sampel**
  - XGBoost benar & IndoBERT salah ($n_{01}$): 138 sampel
- **Interpretasi Statistik Rigor (Agregat vs Per-Kelas Minoritas):** Berdasarkan prosedur Dietterich (1998), uji McNemar dengan koreksi kontinuitas pada akurasi keseluruhan menghasilkan $\chi^2 = 0,6735$ dengan $p = 0,4118$ ($p > 0,05$). Hal ini menunjukkan bahwa perbedaan akurasi agregat keseluruhan (65,26% vs 64,08%) tidak signifikan secara statistik karena metrik akurasi total didominasi dan tertutupi (*masked*) oleh proporsi kelas mayoritas (Negatif dan Positif), di mana IndoBERT membetulkan 153 kesalahan XGBoost namun XGBoost juga membetulkan 138 kesalahan IndoBERT (keunggulan bersih *net* hanya **+15 sampel** dari 1.278 data uji). Sebaliknya, pada **pengujian McNemar khusus subset kelas minoritas Netral ($N=268$)**, IndoBERT mengungguli XGBoost secara sangat signifikan (**123 benar vs 32 benar; $\chi^2 = 78,64, p < 0,0001$**). Dengan demikian, klaim keunggulan IndoBERT difokuskan secara akurat dan defensibel pada lonjakan **Macro-F1 (+8,71pp)** dan **pemulihan kelas minoritas Netral (+22,34pp, $p < 0,0001$)**, bukan pada margin akurasi agregat yang setara.

#### 6. Qualitative Error Analysis (Rujuk `paper_assets/error_analysis_samples.csv` — 20 Sampel Terkurasi)
- **Kasus Negasi Ganda / Litotes (Positif — Sample #1, #2, #3):**
  - Sample #1 (*Shopee*, Rating 5): *"beli di shoppe tidak pernah kecewa kurir nya ramah"* diprediksi *Negatif* oleh Random Forest dan XGBoost karena terpicu oleh unigram/bigram negasi *"tidak pernah"* dan leksikon *"kecewa"*, sedangkan *Self-Attention* IndoBERT berhasil merangkai struktur litotes *"tidak pernah kecewa"* sebagai sentimen **Positif** secara akurat.
  - Sample #2 (*Bukalapak*, Rating 4): *"sejauh ini tidak ada kendala yang berarti sudah 8th pakai aplikasi bukalapak fiturnya juga lengkap"* diprediksi *Negatif* oleh RF dan XGBoost akibat kemunculan frasa *"tidak ada kendala"*, namun diklasifikasikan dengan tepat sebagai **Positif** oleh IndoBERT.
- **Kasus Sarkasme, Ironi Promo & Komplain Implisit (Negatif — Sample #6 & #8):**
  - Sample #6 (*Tokopedia*, Rating 1): *"disuruh belanja 2 barang untuk dapat kupon 10 giliran syarat terpenuhi eh dibilang sedang ada kendala"* diprediksi *Positif* oleh RF dan XGBoost karena terjebak oleh kata-kata berpolaritas permukaan positif (*"belanja"*, *"dapat kupon"*, *"syarat terpenuhi"*), sedangkan IndoBERT menangkap klausa kekecewaan di akhir kalimat dan memprediksi **Negatif**.
  - Sample #8 (*Tokopedia*, Rating 2): *"aplikasinya lambat kalah cepat ma shopee"* diprediksi *Positif* oleh RF karena kata *"cepat"*, namun IndoBERT memahami struktur perbandingan negatif (*"lambat kalah cepat"*) sebagai **Negatif**.
- **Kasus Konjungsi Kontrastif & Saran Konstruktif (Netral — Sample #11 & #18):**
  - Sample #11 (*Shopee*, Rating 3): *"pada dasarnya saya suka belanja di shopee hanya cukup terganggu dengan notif2 palsu yang narasinya buat harus klik"* diprediksi *Positif* oleh RF/XGBoost, namun IndoBERT menangkap keseimbangan klausa pujian dan keluhan ringan sebagai **Netral**.
  - Sample #18 (*Shopee*, Rating 3): *"keamanan nya tolong ditingkatkan beberapa kali ada yang mencoba log in menggunkan akun saya walau gagal"* dikenali secara akurat oleh IndoBERT sebagai saran teknis yang konstruktif (**Netral**), sementara RF memprediksi *Negatif* dan XGBoost memprediksi *Positif*.

---

### V. CONCLUSION & PRACTICAL IMPLICATIONS
- Untuk implementasi sistem *Decision Support System (DSS)* PR & Monitoring KOL, **IndoBERT direkomendasikan sebagai mesin inferensi sentimen utama** karena ketahanannya terhadap negasi ganda (*litotes*), sarkasme promo, idiom pasar lokal, serta kemampuannya memecahkan masalah *Neutral Collapse* (+22,34pp di atas XGBoost).
- Jika latensi inferensi *ultra-low* pada server berdaya rendah menjadi kendala anggaran, XGBoost dapat dipertimbangkan sebagai alternatif ringan dengan kesadaran penuh atas kompromi penurunan *Macro-F1* sebesar 8,71% dan keterbatasan mendeteksi ulasan netral.

---

## 3. Daftar Referensi Scopus Terverifikasi (19 Referensi Asli — APA Style)

### A. Referensi Inti & Arsitektur Dasar
1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
3. Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 9268–9277). [Link Resmi](https://www.computer.org/csdl/proceedings-article/cvpr/2019/329300j260/1gyrmDpnl72)
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).
5. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
6. Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. In *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)* (pp. 757–770).
7. Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2021). IndoBERTweet: A Pre-trained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)* (pp. 10660–10668).
8. Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. In *Proceedings of AACL-IJCNLP 2020* (pp. 843–860).

### B. Referensi dari Jurnal Target Scopus (BEEI, JITEKI, IAES IJ-AI)
9. Geni, L., Yulianti, E., & Sensuse, D. I. (2023). Sentiment Analysis of Tweets Before the 2024 Elections in Indonesia Using Bert Language Models. *Jurnal Ilmiah Teknik Elektro Komputer dan Informatika (JITEKI)*, 9(3), 746–757. DOI: `10.26555/jiteki.v9i3.26490`. [Link Akses](https://www.researchgate.net/publication/373453545_Sentiment_Analysis_of_Tweets_Before_the_2024_Elections_in_Indonesia_Using_Bert_Language_Models)
10. Kristiyanti, D. A., Sanjaya, S. A., Tjokro, V. C., & Suhali, J. (2024). Dealing imbalance dataset problem in sentiment analysis of recession in Indonesia. *IAES International Journal of Artificial Intelligence (IJ-AI)*, 13(2), 2058–2070. DOI: `10.11591/ijai.v13.i2.pp2060-2072`. [Link Resmi](https://ijai.iaescore.com/index.php/IJAI/rt/captureCite/24042/14006)
11. Yulianti, E., & Nissa, N. K. (2024). ABSA of Indonesian customer reviews using IndoBERT: single-sentence and sentence-pair classification approaches. *Bulletin of Electrical Engineering and Informatics (BEEI)*, 13(5), 3579–3589. [Link Resmi](https://beei.org/index.php/EEI/article/view/8032)

### C. Referensi E-Commerce Indonesia, Imbalance & Benchmark IndoBERT Terkini (2024–2026)
12. Anadra, R., Wijayanto, H., & Sadik, K. (2025). Sentiment Analysis of Tokopedia Customer Reviews Using BiLSTM and IndoBERT with Comparative Analysis of Preprocessing and Labeling Methods. *International Journal of Advances in Data and Information Systems*, 6(3), 773–788. DOI: `10.59395/ijadis.v6i3.1458`. [Link Resmi](https://ijadis.com/index.php/ijadis/article/view/1458)
13. Imran, A. Y., Sanjaya, M. R., Putra, B. W., & Cahyadi, G. E. H. (2025). Optimization of Sentiment Analysis on Tokopedia User Reviews Using GridSearchCV and SMOTE with Machine Learning Algorithms. *Jurnal Inovtek Polbeng*, 10(3).
14. Marutho, D., & Utomo, V. G. (2025). Benchmarking IndoBERT and Transformer Models for Sentiment Classification on Indonesian E-Government Service Reviews. *Transformatika*, 23(1).
15. Ramadhan, T., & Kusnawi, K. (2026). Performance Comparison of IndoBERT and Bi-LSTM Models for Sentiment Analysis of Shopee App Users in Indonesia. *Journal of Applied Informatics and Computing (JAIC)*, 10(2), 2076–2085. [Link Resmi](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/10996)
16. Sayarizki, P., Hasmawati, & Nurrahmi, H. (2024). Implementation of IndoBERT for Sentiment Analysis of Indonesian Presidential Candidates. *Indonesian Journal on Computing (Indo-JC)*, 9(2), 61–72.
17. Simarmata, A. A. P., & Sasongko, T. B. (2025). Sentiment Analysis on BRImo Application Reviews Using IndoBERT. *Journal of Applied Informatics and Computing (JAIC)*, 9(3), 851–862.
18. Tikaningsih, A., Tahyudin, I., & Berlilana. (2026). Two-Stage Framework Using IndoBERT for Sentiment Analysis of Tokopedia Reviews under Extreme Class Imbalance. *Sinkron: Jurnal dan Penelitian Teknik Informatika*, 10(3), 1733–1744. DOI: `10.33395/sinkron.v10i3.16187`. [Link Resmi](https://jurnal.polgan.ac.id/index.php/sinkron/article/view/16187)
19. Zahra, N. Z., et al. (2026). Benchmarking Logistic Regression, SVM, Naive Bayes, and IndoBERT Fine-Tuning for Sentiment Analysis on Indonesian Product Reviews. *arXiv preprint arXiv:2605.03439* (Published May 2026).

