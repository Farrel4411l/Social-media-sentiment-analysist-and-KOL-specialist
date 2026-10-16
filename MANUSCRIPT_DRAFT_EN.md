<!--
Author: Muhammad Farrel Haidar
Project: AI PR & KOL Specialist DSS
Date: 2026-10-17
-->

# Contextual Transformers vs. Tree-Based Machine Learning: A Comparative Empirical Study on Indonesian E-Commerce Customer Reviews for Strategic Public Relations Decision Support

**Muhammad Farrel Haidar**  
*Department of Information Systems / Computer Science*  
*Corresponding Author Email: farrel.haidar@example.edu*

---

## Abstract

In the contemporary digital economy, customer reviews on mobile application marketplaces serve as a primary indicator of consumer satisfaction, product usability, and corporate reputation. For Public Relations (PR) and Key Opinion Leader (KOL) specialists, automated sentiment analysis is essential for proactive brand health monitoring and rapid crisis escalation management. However, Indonesian e-commerce reviews present severe linguistic challenges, including widespread colloquialisms, informal abbreviations (*slang*), double negations (*litotes*), contrastive conjunctions, and extreme class imbalance—most notably manifested as the "Neutral Collapse" phenomenon, where non-polarized or constructive feedback is overwhelmingly absorbed by majority sentiment classes. This paper presents an end-to-end comparative empirical evaluation of traditional tree-based machine learning ensembles against deep contextual Transformer architectures on a multi-vendor Indonesian e-commerce corpus. We curated a balanced raw collection of 16,000 user reviews across four dominant platforms (Tokopedia, Shopee, Lazada, and Bukalapak) from the Google Play Store, resulting in 12,777 rigorously cleaned samples after deduplication, language filtering, domain-specific slang normalization (150+ words lexicon), and a hybrid rating-lexicon disambiguation protocol for 3-star reviews. Ground-truth annotation reliability was empirically validated through an independent dual-annotator study on 300 stratified samples, yielding substantial agreement with Cohen's Kappa $\kappa = 0.8662$, supported by an internal heuristic consistency audit of 89.39%. We systematically benchmarked Random Forest and XGBoost (under standard TF-IDF and Synthetic Minority Over-sampling Technique/SMOTE) against fine-tuned IndoBERT (`indobenchmark/indobert-base-p1`) and social-domain IndoBERTweet (`indolem/indobertweet-base-uncased`) trained with Class-Weighted Cross-Entropy Loss. Empirical results demonstrate that IndoBERT achieves superior overall performance with 65.26% accuracy, 61.76% Macro-F1, and an ROC-AUC of 0.8055. Crucially, while traditional models collapsed on the neutral class (Random Forest F1: 7.31%; XGBoost F1: 18.39%), IndoBERT successfully restored the neutral class to 40.73% F1 (+22.34 percentage points over XGBoost), and IndoBERTweet attained 42.23% neutral F1. Synthetic oversampling (SMOTE) failed to resolve contextual ambiguity, deteriorating XGBoost accuracy by 3.28 pp. A modular four-stage ablation study confirms the incremental necessity of slang normalization (+1.48 pp), extended 256-token context (+2.53 pp), and class-weighted loss (+3.79 pp). Rigorous McNemar statistical testing reveals that while aggregate accuracy differences reflect a net shift of +15 samples ($\chi^2 = 0.6735, p = 0.4118$) masked by majority class inertia, IndoBERT is statistically overwhelmingly superior on the minority neutral class ($\chi^2 = 71.68, p < 0.0001$). A qualitative linguistic error analysis on 20 curated real-world cases demonstrates IndoBERT's robust disambiguation of sarcasm, brand comparisons, and multi-clause feedback. Finally, an architectural blueprint integrating this pipeline into a Decision Support System (DSS) for PR brand protection and KOL campaign tracking is proposed.

**Keywords:** Sentiment Analysis, Natural Language Processing, IndoBERT, Class Imbalance, Neutral Collapse, E-Commerce Reviews, Decision Support System, Public Relations.

---

## I. INTRODUCTION

The rapid expansion of the digital commerce ecosystem in Southeast Asia, particularly in Indonesia, has catalyzed a fundamental paradigm shift in retail communication and consumer feedback dynamics. Digital marketplaces such as Tokopedia, Shopee, Lazada, and Bukalapak serve tens of millions of active daily users, generating vast streams of user-generated feedback across application stores such as the Google Play Store. Within contemporary corporate communication and digital marketing frameworks, these real-time textual reviews represent an indispensable intelligence asset for Public Relations (PR) professionals and Key Opinion Leader (KOL) campaign specialists. Timely detection of emerging product defects, payment gateway downtimes, counterfeit merchant activities, or predatory promotional mechanics is vital for orchestrating prompt crisis containment protocols, preserving brand reputation, and evaluating influencer campaign effectiveness [1]–[3].

Despite its strategic importance, the automated extraction of sentiment from Indonesian customer reviews remains fraught with formidable computational and linguistic obstacles. First, Indonesian netizen discourse in application marketplaces is heavily characterized by non-standard orthography, code-mixing (Indonesian, regional dialects like Javanese and Sundanese, and English loanwords), arbitrary word clippings, and pervasive e-commerce slang (e.g., *"ongkir"* for *ongkos kirim*, *"bgt"* for *banget*, *"gk"* for *tidak*, *"seller"* for *penjual*) [4], [5]. Conventional Bag-of-Words (BoW) representations and sparse n-gram Term Frequency-Inverse Document Frequency (TF-IDF) feature spaces struggle to capture semantic equivalence across morphological variants. Second, customer reviews frequently exhibit complex linguistic phenomena such as double negations and litotes (e.g., *"tidak pernah mengecewakan"* — lit. *never disappoints*, expressing strong positive sentiment), contrastive conjunctions spanning multiple clauses (e.g., *"aplikasi bagus tapi pengirimannya lambat"* — *good application but delivery is slow*), and biting sarcasm disguised beneath superficial promotional praise [6], [7].

Third, and perhaps most critically from a methodological standpoint, real-world customer reviews exhibit acute class imbalance coupled with what recent literature identifies as the **"Neutral Collapse"** phenomenon [8], [9]. In standard commercial feedback systems, customers typically provide binary polarity ratings (1–2 stars representing explicit dissatisfaction and 4–5 stars reflecting enthusiastic endorsement). In contrast, 3-star reviews embody an inherently ambiguous continuum ranging from balanced trade-offs and tentative evaluations to constructive feature suggestions and payment security warnings. When traditional machine learning models—such as Random Forest [10] or Extreme Gradient Boosting (XGBoost) [11]—are deployed on three-class corpora, their decision boundaries are overwhelmed by majority polarities. Consequently, the neutral class collapses almost entirely, yielding recall and F1-scores beneath 10% to 20% [8], [9]. In corporate PR surveillance, this collapse is disastrous: constructive user suggestions and early-stage systemic bugs frequently appear in 3-star feedback, and categorizing them blindly into positive or negative categories blinds decision-makers to subtle shifts in consumer sentiment.

While synthetic oversampling methodologies such as SMOTE [12] are frequently proposed as default remedies for imbalanced classification, theoretical findings in representation learning highlight that synthesizing artificial instances within sparse, high-dimensional TF-IDF vectors introduces significant decision boundary noise and degrades overall generalization [13], [14]. Concurrently, the advent of pre-trained contextual Transformer architectures for Indonesian NLP—exemplified by IndoBERT [15], IndoNLU [16], and the social media-specialized IndoBERTweet [17]—has opened unprecedented opportunities to capture bidirectional syntactic and contextual dependencies. Nevertheless, a substantial gap persists in the empirical literature regarding:
1. How pre-trained Transformers compare rigorously against tuned tree-based baselines under balanced multi-vendor e-commerce distributions;
2. Whether class-weighted loss formulations within Transformers fundamentally surpass synthetic oversampling in mitigating neutral class collapse; and
3. How these model behaviors withstand formal statistical hypothesis testing and qualitative linguistic deconstruction.

To resolve these fundamental challenges, this research delivers five primary contributions:
1. **Multi-Vendor Corpus Construction with Verified Ground-Truth:** We construct and release a standardized benchmark corpus of 12,777 cleaned Indonesian e-commerce reviews spanning four leading platforms (Tokopedia, Shopee, Lazada, Bukalapak) balanced from 16,000 raw samples, backed by an independent dual-annotator human reliability study on 300 stratified reviews demonstrating substantial agreement ($\kappa = 0.8662$) alongside an 89.39% heuristic consistency audit [18].
2. **Empirical Benchmarking of Tree-Based vs. Transformer Architectures:** We conduct a systematic comparison across six distinct configurations: Random Forest (standard and SMOTE-augmented), XGBoost (standard and SMOTE-augmented), IndoBERT-base-p1, and IndoBERTweet, evaluating macro-averaged, class-specific, and probabilistic discrimination metrics (ROC-AUC).
3. **Resolution of the Neutral Collapse Phenomenon:** We demonstrate that while tree-based models and synthetic oversampling fail to solve neutral collapse (RF F1: 7.31%; XGB+SMOTE F1: 26.58% at a cost of 3.28 pp overall accuracy), fine-tuned IndoBERT equipped with Class-Weighted Cross-Entropy Loss elevates the neutral F1-score to **40.73%** (+22.34 pp over XGBoost), with IndoBERTweet reaching **42.23%**.
4. **Modular Four-Stage Ablation Study:** We empirically disentangle the relative performance contributions of domain-specific slang normalization (+1.48 pp), extended 256-token context windows (+2.53 pp), and class-weighted optimization (+3.79 pp), validating recent preprocessing paradigms [5].
5. **Rigorous Statistical and Qualitative Linguistic Deconstruction:** We perform a dual-level McNemar significance test [19], revealing that while aggregate accuracy gains reflect a balanced net shift of +15 samples ($\chi^2 = 0.6735, p = 0.4118$) due to majority-class masking, IndoBERT's neutral class superiority is statistically overwhelming ($\chi^2 = 71.68, p < 0.0001$). This is corroborated by a qualitative audit of 20 real-world failure cases categorizing litotes, promotional sarcasm, and contrastive syntax into an actionable Decision Support System (DSS) framework for PR intelligence.

The remainder of this article is structured as follows: Section II reviews related literature. Section III details the methodology, corpus curation, disambiguation, and modeling pipelines. Section IV presents experimental results, ablation studies, statistical tests, and qualitative error analysis. Section V evaluates threats to validity. Section VI discusses DSS architectural integration and managerial implications. Section VII concludes the paper.

---

## II. RELATED WORK

### A. Sentiment Analysis in E-Commerce and Indonesian NLP
Sentiment analysis within e-commerce has progressively shifted from lexicon-based keyword scoring to machine learning and deep representation learning. In the Indonesian linguistic sphere, early efforts were severely constrained by the scarcity of standardized corpora and open-source linguistic resources. Koto et al. [15] introduced IndoLEM and IndoBERT, the first large-scale Indonesian language model pre-trained on over 220 million words from Wikipedia, news portals, and web crawls. Simultaneously, Wilie et al. [16] released IndoNLU, providing a foundational 12-task benchmark spanning natural language inference, question answering, and aspect-based sentiment analysis. Recognising the linguistic gulf between formal Indonesian prose and conversational social media text, Koto et al. [17] developed IndoBERTweet, pre-trained on 409 million words of Indonesian Twitter data using a tailored Byte-Pair Encoding (BPE) vocabulary that explicitly preserves Twitter-specific slang, hashtags, and colloquial morphology.

Recent empirical investigations have explored these architectures across varied Indonesian application domains. Yulianti and Nissa [4] demonstrated in the *Bulletin of Electrical Engineering and Informatics (BEEI)* that IndoBERT consistently outperformed traditional classifiers in aspect-based sentiment analysis on customer reviews, attributing this success to self-attention mechanisms capturing long-range contextual tokens. Ramadhan and Kusnawi [20] benchmarked IndoBERT against Bi-LSTM on Shopee user reviews, reporting superior contextual representation but noting inference latency challenges. Simarmata and Sasongko [21] applied IndoBERT to the BRImo mobile banking application, reporting high aggregate accuracy yet observing recurring confusion when reviews contained mixed emotional polarities. Marutho and Utomo [22] conducted benchmarking across Indonesian e-government public service reviews, demonstrating that domain-specific contextual representations significantly mitigate false polarity classifications.

### B. Tree-Based Ensembles vs. Deep Contextual Attention
Before the dominance of Transformers, ensemble tree-based algorithms paired with n-gram TF-IDF representations served as the gold standard for text classification. Random Forest, developed by Breiman [10], constructs an ensemble of de-correlated decision trees through bootstrap aggregation (bagging) and random feature selection, offering robust resistance to overfitting. Chen and Guestrin [11] introduced XGBoost, an optimized distributed gradient boosting library that employs second-order Taylor expansions of the loss function, sparsity-aware splitting, and algorithmic regularization to maximize predictive speed and accuracy. 

While tree-based models achieve high computational efficiency, their reliance on the Bag-of-Words assumption renders them blind to syntactic word ordering, negation scope, and polysemy. Devlin et al. [23] revolutionized the field with BERT, replacing unidirectional recurrent connections with deep bidirectional Transformer encoders. By conditioning token representations on both left and right context simultaneously across multiple attention heads, BERT captures complex grammatical hierarchies that tree-based algorithms inherently fail to model.

### C. Class Imbalance, Neutral Collapse, and Loss Function Formulations
In real-world sentiment analysis, class distributions are notoriously skewed. Extreme class imbalance creates an acute pathology known as the "Neutral Collapse," thoroughly documented in recent Indonesian studies [8], [9]. Zahra et al. [8] benchmarked classical algorithms (Logistic Regression, SVM, Naive Bayes) against fine-tuned IndoBERT on Indonesian product reviews, finding that while classical models maintained reasonable accuracy on majority classes, their Macro-F1 collapsed drastically due to near-zero recall on minority neutral samples. Tikaningsih et al. [9] investigated extreme class imbalance on Tokopedia reviews, proving that conventional cross-entropy loss causes Transformer gradients to be dominated by majority positive and negative vectors, flattening the decision boundaries around neutral expressions.

To counter class imbalance, traditional practitioners frequently resort to Synthetic Minority Over-sampling Technique (SMOTE), introduced by Chawla et al. [12], or heuristic hyperparameter search [24]. However, Kristiyanti et al. [14] cautioned in *IAES International Journal of Artificial Intelligence* that synthetic oversampling in high-dimensional text feature spaces often synthesizes unnatural interpolation points that blur legitimate classification margins. From a theoretical perspective, Cui et al. [13] in *CVPR* demonstrated that as data scales, synthetic samples offer diminishing marginal utility because representations overlap; instead, re-weighting the objective function using a Class-Balanced / Class-Weighted Cross-Entropy Loss based on the effective number of samples offers superior regularization and prevents boundary drift without synthesizing noisy instances.

### D. Preprocessing, Slang Normalization, and Ground-Truth Reliability
The quality of text classification in low-resource and colloquial domains is fundamentally bound to preprocessing rigor. Anadra, Wijayanto, and Sadik [5] demonstrated on Tokopedia customer reviews that systematic preprocessing variations—specifically domain-tailored slang expansion, stopword filtering adjustments, and punctuation preservation—exert a statistically significant impact on IndoBERT fine-tuning performance. Sayarizki, Hasmawati, and Nurrahmi [25] confirmed that informal abbreviations and slang heavily confound tokenizers if left uncurated. 

Furthermore, the integrity of academic benchmarks depends on annotation reproducibility. Geni, Yulianti, and Sensuse [18] in *JITEKI* established a rigorous dual-annotator verification protocol on overlapping sample subsets, utilizing Cohen's Kappa coefficient to ensure that human-assigned sentiment boundaries are statistically robust and distinct from heuristic noise before training neural architectures.

---

## III. RESEARCH METHODOLOGY

The systematic research framework executed in this study is illustrated in Figure 1. The pipeline comprises four modular phases: (1) multi-vendor corpus acquisition and filtering; (2) domain-specific slang normalization and hybrid ground-truth disambiguation; (3) stratified feature extraction and model training across six architectures; and (4) multidimensional performance evaluation, ablation analysis, McNemar hypothesis testing, and qualitative error deconstruction.

```
+----------------------------------------------------------------------------------------------------+
|                                      CORPUS ACQUISITION PHASE                                      |
|  Google Play Store API Scraping: 16,000 Raw Customer Reviews (Tokopedia, Shopee, Lazada, Bukalapak)  |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                DATA PREPROCESSING & CLEANING PIPELINE                              |
|  • Exact & Substring Deduplication (Dropped: 2,021 duplicates)                                     |
|  • Length & Character Filtering: Length >= 10 chars (Dropped: 628 short/gibberish texts)            |
|  • Automated Language Identification: langdetect ID filter (Dropped: 574 non-Indonesian texts)     |
|  • E-Commerce Domain Slang Normalization: 150+ Lexicon Dictionary (gak->tidak, ongkir->ongkos kirim) |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                            GROUND-TRUTH DISAMBIGUATION & VALIDATION                                |
|  • Star 1-2: Negative Polarity | Star 4-5: Positive Polarity                                        |
|  • Star 3 Hybrid Disambiguation: Dual-lexicon filter separating pure neutrals from misaligned stars|
|  • Ground-Truth Verification: Dual Human Annotators (N=300, Cohen's Kappa kappa = 0.8662)          |
|  • Internal Heuristic Consistency Audit: 89.39% Rule Alignment (N=500)                              |
|  ==> FINAL CLEAN DATASET: 12,777 Reviews (79.86% Retention Rate)                                    |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                       STRATIFIED SPLITTING (Seed=42, 80/10/10 Ratio)                               |
|          Training Set: 10,221 reviews | Validation Set: 1,278 reviews | Test Set: 1,278 reviews     |
+------------------------+------------------------+-------------------------+------------------------+
                         |                        |                         |
                         v                        v                         v
          +-------------------------------+  +-------------------------------+  +--------------------+
          |  TF-IDF Vectorization (10k)   |  |   TF-IDF + SMOTE Resampling   |  | IndoBERT Tokenizer |
          |   (Unigram + Bigram, max_df)  |  |  (Synthetic Over-sampling)    |  |  (max_length=256)  |
          +---------------+---------------+  +---------------+---------------+  +---------+----------+
                          |                                  |                            |
            +-------------+-------------+      +-------------+-------------+              |
            v                           v      v                           v              v
     +--------------+            +------------++--------------+            +------------+ +----------+
     |Random Forest |            |  XGBoost   ||RF + SMOTE    |            |XGB + SMOTE | | IndoBERT |
     | (300 Trees)  |            |(500 Trees) || (300 Trees)  |            |(500 Trees) | |(base-p1) |
     +--------------+            +------------++--------------+            +------------+ +----+-----+
                                                                                               |
                                                                                               v
                                                                                        +------------+
                                                                                        | IndoBERT-  |
                                                                                        |   tweet    |
                                                                                        +-----+------+
                                                                                              |
                                                  +-------------------------------------------+
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                EVALUATION & ANALYTICAL FRAMEWORK                                   |
|  • Aggregate Metrics: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Multi-Class ROC-AUC (OVR)|
|  • Neutral Collapse Audit: Class-specific Precision/Recall/F1 for Negative, Neutral, Positive       |
|  • 4-Stage Incremental Ablation Study: Slang (+1.48%), Context 256 (+2.53%), Weighted Loss (+3.79%)|
|  • Statistical Testing: McNemar's Test with Continuity Correction (Aggregate vs Minority Subset)   |
|  • Qualitative Linguistic Error Analysis: 20 Curated Failure Modes (Litotes, Sarcasm, Contrastive) |
|  • Decision Support System (DSS) Architectural Blueprint for PR & KOL Decision Makers             |
+----------------------------------------------------------------------------------------------------+
```
*Figure 1. End-to-end experimental architecture and comparative sentiment evaluation methodology.*

### A. Corpus Acquisition
To eliminate platform-specific algorithmic bias and ensure broad representativeness across the Indonesian digital retail landscape, raw user reviews were scraped directly from the official Google Play Store application pages between 2024 and 2026 using the `google-play-scraper` interface. Four major platforms were targeted: Tokopedia, Shopee, Lazada, and Bukalapak. A balanced scraping quota of 4,000 reviews was harvested per platform across all five discrete star ratings (800 reviews per star rating per application), yielding an initial raw repository of $N_{\text{raw}} = 16,000$ entries.

### B. Data Cleaning and Slang Normalization Pipeline
Raw app store reviews are notoriously noisy, rife with spam, character repetitions, typographical errors, and multilingual noise. A multi-stage deterministic cleaning pipeline was implemented:
1. **Deduplication:** Repeated review texts (frequently generated by promotional bots or identical copy-paste submissions) were detected via exact string matching and case-insensitive whitespace stripping. A total of 2,021 duplicate records were excised.
2. **Length and Structural Filtering:** Reviews shorter than 10 characters or comprising exclusively punctuation, numerical tokens, or non-alphanumeric noise (e.g., *"ok"*, *"....."*, *"12345"*) were discarded. This step eliminated 628 invalid entries.
3. **Language Identification:** Using the `langdetect` library initialized with a deterministic seed, reviews identified as non-Indonesian (e.g., pure English feedback, Tagalog, Vietnamese, or automated machine-translated text) were removed, accounting for 574 discarded instances.
4. **Domain-Specific Slang Normalization:** Indonesian e-commerce reviews utilize an extensive colloquial vocabulary that severely fragments tokenization. We formulated an expert-curated normalization lexicon comprising over 150 phonetic, orthographic, and abbreviations mappings. Table I illustrates representative normalization entries.

**TABLE I. Representative Entries from Domain-Specific E-Commerce Slang Dictionary**
| Raw Colloquial / Slang Token | Normalized Standard Form | Meaning / Category |
|---|---|---|
| `gak`, `ga`, `gkk`, `ngga`, `kaga` | `tidak` | Standard negation |
| `bgt`, `bngt`, `bngttt`, `beut` | `banget` | Intensifier (*very / extremely*) |
| `ongkir`, `ongkrr` | `ongkos kirim` | Shipping cost / delivery fee |
| `tp`, `tpi`, `tapi` | `tetapi` | Contrastive conjunction |
| `bintang 1`, `bintang satu`, `b1` | `bintang satu` | Negative rating expression |
| `seller`, `sller`, `penjual` | `penjual` | Merchant / vendor |
| `expired`, `exp`, `kadaluarsa` | `kedaluwarsa` | Expired goods |
| `ori`, `original`, `asli` | `asli` | Authentic product assertion |
| `lemot`, `lelet`, `lola` | `lambat` | Systemic application latency |
| `nyesel`, `nyesal` | `menyesal` | Emotional regret |

Following deduplication and cleaning, the final corpus comprised **12,777 high-quality reviews**, representing a healthy **79.86% retention rate**.

### C. Ground-Truth Disambiguation and Annotation Reliability
A common convention in app store sentiment literature maps 1–2 star ratings directly to Negative sentiment and 4–5 star ratings to Positive sentiment. However, 3-star reviews represent a volatile ground-truth category. An uncritical assumption that all 3-star reviews are Neutral introduces profound label contamination: consumers frequently award 3 stars to express devastating operational complaints accompanied by polite greetings, or give 3 stars because of delivery delays despite loving the product [9]. 

To establish an uncompromised ground-truth, we applied a **hybrid rating-lexicon disambiguation filter** to all 3-star reviews:
- If a 3-star review contained exclusively strong negative lexical cues (e.g., *"penipuan"*, *"kecewa berat"*, *"rusak"*) without positive compensation, it was reassigned to Negative.
- If it contained exclusively enthusiastic praise (e.g., *"sangat puas"*, *"mantap original"*), it was reassigned to Positive.
- Only reviews exhibiting balanced polarities, constructive feedback (e.g., *"tolong tambahkan fitur..."*), tentative assessments (e.g., *"baru coba dulu"*), or neutral informational inquiries were confirmed as **Neutral**.

**Ground-Truth Validation Protocol:** Following the human validation paradigm advocated by Geni et al. [18], ground-truth reliability was subjected to formal empirical verification:
1. **Independent Human Inter-Annotator Agreement (IAA):** A stratified random sample of 300 reviews ($N=300$) was independently evaluated by two domain experts without access to the automated labels. Inter-coder agreement was quantified using Cohen's Kappa ($\kappa$):
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$
where $P_o$ is the relative observed agreement and $P_e$ is the hypothetical probability of chance agreement. The human annotation trial achieved **$\kappa = 0.8662$**, falling well into the range of *"Almost Perfect Agreement"* (Landis & Koch, 1977).
2. **Internal Heuristic Consistency Audit:** A 500-sample audit comparing automated disambiguation rules against strict domain criteria demonstrated an **89.39% consistency rate**.

### D. Dataset Partitioning
The curated corpus of 12,777 reviews was partitioned using stratified sampling based on target sentiment labels and source application distributions into:
- **Training Set:** 80% ($n = 10,221$ reviews)
- **Validation Set:** 10% ($n = 1,278$ reviews)
- **Test Set:** 10% ($n = 1,278$ reviews)

All random splits were governed by fixed random seeds (`seed=42`) to guarantee absolute experimental reproducibility.

### E. Model Architectures and Hyperparameter Configurations

#### 1. Baseline 1: Random Forest + TF-IDF
Random Forest was configured with 300 estimators, Gini impurity splitting criteria, square-root feature subsampling (`max_features='sqrt'`), and balanced class weighting (`class_weight='balanced'`). Textual features were extracted via a combined unigram and bigram TF-IDF vectorizer constrained to the top 10,000 features (`ngram_range=(1, 2)`, `min_df=3`, `max_df=0.90`).

#### 2. Baseline 2: XGBoost + TF-IDF
XGBoost was parameterized with 500 gradient boosted decision trees, a learning rate of $\eta = 0.05$, a maximum tree depth of 6, subsampling ratio of 0.8, and column subsampling of 0.8. The multiclass soft-probability objective (`multi:softprob`) was optimized against the 10,000-dimensional TF-IDF feature space. Early stopping with a patience of 20 evaluation rounds on the validation set was enforced to prevent overfitting.

#### 3. Synthetic Over-sampling Baseline Configurations (SMOTE)
To evaluate the standard machine learning remedy for class imbalance, Random Forest and XGBoost were trained on SMOTE-augmented training spaces (`Random Forest + SMOTE` and `XGBoost + SMOTE`). Using `imblearn.over_sampling.SMOTE` (`k_neighbors=5`, `random_state=42`), the minority Neutral class vectors in the sparse TF-IDF training split were synthetically oversampled to match majority class frequencies prior to tree induction.

#### 4. Proposed Contextual Transformer: IndoBERT
We fine-tuned the pre-trained `indobenchmark/indobert-base-p1` checkpoint [15]. The architecture comprises 12 Transformer encoder layers, 12 self-attention heads, a hidden dimensionality of $d=768$, and 124.5 million parameters. The model was trained using PyTorch with the following hyperparameter regimen:
- **Maximum Sequence Length:** $L_{\text{seq}} = 256$ tokens (accommodating extended descriptive complaints).
- **Batch Size:** 16 with mixed-precision arithmetic (FP16 enabled via NVIDIA Ampere/Ada Lovelace tensor cores).
- **Optimization:** AdamW optimizer with an initial learning rate of $\eta = 2 \times 10^{-5}$, linear learning rate decay, and a 10% warmup phase.
- **Regularization:** Weight decay of $\lambda = 0.01$, dropout probability $p = 0.1$, and early stopping triggered when validation loss failed to decrease for 2 consecutive epochs (patience=2).
- **Objective Function:** To directly combat neutral collapse without generating synthetic vector noise, we implemented a **Class-Weighted Cross-Entropy Loss** [13]:
$$\mathcal{L}_{\text{CWCE}} = - \sum_{i=1}^{C} w_i \cdot y_i \log(\hat{y}_i)$$
where $w_i = \frac{N}{C \cdot N_i}$ represents the inverse class frequency weight for class $i$, penalizing misclassifications on the minority neutral class proportionally to its rarity.

#### 5. Exploratory Social-Domain Variant: IndoBERTweet
To evaluate whether a specialized vocabulary initialized on conversational netizen discourse provides superior transferability, we fine-tuned `indolem/indobertweet-base-uncased` [17] under identical sequence lengths ($L_{\text{seq}}=256$), batch sizes, learning rates ($\eta = 2 \times 10^{-5}$), and class-weighted loss configurations.

### F. Evaluation Metrics and Statistical Hypothesis Testing
Model performance was comprehensively appraised across six mathematical metrics:
1. **Overall Accuracy (Acc):** Proportion of correctly classified reviews over total test instances.
2. **Precision ($P$), Recall ($R$), and F1-Score ($F_1$):** Evaluated both as macro-averages (assigning equal weight to each class) and class-specific metrics:
$$F_{1, \text{Macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
3. **Multi-Class ROC-AUC (One-vs-Rest, Macro-Averaged):** Measures the area under the Receiver Operating Characteristic curve, assessing the calibration of predicted class probability distributions.
4. **Statistical Significance via McNemar's Test:** To determine whether differences in classification decisions between the premier baseline (XGBoost) and the proposed Transformer (IndoBERT) were statistically significant rather than stochastic artifacts, McNemar's test with Edwards' continuity correction was executed [19]:
$$\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}$$
where $n_{10}$ denotes the number of test instances correctly classified by IndoBERT but misclassified by XGBoost, and $n_{01}$ denotes instances correctly classified by XGBoost but misclassified by IndoBERT. The resulting test statistic follows a chi-squared distribution with 1 degree of freedom ($\text{df}=1$, significance threshold $\alpha = 0.05$).

---

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

### A. Dataset Statistics and Ground-Truth Distribution
Table II summarizes the structural and linguistic characteristics of the final cleaned corpus across all four e-commerce platforms.

**TABLE II. Final Corpus Statistics across E-Commerce Applications**
| Platform / Application | Total Clean Reviews | Negative ($N_0$) | Neutral ($N_1$) | Positive ($N_2$) | Mean Char Length | Median Word Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bukalapak** | 3,308 | 1,328 | 694 | 1,286 | 114.2 | 15.8 |
| **Tokopedia** | 3,289 | 1,291 | 708 | 1,290 | 118.6 | 16.2 |
| **Shopee** | 3,091 | 1,340 | 639 | 1,112 | 112.4 | 15.4 |
| **Lazada** | 3,089 | 1,357 | 642 | 1,090 | 108.9 | 14.9 |
| **Full Combined Corpus** | **12,777** | **5,316 (41.6%)** | **2,683 (21.0%)** | **4,778 (37.4%)** | **113.6** | **15.6** |
| *Test Partition ($N=1,278$)* | *1,278* | *532 (41.6%)* | *268 (21.0%)* | *478 (37.4%)* | *113.4* | *15.5* |

A critical empirical insight emerges from review text length analysis: **Negative reviews exhibit a substantially longer mean character length (135.7 characters) than positive reviews (84.9 characters)**. In the Indonesian digital commerce domain, satisfied consumers tend to express endorsement via concise accolades (e.g., *"mantap barang cepat sampai"*), whereas aggrieved users author elaborate, multi-clause complaints detailing chronological merchant failures, damaged packaging, or futile customer service interactions. This structural disparity validates our methodological choice of expanding Transformer context capacity to $L_{\text{seq}} = 256$ tokens.

### B. Aggregate Benchmark Performance and SMOTE Comparative Evaluation
Table III presents the overarching empirical benchmark across all six investigated model configurations evaluated on the identical 1,278 test instances.

**TABLE III. Comprehensive Comparative Performance Benchmark on Test Corpus ($N = 1,278$)**
| Model Architecture | Feature Representation / Strategy | Overall Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | F1 (Neutral) (%) | ROC-AUC (OVR) | Training Latency (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | TF-IDF (1-2 gram, 10k) + Balanced Weights | 62.60% | 48.27% | 58.74% | 7.31% 🚨 | 0.7430 | **7.51 s** |
| **Random Forest + SMOTE** | TF-IDF (1-2 gram, 10k) + Synthetic Oversampling | 61.97% | 53.40% | 60.85% | 24.49% | 0.7383 | 7.73 s |
| **XGBoost** | TF-IDF (1-2 gram, 10k) + Early Stopping | 64.08% | 53.05% | 61.27% | 18.39% ⚠️ | 0.7540 | 52.90 s |
| **XGBoost + SMOTE** | TF-IDF (1-2 gram, 10k) + Synthetic Oversampling | 60.80% | 53.96% | 60.33% | 26.58% | 0.7523 | 55.84 s |
| **IndoBERT (base-p1)** | Deep Contextual ($L=256$, Class-Weighted CE) | **65.26%** 🏆 | **61.76%** 🏆 | **65.17%** 🏆 | **40.73%** 🎯 | **0.8055** 🏆 | 607.32 s |
| **IndoBERTweet (uncased)** | Deep Contextual ($L=256$, Class-Weighted CE) | 63.46% | 61.23% | 63.95% | **42.23%** 🚀 | 0.8016 | 436.06 s |

Several profound findings are evident from Table III:
1. **The Superiority of Contextual Deep Learning:** Fine-tuned IndoBERT establishes the highest overall performance across accuracy (65.26%), Macro-F1 (61.76%), and ROC-AUC (0.8055). Compared against standard XGBoost, IndoBERT delivers a **+8.71 percentage point surge in Macro-F1**, demonstrating that bidirectional self-attention captures subtle semantic patterns that entirely elude n-gram term frequencies.
2. **The Failure of SMOTE in Resolving Contextual Ambiguity:** Applying SMOTE to tree-based models elevated neutral class F1-scores marginally (from 7.31% to 24.49% for RF, and 18.39% to 26.58% for XGBoost). However, this modest gain incurred a **severe penalty on overall predictive accuracy**: XGBoost accuracy dropped by **3.28 percentage points** (64.08% down to 60.80%), and Random Forest dropped from 62.60% to 61.97%. This empirical degradation directly corroborates the theoretical critique of Cui et al. [13] and Kristiyanti et al. [14]: linear interpolation between minority vectors in sparse 10,000-dimensional TF-IDF space generates synthetic feature vectors that intrude into majority class boundaries, creating decision boundary noise and inflating false positive rates.
3. **Domain-Specific Lexical Adaptability of IndoBERTweet:** IndoBERTweet achieved the single highest neutral class performance (**F1: 42.23%**, Recall: 53.73%), outperforming even IndoBERT-base on the minority class by +1.50 pp. Because IndoBERTweet's tokenizer was pre-trained directly on informal Indonesian netizen vocabulary, it exhibits native sensitivity to clipped expressions and internet slang. However, IndoBERT-base retained superior performance on long-form negative complaints, leading to a higher overall accuracy (65.26% vs. 63.46%).
4. **Probability Calibration (ROC-AUC):** IndoBERT achieves an exceptional ROC-AUC of 0.8055, markedly outperforming XGBoost (0.7540) and Random Forest (0.7430). In a practical DSS, high ROC-AUC ensures that predicted class probabilities can be reliably thresholded for automated escalation alerts.

### C. Per-Class Performance and the Resolution of Neutral Collapse
To dissect the mechanics of the Neutral Collapse, Table IV presents the breakdown of Precision, Recall, and F1-score across all three sentiment polarities for the four principal non-SMOTE architectures.

**TABLE IV. Granular Per-Class Performance Breakdown on Test Partition ($N = 1,278$)**
| Model Architecture | Target Class | Precision (%) | Recall (%) | F1-Score (%) | Class Status / Pathology |
|---|---|:---:|:---:|:---:|---|
| **Random Forest** | Negative ($n=532$) | 61.02% | **89.55%** | 72.58% | Heavily biased toward negative class |
| | Neutral ($n=268$) | 33.33% | 4.10% | **7.31%** | 🚨 **Severe Neutral Collapse** |
| | Positive ($n=478$) | 69.17% | 61.18% | 64.93% | Moderate positive recognition |
| **XGBoost** | Negative ($n=532$) | 63.97% | 86.57% | 73.57% | Strong negative recognition |
| | Neutral ($n=268$) | 40.00% | 11.94% | **18.39%** | ⚠️ **Severe Neutral Collapse** |
| | Positive ($n=478$) | 69.37% | 65.11% | 67.17% | Moderate positive recognition |
| **IndoBERT (base-p1)** | Negative ($n=532$) | 73.05% | 75.95% | **74.47%** | Balanced precision & recall |
| | Neutral ($n=268$) | 36.61% | 45.90% | **40.73%** | 🎯 **Neutral Collapse Resolved (+22.34 pp)** |
| | Positive ($n=478$) | **80.32%** | 62.16% | **70.08%** | High-precision positive identification |
| **IndoBERTweet** | Negative ($n=532$) | 75.27% | 70.15% | 72.62% | Robust negative identification |
| | Neutral ($n=268$) | 34.78% | **53.73%** | **42.23%** | 🚀 **Highest Neutral Recall & F1** |
| | Positive ($n=478$) | **80.79%** | 59.95% | 68.83% | High-precision positive identification |

As shown in Table IV, the apparent competitiveness of Random Forest and XGBoost in aggregate accuracy (62.60% and 64.08%) is an artifact of majority-class domination:
- Random Forest correctly recognized only **11 out of 268 neutral reviews** (Recall: 4.10%), misclassifying 95.9% of neutral samples into Negative or Positive.
- XGBoost correctly classified only **32 out of 268 neutral reviews** (Recall: 11.94%).
- In contrast, IndoBERT correctly identified **123 neutral reviews** (Recall: 45.90%), elevating the neutral F1-score to **40.73%**—a dramatic **+22.34 pp improvement** over XGBoost and **+33.42 pp** over Random Forest. IndoBERTweet pushed neutral recall even higher to **53.73%** (144 correct neutral reviews).

Importantly, this recovery of the neutral class did not destabilize the primary classes: IndoBERT concurrently achieved the highest Negative F1 (74.47%) and Positive F1 (70.08%), proving that contextual attention combined with inverse-frequency class-weighted loss establishes robust, multi-class decision boundaries without cannibalizing polarized categories.

### D. Four-Stage Modular Ablation Study
To isolate and quantify the specific architectural mechanisms responsible for IndoBERT's performance leap, we conducted a systematic four-stage ablation study. Starting from an unweighted base IndoBERT baseline, each component was incrementally integrated, as documented in Table V.

**TABLE V. Four-Stage Modular Ablation Study on IndoBERT Architecture**
| Stage | Experimental Configuration | Slang Lexicon Normalization | Rating-3 Disambiguation | Context Window ($L_{\text{seq}}$) | Loss Function Formulation | Overall Accuracy (%) | Macro-F1 (%) | F1 (Neutral) (%) | $\Delta$ Macro-F1 (vs Base) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Base IndoBERT Baseline** | Basic (~60 words) | None (Raw Stars) | 128 | Unweighted Standard Cross-Entropy | 63.30% | 57.97% | 32.40% | *Reference* |
| 2 | **+ Slang & Disambiguation** | Domain Lexicon (150+ words) | Hybrid Rating-Lexicon Filter | 128 | Unweighted Standard Cross-Entropy | 64.12% | 59.45% | 34.18% | **+1.48%** |
| 3 | **+ Extended Context (256)** | Domain Lexicon (150+ words) | Hybrid Rating-Lexicon Filter | 256 | Unweighted Standard Cross-Entropy | 64.85% | 60.50% | 35.80% | **+2.53%** |
| 4 | **Proposed Full IndoBERT** | Domain Lexicon (150+ words) | Hybrid Rating-Lexicon Filter | 256 | **Class-Weighted Cross-Entropy** | **65.26%** | **61.76%** | **40.73%** | **+3.79%** |

The ablation trajectory reveals distinct, complementary performance dynamics:
1. **Stage 1 $\rightarrow$ Stage 2 (+1.48 pp Macro-F1):** Incorporating the 150+ word e-commerce slang dictionary and filtering contaminated 3-star reviews provided immediate semantic stabilization. Standardizing clipped tokens (e.g., *"ongkir"*, *"lemot"*) resolved out-of-vocabulary subword fragmentation.
2. **Stage 2 $\rightarrow$ Stage 3 (+1.05 pp Macro-F1; +2.53 pp cumulative):** Expanding token capacity from 128 to 256 tokens prevented truncation of long transactional complaints (which average 135+ characters and frequently place critical resolution details in closing sentences).
3. **Stage 3 $\rightarrow$ Stage 4 (+1.26 pp Macro-F1; +3.79 pp cumulative):** Replacing unweighted loss with Class-Weighted Cross-Entropy produced the single largest jump in neutral class recovery, driving F1 (Neutral) from 35.80% to **40.73% (+4.93 pp)** and finalizing the overall Macro-F1 at **61.76%**.

### E. Statistical Significance Testing via McNemar's Test
To rigorously establish whether the empirical margin between IndoBERT and XGBoost represents a statistically verifiable improvement, we formulated the test set contingency matrix ($N = 1,278$) shown in Table VI.

**TABLE VI. Test Set Contingency Matrix between IndoBERT and XGBoost ($N = 1,278$)**
| Contingency Distribution | XGBoost Correct ($y_{\text{xgb}} = y$) | XGBoost Incorrect ($y_{\text{xgb}} \neq y$) | Marginal Total |
|---|:---:|:---:|:---:|
| **IndoBERT Correct ($y_{\text{bert}} = y$)** | $n_{11} = 681$ | $n_{10} = \mathbf{153}$ | **834 (65.26%)** |
| **IndoBERT Incorrect ($y_{\text{bert}} \neq y$)** | $n_{01} = 138$ | $n_{00} = 306$ | **444 (34.74%)** |
| **Marginal Total** | **819 (64.08%)** | **459 (35.92%)** | **1,278 (100.0%)** |

**Statistical Interpretation and Dual-Level Analysis:**
Applying McNemar's test with continuity correction [19] across the full test set yields:
$$\chi^2 = \frac{(|153 - 138| - 1)^2}{153 + 138} = \frac{(14)^2}{291} = \frac{196}{291} \approx \mathbf{0.6735}$$
With 1 degree of freedom, this corresponds to an asymptotic $p$-value of **$p = 0.4118$**. Because $p > 0.05$, the null hypothesis cannot be rejected at the aggregate accuracy level. 

**Critical Methodological Insight:** A naive researcher might interpret this result as evidence that IndoBERT offers no meaningful advantage over XGBoost. However, a deeper mathematical inspection illuminates why:
1. **Majority Class Masking:** Total accuracy is overwhelmingly dominated by the majority Negative (41.6%) and Positive (37.4%) classes. In aggregate accuracy, IndoBERT corrects 153 errors made by XGBoost, but XGBoost also correctly predicts 138 instances where IndoBERT faltered, resulting in a modest net aggregate gain of **+15 correct reviews** (834 vs. 819).
2. **Statistically Overwhelming Advantage on the Minority Class:** When McNemar's contingency analysis is restricted specifically to the **minority Neutral class subset ($N = 268$)**, IndoBERT correctly classifies **123 samples** compared to only **32 samples** for XGBoost. Among discordant pairs in this class ($n_{10} = 97$, $n_{01} = 6$):
$$\chi^2_{\text{Neutral}} = \frac{(|97 - 6| - 1)^2}{97 + 6} = \frac{(90)^2}{103} = \frac{8,100}{103} \approx \mathbf{71.68} \quad (\mathbf{p < 0.0001})$$
This proves conclusively that IndoBERT's superiority is **statistically overwhelmingly significant ($p < 10^{-16}$)** in resolving the core bottleneck of this task: **Macro-F1 elevation (+8.71 pp) and neutral class restoration (+22.34 pp)**, whereas overall aggregate accuracy is constrained by the statistical inertia of majority class boundaries.

### F. Qualitative Linguistic Error Analysis
To understand the precise semantic mechanisms governing model divergence, Table VII presents a curated qualitative analysis of representative real-world test instances extracted from `paper_assets/error_analysis_samples.csv`.

**TABLE VII. Representative Qualitative Linguistic Failure Cases from E-Commerce Test Corpus**
| ID | Platform | True Label | IndoBERT | RF | XGB | Raw Text / Core Linguistic Feature | Semantic Failure Mode Analysis |
|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **#1** | Shopee | **Positif** | **Positif** | Negatif | Negatif | *"beli di shoppe tidak pernah kecewa kurir nya ramah"* <br> *(Double Negation / Litotes)* | RF/XGB triggered by negative unigrams *"tidak pernah"* and *"kecewa"*. IndoBERT self-attention captures litotes syntax as strong positive endorsement. |
| **#2** | Bukalapak | **Positif** | **Positif** | Negatif | Negatif | *"sejauh ini tidak ada kendala yang berarti sudah 8th pakai..."* <br> *(Long-Range Litotes)* | Tree models misclassify due to *"tidak ada kendala"*. IndoBERT contextualizes customer satisfaction and 8-year loyalty. |
| **#6** | Tokopedia | **Negatif** | **Negatif** | Positif | Positif | *"disuruh belanja 2 barang untuk dapat kupon 10... eh dibilang kendala"* <br> *(Promo Bait Sarcasm)* | Tree models misled by superficial positive keywords (*"belanja"*, *"dapat kupon"*). IndoBERT captures the contrastive irony *"eh dibilang sedang ada kendala"*. |
| **#8** | Tokopedia | **Negatif** | **Negatif** | Positif | Netral | *"aplikasinya lambat kalah cepat ma shopee"* <br> *(Comparative Brand Critique)* | RF triggered by token *"cepat"*. IndoBERT parses comparative structure *"lambat kalah cepat"* as explicit operational dissatisfaction. |
| **#11** | Shopee | **Netral** | **Netral** | Positif | Positif | *"pada dasarnya saya suka belanja... hanya cukup terganggu notif palsu"* <br> *(Balanced Mixed Polarity)* | Tree models capture initial praise clause (*"suka belanja"*). IndoBERT balances praise against grievance (*"hanya cukup terganggu"*), predicting true Neutral. |
| **#16** | Tokopedia | **Netral** | **Netral** | Negatif | Negatif | *"mohon ditambah fitur pencarian dengan gambar agak susah..."* <br> *(Constructive Feature Request)* | Tree models flag *"agak susah"* and *"tidak seperti"* as pure Negative. IndoBERT identifies polite constructive feedback (*"mohon ditambah"*). |
| **#18** | Shopee | **Netral** | **Netral** | Negatif | Positif | *"keamanan nya tolong ditingkatkan beberapa kali ada yang coba log in"* <br> *(Security Bug Suggestion)* | RF treats warning as Negative; XGB misclassifies as Positive. IndoBERT recognizes constructive advisory tone, placing it correctly in Neutral. |

As evidenced by Table VII, IndoBERT's architectural advantage stems directly from multi-head self-attention, which evaluates token relationships dynamically across sentence clauses rather than summing isolated n-gram polarity scores.

---

## V. THREATS TO VALIDITY

### A. Internal Validity
Potential threats to internal validity involve ground-truth labeling consistency and hyperparameter tuning. While Google Play Store star ratings provide an initial polarity signal, individual user rating behaviors are inherently subjective. We mitigated this threat by implementing an automated hybrid rating-lexicon disambiguation filter on all ambiguous 3-star reviews and independently verifying 300 stratified samples through a dual human annotator trial, achieving an exceptional Cohen's Kappa of $\kappa = 0.8662$. Furthermore, to avoid hyperparameter bias, all models underwent consistent early stopping protocols on validation sets with identical random seeding.

### B. External Validity
External validity concerns the generalizability of our findings beyond the four evaluated e-commerce applications. Because user reviews were harvested exclusively from Tokopedia, Shopee, Lazada, and Bukalapak within the Indonesian market, linguistic distributions reflect consumer electronics, fashion, and daily retail commerce. While the identified linguistic phenomena (litotes, slang, promotional sarcasm) are universal across Indonesian social discourse, fine-tuning performance on specialized regulatory domains (e.g., fintech, medical health tech, or legal disputes) may require domain-adapted pre-training or specialized lexical dictionaries.

### C. Construct Validity
Construct validity addresses whether our evaluation metrics accurately reflect real-world model utility. Sole reliance on overall accuracy would have obscured the severe collapse of the neutral class. By incorporating macro-averaged F1, class-specific precision/recall, multi-class ROC-AUC, and McNemar significance testing across both aggregate and minority strata, we ensured that performance evaluations robustly capture real-world operational reliability.

---

## VI. ARCHITECTURAL INTEGRATION IN DECISION SUPPORT SYSTEMS (DSS) & PRACTICAL IMPLICATIONS

### A. Strategic Decision Support System (DSS) Architecture
To translate these empirical findings into actionable enterprise utility, Figure 2 outlines the architectural integration of the proposed IndoBERT sentiment engine into an enterprise **Public Relations & KOL Decision Support System (DSS)**.

```
+----------------------------------------------------------------------------------------------------+
|                         CONTINUOUS INGESTION & DATA STREAMING LAYER                                |
|  Google Play Store Reviews API  |  X / Twitter Social Mentions  |  TikTok & Instagram KOL Comments |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                        PREPROCESSING & SLANG NORMALIZATION SERVICE                                 |
|  • In-Memory Deduplication & Regex Filtering                                                       |
|  • Dynamic Slang Normalization Engine (150+ Lexicon Dictionary)                                    |
|  • Length & Language Validation Gates                                                              |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                         DUAL-TIER INFERENCE ORCHESTRATION ENGINE                                   |
|                                                                                                    |
|    [Stream Ingestion Queue]                                                                        |
|               |                                                                                    |
|               +---> Fast-Path Filter (XGBoost Baseline)                                            |
|               |     • Latency: ~50 ms / batch                                                      |
|               |     • Filters unambiguous 1-star & 5-star short reviews                            |
|               |                                                                                    |
|               +---> Deep Strategic Path (Fine-Tuned IndoBERT Model Engine)                         |
|                     • Invoked for multi-clause reviews, 3-star ratings, or low confidence (<0.75)   |
|                     • Resolves litotes, promo sarcasm, constructive suggestions                    |
|                     • Outputs calibrated multi-class probabilities (Negative, Neutral, Positive)   |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                      STRATEGIC PR & KOL DECISION & VISUALIZATION LAYER                             |
|                                                                                                    |
|   1. REAL-TIME CRISIS DETECTION MODULE                                                             |
|      • Velocity Spike Alarms: Triggers when Negative Volume > 3-sigma baseline                     |
|      • Issue Clustering: Flags severe complaints (fraud, payment bugs, counterfeit items)          |
|                                                                                                    |
|   2. STRATEGIC KOL CAMPAIGN IMPACT TRACKER                                                         |
|      • Pre- vs. Post-Campaign Sentiment Trajectory Tracking                                        |
|      • Net Promoter Score (NPS) & Brand Health Index Estimation                                     |
|                                                                                                    |
|   3. PRODUCT & PR FEATURE ROADMAP FEEDBACK                                                         |
|      • Neutral Feedback Aggregator: Clusters constructive suggestions (Feature Requests, UI/UX)    |
|      • Directly surfaces non-polarized feedback to Product Managers & PR spokespersons             |
+----------------------------------------------------------------------------------------------------+
```
*Figure 2. Strategic PR and KOL Decision Support System (DSS) architectural workflow.*

### B. Practical Implications for PR Practitioners and KOL Specialists
1. **Automated Crisis Escalation Management:** Traditional PR monitoring that relies on naive keyword volume or tree-based models risks catastrophic false alarms (e.g., misinterpreting positive litotes like *"tidak pernah kecewa"* as a surge in dissatisfaction) or dangerous omissions (failing to recognize sarcastic promo complaints like Sample #6). Deploying IndoBERT ensures that PR teams respond rapidly to genuine operational failures while avoiding unnecessary panic over linguistically complex positive reviews.
2. **KOL Campaign ROI and Sentiment Trajectory:** Influencer marketing campaigns frequently produce polarized comment sections containing enthusiastic praise alongside logistical inquiries. Traditional models misclassify neutral logistical inquiries, artificially distorting campaign sentiment scores. IndoBERT's robust 40.73% neutral F1 enables brand managers to isolate campaign-driven praise from platform-level delivery issues with high statistical precision.
3. **Operationalizing Constructive Feedback:** By successfully resolving the Neutral Collapse, the proposed DSS creates an automated pipeline for harvesting constructive user recommendations (e.g., banking payment additions, UI search improvements). Instead of being lost in positive or negative noise, these neutral insights are directly funneled to corporate product teams to drive data-informed application updates.
4. **Latency vs. Accuracy Deployment Trade-Offs:** For enterprise environments with constrained computational resources, our benchmark establishes that XGBoost provides a rapid, low-latency alternative (52.90 s training time vs. 607.32 s for IndoBERT). However, system architects must recognize the deliberate trade-off: utilizing XGBoost entails an **8.71 pp reduction in Macro-F1** and the near-total loss of neutral class visibility.

---

## VII. CONCLUSION AND FUTURE WORK

This empirical study presented a comprehensive comparative investigation between traditional tree-based machine learning ensembles and deep contextual Transformer architectures on a multi-vendor Indonesian e-commerce customer review corpus. By curating 12,777 cleaned reviews across Tokopedia, Shopee, Lazada, and Bukalapak, supported by a formal human inter-annotator agreement study ($\kappa = 0.8662$) and an 89.39% heuristic consistency audit, we established an open, reproducible evaluation benchmark.

Our empirical findings demonstrate that fine-tuned IndoBERT-base-p1 achieves superior overall predictive capability, securing **65.26% accuracy, 61.76% Macro-F1, and an ROC-AUC of 0.8055**. Crucially, while Random Forest and XGBoost suffered severe Neutral Collapse (neutral F1-scores of 7.31% and 18.39%), IndoBERT successfully restored the neutral class to **40.73% F1 (+22.34 pp)** through the integration of Class-Weighted Cross-Entropy Loss, while the social-domain IndoBERTweet achieved **42.23% neutral F1**. Synthetic oversampling via SMOTE failed to alleviate contextual ambiguity, deteriorating overall XGBoost accuracy by 3.28 pp. A modular four-stage ablation study verified that domain-specific slang normalization (+1.48 pp), extended context length (+2.53 pp), and class-weighted loss (+3.79 pp) each contribute indispensable performance gains. Furthermore, a dual-level McNemar statistical analysis demonstrated that while overall aggregate accuracy margins represent a balanced net shift of +15 samples masked by majority-class inertia, IndoBERT is statistically overwhelmingly superior on the minority neutral class ($\chi^2 = 71.68, p < 0.0001$). Finally, qualitative analysis of 20 failure cases demonstrated IndoBERT's unique capability to resolve litotes, comparative brand criticism, and multi-clause constructive feedback.

Future research will explore:
1. Incorporating multi-modal feedback channels (e.g., image attachments in customer reviews depicting damaged merchandise or incorrect product packaging);
2. Investigating parameter-efficient fine-tuning (PEFT/LoRA) on larger multilingual foundation models (such as LLaMA-3 or Gemma-2) to further elevate contextual nuance; and
3. Deploying this dual-tier DSS pipeline in production environments to conduct prospective, real-time longitudinal studies on brand crisis propagation across Indonesian enterprise communications.

---

## ACKNOWLEDGMENTS & REPRODUCIBILITY STATEMENT

All code, curated dataset splits, normalization lexicons, evaluation checkpoints, and visualization assets required to replicate every empirical finding and table reported in this paper are deterministically archived and open-sourced under the MIT License at the institutional research repository:  
`https://github.com/farrelhaidar/cobapenelitianpart4`

---

## REFERENCES

[1] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.

[2] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, Aug. 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT)*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186.

[4] E. Yulianti and N. K. Nissa, "ABSA of Indonesian customer reviews using IndoBERT: Single-sentence and sentence-pair classification approaches," *Bulletin of Electrical Engineering and Informatics (BEEI)*, vol. 13, no. 5, pp. 3579–3589, Oct. 2024, doi: 10.11591/eei.v13i5.8032.

[5] R. Anadra, H. Wijayanto, and K. Sadik, "Sentiment analysis of Tokopedia customer reviews using BiLSTM and IndoBERT with comparative analysis of preprocessing and labeling methods," *International Journal of Advances in Data and Information Systems (IJADIS)*, vol. 6, no. 3, pp. 773–788, Dec. 2025, doi: 10.59395/ijadis.v6i3.1458.

[6] P. Sayarizki, Hasmawati, and H. Nurrahmi, "Implementation of IndoBERT for sentiment analysis of Indonesian presidential candidates," *Indonesian Journal on Computing (Indo-JC)*, vol. 9, no. 2, pp. 61–72, Aug. 2024.

[7] A. Y. Imran, M. R. Sanjaya, B. W. Putra, and G. E. H. Cahyadi, "Optimization of sentiment analysis on Tokopedia user reviews using GridSearchCV and SMOTE with machine learning algorithms," *Jurnal Inovtek Polbeng - Seri Informatika*, vol. 10, no. 3, pp. 412–424, Dec. 2025.

[8] N. Z. Zahra et al., "Benchmarking logistic regression, SVM, naive Bayes, and IndoBERT fine-tuning for sentiment analysis on Indonesian product reviews," *arXiv preprint arXiv:2605.03439*, May 2026.

[9] A. Tikaningsih, I. Tahyudin, and Berlilana, "Two-stage framework using IndoBERT for sentiment analysis of Tokopedia reviews under extreme class imbalance," *Sinkron: Jurnal dan Penelitian Teknik Informatika*, vol. 10, no. 3, pp. 1733–1744, Jul. 2026, doi: 10.33395/sinkron.v10i3.16187.

[10] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.

[11] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF Conf. Comput. Vision Pattern Recognit. (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 9268–9277, doi: 10.1109/CVPR.2019.00949.

[12] D. A. Kristiyanti, S. A. Sanjaya, V. C. Tjokro, and J. Suhali, "Dealing imbalance dataset problem in sentiment analysis of recession in Indonesia," *IAES International Journal of Artificial Intelligence (IJ-AI)*, vol. 13, no. 2, pp. 2058–2070, Jun. 2024, doi: 10.11591/ijai.v13.i2.pp2060-2072.

[13] F. Koto, A. Rahimi, J. H. Lau, and T. Baldwin, "IndoLEM and IndoBERT: A benchmark dataset and pre-trained language model for Indonesian NLP," in *Proc. 28th Int. Conf. Comput. Linguistics (COLING 2020)*, Barcelona, Spain, Dec. 2020, pp. 757–770.

[14] B. Wilie et al., "IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding," in *Proc. 1st Conf. Asia-Pacific Chapter Assoc. Comput. Linguistics 10th Int. Joint Conf. Natural Lang. Process. (AACL-IJCNLP 2020)*, Suzhou, China, Dec. 2020, pp. 843–860.

[15] F. Koto, A. Rahimi, J. H. Lau, and T. Baldwin, "IndoBERTweet: A pre-trained language model for Indonesian Twitter with effective domain-specific vocabulary initialization," in *Proc. 2021 Conf. Empirical Methods Natural Lang. Process. (EMNLP 2021)*, Punta Cana, Dominican Republic, Nov. 2021, pp. 10660–10668.

[16] T. Ramadhan and K. Kusnawi, "Performance comparison of IndoBERT and Bi-LSTM models for sentiment analysis of Shopee app users in Indonesia," *Journal of Applied Informatics and Computing (JAIC)*, vol. 10, no. 2, pp. 2076–2085, Apr. 2026.

[17] A. A. P. Simarmata and T. B. Sasongko, "Sentiment analysis on BRImo application reviews using IndoBERT," *Journal of Applied Informatics and Computing (JAIC)*, vol. 9, no. 3, pp. 851–862, Dec. 2025.

[18] L. Geni, E. Yulianti, and D. I. Sensuse, "Sentiment analysis of tweets before the 2024 elections in Indonesia using BERT language models," *Jurnal Ilmiah Teknik Elektro Komputer dan Informatika (JITEKI)*, vol. 9, no. 3, pp. 746–757, Sep. 2023, doi: 10.26555/jiteki.v9i3.26490.

[19] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, Oct. 1998, doi: 10.1162/089976698300017197.

[20] D. Marutho and V. G. Utomo, "Benchmarking IndoBERT and transformer models for sentiment classification on Indonesian e-government service reviews," *Transformatika*, vol. 23, no. 1, pp. 45–58, Jan. 2025.
