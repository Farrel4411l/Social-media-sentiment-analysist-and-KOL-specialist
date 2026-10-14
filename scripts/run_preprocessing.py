# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-14

"""
Pipeline Preprocessing, Normalisasi Slang, Anotasi Sentimen, dan Partisi Stratified Data (Fase F2).
Target Output:
- data/processed/dataset_clean.csv
- data/splits/{train,val,test}.csv (80/10/10)
- data/manual_validation_sample.csv (10% sample)
- paper_assets/table1_dataset_statistics.csv
- paper_assets/figure1_data_distribution.png
"""

import os
import re
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from langdetect import detect, DetectorFactory

# Set seed untuk reproduksibilitas
DetectorFactory.seed = 42
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreprocessPipeline")

# Kamus Slang Bahasa Indonesia (Komprehensif untuk Ulasan E-Commerce)
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak", "tdk": "tidak",
    "g": "tidak", "nda": "tidak", "ndak": "tidak", "kagak": "tidak", "kaga": "tidak",
    "bgt": "banget", "bgtt": "banget", "bgtz": "banget", "bnget": "banget",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
    "bs": "bisa", "bisaa": "bisa", "bsa": "bisa",
    "bener": "benar", "bnr": "benar", "beneran": "benar",
    "krn": "karena", "karna": "karena", "coz": "karena", "krena": "karena",
    "tp": "tapi", "tpi": "tapi", "tapii": "tapi",
    "sy": "saya", "gua": "saya", "gw": "saya", "aku": "saya", "gue": "saya", "gwe": "saya",
    "km": "kamu", "lu": "kamu", "loe": "kamu", "ente": "kamu",
    "bgs": "bagus", "bgus": "bagus", "bagusss": "bagus", "mantab": "mantap", "mantul": "mantap",
    "top": "mantap", "keren": "bagus", "kece": "bagus", "ok": "oke", "okeey": "oke",
    "kecewaa": "kecewa", "kcwa": "kecewa", "jelek": "buruk", "jlk": "buruk", "parah": "buruk", "hancur": "buruk",
    "batal": "batal", "cancel": "batal",
    "slow": "lambat", "lemot": "lambat", "lola": "lambat", "lelet": "lambat",
    "makasih": "terima kasih", "tq": "terima kasih", "thx": "terima kasih", "thanks": "terima kasih", "tengkyu": "terima kasih",
    "ongkir": "ongkos kirim", "freeongkir": "gratis ongkos kirim", "gratisongkir": "gratis ongkos kirim",
    "seller": "penjual", "buyer": "pembeli", "cs": "layanan pelanggan",
    "respon": "respons", "bales": "balas", "bls": "balas", "dibalas": "direspons",
    "dgn": "dengan", "dg": "dengan", "utk": "untuk", "yg": "yang", "yng": "yang",
    "aja": "saja", "ajah": "saja", "aj": "saja", "doang": "saja", "tok": "saja",
    "emg": "memang", "emang": "memang", "pdhl": "padahal", "sm": "sama", "sma": "sama",
    "blm": "belum", "blom": "belum", "belom": "belum",
    "bikin": "buat", "brg": "barang", "brang": "barang", "dtg": "datang", "ampe": "sampai",
    "nyampe": "sampai", "sampe": "sampai", "smpe": "sampai", "expedisi": "ekspedisi",
    "kurir": "kurir", "rusak": "rusak", "pecah": "rusak", "ilang": "hilang", "ilang": "hilang",
    "apk": "aplikasi", "apikasi": "aplikasi", "app": "aplikasi", "apps": "aplikasi",
    "diskon": "diskon", "promo": "promosi", "voucher": "kupon", "murah": "terjangkau",
    "mahal": "mahal", "kemahalan": "terlalu mahal", "rugi": "rugi", "penipuan": "penipuan",
    "tipu": "tipu", "bohong": "bohong", "nipuu": "menipu"
}

# Stopwords & kosakata umum penanda Bahasa Indonesia
INDONESIAN_MARKERS = set([
    "yang", "dan", "di", "ini", "ke", "dari", "bisa", "bagus", "banget", "tidak",
    "kecewa", "paket", "barang", "ongkir", "aplikasi", "beli", "pesan", "toko",
    "kurir", "pengiriman", "harga", "murah", "mantap", "jelek", "rusak", "puas",
    "lama", "cepat", "admin", "update", "promo", "gratis", "suka", "tolong",
    "gagal", "bintang", "saya", "aku", "kami", "kamu", "gak", "udah", "bgt",
    "nya", "kalo", "kalau", "tapi", "tp", "sy", "ada", "itu", "lagi", "sudah",
    "buat", "mau", "sama", "kok", "padahal", "belum", "masih", "selalu"
])


def clean_text(text: str) -> str:
    """
    Normalisasi teks:
    1. Lowercase
    2. Hapus URL, Mention, Hashtag
    3. Hapus Karakter Non-ASCII / Emoji
    4. Reduksi Karakter Berulang (misal: 'baaaagus' -> 'bagus')
    5. Normalisasi Slang E-Commerce
    6. Pembersihan Spasi
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Hapus Mention (@username) dan Hashtag (#tag)
    text = re.sub(r"@\w+|#\w+", " ", text)

    # 4. Hapus Emoji dan karakter khusus non-alfanumerik standar (simpan tanda baca dasar sementara atau buang)
    text = re.sub(r"[^\w\s]", " ", text)

    # 5. Reduksi karakter berulang lebih dari 2 kali berturut-turut (misal: 'kerennnn' -> 'keren')
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # 6. Ganti Slang Bahasa Indonesia
    words = text.split()
    normalized_words = [SLANG_DICT.get(w, w) for w in words]
    text = " ".join(normalized_words)

    # 7. Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_indonesian_language(text: str) -> bool:
    """
    Validasi bahasa Indonesia menggunakan kombinasi leksikon & langdetect.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False

    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    # Jika mengandung minimal 1 token khas Indonesia
    if len(tokens.intersection(INDONESIAN_MARKERS)) >= 1:
        return True

    try:
        lang = detect(text)
        return lang in ["id", "jw", "su"]
    except Exception:
        return False


def run_preprocessing_pipeline():
    logger.info("=" * 70)
    logger.info("FASE F2: PREPROCESSING & LABELING PIPELINE")
    logger.info("=" * 70)

    # 1. Load semua CSV mentah
    raw_files = glob.glob("data/raw/*_reviews.csv")
    if not raw_files:
        raise FileNotFoundError("Tidak ditemukan file CSV di data/raw/!")

    dfs = []
    for f in raw_files:
        df_temp = pd.read_csv(f)
        dfs.append(df_temp)
        logger.info(f"Loaded {os.path.basename(f)}: {len(df_temp):,} baris")

    raw_df = pd.concat(dfs, ignore_index=True)
    initial_count = len(raw_df)
    logger.info(f"Total Ulasan Mentah Tergabung: {initial_count:,} baris")

    # 2. Filtering: Drop duplikat berdasarkan content
    df_dedup = raw_df.drop_duplicates(subset=["content"]).copy()
    dedup_count = len(df_dedup)
    logger.info(f"Setelah Deduplikasi Teks: {dedup_count:,} baris (Dihapus: {initial_count - dedup_count:,})")

    # 3. Filtering: Drop ulasan < 10 karakter
    df_len = df_dedup[df_dedup["content"].fillna("").astype(str).str.len() >= 10].copy()
    len_count = len(df_len)
    logger.info(f"Setelah Filter Panjang >= 10 Karakter: {len_count:,} baris (Dihapus: {dedup_count - len_count:,})")

    # 4. Filtering: Language Detection (Keep only Indonesian)
    logger.info("Melakukan Language Identification (Bahasa Indonesia)...")
    df_len["is_indonesian"] = df_len["content"].apply(is_indonesian_language)
    df_clean_lang = df_len[df_len["is_indonesian"]].copy()
    lang_count = len(df_clean_lang)
    logger.info(f"Setelah Language Filter: {lang_count:,} baris (Dihapus: {len_count - lang_count:,})")

    # 5. Normalisasi Teks
    logger.info("Menjalankan Normalisasi Teks (Slang replacement, URL/Emoji strip)...")
    df_clean_lang["cleaned_text"] = df_clean_lang["content"].apply(clean_text)

    # Pastikan hasil cleaned_text tidak kosong
    df_final = df_clean_lang[df_clean_lang["cleaned_text"].str.len() >= 5].copy()

    # 6. Labeling Rule Sesuai Standar Riset Sentimen:
    # 1-2 = Negatif (0), 3 = Netral (1), 4-5 = Positif (2)
    def map_sentiment(score: int):
        if score in [1, 2]:
            return 0, "negatif"
        elif score == 3:
            return 1, "netral"
        else:
            return 2, "positif"

    df_final["label"] = df_final["score"].apply(lambda s: map_sentiment(s)[0])
    df_final["sentiment"] = df_final["score"].apply(lambda s: map_sentiment(s)[1])

    final_count = len(df_final)
    retention_rate = (final_count / initial_count) * 100
    logger.info(f"Total Dataset Bersih Siap Latih: {final_count:,} baris (Retensi: {retention_rate:.2f}%)")

    # 7. Simpan Dataset Clean
    os.makedirs("data/processed", exist_ok=True)
    clean_csv_path = "data/processed/dataset_clean.csv"
    columns_to_keep = [
        "reviewId", "userName", "content", "cleaned_text", "score",
        "label", "sentiment", "thumbsUpCount", "at", "appVersion", "app_name"
    ]
    df_final[columns_to_keep].to_csv(clean_csv_path, index=False, encoding="utf-8")
    logger.info(f"Disimpan: {clean_csv_path}")

    # 8. Stratified Split 80/10/10 (Train / Val / Test)
    os.makedirs("data/splits", exist_ok=True)
    # 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(
        df_final,
        test_size=0.20,
        random_state=42,
        stratify=df_final["label"]
    )
    # Split temp_df menjadi 50/50 -> 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_path = "data/splits/train.csv"
    val_path = "data/splits/val.csv"
    test_path = "data/splits/test.csv"

    train_df[columns_to_keep].to_csv(train_path, index=False, encoding="utf-8")
    val_df[columns_to_keep].to_csv(val_path, index=False, encoding="utf-8")
    test_df[columns_to_keep].to_csv(test_path, index=False, encoding="utf-8")

    logger.info(f"Split Partisi Tersimpan:")
    logger.info(f"  - Train Set : {len(train_df):,} baris ({len(train_df)/final_count*100:.1f}%) -> {train_path}")
    logger.info(f"  - Val Set   : {len(val_df):,} baris ({len(val_df)/final_count*100:.1f}%) -> {val_path}")
    logger.info(f"  - Test Set  : {len(test_df):,} baris ({len(test_df)/final_count*100:.1f}%) -> {test_path}")

    # 9. Export 10% Manual Validation Sample
    val_sample_10pct = df_final.sample(frac=0.10, random_state=42).copy()
    val_sample_10pct["label_manual"] = ""
    sample_path = "data/manual_validation_sample.csv"
    sample_cols = ["reviewId", "app_name", "score", "label", "sentiment", "cleaned_text", "label_manual"]
    val_sample_10pct[sample_cols].to_csv(sample_path, index=False, encoding="utf-8")
    logger.info(f"Disimpan Sample Validasi Manual (10%): {len(val_sample_10pct):,} baris -> {sample_path}")

    # 10. Generate Table 1: Dataset Statistics
    os.makedirs("paper_assets", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

    # Hitung metrik panjang teks
    df_final["char_len"] = df_final["cleaned_text"].str.len()
    df_final["word_len"] = df_final["cleaned_text"].apply(lambda t: len(t.split()))

    stats_list = []
    # Baris Total Keseluruhan
    stats_list.append({
        "Category": "Full Dataset",
        "Subcategory": "Total",
        "Raw_Records": initial_count,
        "Clean_Records": final_count,
        "Retention_Pct": f"{retention_rate:.2f}%",
        "Avg_Char_Length": f"{df_final['char_len'].mean():.1f}",
        "Avg_Word_Length": f"{df_final['word_len'].mean():.1f}"
    })

    # Per Aplikasi
    for app in ["tokopedia", "shopee", "lazada", "bukalapak"]:
        app_sub = df_final[df_final["app_name"] == app]
        raw_app_count = len(raw_df[raw_df["app_name"] == app])
        stats_list.append({
            "Category": "Application",
            "Subcategory": app.capitalize(),
            "Raw_Records": raw_app_count,
            "Clean_Records": len(app_sub),
            "Retention_Pct": f"{(len(app_sub)/raw_app_count*100):.2f}%",
            "Avg_Char_Length": f"{app_sub['char_len'].mean():.1f}",
            "Avg_Word_Length": f"{app_sub['word_len'].mean():.1f}"
        })

    # Per Kelas Sentimen
    for lbl, sent_name in [(0, "Negatif (0)"), (1, "Netral (1)"), (2, "Positif (2)")]:
        sent_sub = df_final[df_final["label"] == lbl]
        stats_list.append({
            "Category": "Sentiment Class",
            "Subcategory": sent_name,
            "Raw_Records": "-",
            "Clean_Records": len(sent_sub),
            "Retention_Pct": f"{(len(sent_sub)/final_count*100):.2f}% of clean",
            "Avg_Char_Length": f"{sent_sub['char_len'].mean():.1f}",
            "Avg_Word_Length": f"{sent_sub['word_len'].mean():.1f}"
        })

    # Per Split Dataset
    for split_name, s_df in [("Train (80%)", train_df), ("Validation (10%)", val_df), ("Test (10%)", test_df)]:
        stats_list.append({
            "Category": "Data Split",
            "Subcategory": split_name,
            "Raw_Records": "-",
            "Clean_Records": len(s_df),
            "Retention_Pct": f"{(len(s_df)/final_count*100):.2f}% of clean",
            "Avg_Char_Length": f"{s_df['cleaned_text'].str.len().mean():.1f}",
            "Avg_Word_Length": f"{s_df['cleaned_text'].apply(lambda t: len(t.split())).mean():.1f}"
        })

    table1_df = pd.DataFrame(stats_list)
    table1_path = "paper_assets/table1_dataset_statistics.csv"
    table1_df.to_csv(table1_path, index=False, encoding="utf-8")
    table1_df.to_csv("data/results/table1_dataset_statistics.csv", index=False, encoding="utf-8")
    logger.info(f"Disimpan Table 1: {table1_path}")

    # 11. Generate Figure 1: Data Distribution Grouped Bar Chart (Publication Ready 300 DPI)
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    palette = {"negatif": "#d9534f", "netral": "#f0ad4e", "positif": "#5cb85c"}
    ax = sns.countplot(
        data=df_final,
        x="app_name",
        hue="sentiment",
        palette=palette,
        order=["tokopedia", "shopee", "lazada", "bukalapak"]
    )

    plt.title("Figure 1: Distribution of Sentiment Labels Across E-Commerce Applications", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("E-Commerce Application", fontsize=12, labelpad=10)
    plt.ylabel("Number of Reviews", fontsize=12, labelpad=10)
    plt.legend(title="Sentiment Class", title_fontsize="11", loc="upper right")
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Tokopedia", "Shopee", "Lazada", "Bukalapak"], fontsize=11)

    # Tambahkan label angka di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom",
                fontsize=9, xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    fig1_path = "paper_assets/figure1_data_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Disimpan Figure 1 (300 DPI): {fig1_path}")

    logger.info("=" * 70)
    logger.info("FASE F2 SELESAI DENGAN SEMPURNA!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_preprocessing_pipeline()
