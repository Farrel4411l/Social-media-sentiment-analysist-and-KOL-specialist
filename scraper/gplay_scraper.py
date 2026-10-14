# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-14

"""
Scraper Google Play Store untuk Pengumpulan Dataset Ulasan E-Commerce Indonesia.
Target: 15.000 ulasan dari 4 aplikasi e-commerce teratas (Tokopedia, Shopee, Lazada, Bukalapak),
masing-masing ~3.750 ulasan dengan distribusi rating merata (1 s.d. 5 bintang).
"""

import os
import sys
import time
import random
import logging
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
from google_play_scraper import Sort, reviews

# Set random seed untuk reproduksibilitas
random.seed(42)

# Konfigurasi logging standar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GPlayScraper")

# Daftar Aplikasi Target (E-Commerce Indonesia)
TARGET_APPS: Dict[str, str] = {
    "tokopedia": "com.tokopedia.tkpd",
    "shopee": "com.shopee.id",
    "lazada": "com.lazada.android",
    "bukalapak": "com.bukalapak.android"
}

# Kolom yang wajib disimpan untuk dataset akademis
REQUIRED_COLUMNS: List[str] = [
    "reviewId",
    "userName",
    "content",
    "score",
    "thumbsUpCount",
    "at",
    "appVersion",
    "app_name"
]


def fetch_reviews_with_retry(
    app_id: str,
    score: int,
    count: int,
    token: Optional[Any] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0
) -> tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Mengambil batch ulasan dari Google Play Store dengan penanganan Exponential Backoff.
    """
    retries = 0
    backoff = initial_backoff

    while retries < max_retries:
        try:
            batch_result, next_token = reviews(
                app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=count,
                filter_score_with=score,
                continuation_token=token
            )
            return batch_result, next_token
        except Exception as exc:
            retries += 1
            logger.warning(
                f"Rate limit / koneksi terganggu untuk {app_id} (Score {score}). "
                f"Percobaan {retries}/{max_retries}. Menunggu {backoff:.1f} detik... Error: {exc}"
            )
            time.sleep(backoff)
            backoff *= 2.0

    logger.error(f"Gagal mengambil batch untuk {app_id} (Score {score}) setelah {max_retries} percobaan.")
    return [], None


def scrape_app_balanced(
    app_name: str,
    app_id: str,
    target_total: int = 3750,
    batch_size: int = 200,
    output_dir: str = "data/raw"
) -> str:
    """
    Mengambil ulasan untuk 1 aplikasi dengan distribusi rating seimbang (1 s.d. 5 bintang).
    Target per rating = target_total // 5.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_per_score = target_total // 5
    logger.info(
        f"\n{'='*60}\n"
        f"Memulai scraping: {app_name.upper()} ({app_id})\n"
        f"Target Total: {target_total} ulasan ({target_per_score} ulasan per skor 1-5)\n"
        f"{'='*60}"
    )

    all_reviews: List[Dict[str, Any]] = []

    for score in range(1, 6):
        score_collected: List[Dict[str, Any]] = []
        token: Optional[Any] = None
        consecutive_empty = 0

        logger.info(f"[{app_name}] Mengumpulkan ulasan Score {score} (Target: {target_per_score})...")

        while len(score_collected) < target_per_score:
            remaining = target_per_score - len(score_collected)
            fetch_count = min(batch_size, remaining)

            batch, token = fetch_reviews_with_retry(
                app_id=app_id,
                score=score,
                count=fetch_count,
                token=token
            )

            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.warning(
                        f"[{app_name}] Tidak ada ulasan lagi yang ditemukan untuk Score {score}. "
                        f"Berhenti pada {len(score_collected)} ulasan."
                    )
                    break
                time.sleep(2.0)
                continue

            consecutive_empty = 0
            for item in batch:
                item["app_name"] = app_name
                score_collected.append(item)

            logger.info(
                f"[{app_name}] Score {score}: {len(score_collected)}/{target_per_score} terkumpul "
                f"({(len(score_collected) / target_per_score * 100):.1f}%)"
            )

            if not token:
                logger.info(f"[{app_name}] Token habis untuk Score {score}.")
                break

            # Sleep adaptif 1.0 - 1.5 detik untuk menghormati Google Play rate limits
            time.sleep(random.uniform(1.0, 1.5))

        all_reviews.extend(score_collected)

    # Format ke DataFrame
    df = pd.DataFrame(all_reviews)

    if not df.empty:
        # Filter kolom yang dibutuhkan
        available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
        df = df[available_cols]

        # Deduplikasi berdasarkan reviewId
        initial_len = len(df)
        df = df.drop_duplicates(subset=["reviewId"])
        dedup_len = len(df)

        if initial_len != dedup_len:
            logger.info(f"[{app_name}] Dihapus {initial_len - dedup_len} ulasan duplikat.")

    # Simpan ke CSV
    output_path = os.path.join(output_dir, f"{app_name}_reviews.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(
        f"[{app_name}] Berhasil disimpan ke: {output_path} "
        f"({len(df)} baris total)\n"
        f"Distribusi skor:\n{df['score'].value_counts().sort_index().to_string()}\n"
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Google Play Store E-Commerce Review Scraper (Scopus Research Pipeline)"
    )
    parser.add_argument(
        "--target-per-app",
        type=int,
        default=3750,
        help="Target jumlah ulasan per aplikasi (default: 3750 untuk total 15.000 dari 4 aplikasi)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Ukuran batch per request (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Folder penyimpanan CSV mentah (default: data/raw)"
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        choices=list(TARGET_APPS.keys()),
        help="Hanya scrape 1 aplikasi spesifik (default: scrape semua 4 aplikasi)"
    )

    args = parser.parse_args()

    apps_to_scrape = (
        {args.app: TARGET_APPS[args.app]}
        if args.app
        else TARGET_APPS
    )

    logger.info("=" * 70)
    logger.info("GOOGLE PLAY STORE SCRAPER — E-COMMERCE SENTIMENT DATASET")
    logger.info(f"Jumlah Aplikasi Target : {len(apps_to_scrape)} ({', '.join(apps_to_scrape.keys())})")
    logger.info(f"Target per Aplikasi    : {args.target_per_app} ulasan")
    logger.info(f"Total Target Dataset   : {len(apps_to_scrape) * args.target_per_app} ulasan")
    logger.info(f"Output Directory       : {args.output_dir}")
    logger.info("=" * 70)

    start_time = time.time()
    generated_files = []

    for app_name, app_id in apps_to_scrape.items():
        csv_file = scrape_app_balanced(
            app_name=app_name,
            app_id=app_id,
            target_total=args.target_per_app,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        generated_files.append(csv_file)

    elapsed_min = (time.time() - start_time) / 60.0
    logger.info("=" * 70)
    logger.info("SCRAPING SELESAI DENGAN SUKSES!")
    logger.info(f"Waktu Eksekusi: {elapsed_min:.2f} menit")
    logger.info("File yang dihasilkan:")
    for f in generated_files:
        logger.info(f"  - {f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
