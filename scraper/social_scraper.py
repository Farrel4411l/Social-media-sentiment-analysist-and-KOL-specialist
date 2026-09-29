# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-09-24

import asyncio
import os
from typing import List
from apify_client import ApifyClientAsync
from dotenv import load_dotenv

load_dotenv()

class SocialMediaScraper:
    """
    Modul untuk melakukan penarikan data (scraping) dari media sosial 
    menggunakan layanan Apify (Instagram Scraper).
    """
    def __init__(self):
        # Menginisialisasi Apify Client dengan token API dari .env
        self.apify_token = os.getenv("APIFY_API_TOKEN", "")
        self.client = ApifyClientAsync(self.apify_token)

    async def scrape_by_keyword(self, keyword: str, limit: int = 5) -> List[str]:
        """
        Melakukan scraping riil ke Instagram (via Apify) berdasarkan keyword/hashtag.
        Mengembalikan list of strings berisi caption/teks.
        """
        print(f"[Scraper] Memulai penarikan data riil via Apify untuk keyword: '{keyword}'...")
        
        # Konfigurasi input untuk apify/instagram-scraper
        # Menggunakan directUrls untuk mengekstrak postingan, bukan sekadar metadata hashtag
        run_input = {
            "directUrls": [f"https://www.instagram.com/explore/tags/{keyword}/"],
            "resultsType": "posts",
            "resultsLimit": limit
        }
        
        try:
            # Memanggil Actor Apify (apify/instagram-scraper)
            # Proses ini membutuhkan waktu beberapa saat tergantung dari sisi server Apify
            run = await self.client.actor("apify/instagram-scraper").call(run_input=run_input)
            
            # Mendapatkan dataset ID (kompatibilitas dengan apify-client v3)
            dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "defaultDatasetId", getattr(run, "default_dataset_id", None))
            
            # Mengambil hasil dari default dataset
            dataset_items = await self.client.dataset(dataset_id).list_items()
            
            results = []
            for item in dataset_items.items:
                # Mengekstrak teks/caption dari postingan Instagram
                caption = item.get("caption", "")
                if caption:
                    # Bersihkan sedikit (ambil 200 karakter pertama agar SLM tidak kepenuhan context)
                    results.append(caption[:200])
                    
            print(f"[Scraper] Berhasil menarik {len(results)} postingan riil dari Instagram terkait '{keyword}'.")
            
            # Jika dataset kosong (tidak ada hasil)
            if not results:
                raise Exception(f"Penarikan data selesai, namun tidak ada postingan/caption riil yang ditemukan untuk keyword: '{keyword}'.")
                
            return results
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Scraper Error] Terjadi kendala saat memanggil Apify: {error_msg}")
            
            # Cek apakah error disebabkan oleh limitasi kuota Apify
            if "quota" in error_msg.lower() or "limit" in error_msg.lower() or "rate" in error_msg.lower() or "payment" in error_msg.lower():
                raise Exception("Gagal menarik data: Kuota/Limit API Apify Anda telah habis atau sedang dibatasi (Rate Limit).")
            
            # Melemparkan error asli ke endpoint tanpa mock data
            raise Exception(f"Gagal menarik data dari media sosial riil: {error_msg}")

# Singleton instance untuk diimpor ke file lain
social_scraper = SocialMediaScraper()
