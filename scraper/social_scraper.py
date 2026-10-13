# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-04

import asyncio
import os
import requests
from typing import List, Dict, Any
from apify_client import ApifyClientAsync
from dotenv import load_dotenv
import random

load_dotenv()

class SocialMediaScraper:
    """
    Modul untuk melakukan penarikan data (scraping) riil secara konkuren. 
    Mendukung Instagram (Apify) dan Media Massa (NewsAPI).
    Semua fungsi mock/setup telah dihapus untuk tahap produksi awal.
    """
    def __init__(self):
        self.apify_token = os.getenv("APIFY_API_TOKEN", "")
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.client = ApifyClientAsync(self.apify_token)

    async def scrape_instagram(self, keyword: str, limit: int) -> List[str]:
        """Scraping Instagram via Apify (Real Data)"""
        print(f"[Scraper: Instagram] Memulai penarikan data riil via Apify untuk '{keyword}'...")
        run_input = {
            "directUrls": [f"https://www.instagram.com/explore/tags/{keyword}/"],
            "resultsType": "posts",
            "resultsLimit": limit
        }
        try:
            run = await self.client.actor("apify/instagram-scraper").call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "defaultDatasetId", getattr(run, "default_dataset_id", None))
            dataset_items = await self.client.dataset(dataset_id).list_items()
            
            results = []
            for item in dataset_items.items:
                caption = item.get("caption", "")
                if caption:
                    results.append(caption[:200])
            print(f"[Scraper: Instagram] Berhasil menarik {len(results)} postingan.")
            return results
        except Exception as e:
            print(f"[Scraper: Instagram] Error: {e}")
            return []

    async def scrape_news(self, keyword: str, limit: int) -> List[str]:
        """Scraping Berita / Media Massa riil menggunakan NewsAPI"""
        print(f"[Scraper: NewsAPI] Memindai berita utama terkait '{keyword}'...")
        if not self.news_api_key:
            print("[Scraper: NewsAPI] Error: NEWS_API_KEY tidak ditemukan di .env")
            return []
            
        url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={self.news_api_key}&sortBy=relevancy&pageSize={limit}"
        
        try:
            # Gunakan to_thread agar requests.get (blocking) berjalan sinkron dalam loop async
            response = await asyncio.to_thread(requests.get, url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                results = []
                for article in articles:
                    title = article.get("title", "")
                    desc = article.get("description", "")
                    if title or desc:
                        content = f"Berita: {title}. {desc}"
                        results.append(content[:250])
                        
                print(f"[Scraper: NewsAPI] Berhasil menarik {len(results)} artikel berita.")
                return results
            else:
                print(f"[Scraper: NewsAPI] Gagal: HTTP {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"[Scraper: NewsAPI] Error saat request: {e}")
            return []

    async def scrape_by_keyword(self, keyword: str, limit: int = 5) -> List[str]:
        """
        Orchestrator: Menarik data riil dari Instagram dan NewsAPI secara bersamaan.
        """
        print(f"\n[Scraper Master] Memulai Multi-Source Scraping (Real APIs) untuk: '{keyword}'")
        
        # Eksekusi secara konkuren
        ig_task = self.scrape_instagram(keyword, limit)
        news_task = self.scrape_news(keyword, limit)

        # Menunggu semua tasks selesai
        results = await asyncio.gather(ig_task, news_task, return_exceptions=True)
        
        aggregated_data = []
        for res in results:
            if isinstance(res, list):
                aggregated_data.extend(res)
                
        # Acak urutan agar sentimen tersebar merata
        random.shuffle(aggregated_data)
        
        print(f"[Scraper Master] Total {len(aggregated_data)} data berhasil ditarik.\n")
        return aggregated_data

# Singleton instance
social_scraper = SocialMediaScraper()
