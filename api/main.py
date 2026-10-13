# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-09-22

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from models.sentiment_engine import NLPSentimentEngine
from rag.orchestrator import RAGOrchestrator
from models.budget_predictor import KOLBudgetPredictor

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="AI PR & KOL Specialist API",
    description="Backend API untuk Decision Support System (DSS) Digital Marketing",
    version="1.0.0"
)

# Inisialisasi engine & pipeline 
# Dilakukan secara global (singleton-like) agar dipanggil sekali saat server start
try:
    sentiment_engine = NLPSentimentEngine()
    rag_orchestrator = RAGOrchestrator(persist_directory="data/chroma_db")
    budget_predictor = KOLBudgetPredictor(model_path="data/budget_model.joblib")
except Exception as e:
    print(f"Error initializing services: {e}")

from scraper.social_scraper import social_scraper

# ==========================================
# PYDANTIC VALIDATION SCHEMAS
# ==========================================

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Teks keluhan/opini publik yang ingin diekstrak sentimennya.")

class BriefRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Kumpulan masalah/opini publik untuk diubah menjadi brief KOL (End-to-End).")

class BudgetRequest(BaseModel):
    followers: int = Field(..., gt=0, description="Jumlah followers KOL")
    engagement_rate: float = Field(..., ge=0.0, le=100.0, description="Engagement rate KOL dalam persentase (contoh: 3.5)")
    target_reach: int = Field(..., gt=0, description="Estimasi jangkauan audiens yang ditargetkan")
    sentiment_score: float = Field(0.0, description="Skor compound sentimen publik (-1.0 to 1.0)")
    dominant_emotion: str = Field("Neutral", description="Emosi dominan audiens")

class AutoCampaignRequest(BaseModel):
    keyword: str = Field(..., min_length=2, description="Nama brand atau isu yang ingin ditarik datanya untuk diproses otomatis")


# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest):
    """
    [PHASE 1] Mengekstrak sentimen dan mendeteksi pain points dari keluhan audiens.
    """
    try:
        result = sentiment_engine.analyze(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan analisis sentimen: {str(e)}")

@app.post("/generate-brief")
def generate_brief(request: BriefRequest):
    """
    [PHASE 2 & 3] Pipeline Lengkap:
    1. Sentiment Extraction
    2. RAG Context Retrieval (Marketing Psychology Frameworks)
    3. SLM Generation (JSON format KOL Brief)
    """
    try:
        # Ekstrak sentimen awal
        sentiment_result = sentiment_engine.analyze(request.text)
        
        # Proses pipeline (Retrieval + SLM Generation)
        brief_result = rag_orchestrator.process_sentiment_for_brief(sentiment_result)
        
        return {"status": "success", "data": brief_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses KOL Brief: {str(e)}")

@app.post("/predict-budget")
def predict_budget(request: BudgetRequest):
    """
    [PHASE 4 ML] Memprediksi anggaran yang wajar untuk seorang KOL
    serta memproyeksikan ROI berdasarkan skor sentimen.
    """
    try:
        est_budget_data = budget_predictor.predict_budget(
            followers=request.followers,
            engagement_rate=request.engagement_rate,
            target_reach=request.target_reach,
            sentiment_score=request.sentiment_score,
            dominant_emotion=request.dominant_emotion
        )
        return {
            "status": "success",
            "data": est_budget_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memprediksi budget KOL: {str(e)}")

from fastapi import FastAPI, HTTPException, Request
import os
from dotenv import load_dotenv

load_dotenv()

@app.post("/generate-campaign-auto")
async def generate_campaign_auto(request: AutoCampaignRequest, req: Request):
    """
    [PHASE 5] Auto-Scrape & Generate Campaign:
    Menerima keyword, melakukan penarikan data opini publik di media sosial,
    menganalisis agregat sentimennya, lalu men-generate KOL Brief yang sesuai.
    """
    try:
        # Baca Gemini API Key dari Header (prioritas), jika kosong ambil dari .env
        gemini_api_key = req.headers.get("X-Gemini-Key") or os.getenv("GEMINI_API_KEY")
        
        # 1. Scrape data menggunakan keyword (simulasi async)
        scraped_texts = await social_scraper.scrape_by_keyword(request.keyword, limit=5)
        
        # 2. Gabungkan hasil scrape untuk dianalisa sebagai satu narasi publik utuh
        combined_text = " ".join(scraped_texts)
        
        # 3. Sentiment Analysis
        sentiment_result = sentiment_engine.analyze(combined_text)
        
        # 4. RAG Context Retrieval & SLM Generation
        brief_result = rag_orchestrator.process_sentiment_for_brief(
            sentiment_result, 
            keyword=request.keyword,
            api_key=gemini_api_key
        )
        
        # 5. ML Budget Predictor (Automated Estimate for a standard Macro KOL)
        budget_projection = budget_predictor.predict_budget(
            followers=500000,
            engagement_rate=4.5,
            target_reach=150000,
            sentiment_score=sentiment_result.get("polarity_score", 0.0),
            dominant_emotion=sentiment_result.get("dominant_emotion", "Neutral")
        )
        
        brief_result["ml_budget_projection"] = budget_projection
        
        return {
            "status": "success",
            "keyword_tracked": request.keyword,
            "data_points_analyzed": len(scraped_texts),
            "data": brief_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses auto campaign: {str(e)}")

@app.get("/")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "online", "message": "AI PR & KOL Specialist API is running."}

# Entry point untuk menjalankan uvicorn secara langsung
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
