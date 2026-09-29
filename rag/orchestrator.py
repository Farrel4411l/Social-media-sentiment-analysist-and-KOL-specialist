# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-09-22

from typing import List, Dict, Any
from rag.vectorstore import VectorDBInitializer
from models.slm_generator import StrategistSLMGenerator

class RAGOrchestrator:
    """
    Kelas Orchestrator yang menerima query (berupa analisis sentimen/kondisi),
    melakukan pencarian framework ke Vector DB, dan akan menyuntikkannya ke prompt SLM.
    """
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.vector_initializer = VectorDBInitializer(persist_directory=persist_directory)
        self.vector_db = self.vector_initializer.get_vector_db()
        self.slm_generator = StrategistSLMGenerator()
        
    def retrieve_context(self, query: str, k: int = 2) -> List[str]:
        """
        Mencari top-K framework psikologi marketing yang paling relevan dengan permasalahan audiens.
        """
        docs = self.vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def process_sentiment_for_brief(self, sentiment_data: Dict[str, Any], keyword: str = "", api_key: str = None) -> Dict[str, Any]:
        """
        Menerima ekstraksi sentimen, melakukan retrieval RAG, dan 
        memanggil model SLM untuk menghasilkan teks KOL Brief JSON.
        """
        pain_points = sentiment_data.get('pain_points', '')
        overall_sentiment = sentiment_data.get('sentiment', 'neutral')
        
        # 1. Retrieval (RAG)
        search_query = f"Pain points: {pain_points}. Audience sentiment: {overall_sentiment}. We need a framework to address this."
        retrieved_frameworks = self.retrieve_context(search_query, k=2)
        
        # 2. Generation (SLM Prompting)
        generated_brief = self.slm_generator.generate_brief(
            sentiment_data=sentiment_data,
            rag_context=retrieved_frameworks,
            keyword=keyword,
            api_key=api_key
        )
        
        return {
            "input_sentiment": sentiment_data,
            "retrieved_frameworks": retrieved_frameworks,
            "kol_brief": generated_brief
        }

if __name__ == "__main__":
    orchestrator = RAGOrchestrator()
    dummy_sentiment = {
        "sentiment": "negative",
        "pain_points": "Konsumen ragu-ragu dan tidak percaya produk ini asli, banyak masalah produk palsu di pasar."
    }
    result = orchestrator.process_sentiment_for_brief(dummy_sentiment)
    print("--- RAG Context ---")
    for fw in result["retrieved_frameworks"]:
        print("-", fw)
    print("\n--- SLM Generated KOL Brief ---")
    print(result["kol_brief"])
