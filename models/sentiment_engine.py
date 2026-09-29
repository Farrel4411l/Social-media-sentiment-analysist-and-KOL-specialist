from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from typing import Dict, Any
import re

class NLPSentimentEngine:
    """
    Mesin NLP untuk mengekstrak opini publik dan mengklasifikasikan sentimen
    dari input raw text. Menggunakan VADER yang lebih optimal untuk bahasa media sosial.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.translator = GoogleTranslator(source='auto', target='en')

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Melakukan analisis sentimen menggunakan VADER.
        Teks diterjemahkan ke bahasa Inggris terlebih dahulu.
        """
        try:
            # Membatasi karakter karena batasan free API
            translated_text = self.translator.translate(text[:4000])
        except Exception:
            translated_text = text # Fallback

        # Hitung skor sentimen
        score = self.analyzer.polarity_scores(translated_text)
        polarity = score['compound']
        
        # Penyesuaian threshold
        if polarity > 0.05:
            sentiment = "positive"
        elif polarity < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Ekstraksi Pain Points: Ambil porsi kecil dari teks jika panjang
        clean_text = re.sub(r'http\S+', '', text).strip()
        pain_points = clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
            
        return {
            "original_text": text,
            "sentiment": sentiment,
            "polarity_score": round(polarity, 2),
            "pain_points": pain_points
        }
