# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-03

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from textblob import TextBlob
from typing import Dict, Any, List
import re
from collections import Counter

class NLPSentimentEngine:
    """
    Mesin NLP Tingkat Lanjut untuk mengekstrak opini publik, klasifikasi sentimen (VADER),
    mendeteksi Emosi Dominan, dan Aspect-Based Sentiment.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.translator = GoogleTranslator(source='auto', target='en')
        
        # Lexicon Emosi Dasar
        self.emotion_lexicon = {
            "Joy": ["good", "great", "love", "awesome", "excellent", "happy", "brilliant", "amazing", "beautiful", "satisfied"],
            "Anger": ["bad", "terrible", "worst", "angry", "mad", "hate", "scam", "stupid", "fraud", "annoying", "disgusting"],
            "Disappointment": ["disappointed", "sad", "unfortunately", "fail", "letdown", "overhyped", "poor", "boring"],
            "Fear": ["scary", "fear", "panic", "worry", "dangerous", "threat", "risk"],
            "Surprise": ["wow", "shocking", "unexpected", "surprise", "unbelievable"]
        }

    def detect_dominant_emotion(self, text: str) -> str:
        """Mendeteksi emosi dominan berdasarkan frekuensi lexicon."""
        words = re.findall(r'\w+', text.lower())
        emotion_scores = Counter()
        
        for emotion, keywords in self.emotion_lexicon.items():
            for word in words:
                if word in keywords:
                    emotion_scores[emotion] += 1
                    
        if emotion_scores:
            return emotion_scores.most_common(1)[0][0]
        return "Neutral / Mixed"

    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Aspect-Based Sentiment: Mengekstrak kata benda (noun phrases) apa yang 
        sering dibicarakan, lalu mengelompokkannya ke disukai/dikeluhkan.
        """
        blob = TextBlob(text)
        aspects = blob.noun_phrases
        
        liked = []
        disliked = []
        
        # Evaluasi sentimen untuk tiap noun phrase berdasarkan kata di sekitarnya
        for aspect in aspects:
            # Lewati aspect yang terlalu pendek atau terlalu umum
            if len(aspect) < 3: continue
            
            # Cari kalimat yang mengandung aspect ini
            for sentence in blob.sentences:
                if aspect in sentence.string.lower():
                    score = self.analyzer.polarity_scores(sentence.string)['compound']
                    if score > 0.1 and aspect not in liked:
                        liked.append(aspect)
                    elif score < -0.1 and aspect not in disliked:
                        disliked.append(aspect)
        
        # Batasi maksimal 5 aspek utama
        return {
            "liked_aspects": liked[:5],
            "disliked_aspects": disliked[:5]
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Melakukan analisis mendalam secara menyeluruh.
        """
        try:
            # Membatasi karakter karena batasan free API
            translated_text = self.translator.translate(text[:4000])
        except Exception:
            translated_text = text # Fallback

        # 1. Skor Sentimen Keseluruhan (VADER)
        score = self.analyzer.polarity_scores(translated_text)
        polarity = score['compound']
        
        if polarity > 0.05:
            sentiment = "positive"
        elif polarity < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # 2. Emosi Dominan
        dominant_emotion = self.detect_dominant_emotion(translated_text)
        
        # 3. Aspect-Based Sentiment
        aspects = self.extract_aspects(translated_text)
            
        # 4. Ekstraksi Pain Points
        clean_text = re.sub(r'http\S+', '', text).strip()
        pain_points = clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
            
        return {
            "original_text": text,
            "sentiment": sentiment,
            "polarity_score": round(polarity, 2),
            "dominant_emotion": dominant_emotion,
            "aspects": aspects,
            "pain_points": pain_points
        }
