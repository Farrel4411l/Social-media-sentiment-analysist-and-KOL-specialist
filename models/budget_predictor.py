# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-03

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, Any

class KOLBudgetPredictor:
    """
    Model Regresi berbasis XGBoost untuk memprediksi estimasi budget KOL (Key Opinion Leader)
    berdasarkan metrik audiens dan skor sentimen publik terkini.
    """
    def __init__(self, model_path: str = "data/budget_model.joblib"):
        self.model_path = model_path
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self.is_trained = False
        
        # Load model jika sudah ada
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                # Pastikan model yang di-load adalah model versi baru (4 fitur)
                if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != 4:
                    print("[ML Predictor] Model lama terdeteksi (3 fitur). Melatih ulang model 4 fitur...")
                    self.train_dummy_model()
                else:
                    self.is_trained = True
            except Exception:
                pass

    def train_dummy_model(self):
        """
        Fungsi untuk melatih model regresi dengan dummy dataset yang mencakup Sentimen.
        Features: [followers, engagement_rate, target_reach, sentiment_score]
        """
        print("[ML Predictor] Membangkitkan data sintetik untuk pelatihan model...")
        np.random.seed(42)
        n_samples = 1000
        
        followers = np.random.randint(10000, 5000000, n_samples)
        engagement_rate = np.random.uniform(0.5, 15.0, n_samples)
        target_reach = followers * np.random.uniform(0.1, 0.3, n_samples)
        sentiment_score = np.random.uniform(-1.0, 1.0, n_samples)
        
        X = np.column_stack((followers, engagement_rate, target_reach, sentiment_score))
        
        # Logic: Sentimen negatif -> budget mahal (KOL Premium untuk damage control)
        # Sentimen positif -> budget standar (Organik)
        risk_multiplier = np.where(sentiment_score < 0, 1.5 + abs(sentiment_score), 1.0 - (sentiment_score * 0.2))
        
        base_budget = (followers * 50) + (followers * (engagement_rate/100) * 5000)
        budget = base_budget * risk_multiplier
        
        noise = np.random.normal(0, 1000000, n_samples)
        y = np.clip(budget + noise, a_min=500000, a_max=None)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        return {"status": "trained"}

    def predict_budget(self, followers: int, engagement_rate: float, target_reach: int, sentiment_score: float, dominant_emotion: str) -> Dict[str, Any]:
        """
        Melakukan prediksi harga KOL dan memproyeksikan estimasi ROI.
        """
        if not self.is_trained:
            self.train_dummy_model()
            
        features = np.array([[followers, engagement_rate, target_reach, sentiment_score]])
        predicted_budget = float(self.model.predict(features)[0])
        predicted_budget = max(predicted_budget, 500000.0) # Floor di 500k
        
        # Range estimasi (bawah - atas)
        min_budget = predicted_budget * 0.9
        max_budget = predicted_budget * 1.2
        
        # Proyeksi ROI (Berapa konversi yang dihasilkan berbanding budget)
        strategy = "Damage Control & Trust Recovery" if sentiment_score < 0 else "Brand Amplification & Conversion"
        
        base_conversion_rate = 0.05
        # Emosi menentukan seberapa tinggi konversi
        if dominant_emotion in ["Joy", "Surprise"]:
            base_conversion_rate = 0.08
        elif dominant_emotion in ["Anger", "Disappointment"]:
            base_conversion_rate = 0.02
            
        projected_conversions = int(target_reach * base_conversion_rate)
        cpa = predicted_budget / projected_conversions if projected_conversions > 0 else predicted_budget
        
        return {
            "predicted_budget_mean": predicted_budget,
            "budget_range_str": f"Rp {int(min_budget):,} - Rp {int(max_budget):,}",
            "recommended_strategy": strategy,
            "projected_roi": {
                "estimated_conversions": projected_conversions,
                "cost_per_acquisition": f"Rp {int(cpa):,}"
            }
        }

# Global Instance
budget_predictor = KOLBudgetPredictor()
