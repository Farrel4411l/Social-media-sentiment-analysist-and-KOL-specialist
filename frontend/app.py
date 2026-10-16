# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-03

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI PR & KOL Specialist DSS",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        background-color: #2e66ff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1a4cd9;
        border-color: #1a4cd9;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI PR & KOL Specialist - DSS Campaign Generator")
st.markdown("Decision Support System (DSS) untuk Digital Marketing yang secara otomatis mengeruk opini publik, mengaitkannya dengan literatur psikologi via RAG, mendesain KOL Brief, dan memproyeksikan **Budget & ROI** dengan *Machine Learning*.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("FastAPI Endpoint URL", value="http://localhost:9998")
    default_gemini = os.getenv("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API Key (Opsional)", value=default_gemini, type="password", help="Masukkan API Key Google Gemini untuk mengaktifkan True AI.")
    
    st.markdown("---")
    st.subheader("🌐 Data Sources (Real APIs)")
    st.checkbox("Instagram (Apify)", value=True, disabled=True)
    st.checkbox("NewsAPI", value=True, disabled=True)
    
    st.markdown("---")
    st.subheader("👨‍💻 Lead Researcher")
    st.markdown("**Muhammad Farrel Haidar**")
    st.caption("Teknik Informatika (Angkatan 2024, Semester 5)")
    st.markdown("📧 [muhammadfarrelhaidar@gmail.com](mailto:muhammadfarrelhaidar@gmail.com)")
    st.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/muhammad-farrel-haidar-69364238a)")
    st.caption("Pastikan backend FastAPI Anda sedang berjalan.")

st.markdown("### 🔍 Generate KOL Campaign Secara Otomatis")
keyword = st.text_input("Masukkan Keyword (Misal: Nama Brand, Isu Viral, Hashtag):", placeholder="Contoh: Tolak Angin")

if st.button("🚀 Scrape & Generate Campaign", type="primary", use_container_width=True):
    if not keyword:
        st.warning("Mohon masukkan keyword terlebih dahulu.")
    else:
        with st.spinner(f"Memproses '{keyword}'... (Multi-Source Scraper -> Deep Sentiment -> RAG -> ML Predictor)"):
            try:
                headers = {"X-Gemini-Key": gemini_key} if gemini_key else {}
                
                response = requests.post(
                    f"{api_url}/generate-campaign-auto",
                    json={"keyword": keyword},
                    headers=headers,
                    timeout=150
                )
                
                if response.status_code == 200:
                    res_json = response.json()
                    st.success(f"🎉 Campaign berhasil di-generate dari {res_json.get('data_points_analyzed', 0)} data poin opini publik!")
                    
                    data = res_json.get("data", {})
                    
                    col1, col2 = st.columns([1, 1.2])
                    
                    with col1:
                        # SECTION 1: Deep Sentiment
                        with st.container(border=True):
                            st.subheader("📊 1. Analisis Sentimen Mendalam")
                            sentiment = data.get("input_sentiment", {})
                            
                            sent_val = sentiment.get('sentiment', 'Unknown').upper()
                            sent_color = "green" if sent_val == "POSITIVE" else "red" if sent_val == "NEGATIVE" else "gray"
                            
                            st.markdown(f"**Overall Sentiment**: <span style='color:{sent_color}; font-weight:bold;'>{sent_val}</span> (Score: `{sentiment.get('polarity_score', 0)}`)", unsafe_allow_html=True)
                            
                            # Dominant Emotion
                            st.markdown(f"**🎭 Dominant Emotion**: `{sentiment.get('dominant_emotion', 'Neutral')}`")
                            
                            # Aspects
                            aspects = sentiment.get("aspects", {})
                            if aspects.get("liked_aspects"):
                                st.markdown("**🟢 Liked Aspects (Top)**: " + ", ".join([f"`{x}`" for x in aspects.get("liked_aspects")]))
                            if aspects.get("disliked_aspects"):
                                st.markdown("**🔴 Disliked Aspects (Top)**: " + ", ".join([f"`{x}`" for x in aspects.get("disliked_aspects")]))
                            
                            st.markdown("**🔍 Extracted Pain Points / Highlights:**")
                            st.info(sentiment.get("pain_points", "Tidak ada pain points menonjol."))
                        
                        # SECTION 2: RAG
                        with st.container(border=True):
                            st.subheader("📚 2. RAG Psychological Frameworks")
                            st.caption("Referensi psikologi marketing berdasarkan masalah audiens.")
                            frameworks = data.get("retrieved_frameworks", [])
                            for i, fw in enumerate(frameworks):
                                with st.expander(f"📖 Lihat Framework #{i+1}", expanded=(i==0)):
                                    st.write(fw)
                                
                    with col2:
                        # SECTION 3: KOL Brief
                        with st.container(border=True):
                            st.subheader("📝 3. AI Generated KOL Brief")
                            brief = data.get("kol_brief", {})
                            
                            if isinstance(brief, dict):
                                st.markdown(f"**🎯 Campaign Objective:**\n\n{brief.get('Campaign Objective', '')}")
                                st.divider()
                                st.markdown(f"**🧠 Psychological Angle:**\n\n{brief.get('Psychological Angle', '')}")
                                st.divider()
                                st.markdown(f"**🎭 Ideal Persona:**\n\n{brief.get('Persona', '')}")
                                st.divider()
                                st.markdown(f"**🎬 Storyline Brief:**\n\n{brief.get('Storyline', '')}")
                            else:
                                st.warning("Format Brief Gagal Diparsing. Raw Output:")
                                st.write(brief)
                                
                        # SECTION 4: ML Budget Predictor
                        with st.container(border=True):
                            st.subheader("📈 4. ML Budget & ROI Projection")
                            ml_data = data.get("ml_budget_projection", {})
                            
                            if ml_data:
                                st.caption("Estimasi otomatis untuk KOL tingkat Makro (Asumsi: 500K Followers).")
                                st.markdown(f"**Rekomendasi Strategi:** `{ml_data.get('recommended_strategy', '')}`")
                                
                                m1, m2 = st.columns(2)
                                m1.metric("Recommended Budget Range", ml_data.get("budget_range_str", ""))
                                
                                roi = ml_data.get("projected_roi", {})
                                m2.metric("Projected Conversions", roi.get("estimated_conversions", 0))
                                
                                st.info(f"**Cost Per Acquisition (CPA):** {roi.get('cost_per_acquisition', '')} per konversi.")
                            else:
                                st.write("Data ML Budget Predictor tidak tersedia.")

                else:
                    error_detail = response.json().get('detail', response.text)
                    st.error(f"❌ Gagal memproses data.\n\nDetail: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Tidak dapat terhubung ke Backend FastAPI. Pastikan server uvicorn sedang berjalan.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>© 2026 <b>Muhammad Farrel Haidar</b> (Teknik Informatika '24, Semester 5) — AI PR & KOL Specialist DSS<br><a href='mailto:muhammadfarrelhaidar@gmail.com' style='color:#2e66ff;'>Email</a> | <a href='https://www.linkedin.com/in/muhammad-farrel-haidar-69364238a' target='_blank' style='color:#2e66ff;'>LinkedIn</a></div>", unsafe_allow_html=True)
