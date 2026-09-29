# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-09-24

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi Halaman Dasar
st.set_page_config(
    page_title="AI PR & KOL Specialist",
    page_icon="🤖",
    layout="wide"
)

# Kustomisasi CSS untuk tombol yang lebih bersih
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
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI PR & KOL Specialist - Campaign Generator")
st.markdown("Decision Support System (DSS) untuk Digital Marketing yang secara otomatis mengeruk opini publik, mengaitkannya dengan literatur psikologi via RAG, dan mendesain KOL Brief yang persuasif.")

# Konfigurasi Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("FastAPI Endpoint URL", value="http://localhost:9998")
    default_gemini = os.getenv("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API Key (Opsional)", value=default_gemini, type="password", help="Masukkan API Key Google Gemini (gratis dari Google AI Studio) untuk mendapatkan hasil AI yang sesungguhnya.")
    st.markdown("---")
    st.markdown("Pastikan backend FastAPI Anda sedang berjalan di port yang sesuai.")

st.markdown("### 🔍 Generate KOL Campaign Secara Otomatis")
keyword = st.text_input("Masukkan Keyword (Misal: Nama Brand, Isu Viral, Hashtag):", placeholder="Contoh: Tolak Angin")

if st.button("🚀 Scrape & Generate Campaign", type="primary", use_container_width=True):
    if not keyword:
        st.warning("Mohon masukkan keyword terlebih dahulu.")
    else:
        with st.spinner(f"Memproses data untuk '{keyword}'... (Menarik API Sosmed, Analisis NLP, RAG Retrieval, SLM Generation)"):
            try:
                # Menyiapkan headers
                headers = {}
                if gemini_key:
                    headers["X-Gemini-Key"] = gemini_key
                
                # Memanggil Backend
                response = requests.post(
                    f"{api_url}/generate-campaign-auto",
                    json={"keyword": keyword},
                    headers=headers,
                    timeout=120
                )
                
                if response.status_code == 200:
                    res_json = response.json()
                    st.success("🎉 Campaign berhasil di-generate berdasarkan opini publik riil!")
                    
                    data = res_json.get("data", {})
                    
                    # Split Layout
                    col1, col2 = st.columns([1, 1.2])
                    
                    with col1:
                        with st.container(border=True):
                            st.subheader("📊 1. Analisis Sentimen & Pain Points")
                            sentiment = data.get("input_sentiment", {})
                            
                            sent_val = sentiment.get('sentiment', 'Unknown').upper()
                            sent_color = "green" if sent_val == "POSITIVE" else "red" if sent_val == "NEGATIVE" else "gray"
                            
                            st.markdown(f"**Overall Sentiment**: <span style='color:{sent_color}; font-weight:bold;'>{sent_val}</span>", unsafe_allow_html=True)
                            st.write(f"**Polarity Score**: `{sentiment.get('polarity_score', 0)}`")
                            
                            st.markdown("**🔍 Extracted Pain Points / Highlights:**")
                            st.info(sentiment.get("pain_points", "Tidak ada pain points menonjol."))
                        
                        with st.container(border=True):
                            st.subheader("📚 2. RAG Psychological Frameworks")
                            st.caption("Referensi framework yang ditarik otomatis berdasarkan masalah audiens.")
                            frameworks = data.get("retrieved_frameworks", [])
                            for i, fw in enumerate(frameworks):
                                with st.expander(f"📖 Lihat Framework #{i+1}", expanded=(i==0)):
                                    st.write(fw)
                                
                    with col2:
                        with st.container(border=True):
                            st.subheader("📝 3. AI Generated KOL Brief")
                            st.caption("KOL Brief Terstruktur Berdasarkan Data Medsos")
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

                        
                else:
                    # Menangkap error 500 jika limit API Apify habis atau tidak ada data
                    error_detail = response.json().get('detail', response.text)
                    st.error(f"❌ Gagal memproses data.\n\nDetail: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Tidak dapat terhubung ke Backend FastAPI. Pastikan server uvicorn sedang berjalan.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>© 2026 Muhammad Farrel Haidar - AI PR & KOL Specialist DSS</div>", unsafe_allow_html=True)
