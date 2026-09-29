# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-09-22

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class StrategistSLMGenerator:
    """
    Modul SLM (Small Language Model) yang bertindak sebagai PR Strategist.
    Menerima context dari RAG dan hasil sentimen, lalu men-generate KOL Brief
    dalam format JSON terstruktur berkat prompt engineering yang presisi.
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name_or_path
        self.llm = self._initialize_mock_llm()
        self.parser = JsonOutputParser()
        self.prompt_template = self._build_prompt_template()

    def _initialize_mock_llm(self):
        """
        Inisialisasi LLM/SLM. Karena ini tahap scaffolding/DSS boilerplate, 
        kita menggunakan Mock Class agar eksekusi sistem tidak terblokir oleh download model besar.
        Di tahap production, ini diganti dengan model asli via HuggingFacePipeline.
        """
        # --- PROD CODE SNIPPET ---
        # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        # return HuggingFacePipeline(pipeline=pipe)
        
        class MockLLM:
            def invoke_with_keyword(self, prompt_text: str, keyword: str):
                import re
                
                # Mengekstrak teks pain_points atau original_text dari prompt
                pain_points = ""
                match = re.search(r'"original_text":\s*"([^"]+)"', prompt_text)
                if match:
                    pain_points = match.group(1).lower()
                else:
                    pain_points = prompt_text.lower()
                
                # Deteksi Konteks Entitas Berdasarkan Teks
                context_type = "produk" # Default
                
                # Gunakan keyword sebagai default entitas, atau 'brand ini'
                entity_name = keyword.upper() if keyword else "tokoh/brand tersebut"
                
                # Kategori Tokoh Politik / Pemerintahan
                if any(k in pain_points for k in ["jokowi", "prabowo", "gibran", "presiden", "menteri", "pemerintah", "pemilu", "politik", "pejabat", "ijazah", "kpu", "trump", "putin", "biden"]):
                    context_type = "tokoh_politik"
                # Kategori Jasa / Layanan / Tech
                elif any(k in pain_points for k in ["jasa", "layanan", "service", "aplikasi", "gojek", "grab", "bank", "internet", "wifi"]):
                    context_type = "jasa"
                # Kategori Gerakan / Sosial
                elif any(k in pain_points for k in ["kampanye", "gerakan", "donasi", "sosial", "lingkungan", "climate", "aksi"]):
                    context_type = "campaign_sosial"
                # Kategori Public Figure / Artis
                elif any(k in pain_points for k in ["artis", "penyanyi", "band", "aktor", "konser", "film"]):
                    context_type = "public_figure"
                
                # Generate Dynamic JSON based on context and WEAVE the keyword!
                if context_type == "tokoh_politik":
                    mock_response = {
                        "Campaign Objective": f"Memperbaiki citra publik {entity_name}, meredam misinformasi yang beredar, dan membangun sentimen positif berdasarkan rekam jejak nyata.",
                        "Psychological Angle": "Menggunakan pendekatan Empathy & Authority. Menyadari keresahan masyarakat (PAS) lalu memposisikan tokoh sebagai pendengar yang membawa solusi riil dan stabilitas.",
                        "Persona": "Netral, intelektual, namun merakyat (Grassroots Empathy) dan terpercaya.",
                        "Storyline": f"KOL (misalnya pengamat atau masyarakat sipil) memulai dengan membahas opini publik atau polemik seputar {entity_name} yang sedang hangat di masyarakat. Kemudian, KOL membedah fakta/data secara objektif mengenai langkah konkrit yang diambil oleh {entity_name}, lalu mengajak audiens berdiskusi sehat di kolom komentar tanpa terbawa hoaks atau provokasi."
                    }
                elif context_type == "jasa":
                    mock_response = {
                        "Campaign Objective": f"Meningkatkan user acquisition untuk layanan {entity_name} dan membuktikan keandalan serta kemudahannya.",
                        "Psychological Angle": "Menekan pain point audiens (keribetan/inefisiensi) dan menawarkan layanan ini sebagai 'Life Hack' yang menghemat waktu dan biaya (AIDA).",
                        "Persona": "Modern, efisien, tech-savvy, dan dapat dipercaya (Relatable Professional).",
                        "Storyline": f"KOL menunjukkan skenario nyata betapa repotnya mengurus suatu masalah secara manual. Kemudian KOL mendemonstrasikan menggunakan {entity_name} melalui layar, menonjolkan betapa cepat masalah selesai berkat {entity_name}, dan mengajak audiens mencoba dengan link/kode promo."
                    }
                elif context_type == "campaign_sosial":
                    mock_response = {
                        "Campaign Objective": f"Meningkatkan kesadaran massal (awareness) terkait isu {entity_name} dan mendorong partisipasi aktif/donasi.",
                        "Psychological Angle": "Menggunakan Emotional Appeal dan Urgency (Framework AIDA) agar audiens merasa terhubung secara emosional dan merasa harus bertindak sekarang juga.",
                        "Persona": "Inspiratif, peduli, tulus, dan komunikator yang menyentuh hati.",
                        "Storyline": f"KOL menceritakan fakta menyedihkan atau statistik mengejutkan terkait {entity_name}. KOL menunjukkan visual yang menggugah emosi, lalu mengarahkan audiens untuk ikut mengambil bagian, menandatangani petisi, atau berdonasi untuk aksi {entity_name} melalui link di bio."
                    }
                elif context_type == "public_figure":
                    mock_response = {
                        "Campaign Objective": f"Membangun hype untuk {entity_name}, membersihkan nama baik dari rumor, dan memperkuat hubungan emosional dengan fanbase.",
                        "Psychological Angle": "Menggunakan Halo Effect dan Relatability. Menunjukkan sisi otentik yang jarang terlihat (Behind the Scenes).",
                        "Persona": "Hangat, otentik (Authentic), enerjik, dan bersahabat.",
                        "Storyline": f"KOL menceritakan perjalanannya mengikuti {entity_name} sejak lama. Memasukkan unsur nostalgia atau behind-the-scenes yang menyentuh, meredam rumor negatif dengan fakta positif, dan mengajak audiens meramaikan project terbaru dari {entity_name}."
                    }
                else: # Produk (Default)
                    mock_response = {
                        "Campaign Objective": f"Memulihkan kepercayaan audiens terhadap kualitas produk {entity_name} dan membuktikan fungsi serta keasliannya.",
                        "Psychological Angle": "Menggunakan Cialdini's Social Proof dipadukan dengan PAS (Problem, Agitate, Solution) untuk menjawab keraguan konsumen.",
                        "Persona": "Otoritatif namun berempati (Authority & Empathetic), jujur, dan aplikatif.",
                        "Storyline": f"KOL memulai dengan menceritakan pengalaman buruk (pain point) audiens dalam mencari atau menggunakan barang sejenis. Kemudian KOL menunjukkan produk {entity_name} yang asli sebagai solusi terpercaya, mendemonstrasikan penggunaannya secara nyata, dan memandu audiens untuk hanya membeli {entity_name} di official store agar terhindar dari barang palsu."
                    }
                    
                return json.dumps(mock_response)

            def invoke(self, prompt_text: str):
                return self.invoke_with_keyword(prompt_text, "")
                
        return MockLLM()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """
        Merancang System Prompt dan User Prompt secara ketat untuk 
        memastikan keluaran (output) adalah valid JSON dengan keys spesifik.
        """
        system_instructions = (
            "You are an elite PR & KOL Strategist AI. Your goal is to design highly persuasive KOL (Key Opinion Leader) "
            "marketing briefs based on public sentiment and specific marketing psychology frameworks.\n\n"
            "You MUST output the result EXCLUSIVELY as a valid JSON object. Do not add markdown blocks (like ```json), "
            "do not add conversational text. The JSON must exactly have the following 4 keys:\n"
            "- \"Campaign Objective\": (string) The main marketing goal of this KOL campaign.\n"
            "- \"Psychological Angle\": (string) How the provided framework is applied to the audience's pain points.\n"
            "- \"Persona\": (string) The ideal characteristics or vibe of the KOL (e.g., Empathetic, Authority, Energetic).\n"
            "- \"Storyline\": (string) A brief, step-by-step storyline of what the KOL should say or act out in the content.\n"
        )
        
        user_message = (
            "Based on the following input data, generate the KOL Brief.\n\n"
            "--- AUDIENCE SENTIMENT & PAIN POINTS ---\n"
            "{sentiment_analysis}\n\n"
            "--- PSYCHOLOGICAL MARKETING FRAMEWORKS (RAG CONTEXT) ---\n"
            "{rag_context}\n"
        )
        
        return ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("user", user_message)
        ])

    def generate_brief(self, sentiment_data: dict, rag_context: list, keyword: str = "", api_key: str = None) -> dict:
        """
        Menjalankan prompt engineering ke SLM dan mem-parsing output JSON.
        Jika api_key diberikan, gunakan API Google Gemini yang sebenarnya (google.genai).
        Jika tidak, gunakan simulasi MockLLM dinamis.
        """
        # Format input data
        sentiment_str = json.dumps(sentiment_data, indent=2)
        context_str = "\n".join([f"- {ctx}" for ctx in rag_context])
        
        # Menghasilkan prompt utuh
        messages = self.prompt_template.format_messages(
            sentiment_analysis=sentiment_str,
            rag_context=context_str
        )
        
        full_prompt = "\n".join([m.content for m in messages])
        
        # ==========================================
        # EXECUTE INFERENCE
        # ==========================================
        raw_output = ""
        
        if api_key:
            # Gunakan AI Asli (Gemini API)
            try:
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key)
                # Gunakan model terbaru yang tersedia di free tier
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                )
                raw_output = response.text
            except Exception as e:
                # Fallback jika model gagal/key salah
                print(f"Gemini API Error: {e}")
                if hasattr(self.llm, 'invoke_with_keyword'):
                    raw_output = self.llm.invoke_with_keyword(full_prompt, keyword)
                else:
                    raw_output = self.llm.invoke(full_prompt)
        else:
            # Simulasi Mock Dinamis
            if hasattr(self.llm, 'invoke_with_keyword'):
                raw_output = self.llm.invoke_with_keyword(full_prompt, keyword)
            else:
                raw_output = self.llm.invoke(full_prompt)
        
        # Parsing string menjadi JSON / Python Dictionary
        try:
            structured_brief = self.parser.parse(raw_output)
            return structured_brief
        except Exception as e:
            return {
                "error": "Failed to parse SLM output as JSON",
                "raw_output": raw_output
            }
