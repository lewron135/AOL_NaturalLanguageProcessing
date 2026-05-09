import streamlit as st
import sys
import os
import pickle

# Memastikan Python dapat membaca modul dari folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.models.recommender import ResumeRecommender

# Konfigurasi Halaman (Bersih dan Profesional)
st.set_page_config(page_title="Candidate Screening System", layout="centered")

# Menggunakan custom CSS sederhana untuk memastikan tampilan flat dan bersih
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: bold;
    }
    .snippet-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-family: monospace;
        font-size: 14px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Candidate Screening System")
st.markdown("Enter the job description and requirements below to retrieve the most relevant resumes from the database.")
st.markdown("---")

# Memuat model yang sudah dilatih (di-cache agar tidak loading ulang terus menerus)
@st.cache_resource
def load_model():
    model_path = 'models/tfidf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            vectorizer, resume_matrix, df_resume = pickle.load(f)
        
        # Inisialisasi ulang engine dengan data yang sudah di-load
        engine = ResumeRecommender()
        engine.vectorizer = vectorizer
        engine.resume_matrix = resume_matrix
        engine.df_resume = df_resume
        return engine
    else:
        return None

engine = load_model()

if engine is None:
    st.error("System Error: Model file not found. Ensure 'tfidf_model.pkl' exists in the models directory.")
else:
    # Bagian Input
    st.subheader("Job Description Input")
    job_desc = st.text_area("Paste job responsibilities and required skills here:", height=200)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.number_input("Candidates to retrieve", min_value=1, max_value=20, value=5)
    
    # Tombol Eksekusi
    if st.button("Run Screening"):
        if job_desc.strip() == "":
            st.warning("Input cannot be empty.")
        else:
            st.markdown("---")
            st.subheader("Screening Results")
            
            # Memanggil fungsi rekomendasi dari modul src
            results = engine.get_recommendations(job_desc, top_n=top_n)
            
            # Menampilkan hasil
            for i, r in enumerate(results):
                score_percentage = round(r['score'] * 100, 2)
                
                st.markdown(f"#### Rank {i+1} - Category: {r['category']}")
                st.write(f"**Match Score:** {score_percentage}%")
                
                # Menampilkan potongan teks resume di dalam box abu-abu yang bersih
                st.markdown(f'<div class="snippet-box">{r["resume_preview"]}</div>', unsafe_allow_html=True)