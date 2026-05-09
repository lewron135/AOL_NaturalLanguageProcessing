import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ResumeRecommender:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.resume_matrix = None
        self.df_resume = None

    def fit(self, df_resume, text_column='Resume_Cleaned_v2'):
        """Melatih mesin dengan data resume yang sudah ada"""
        self.df_resume = df_resume
        self.resume_matrix = self.vectorizer.fit_transform(df_resume[text_column])
        print(f"Model berhasil dilatih dengan {len(df_resume)} resume.")

    def get_recommendations(self, job_description_text, top_n=5):
        """Mencari kandidat terbaik berdasarkan teks lowongan"""
        # Transformasi job desc menjadi vektor (angka)
        job_vector = self.vectorizer.transform([job_description_text])
        
        # Hitung kemiripan
        scores = cosine_similarity(job_vector, self.resume_matrix).flatten()
        
        # Ambil indeks teratas
        top_indices = scores.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'category': self.df_resume.iloc[idx]['Category'],
                'score': scores[idx],
                'resume_preview': self.df_resume.iloc[idx]['Resume'][:200] + "..."
            })
        return results

    def save_model(self, path='models/'):
        """Menyimpan model agar tidak perlu training ulang tiap aplikasi jalan"""
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}tfidf_model.pkl', 'wb') as f:
            pickle.dump((self.vectorizer, self.resume_matrix, self.df_resume), f)
        print("Model berhasil disimpan ke folder models/")