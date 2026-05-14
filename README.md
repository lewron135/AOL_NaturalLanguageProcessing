# Smart CV Analyzer

Sistem pencocokan resume ke deskripsi pekerjaan (job description) yang dibangun sebagai proyek akhir untuk mata kuliah COMP6885001 Natural Language Processing di BINUS University (2025/2026).

Aplikasi ini menggunakan pemahaman semantik untuk menjembatani perbedaan antara cara kandidat menulis pengalaman mereka dan cara perekrut menulis deskripsi pekerjaan. Jika CV menyebutkan "Deep Learning" dan lowongan meminta "Neural Networks," sistem memahami bahwa keduanya memiliki keterkaitan semantik yang kuat.

---

## Fitur Utama

- Hybrid NER Extraction: Menggabungkan EntityRuler spaCy (berbasis kamus) dengan Semantic Filtering untuk menangkap skill teknis secara presisi sambil mengabaikan noise seperti tunjangan atau lokasi kantor.
- Semantic Match Scoring: Menggunakan Sentence-BERT (SBERT) untuk menghitung cosine similarity antara profil kandidat dan kebutuhan pekerjaan.
- Automated Gap Analysis: Mengidentifikasi kekurangan skill (Missing Skills) secara instan dengan membandingkan entitas yang diekstrak, membantu perekrut melihat area mana yang belum dipenuhi kandidat.
- Mojibake and Encoding Repair: Integrasi ftfy (Fixes Text For You) dan normalisasi Unicode NFC untuk menangani teks berantakan yang sering ditemukan pada ekstraksi PDF.
- Visual Entity Highlighting: Memberikan feedback visual yang menyoroti Skill, Organisasi, dan Lokasi di dalam teks asli resume.

---

## Alur Pipeline NLP

1. Normalisasi Teks: Teks mentah dari PDF dibersihkan menggunakan ftfy untuk memperbaiki artifak encoding dan NLTK untuk proses lemmatization.
2. Ekstraksi Fitur: Sistem mengekstrak Noun Chunks dan Named Entities.
3. Semantic Filtering: Untuk menghindari fitur sampah (seperti "competitive salary"), setiap frasa yang diekstrak dibandingkan dengan Technical Anchors. Hanya frasa yang secara semantik lebih dekat ke konteks teknologi yang dipertahankan.
4. Vector Embedding: Deskripsi pekerjaan dan CV yang telah dibersihkan diubah menjadi vektor dimensi tinggi menggunakan model all-mpnet-base-v2.
5. Perhitungan Similarity: Menggunakan Cosine Similarity untuk menentukan persentase kecocokan akhir.

---

## Evaluasi Sistem

Berdasarkan pengujian pada dataset validasi, komponen ekstraksi entitas (NER) mencapai performa sebagai berikut:

- Precision: 0.9500
- Recall: 0.7308
- F1-Score: 0.8261

Hasil ini menunjukkan bahwa sistem sangat akurat dalam memfilter informasi yang tidak relevan (High Precision) dan memiliki sensitivitas yang kuat dalam menangkap skill teknis yang tertulis di resume.

---

## Struktur Proyek

Proyek ini mengikuti arsitektur modular untuk kemudahan pemeliharaan:

```
AOL_NLP/
├── app.py                  # Entry point aplikasi Streamlit
├── src/
│   ├── extraction/
│   │   ├── engine.py       # Logika pemuatan NER dan SBERT
│   │   └── filters.py      # Filter relevansi semantik
│   ├── utils/
│   │   └── preprocessor.py  # Perbaikan encoding (ftfy) dan pembersihan teks
│   └── ui.py               # Komponen antarmuka Streamlit
├── data/
│   └── processed/          # Berisi cleaned_resumes.csv untuk sistem rekomendasi
├── notebooks/
│   └── 01_data_merging_and_eda.ipynb  # Evaluasi dan EDA
└── requirements.txt        # Dependensi proyek
```

---

## Cara Menjalankan

1. Clone repositori:
   git clone https://github.com/username/AOL_NLP.git
   cd AOL_NLP

2. Instal dependensi:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Jalankan aplikasi:
   streamlit run app.py

---

## Stack Teknologi

- NLP Utama: spaCy (Hybrid NER), Sentence-Transformers (all-mpnet-base-v2)
- Text Repair: ftfy (Mojibake repair), unicodedata (NFC Normalization)
- Operasi Vektor: PyTorch dan Scikit-learn (Cosine Similarity)
- Antarmuka: Streamlit
- Pengolahan Data: Pandas dan NumPy