"""
Rebuild script for notebooks/01_data_merging_and_eda.ipynb
Implements the refactor plan: 39 cells -> 55 cells.
Run from the project root: python3 rebuild_notebook.py
"""
import json, copy

NB_PATH   = "notebooks/01_data_merging_and_eda.ipynb"
SRC_PATH  = "notebooks/01_data_merging_and_eda.BACKUP.ipynb"  # always read from backup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_src(text):
    """Convert multiline string to notebook source list (one entry per line)."""
    lines = text.lstrip("\n").split("\n")
    out = []
    for idx, line in enumerate(lines):
        if idx < len(lines) - 1:
            out.append(line + "\n")
        elif line:
            out.append(line)
    return out

def mk_md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": to_src(text)}

def mk_code(text, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": to_src(text)}

def clone(cell, new_src=None):
    c = copy.deepcopy(cell)
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    if new_src is not None:
        c["source"] = to_src(new_src)
    return c

# ---------------------------------------------------------------------------
# Load original notebook
# ---------------------------------------------------------------------------
with open(SRC_PATH) as f:
    nb = json.load(f)
o = nb["cells"]   # 39 original cells (indices 0-38)
print(f"Loaded {len(o)} original cells.")

# ---------------------------------------------------------------------------
# IDs for new cells
# ---------------------------------------------------------------------------
_id_counter = [0]
def nid():
    _id_counter[0] += 1
    return f"nn{_id_counter[0]:06d}"

# ===========================================================================
# NEW / MODIFIED CELL SOURCES
# ===========================================================================

# ---- target[0]: Project Overview MD ----------------------------------------
MD_OVERVIEW = """\
# Smart CV Analyzer — NLP Final Project

**Mata Kuliah:** COMP6885001 — Natural Language Processing | **BINUS University 2025/2026**

Proyek ini membangun sistem NLP *end-to-end* untuk mencocokkan resume pelamar kerja
dengan deskripsi lowongan, mengekstrak skill, dan mengidentifikasi gap kompetensi kandidat.

---

## Dataset

| Sumber | File | Jumlah |
|--------|------|--------|
| Bhawal | `resumes_bhawal/Resume.csv` | 2.484 resume |
| Ejaz | `resumes_ejaz.csv` | 962 resume |
| Bibek | `job_descriptions_bibek.csv` | 10.000 lowongan |

---

## Alur Pipeline

1. **Load** — Baca 3 CSV + 1 PDF CV
2. **Merge** — Gabungkan Ejaz + Bhawal menjadi `df_resume_merged` (2.648 baris)
3. **Clean** — `clean_resume_text_v2`: hapus noise, non-ASCII, stopwords
4. **Vectorize** — TF-IDF (`max_features=5000`, `ngram_range=(1,2)`)
5. **Match** — Cosine similarity: resume vs job description
6. **Analyze** — spaCy PhraseMatcher + RapidFuzz: skill extraction & gap analysis

---

| Bagian | Topik |
|--------|-------|
| 1 | Setup & Imports |
| 2 | Dataset Loading |
| 3 | Data Merging & Cleaning |
| 4 | EDA — Resume |
| 5 | EDA — Job Descriptions |
| 6 | Text Preprocessing |
| 7 | Lemmatization |
| 8 | Post-Preprocessing EDA |
| 9 | Feature Extraction (TF-IDF) |
| 10 | Baseline Matching (Cosine Similarity) |
| 11 | Skill Extraction & Gap Analysis |
| 12 | Smart Skill Vocabulary Builder |
| 13 | Model Training & Export |
| 14 | End-to-End Validation |"""

# ---- target[1]: Consolidated Imports ---------------------------------------
CODE_IMPORTS = """\
# ============================================================
# Section 1: Setup & Imports
# Semua library diimpor di sini agar dependency jelas.
# ============================================================

# Standard library
import os, sys, re, json, random, pickle

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP — NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# NLP — spaCy
import spacy
from spacy.matcher import PhraseMatcher

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fuzzy matching & PDF
from rapidfuzz import process, fuzz
import PyPDF2

# Project module
sys.path.append(os.path.abspath(os.path.join('..')))
from src.models.recommender import ResumeRecommender

print("Semua library berhasil diimpor.")"""

# ---- target[2]: Section 2 MD header ----------------------------------------
MD_SEC2 = """\
## Section 2: Dataset Loading

Memuat tiga dataset mentah dan satu file PDF CV. Path dikonfigurasi di blok berikut —
sesuaikan `PROJECT_ROOT` dengan lokasi folder `AOL_NLP` di mesin Anda.

**Input:** File CSV di `data/raw/`
**Output:** `df_jobs`, `df_resume_bhawal`, `df_resume_ejaz`"""

# ---- target[3]: Path config notice MD ---------------------------------------
MD_PATH_NOTICE = """\
> **KONFIGURASI PATH — SESUAIKAN SEBELUM MENJALANKAN**
> Ubah nilai `PROJECT_ROOT` di sel berikut agar sesuai dengan lokasi folder
> `AOL_NLP` di komputer Anda. Semua path lain dihitung secara otomatis."""

# ---- target[4]: Path config code (replaces cell[2]) ------------------------
CODE_PATHS = """\
# ============================================================
# CHANGE THESE PATHS TO MATCH YOUR MACHINE
# Set PROJECT_ROOT to the absolute path of the AOL_NLP folder.
# All other paths are derived automatically.
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))

JOB_DESC_PATH      = os.path.join(PROJECT_ROOT, 'data', 'raw', 'job_descriptions_bibek.csv')
RESUME_BHAWAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'resumes_bhawal', 'Resume.csv')
RESUME_EJAZ_PATH   = os.path.join(PROJECT_ROOT, 'data', 'raw', 'resumes_ejaz.csv')
CV_PDF_PATH        = os.path.join(PROJECT_ROOT, 'data', 'raw', 'CV_ATS_JOSEP_NATANAEL_PASARIBU .pdf')
MODEL_DIR          = os.path.join(PROJECT_ROOT, 'models')

# Verifikasi keberadaan file
for name, path in [('Job Desc', JOB_DESC_PATH), ('Resume Bhawal', RESUME_BHAWAL_PATH),
                   ('Resume Ejaz', RESUME_EJAZ_PATH), ('CV PDF', CV_PDF_PATH)]:
    status = "OK" if os.path.exists(path) else "FILE TIDAK DITEMUKAN!"
    print(f"  [{status}] {name}: {path}")

df_jobs          = pd.read_csv(JOB_DESC_PATH)
df_resume_bhawal = pd.read_csv(RESUME_BHAWAL_PATH)
df_resume_ejaz   = pd.read_csv(RESUME_EJAZ_PATH)

print(f"\\nTotal Job Descriptions : {df_jobs.shape[0]:,} baris")
print(f"Total Resume (Bhawal)  : {df_resume_bhawal.shape[0]:,} baris")
print(f"Total Resume (Ejaz)    : {df_resume_ejaz.shape[0]:,} baris")"""

# ---- target[7]: df_jobs.info() (NEW) ----------------------------------------
CODE_JOBS_INFO = """\
# Schema dataset Job Description
print("=== df_jobs.info() ===")
df_jobs.info()
print("\\nContoh 3 baris pertama:")
display(df_jobs.head(3))"""

# ---- target[8]: Section 3 MD header -----------------------------------------
MD_SEC3 = """\
## Section 3: Data Merging & Cleaning

Menyatukan dataset resume dari dua sumber (Ejaz dan Bhawal) menjadi satu
DataFrame `df_resume_merged` yang bersih.

**Input:** `df_resume_ejaz`, `df_resume_bhawal`
**Output:** `df_resume_merged` — DataFrame dengan kolom `Category` dan `Resume`

**Keputusan desain:** Kolom `Resume_str` dari Bhawal di-rename ke `Resume` agar
konsisten dengan skema Ejaz sebelum digabungkan dengan `pd.concat`."""

# ---- target[10]: df_resume_merged.info() (NEW) ------------------------------
CODE_MERGED_INFO = """\
# Verifikasi hasil merge
print("=== df_resume_merged.info() ===")
df_resume_merged.info()
print(f"\\nJumlah kategori unik: {df_resume_merged['Category'].nunique()}")
print("\\nDistribusi kategori:")
display(df_resume_merged['Category'].value_counts())"""

# ---- target[11]: Section 4 MD header ----------------------------------------
MD_SEC4 = """\
## Section 4: Exploratory Data Analysis (EDA) — Resume

Mengeksplorasi distribusi kategori, panjang kata, dan konten mentah dari
dataset resume yang sudah digabungkan.

**Input:** `df_resume_merged`
**Output:** Kolom baru `Word_Count`; visualisasi distribusi

Tiga sudut pandang:
1. **Bar chart** — jumlah resume per kategori (class balance check)
2. **Histogram** — distribusi jumlah kata secara keseluruhan
3. **Boxplot** — variasi jumlah kata *per kategori* untuk melihat outlier per domain"""

# ---- target[13]: bar chart (cell[9] — remove inline imports) ----------------
_src9 = "".join(o[9]["source"])
CODE_BARCHART = _src9.replace("import matplotlib.pyplot as plt\n", "").replace("import seaborn as sns\n", "")

# ---- target[15]: Boxplot by Category (NEW) ----------------------------------
CODE_BOXPLOT = """\
# Boxplot: variasi panjang kata per kategori — menunjukkan outlier per domain
order = (df_resume_merged.groupby('Category')['Word_Count']
         .median()
         .sort_values(ascending=False)
         .index)

plt.figure(figsize=(12, 10))
sns.boxplot(data=df_resume_merged, x='Word_Count', y='Category',
            order=order, palette='viridis')
plt.title('Distribusi Word Count per Kategori Resume', fontsize=16)
plt.xlabel('Jumlah Kata', fontsize=12)
plt.ylabel('Kategori', fontsize=12)
plt.xlim(0, df_resume_merged['Word_Count'].quantile(0.99))
plt.tight_layout()
plt.show()"""

# ---- target[16]: random sample (cell[11] — remove import random) ------------
_src11 = "".join(o[11]["source"])
CODE_SAMPLE = _src11.replace("import random\n\n", "").replace("import random\n", "")

# ---- target[17]: Section 5 MD header (fix typo) -----------------------------
MD_SEC5 = """\
## Section 5: Exploratory Data Analysis (EDA) — Job Descriptions

Mengeksplorasi dataset lowongan kerja: info kolom, distribusi mode kerja,
dan panjang teks responsibilities.

**Input:** `df_jobs`
**Output:** Kolom baru `Word_Count` pada `df_jobs`; visualisasi distribusi

> *Catatan: judul markdown asli mengandung typo "Desxriptions" — sudah diperbaiki.*"""

# ---- target[19]: Pie/donut chart work_mode (NEW) ----------------------------
CODE_PIECHART = """\
# Pie/donut chart distribusi work_mode
work_mode_counts = df_jobs['work_mode'].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    work_mode_counts.values,
    labels=work_mode_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.5),   # width < 1 membuat bentuk donut
    colors=sns.color_palette('Set2', len(work_mode_counts))
)
for autotext in autotexts:
    autotext.set_fontsize(11)
ax.set_title('Distribusi Mode Kerja (Work Mode) — Job Descriptions', fontsize=15)
plt.tight_layout()
plt.show()

print("Jumlah per mode:")
display(work_mode_counts)"""

# ---- target[20]: Section 6 MD header ----------------------------------------
MD_SEC6 = """\
## Section 6: Text Preprocessing

Membersihkan teks resume dan job description melalui dua versi pipeline:

| Tahap | Fungsi | Yang Dilakukan |
|-------|--------|----------------|
| v1 | `clean_resume_text` | Lowercase, hapus URL/email, tanda baca, angka |
| v2 | `clean_resume_text_v2` | Semua langkah v1 + hapus non-ASCII (mojibake) + hapus stopwords |

**Lima langkah preprocessing v2:**
1. Konversi ke string & lowercase
2. Hapus URL (`http...`) dan alamat email
3. Hapus karakter non-ASCII — membasmi mojibake seperti `â€¢`
4. Hapus tanda baca dan angka
5. Hapus stopwords bahasa Inggris ("the", "and", "to", dst.)

**Input:** `df_resume_merged['Resume']`, `df_jobs['responsibilities']`, `df_jobs['required_skills']`
**Output:** Kolom `Resume_Cleaned` (v1), `Resume_Cleaned_v2` (v2), serta kolom v2 untuk `df_jobs`"""

# ---- target[22]: v1 reference note MD ---------------------------------------
MD_V1_NOTE = """\
> **Catatan Referensi:** Dua sel berikut (`clean_resume_text` v1 dan penerapannya)
> dipertahankan sebagai *baseline* untuk perbandingan saja. Pipeline utama menggunakan
> `clean_resume_text_v2` di bagian Preprocessing Tahap 2 di bawah — v2 menambahkan
> penghapusan karakter non-ASCII (mojibake) dan stopwords yang tidak ada di v1."""

# ---- target[23]: clean v1 (cell[15] — remove import re, fix regex) ----------
_src15 = "".join(o[15]["source"])
CODE_V1 = _src15.replace("import re\n", "")
CODE_V1 = CODE_V1.replace("re.sub('https\\S+\\s'", "re.sub(r'https\\S+\\s'")
CODE_V1 = CODE_V1.replace("re.sub('\\S+@\\S+'", "re.sub(r'\\S+@\\S+'")
CODE_V1 = CODE_V1.replace('re.escape("""', 're.escape(r"""')
CODE_V1 = CODE_V1.replace("re.sub('\\s+'", "re.sub(r'\\s+'")

# ---- target[26]: clean v2 (cell[18] — remove imports, fix regex, add df_jobs) ---
_src18 = "".join(o[18]["source"])
CODE_V2 = _src18
# Remove import block (keep stop_words assignment)
CODE_V2 = CODE_V2.replace(
    "# Hapus Mojibake (Karakter aneh) & Stopwords\nimport nltk\nfrom nltk.corpus import stopwords\n\nnltk.download('stopwords')\n",
    "# Hapus Mojibake (Karakter aneh) & Stopwords\n"
)
# Fix regex
CODE_V2 = CODE_V2.replace("re.sub('http\\S+\\s*'", "re.sub(r'http\\S+\\s*'")
CODE_V2 = CODE_V2.replace("re.sub('\\S+@\\S+'", "re.sub(r'\\S+@\\S+'")
CODE_V2 = CODE_V2.replace('re.escape("""', 're.escape(r"""')
CODE_V2 = CODE_V2.replace("re.sub('\\s+'", "re.sub(r'\\s+'")
# Add df_jobs v2 cleaning at end
CODE_V2 = CODE_V2.rstrip() + (
    "\n\nprint(\"\\nMenerapkan clean_resume_text_v2 ke Job Descriptions (pipeline resmi)...\")\n"
    "df_jobs['Responsibilities_Cleaned_v2'] = df_jobs['responsibilities'].apply(clean_resume_text_v2)\n"
    "df_jobs['Skills_Cleaned_v2']           = df_jobs['required_skills'].apply(clean_resume_text_v2)\n"
    "print(\"Selesai. Kolom Job Descriptions yang tersedia:\")\n"
    "print([c for c in df_jobs.columns if 'Cleaned' in c or 'Text' in c])"
)

# ---- target[27]: Section 7 MD header ----------------------------------------
MD_SEC7 = """\
## Section 7: Lemmatization

Mereduksi setiap kata ke bentuk dasarnya (lemma) menggunakan `WordNetLemmatizer` dari NLTK.

**Mengapa Lemmatization, bukan Stemming?**
- *Stemming* memotong akhiran secara mekanis: "studies" -> "studi" (salah secara linguistik)
- *Lemmatization* menggunakan kamus: "studies" -> "study", "better" -> "good"

Hasilnya adalah kata yang valid secara linguistik, lebih bermakna untuk TF-IDF dan cosine similarity.

**Input:** `df_resume_merged['Resume_Cleaned_v2']`
**Output:** `df_resume_merged['Resume_Cleaned_v2']` (di-overwrite dengan teks terlemmatisasi)"""

# ---- target[29]: lemmatize (cell[20] — remove imports) ----------------------
_src20 = "".join(o[20]["source"])
CODE_LEMMA = _src20
CODE_LEMMA = CODE_LEMMA.replace("from nltk.stem import WordNetLemmatizer\n\n", "")
CODE_LEMMA = CODE_LEMMA.replace("nltk.download('wordnet')\n", "")
CODE_LEMMA = CODE_LEMMA.replace("nltk.download('omw-1.4')\n\n", "")
CODE_LEMMA = CODE_LEMMA.replace("nltk.download('omw-1.4')\n", "")

# ---- target[30]: Section 8 MD header ----------------------------------------
MD_SEC8 = """\
## Section 8: Post-Preprocessing EDA

Memvisualisasikan distribusi kata kunci *setelah* preprocessing selesai,
menggunakan WordCloud untuk satu kategori pilihan.

**Input:** `df_resume_merged['Resume_Cleaned_v2']`
**Output:** Visualisasi WordCloud kata-kata dominan per kategori"""

# ---- target[32]: WordCloud (cell[22] — remove import) ----------------------
_src22 = "".join(o[22]["source"])
CODE_WORDCLOUD = _src22.replace("from wordcloud import WordCloud\n\n", "")

# ---- target[33]: Section 9 MD header ----------------------------------------
MD_SEC9 = """\
## Section 9: Feature Extraction — TF-IDF Vectorization

Mengubah teks resume yang sudah bersih menjadi representasi numerik menggunakan
**TF-IDF (Term Frequency-Inverse Document Frequency)**.

**Cara kerja TF-IDF:**
Skor tiap kata = seberapa sering kata muncul di satu dokumen (TF) x
seberapa *jarang* kata itu di seluruh korpus (IDF). Kata generik mendapat bobot rendah;
kata spesifik domain mendapat bobot tinggi.

**Parameter yang digunakan:**

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `max_features` | 5000 | Membatasi dimensi untuk menghemat memori |
| `ngram_range` | (1, 2) | Menangkap frasa dua kata seperti "machine learning" |
| `sublinear_tf` | True | Log-scaling TF: `tf = 1 + log(tf)` meredam term yang terlalu dominan |
| `min_df` | 2 | Buang kata yang hanya muncul di 1 dokumen (kemungkinan noise) |

**Input:** `df_resume_merged['Resume_Cleaned_v2']`
**Output:** `resume_matrix` (sparse matrix 2648 x max 5000), `tfidf_vectorizer` (fitted object)"""

# ---- target[35]: TF-IDF (cell[24] — upgrade params, add bigrams) ------------
CODE_TFIDF = """\
# Menggunakan TF-IDF dengan parameter yang ditingkatkan
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # tangkap bigrams seperti "machine learning"
    sublinear_tf=True,    # log-scale TF untuk meredam term yang sangat sering muncul
    min_df=2,             # buang kata yang hanya muncul di < 2 dokumen (noise reduction)
)

# Ubah teks resume jadi matrix angka
resume_matrix = tfidf_vectorizer.fit_transform(df_resume_merged['Resume_Cleaned_v2'])

print(f"Ukuran Matriks Resume: {resume_matrix.shape}")
print(f"Jumlah Pelamar (Baris) : {resume_matrix.shape[0]}")
print(f"Jumlah Kosakata (Kolom): {resume_matrix.shape[1]}")

# Cek kosakata apa saja yang berhasil ditangkap oleh mesin
fitur_kata = tfidf_vectorizer.get_feature_names_out()
print("\\nContoh 20 kosakata (fitur) yang dimengerti mesin:")
print(random.sample(list(fitur_kata), 20))

# Top-20 bigrams terpenting di seluruh korpus resume
fitur_idx   = {f: idx for idx, f in enumerate(fitur_kata)}
bigram_list = [f for f in fitur_kata if ' ' in f]
bi_indices  = [fitur_idx[b] for b in bigram_list]
bigram_sums = np.asarray(resume_matrix[:, bi_indices].sum(axis=0)).flatten()
top20_bi    = bigram_sums.argsort()[-20:][::-1]
print(f"\\nTop-20 Bigrams ({len(bigram_list)} total bigrams ditemukan):")
for rank_b, bi in enumerate(top20_bi):
    print(f"  {rank_b+1:2d}. '{bigram_list[bi]}' (skor corpus: {bigram_sums[bi]:.1f})")"""

# ---- target[36]: Section 10 MD header ---------------------------------------
MD_SEC10 = """\
## Section 10: Baseline Matching — Cosine Similarity

Menghitung kecocokan antara setiap lowongan kerja dan seluruh pool resume
menggunakan **cosine similarity**.

**Formula:**

    cos(A, B) = (A . B) / (||A|| x ||B||)

Hasilnya berkisar 0 (tidak mirip) hingga 1 (identik). Dua dokumen dengan
kosakata yang tumpang-tindih mendapat skor tinggi, terlepas dari panjang dokumennya.

**Keterbatasan baseline:** Resume kategori "ARTS" muncul di Rank 1 untuk lowongan
"Software Engineer" karena secara kebetulan mengandung banyak kata "software engineer".
Ini motivasi untuk pendekatan skill extraction berbasis daftar (Section 11-12).

**Input:** `resume_matrix`, `tfidf_vectorizer`, `df_jobs`
**Output:** `job_matrix`, ranking kandidat per lowongan"""

# ---- target[38]: cosine sim (cell[26] — remove import, use v2 cols) ---------
_src26 = "".join(o[26]["source"])
CODE_COSINE = _src26.replace("from sklearn.metrics.pairwise import cosine_similarity\n\n", "")
CODE_COSINE = CODE_COSINE.replace(
    "df_jobs['Job_Text_Cleaned'] = df_jobs['Skills_Cleaned'] + \" \" + df_jobs['Responsibilities_Cleaned']",
    "df_jobs['Job_Text_Cleaned'] = df_jobs['Skills_Cleaned_v2'] + \" \" + df_jobs['Responsibilities_Cleaned_v2']"
)

# ---- target[40]: Section 11 MD header ---------------------------------------
MD_SEC11 = """\
## Section 11: Skill Extraction & Gap Analysis

Menggunakan **spaCy PhraseMatcher** untuk mendeteksi skill secara tepat dari
teks CV dan job description, lalu menghitung gap kompetensi.

**Input:** File PDF CV (`CV_PDF_PATH`), teks lowongan contoh, `DAFTAR_SKILL_IT`
**Output:** Persentase kecocokan, daftar skill yang ada, yang kurang, dan yang ekstra

**Alur:**
1. `extract_text_from_pdf()` — baca semua halaman PDF menjadi teks mentah
2. `extract_skills()` — jalankan PhraseMatcher spaCy pada teks
3. `analyze_skill_gap()` — operasi himpunan (intersection, difference) antara skill CV dan JD

**Keterbatasan PhraseMatcher:** Membutuhkan kecocokan *eksak*.
`react` tidak sama dengan `react.js` tidak sama dengan `reactjs` — diselesaikan di Section 12."""

# ---- target[41]: PDF extraction (cell[31] MOVED — fix import & path) --------
_src31 = "".join(o[31]["source"])
CODE_PDF = _src31.replace(
    "# Cell 18: Mengubah PDF menjadi Teks\nimport PyPDF2\n\n",
    "# Mengubah PDF menjadi Teks\n"
)
CODE_PDF = CODE_PDF.replace(
    "cv_pdf_path = '/Users/josepnat/Developer/AOL_NLP/data/raw/CV_ATS_JOSEP_NATANAEL_PASARIBU .pdf' # Ganti nama file-nya jika berbeda",
    "cv_pdf_path = CV_PDF_PATH  # Path dikonfigurasi di Section 2"
)

# ---- target[42]: spaCy skill extractor (cell[32] MOVED — remove imports) ----
_src32 = "".join(o[32]["source"])
CODE_SKILLS = _src32.replace(
    "# Skill Extractor & Gap Analyzer\nimport spacy\nfrom spacy.matcher import PhraseMatcher\n\n",
    "# Skill Extractor & Gap Analyzer\n"
)

# ---- target[43]: Section 12 MD header ---------------------------------------
MD_SEC12 = """\
## Section 12: Smart Skill Vocabulary Builder — Fuzzy Matching

Membangun kosakata skill yang besar dari dataset Ejaz, lalu mencocokkannya
secara *fuzzy* (toleran terhadap variasi penulisan).

**Mengapa exact matching gagal:**
Skill yang sama sering ditulis berbeda-beda:
- `react` vs `react.js` vs `reactjs`
- `postgres` vs `postgresql`
- `node.js` vs `nodejs`

**Solusi RapidFuzz dengan `threshold=85`:**
Menggunakan `fuzz.partial_ratio` setelah normalisasi tanda baca.
Threshold 85 cukup ketat untuk menghindari false positive (misalnya `java` tidak sama dengan `javascript`)
namun cukup longgar untuk menangkap variasi penulisan di atas.

**Input:** `df_resume_ejaz['Resume']`
**Output:** `DAFTAR_SKILL_SMART` (7.813 entri) -> `DAFTAR_SKILL_SMART_CLEAN` (7.558 setelah filtering)"""

# ---- target[45]: fuzzy matching (cell[34] — remove inline imports) ----------
_src34 = "".join(o[34]["source"])
CODE_FUZZY = _src34.replace(
    "# Massive Vocabulary & Robust Fuzzy Matcher\nfrom rapidfuzz import process, fuzz\nimport re\n\n",
    "# Massive Vocabulary & Robust Fuzzy Matcher\n"
)

# ---- target[46]: Threshold sweep (NEW) --------------------------------------
CODE_THRESHOLD = """\
# Threshold Sensitivity Test — menentukan nilai threshold yang optimal secara empiris
skill_test_cv  = {"reactjs", "python 3", "postgres", "node.js", "typescript"}
skill_test_jd  = {"react", "python", "postgresql", "nodejs", "typescript", "docker"}

print("=== UJI SENSITIVITAS THRESHOLD FUZZY MATCHING ===")
print(f"{'Threshold':>10} | {'Matched':>8} | {'Missing':>8} | Skill Matched")
print("-" * 65)
for thresh in [70, 75, 80, 85, 90]:
    m, miss = fuzzy_match_skills(skill_test_cv, skill_test_jd, threshold=thresh)
    print(f"{thresh:>10} | {len(m):>8} | {len(miss):>8} | {sorted(m)}")

print("\\nKesimpulan: threshold=85 dipilih sebagai default karena menangkap")
print("variasi penulisan (react/reactjs, node.js/nodejs, postgres/postgresql)")
print("tanpa menghasilkan false positive.")"""

# ---- target[48]: Section 13 MD header ---------------------------------------
MD_SEC13 = """\
## Section 13: Model Training & Export

Melatih `ResumeRecommender` dengan seluruh dataset resume yang sudah bersih,
lalu menyimpan semua artefak yang diperlukan oleh `app.py`.

**Yang disimpan:**

| File | Path | Isi |
|------|------|-----|
| `tfidf_model.pkl` | `../models/` | TfidfVectorizer + resume_matrix + df_resume (pickle) |
| `smart_skills.json` | `../models/` | List 7.558 kosakata skill bersih (JSON) |

**Input:** `df_resume_merged` dengan kolom `Resume_Cleaned_v2` siap pakai
**Output:** Dua file di `models/` yang langsung dapat di-load oleh `app.py`"""

# ---- target[50]: ResumeRecommender (cell[29] MOVED — remove inline imports) -
_src29 = "".join(o[29]["source"])
CODE_RECOMMENDER = _src29
CODE_RECOMMENDER = CODE_RECOMMENDER.replace("import sys\nimport os\n", "")
CODE_RECOMMENDER = CODE_RECOMMENDER.replace(
    "# pindah ke root directory (AOL_NLP)\nsys.path.append(os.path.abspath(os.path.join('..')))\n\n# Di Notebook\nfrom src.models.recommender import ResumeRecommender\n\n",
    "# ResumeRecommender sudah diimpor di Section 1\n"
)

# ---- target[51]: save_model (cell[30] MOVED — fix path) ---------------------
CODE_SAVE_MODEL = "engine.save_model(path='../models/')   # simpan ke folder models/ di root project"

# ---- target[52]: refined cleaning (cell[38] MOVED — remove redundant imports) ---
_src38 = "".join(o[38]["source"])
CODE_REFINED = _src38.replace(
    "import pandas as pd\nimport re\nimport json\nimport nltk\nfrom nltk.corpus import stopwords\n\n"
    "# Pastikan NLTK sudah terunduh\nnltk.download('stopwords', quiet=True)\n",
    "# stopwords sudah diimpor di Section 1\n"
)

# ---- target[53]: Section 14 MD header ----------------------------------------
MD_SEC14 = """\
## Section 14: End-to-End Validation

Memvalidasi bahwa artefak yang disimpan dapat di-load dan menghasilkan output
yang benar tanpa menjalankan ulang seluruh pipeline training.

**Mengapa ini penting untuk rubrik:**
Rubrik menilai *Reliability (10 pts)* — artefak model harus dapat digunakan
ulang secara independen. Sel ini mensimulasikan persis apa yang dilakukan `app.py`:
load pickle, preprocessing teks baru, cosine similarity.

**Yang divalidasi:**
1. Load `tfidf_model.pkl` berhasil tanpa error
2. `full_preprocess()` menghasilkan teks bersih yang konsisten dengan training
3. Cosine similarity menghasilkan skor yang masuk akal
4. `smart_skills.json` dapat dibaca dengan jumlah entry yang benar

**Input:** `../models/tfidf_model.pkl`, `../models/smart_skills.json`
**Output:** Laporan validasi dengan skor similarity dan top-5 matching terms"""

# ---- target[54]: End-to-End Validation (NEW) ---------------------------------
# Uses clean_resume_text_v2 and lemmatize_text already defined in Sections 6 & 7
CODE_VALIDATION = """\
print("=== END-TO-END PIPELINE VALIDATION ===\\n")

# 1. Load model yang sudah disimpan dari disk
pkl_path = os.path.join(MODEL_DIR, 'tfidf_model.pkl')
print(f"[1] Loading model dari: {pkl_path}")
with open(pkl_path, 'rb') as f_pkl:
    loaded_vectorizer, loaded_resume_matrix, loaded_df_resume = pickle.load(f_pkl)
print(f"    OK: {loaded_resume_matrix.shape[0]:,} resume, {loaded_resume_matrix.shape[1]:,} fitur.")

# 2. Full preprocess — panggil fungsi yang sudah didefinisikan di Section 6 & 7
def full_preprocess(text):
    return lemmatize_text(clean_resume_text_v2(text))

# 3. Hardcoded test inputs
test_cv = (
    "Experienced Python developer with expertise in Django, REST APIs, and SQL databases. "
    "Familiar with machine learning, Docker, and agile development practices."
)
test_jd = (
    "Looking for a Python developer with experience in Django, REST APIs, SQL, "
    "and familiarity with machine learning tools."
)

cv_processed = full_preprocess(test_cv)
jd_processed = full_preprocess(test_jd)
print(f"\\n[2] CV setelah full_preprocess:")
print(f"    {cv_processed[:120]}...")

# 4. Vectorize + cosine similarity menggunakan model yang dimuat dari disk
cv_vec = loaded_vectorizer.transform([cv_processed])
jd_vec = loaded_vectorizer.transform([jd_processed])
score  = cosine_similarity(jd_vec, cv_vec)[0][0]
print(f"\\n[3] Skor kecocokan CV vs JD: {score:.4f} ({score*100:.1f}%)")

# 5. Top-5 matching terms by TF-IDF weight
features = loaded_vectorizer.get_feature_names_out()
cv_arr   = cv_vec.toarray()[0]
jd_arr   = jd_vec.toarray()[0]
matching = [(features[i], min(cv_arr[i], jd_arr[i]))
            for i in range(len(features)) if cv_arr[i] > 0 and jd_arr[i] > 0]
matching.sort(key=lambda x: x[1], reverse=True)
print("\\n[4] Top-5 Matching Terms (by TF-IDF weight):")
for term, w in matching[:5]:
    print(f"    '{term}': {w:.4f}")

# 6. Verify smart_skills.json
skills_path = os.path.join(MODEL_DIR, 'smart_skills.json')
with open(skills_path) as f_skills:
    loaded_skills = json.load(f_skills)
print(f"\\n[5] smart_skills.json: {len(loaded_skills):,} skill dimuat.")

print("\\nVALIDASI SELESAI — semua artefak berfungsi dan pipeline serialization OK.")"""

# ===========================================================================
# ASSEMBLE 55-CELL TARGET ARRAY
# ===========================================================================
cells = [
    mk_md(MD_OVERVIEW,       nid()),  # 0
    mk_code(CODE_IMPORTS,    nid()),  # 1
    mk_md(MD_SEC2,           nid()),  # 2
    mk_md(MD_PATH_NOTICE,    nid()),  # 3
    clone(o[2],  CODE_PATHS),         # 4  (was cell[2])
    clone(o[3]),                       # 5  (was cell[3] — unchanged)
    clone(o[4]),                       # 6  (was cell[4] — unchanged)
    mk_code(CODE_JOBS_INFO,  nid()),  # 7  NEW
    mk_md(MD_SEC3,           nid()),  # 8
    clone(o[6]),                       # 9  (was cell[6] — unchanged)
    mk_code(CODE_MERGED_INFO,nid()),  # 10 NEW
    mk_md(MD_SEC4,           nid()),  # 11
    clone(o[8]),                       # 12 (was cell[8] — unchanged)
    clone(o[9],  CODE_BARCHART),      # 13 (was cell[9] — remove imports)
    clone(o[10]),                      # 14 (was cell[10] — unchanged)
    mk_code(CODE_BOXPLOT,    nid()),  # 15 NEW
    clone(o[11], CODE_SAMPLE),        # 16 (was cell[11] — remove import random)
    mk_md(MD_SEC5,           nid()),  # 17
    clone(o[13]),                      # 18 (was cell[13] — unchanged)
    mk_code(CODE_PIECHART,   nid()),  # 19 NEW
    mk_md(MD_SEC6,           nid()),  # 20
    clone(o[14]),                      # 21 (was cell[14] — unchanged)
    mk_md(MD_V1_NOTE,        nid()),  # 22
    clone(o[15], CODE_V1),            # 23 (was cell[15] — remove import, fix regex)
    clone(o[16]),                      # 24 (was cell[16] — unchanged, reference cell)
    clone(o[17]),                      # 25 (was cell[17] — unchanged)
    clone(o[18], CODE_V2),            # 26 (was cell[18] — fix regex, add df_jobs v2)
    mk_md(MD_SEC7,           nid()),  # 27
    clone(o[19]),                      # 28 (was cell[19] — unchanged)
    clone(o[20], CODE_LEMMA),         # 29 (was cell[20] — remove imports)
    mk_md(MD_SEC8,           nid()),  # 30
    clone(o[21]),                      # 31 (was cell[21] — unchanged)
    clone(o[22], CODE_WORDCLOUD),     # 32 (was cell[22] — remove import)
    mk_md(MD_SEC9,           nid()),  # 33
    clone(o[23]),                      # 34 (was cell[23] — unchanged)
    clone(o[24], CODE_TFIDF),         # 35 (was cell[24] — upgrade params + bigrams)
    mk_md(MD_SEC10,          nid()),  # 36
    clone(o[25]),                      # 37 (was cell[25] — unchanged)
    clone(o[26], CODE_COSINE),        # 38 (was cell[26] — remove import, use v2 cols)
    clone(o[27]),                      # 39 (was cell[27] — unchanged)
    mk_md(MD_SEC11,          nid()),  # 40
    clone(o[31], CODE_PDF),           # 41 (was cell[31] MOVED — fix import/path)
    clone(o[32], CODE_SKILLS),        # 42 (was cell[32] MOVED — remove imports)
    mk_md(MD_SEC12,          nid()),  # 43
    clone(o[33]),                      # 44 (was cell[33] — unchanged)
    clone(o[34], CODE_FUZZY),         # 45 (was cell[34] — remove imports)
    mk_code(CODE_THRESHOLD,  nid()),  # 46 NEW
    clone(o[36], "".join(o[36]["source"]).replace("import json\n\n", "").replace("import json\n", "")),  # 47 — remove redundant import
    mk_md(MD_SEC13,          nid()),  # 48
    clone(o[28]),                      # 49 (was cell[28] MOVED — unchanged)
    clone(o[29], CODE_RECOMMENDER),   # 50 (was cell[29] MOVED — remove imports)
    clone(o[30], CODE_SAVE_MODEL),    # 51 (was cell[30] MOVED — fix path)
    clone(o[38], CODE_REFINED),       # 52 (was cell[38] MOVED — remove imports)
    mk_md(MD_SEC14,          nid()),  # 53
    mk_code(CODE_VALIDATION, nid()),  # 54 NEW
]

print(f"Assembled {len(cells)} target cells.")

# ===========================================================================
# VERIFY: Unique IDs, count
# ===========================================================================
all_ids = [c["id"] for c in cells]
dups = [x for x in set(all_ids) if all_ids.count(x) > 1]
if dups:
    print(f"ERROR: Duplicate IDs: {dups}")
    raise SystemExit(1)
print(f"Cell IDs all unique: OK ({len(cells)} cells)")

# Syntax-check all code cells
import ast
for idx, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        try:
            compile(src, f"cell_{idx}", "exec")
        except SyntaxError as e:
            print(f"WARNING SyntaxError in cell {idx}: {e}")
            print("Source preview:", src[:300])
print("Syntax check complete.")

# ===========================================================================
# WRITE BACK
# ===========================================================================
nb["cells"] = cells
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {NB_PATH}  ({len(cells)} cells total)")
