# Smart CV Analyzer

A resume-to-job-description matching app built as the final project for COMP6885001 Natural Language Processing at BINUS University (2025/2026).

The basic idea: upload a PDF resume, paste a job description, and the app tells you how well they match — and more importantly, *what's missing*. Instead of counting exact keyword matches (which is fragile), it uses semantic similarity so it understands that "neural network experience" and "deep learning background" mean basically the same thing.

---

## What it does

- Extracts skills and technical terms from your CV using spaCy
- Extracts requirements from the job description the same way
- Compares them using SBERT (Sentence-BERT) cosine similarity
- Shows you a match score (0–100%), what matched, what didn't, and where the gaps are
- Highlights named entities (skills, orgs, people, locations) directly on the resume text
- Optional TF-IDF lexical score as a secondary signal if the model artifact is available

---

## Why not just keyword matching?

Simple keyword matching breaks easily. If your CV says "deep learning" but the job posting says "neural networks", a keyword matcher says no match. SBERT encodes both phrases into vector space and understands they're semantically close — so it correctly counts that as a match.

The harder problem is noise. Raw noun phrase extraction from job descriptions picks up a lot of garbage — "paid time off", "health insurance", "office location". These are not skills. To filter them out without a hardcoded blacklist, I encode both the extracted phrases and a set of "anchor phrases" representing technical vs administrative content, then only keep phrases that are semantically closer to the technical cluster.

---

## Project structure

```
AOL_NLP/
├── app.py              # original version (kept for reference)
├── app_v2.py           # current entry point — run this one
│
├── src/
│   ├── ui.py                       # all Streamlit rendering
│   ├── extraction/
│   │   ├── engine.py               # main NLP pipeline
│   │   └── filters.py              # semantic relevance filter
│   └── utils/
│       └── preprocessor.py         # text cleaning + lemmatization
│
├── models/
│   └── tfidf_model.pkl             # exported from the notebook
│
├── notebooks/
│   └── 01_data_merging_and_eda.ipynb
│
└── data/raw/           # datasets (excluded from git — see .gitignore)
```

---

## How to run it

**Requirements:** Python 3.10+

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd AOL_NLP

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy model
python -m spacy download en_core_web_md

# 5. Run the app
streamlit run app_v2.py
```

Then open http://localhost:8501.

> **Note on TF-IDF score:** The sidebar will show an amber badge if `models/tfidf_model.pkl` is missing. To generate it, run the notebook (`notebooks/01_data_merging_and_eda.ipynb`) all the way through — it exports the artifact at the end. The app still works without it; the TF-IDF score just won't appear.

---

## Datasets used

| Dataset | Description |
|---|---|
| Bhawal (Kaggle) | ~2,400 real resumes across 24 job categories |
| Ejaz (Kaggle) | ~960 resumes with category labels |
| Bibek (Kaggle) | 10,000 job descriptions (used for notebook baseline demo) |

All three are excluded from version control via `.gitignore` because of file size.

---

## Tech stack

| What | Why |
|---|---|
| spaCy `en_core_web_md` | NER and noun phrase extraction |
| `sentence-transformers` (all-MiniLM-L6-v2) | semantic similarity between CV and JD features |
| scikit-learn TF-IDF | lexical similarity as a secondary signal |
| PyPDF2 | PDF text extraction |
| NLTK | stopword removal and lemmatization |
| Streamlit | web UI |

---

## Known issues / limitations

- PDFs with complex layouts (multi-column, tables, heavy formatting) often produce garbled text after extraction — PyPDF2 isn't great at those
- Very short CVs produce too few features for a reliable score
- The semantic filter threshold (0.30) is tuned for general tech roles — highly specialized domains (biomedical, legal) might get over-filtered
- No multilingual support; the entire pipeline assumes English input

---

## Author

Josep Natanael Pasaribu · BINUS University · COMP6885001 NLP · 2025/2026
