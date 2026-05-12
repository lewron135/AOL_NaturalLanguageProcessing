import pickle
from pathlib import Path

import spacy
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from spacy import displacy

from src.extraction.filters import semantic_relevance_filter
from src.utils.preprocessor import full_preprocess, pdf_clean

TECH_TERM_LOCK = [
    "Python", "Java", "SQL", "MariaDB", "PHP", "C", "C++", "JavaScript", "TypeScript", "Go",
    "Figma", "Canva", "Adobe Premiere", "DaVinci Resolve", "Machine Learning",
    "Deep Learning", "Artificial Intelligence", "Computer Vision", "NLP",
    "Data Science", "TensorFlow", "PyTorch", "Scikit-learn", "Docker", "Kubernetes",
    "REST API", "Git", "Linux", "React", "Node.js", "FastAPI", "Flask", "Streamlit",
    "BERT", "Transformer", "RapidMiner", "Pandas", "NumPy", "Tableau", "Power BI",
]

NER_OPTIONS = {
    "ents": ["SKILL", "ORG", "GPE", "PERSON"],
    "colors": {
        "SKILL": "#2d4a1e",
        "ORG": "#1e2d4a",
        "GPE": "#3a2d1e",
        "PERSON": "#2d1e3a",
    },
}


@st.cache_resource
def load_nlp() -> spacy.Language:
    nlp = spacy.load("en_core_web_md")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "SKILL", "pattern": [{"LOWER": t.lower()} for t in term.split()], "id": term}
            for term in TECH_TERM_LOCK
        ]
        ruler.add_patterns(patterns)
    return nlp


@st.cache_resource
def load_sbert() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_tfidf():
    path = Path(__file__).parent.parent.parent / "models" / "tfidf_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_features(doc, sbert_model) -> set:
    entities = {
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"SKILL", "PRODUCT", "WORK_OF_ART"} and len(ent.text.strip()) > 1
    }
    noun_chunks = {
        chunk.text.strip()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) >= 2
        and len(chunk.text.strip()) > 3
        and not chunk.text.strip().isdigit()
        and not chunk.root.is_stop
    }
    combined = entities | noun_chunks
    return semantic_relevance_filter(combined, sbert_model)


def calculate_semantic_score(cv_features: set, jd_features: set, sbert_model) -> tuple[float, list]:
    if not jd_features or not cv_features:
        return 0.0, []

    cv_list = list(cv_features)
    jd_list = list(jd_features)

    cv_embeddings = sbert_model.encode(cv_list, convert_to_tensor=True)
    jd_embeddings = sbert_model.encode(jd_list, convert_to_tensor=True)

    cosine_scores = util.cos_sim(jd_embeddings, cv_embeddings)

    details = []
    total_sim = 0.0

    for i, jd_skill in enumerate(jd_list):
        max_score, max_idx = torch.max(cosine_scores[i], dim=0)
        score = max_score.item()
        best_match = cv_list[max_idx]

        if score > 0.65:
            details.append({"jd": jd_skill, "cv": best_match, "score": score})
            total_sim += score
        else:
            details.append({"jd": jd_skill, "cv": None, "score": score})

    final_score = (total_sim / len(jd_list)) * 100
    return final_score, details


def tfidf_similarity(cv_text: str, jd_text: str, tfidf_vectorizer) -> float | None:
    if tfidf_vectorizer is None:
        return None
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    cv_clean = full_preprocess(cv_text)
    jd_clean = full_preprocess(jd_text)

    try:
        vectors = tfidf_vectorizer.transform([cv_clean, jd_clean])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(score)
    except Exception:
        return None


def render_ner_html(doc) -> str:
    return displacy.render(doc, style="ent", options=NER_OPTIONS, jupyter=False)


def analyze(cv_text_raw: str, jd_text_raw: str, nlp, sbert_model, tfidf_vectorizer=None) -> dict:
    clean_cv = pdf_clean(cv_text_raw)
    clean_jd = pdf_clean(jd_text_raw)

    doc_cv = nlp(clean_cv)
    doc_jd = nlp(clean_jd)

    cv_features = extract_features(doc_cv, sbert_model)
    jd_features = extract_features(doc_jd, sbert_model)

    score, details = calculate_semantic_score(cv_features, jd_features, sbert_model)
    tfidf_score = tfidf_similarity(clean_cv, clean_jd, tfidf_vectorizer)

    return {
        "score": score,
        "details": details,
        "tfidf_score": tfidf_score,
        "cv_features": cv_features,
        "jd_features": jd_features,
        "doc_cv": doc_cv,
        "ner_html": render_ner_html(doc_cv),
    }
