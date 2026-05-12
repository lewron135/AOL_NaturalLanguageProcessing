import streamlit as st
import PyPDF2
import spacy
from spacy import displacy
from sentence_transformers import SentenceTransformer, util
import torch
import re

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="CV Analyzer — NLP Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CORE ENGINE: SEMANTIC HYBRID SYSTEM
# ==========================================

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_md")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Custom Entity Ruler to prevent mislabeling technical terms
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        tech_terms = [
            "Python", "Java", "SQL", "MariaDB", "PHP", "C", "C++", "JavaScript",
            "Figma", "Canva", "Capcut", "Adobe Premier", "Davinci Resolve",
            "Machine Learning", "Deep Learning", "Artificial Intelligence", "AI",
            "MobileNetV2", "Computer Vision", "NLP", "Data Science", "Computer Science",
            "Jaccard Similarity", "Content-Based Filtering", "RapidMiner",
            "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
            "Docker", "Kubernetes", "REST API", "GraphQL", "Git", "Linux",
            "React", "Vue", "Angular", "Node.js", "FastAPI", "Flask", "Django",
            "Tableau", "Power BI", "Excel", "MATLAB", "R", "Hadoop", "Spark"
        ]
        patterns = [
            {
                "label": "SKILL",
                "pattern": [{"LOWER": t.lower()} for t in s.split()],
                "id": s
            }
            for s in tech_terms
        ]
        ruler.add_patterns(patterns)

    return nlp, sbert_model


def advanced_text_cleaning(text):
    """
    Fix common PDF extraction artifacts:
    - Spaced-out letters: 'C o n s t r u c t i o n' -> 'Construction'
    - Normalize excess whitespace
    """
    text = re.sub(r'(?<=\b\w)\s(?=\w\b)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==========================================
# 2. SEMANTIC RELEVANCE FILTERING (CORE FIX)
# ==========================================

# These anchor phrases define "Technical Competency" in semantic space.
# SBERT will use these to filter out administrative noise dynamically.
# This is scalable — no rigid keyword lists needed.
TECHNICAL_ANCHOR_PHRASES = [
    "programming language",
    "software development skill",
    "technical competency",
    "engineering framework",
    "machine learning algorithm",
    "data analysis tool",
    "database management",
    "development methodology",
    "computer science concept",
    "software engineering practice",
]

ADMINISTRATIVE_ANCHOR_PHRASES = [
    "employee benefit",
    "work schedule",
    "office location",
    "salary compensation",
    "paid time off",
    "health insurance",
    "job requirement administrative",
    "company policy",
    "employment terms",
]


@st.cache_resource
def get_anchor_embeddings(_sbert_model):
    """Pre-compute anchor embeddings once and cache them."""
    tech_embeddings = _sbert_model.encode(TECHNICAL_ANCHOR_PHRASES, convert_to_tensor=True)
    admin_embeddings = _sbert_model.encode(ADMINISTRATIVE_ANCHOR_PHRASES, convert_to_tensor=True)
    return tech_embeddings, admin_embeddings


def semantic_relevance_filter(entities: set, sbert_model, threshold: float = 0.30) -> set:
    """
    Filter entities by semantic proximity to 'Technical Competency' anchors.

    An entity passes if:
    - Its max similarity to any TECHNICAL anchor > threshold, AND
    - Its technical score > administrative score (not more 'admin-like' than 'tech-like')

    This removes noise like 'PTO', 'health insurance', 'location', '8-to-65'
    without relying on hardcoded blacklists.

    Args:
        entities: Raw set of candidate skill strings
        sbert_model: Loaded SBERT model
        threshold: Minimum cosine similarity to be considered technical (0.30 is permissive)

    Returns:
        Filtered set of technically-relevant entities
    """
    if not entities:
        return set()

    entity_list = list(entities)
    entity_embeddings = sbert_model.encode(entity_list, convert_to_tensor=True)

    tech_embeddings, admin_embeddings = get_anchor_embeddings(sbert_model)

    # Similarity to technical anchors: shape (num_entities, num_tech_anchors)
    tech_sim = util.cos_sim(entity_embeddings, tech_embeddings)
    # Similarity to administrative anchors: shape (num_entities, num_admin_anchors)
    admin_sim = util.cos_sim(entity_embeddings, admin_embeddings)

    # Max score across all anchors per entity
    max_tech_scores = tech_sim.max(dim=1).values
    max_admin_scores = admin_sim.max(dim=1).values

    filtered = set()
    for i, entity in enumerate(entity_list):
        tech_score = max_tech_scores[i].item()
        admin_score = max_admin_scores[i].item()

        # Pass filter if: sufficiently technical AND more technical than administrative
        if tech_score >= threshold and tech_score > admin_score:
            filtered.add(entity)

    return filtered


def extract_features(doc, sbert_model) -> set:
    """
    Extract candidate features from a spaCy doc and apply semantic filtering.

    Strategy:
    1. Pull named entities with technical labels (SKILL, ORG, PRODUCT)
    2. Pull noun phrases that are not stopwords and have enough tokens (>=2)
       to avoid single-word noise like 'location', 'role', 'PTO'
    3. Apply semantic relevance filter to the combined pool
    """
    # Named entities with technical labels
    entities = set([
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ["SKILL", "PRODUCT", "WORK_OF_ART"]
           and len(ent.text.strip()) > 1
    ])

    # Noun chunks: multi-word phrases only (avoids single generic nouns)
    # Also filter out purely numeric or very short strings
    noun_chunks = set([
        chunk.text.strip()
        for chunk in doc.noun_chunks
        if not chunk.root.is_stop
           and len(chunk.text.strip()) > 3
           and not chunk.text.strip().isdigit()
           and len(chunk.text.split()) >= 2  # at least 2-word phrases
    ])

    combined = entities.union(noun_chunks)

    # Apply semantic relevance filter — this is where noise is removed
    filtered = semantic_relevance_filter(combined, sbert_model)

    return filtered


def calculate_semantic_score(cv_features: set, jd_features: set, sbert_model):
    """
    Match CV features against JD requirements using cosine similarity.
    Returns a score (0-100) and detailed match breakdown.
    """
    if not jd_features or not cv_features:
        return 0.0, []

    cv_list = list(cv_features)
    jd_list = list(jd_features)

    cv_embeddings = sbert_model.encode(cv_list, convert_to_tensor=True)
    jd_embeddings = sbert_model.encode(jd_list, convert_to_tensor=True)

    cosine_scores = util.cos_sim(jd_embeddings, cv_embeddings)

    matched_details = []
    total_sim = 0.0

    for i, jd_skill in enumerate(jd_list):
        max_score, max_idx = torch.max(cosine_scores[i], dim=0)
        score = max_score.item()
        best_match = cv_list[max_idx]

        if score > 0.65:
            matched_details.append({"jd": jd_skill, "cv": best_match, "score": score})
            total_sim += score
        else:
            matched_details.append({"jd": jd_skill, "cv": None, "score": score})

    final_score = (total_sim / len(jd_list)) * 100
    return final_score, matched_details


# ==========================================
# 3. UI STYLING — EDITORIAL / REFINED DARK
# ==========================================

def apply_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0d0d0d;
            color: #e8e4dd;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #1e1e1e;
        }
        [data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

        /* ── Hide Streamlit branding ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0d0d0d; }
        ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

        /* ── Typography ── */
        .page-title {
            font-family: 'Instrument Serif', serif;
            font-size: 2.8rem;
            font-weight: 400;
            color: #e8e4dd;
            letter-spacing: -0.5px;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }
        .page-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            color: #4a4a4a;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 2.5rem;
        }
        .section-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 0.6rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1e1e1e;
        }

        /* ── Cards ── */
        .card {
            background: #111111;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 16px;
        }
        .card-accent-green {
            border-left: 3px solid #2d6a4f;
        }
        .card-accent-red {
            border-left: 3px solid #6b2737;
        }

        /* ── Score block ── */
        .score-block {
            background: #111111;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 48px 32px;
            text-align: center;
            margin: 24px 0;
            position: relative;
            overflow: hidden;
        }
        .score-block::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #c8b560, transparent);
        }
        .score-number {
            font-family: 'Instrument Serif', serif;
            font-size: 5.5rem;
            font-weight: 400;
            color: #c8b560;
            line-height: 1;
            margin: 0;
        }
        .score-unit {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            color: #555;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-top: 8px;
        }

        /* ── Match items ── */
        .match-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #161616;
            font-size: 0.85rem;
        }
        .match-row:last-child { border-bottom: none; }
        .match-skill { color: #c8b560; font-weight: 500; }
        .match-cv { color: #555; font-size: 0.75rem; font-family: 'DM Mono', monospace; }
        .gap-skill { color: #9a5a5a; font-weight: 400; }
        .match-badge {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            background: #1a2a1a;
            color: #5a9a5a;
            padding: 2px 8px;
            border-radius: 3px;
            flex-shrink: 0;
        }

        /* ── NER display ── */
        .ner-container {
            background: #111111;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 24px;
            line-height: 2.4;
            font-size: 0.88rem;
            color: #a0998f;
        }

        /* ── Button ── */
        .stButton > button {
            background: #c8b560 !important;
            color: #0d0d0d !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.7rem !important;
            font-weight: 500 !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            border: none !important;
            border-radius: 6px !important;
            height: 48px !important;
            width: 100% !important;
            transition: opacity 0.2s ease !important;
        }
        .stButton > button:hover {
            opacity: 0.85 !important;
            transform: none !important;
            box-shadow: none !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploaderDropzone"] {
            background: #111111 !important;
            border: 1px dashed #2a2a2a !important;
            border-radius: 8px !important;
        }

        /* ── Text area ── */
        textarea {
            background: #111111 !important;
            border: 1px solid #1e1e1e !important;
            border-radius: 6px !important;
            color: #e8e4dd !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.88rem !important;
        }
        textarea:focus {
            border-color: #c8b560 !important;
            box-shadow: none !important;
        }

        /* ── Sidebar nav ── */
        .nav-item {
            display: block;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.82rem;
            color: #555;
            cursor: pointer;
            margin-bottom: 4px;
            transition: all 0.15s;
            text-decoration: none;
            letter-spacing: 0.3px;
        }
        .nav-item:hover, .nav-item.active {
            background: #1a1a1a;
            color: #e8e4dd;
        }
        .nav-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem;
            color: #333;
            letter-spacing: 2px;
            text-transform: uppercase;
            padding: 0 14px;
            margin-bottom: 8px;
            margin-top: 20px;
        }

        /* ── Divider ── */
        hr { border: none; border-top: 1px solid #1a1a1a; margin: 20px 0; }

        /* ── Spinner ── */
        [data-testid="stSpinner"] { color: #c8b560 !important; }

        /* ── Metric tag ── */
        .metric-tag {
            display: inline-block;
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            color: #555;
            border: 1px solid #1e1e1e;
            padding: 3px 10px;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 6px;
        }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 4. PAGE RENDERING
# ==========================================

def page_about():
    st.markdown('<p class="page-title">Methodology</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">How this system works</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Architecture</div>
        <p style="color:#7a7a7a; font-size:0.9rem; line-height:1.8; margin:0;">
            This system combines two NLP components: <strong style="color:#e8e4dd;">spaCy</strong>
            for named entity recognition and noun phrase extraction,
            and <strong style="color:#e8e4dd;">Sentence-BERT (SBERT)</strong> for computing
            semantic similarity between extracted features and job requirements.
            Unlike traditional keyword matching, this architecture understands
            the conceptual meaning behind terms rather than relying on exact string matches.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Semantic Relevance Filtering</div>
        <p style="color:#7a7a7a; font-size:0.9rem; line-height:1.8; margin:0;">
            Raw noun phrase extraction produces significant noise — administrative terms such as
            <em>paid time off</em>, <em>health insurance</em>, and <em>office location</em>
            get captured alongside genuine technical skills. This system resolves that by
            encoding both candidate entities and a set of <strong style="color:#e8e4dd;">
            semantic anchor phrases</strong> (representing "technical competency" and
            "administrative information") into vector space, then filtering out anything
            that is semantically closer to the administrative cluster than the technical one.
            No hardcoded blacklists — the filter generalizes automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Scoring Method</div>
        <p style="color:#7a7a7a; font-size:0.9rem; line-height:1.8; margin:0;">
            Each requirement extracted from the job description is compared against all
            CV features using <strong style="color:#e8e4dd;">cosine similarity</strong>.
            The best-matching CV feature is paired with each requirement.
            A match is counted when similarity exceeds 0.65 (65%).
            The final score is the mean similarity across all requirements, scaled to 100.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Evaluation Metrics</div>
        <div style="margin-top:12px;">
            <span class="metric-tag">Cosine Similarity</span>
            <span class="metric-tag">Named Entity Recognition</span>
            <span class="metric-tag">Noun Phrase Chunking</span>
            <span class="metric-tag">Sentence Transformers</span>
            <span class="metric-tag">Semantic Vector Space</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_analyzer(nlp, sbert_model):
    st.markdown('<p class="page-title">CV Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a resume and enter job requirements</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Resume — PDF</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop PDF here",
            type=["pdf"],
            label_visibility="collapsed"
        )

    with col_right:
        st.markdown('<div class="section-label">Job Requirements</div>', unsafe_allow_html=True)
        jd_input = st.text_area(
            "Paste the job description or qualifications",
            height=140,
            placeholder="e.g. Required: Python, 3+ years experience in machine learning, proficiency in SQL...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Analysis")

    if run:
        if not uploaded_file:
            st.warning("Please upload a PDF resume.")
            return
        if not jd_input.strip():
            st.warning("Please enter the job requirements.")
            return

        with st.spinner("Extracting and filtering features..."):
            # 1. Extract and clean text
            reader = PyPDF2.PdfReader(uploaded_file)
            raw_cv_text = " ".join([
                page.extract_text() or ""
                for page in reader.pages
            ])
            clean_cv = advanced_text_cleaning(raw_cv_text)
            clean_jd = advanced_text_cleaning(jd_input)

            # 2. NER + noun phrase extraction (spaCy)
            doc_cv = nlp(clean_cv)
            doc_jd = nlp(clean_jd)

            # 3. Feature extraction with semantic relevance filtering
            cv_features = extract_features(doc_cv, sbert_model)
            jd_features = extract_features(doc_jd, sbert_model)

            # 4. Semantic scoring
            score, details = calculate_semantic_score(cv_features, jd_features, sbert_model)

        # ── Score display ──
        label = "Strong Match" if score >= 70 else "Moderate Match" if score >= 45 else "Weak Match"
        st.markdown(f"""
        <div class="score-block">
            <p class="score-number">{score:.1f}<span style="font-size:2rem; color:#555;">%</span></p>
            <p class="score-unit">Semantic Match Score — {label}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Match breakdown ──
        col_match, col_gap = st.columns(2, gap="large")

        strong = [d for d in details if d["score"] > 0.75]
        moderate = [d for d in details if 0.50 <= d["score"] <= 0.75]
        gaps = [d for d in details if d["score"] < 0.50]

        with col_match:
            st.markdown(f"""
            <div class="card card-accent-green">
                <div class="section-label">Strong Matches — {len(strong)}</div>
            """, unsafe_allow_html=True)
            if strong:
                for d in strong:
                    cv_hint = f'<span class="match-cv">{d["cv"]}</span>' if d["cv"] else ''
                    st.markdown(f"""
                    <div class="match-row">
                        <span class="match-skill">{d['jd']}</span>
                        <span class="match-badge">{d['score']:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#333; font-size:0.85rem;">No strong matches found.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if moderate:
                st.markdown(f"""
                <div class="card" style="margin-top:0;">
                    <div class="section-label">Partial Matches — {len(moderate)}</div>
                """, unsafe_allow_html=True)
                for d in moderate:
                    st.markdown(f"""
                    <div class="match-row">
                        <span style="color:#8a8060; font-size:0.85rem;">{d['jd']}</span>
                        <span class="match-badge" style="background:#1a1a0a; color:#8a8060;">{d['score']:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_gap:
            st.markdown(f"""
            <div class="card card-accent-red">
                <div class="section-label">Requirements Gap — {len(gaps)}</div>
            """, unsafe_allow_html=True)
            if gaps:
                for d in gaps:
                    st.markdown(f"""
                    <div class="match-row">
                        <span class="gap-skill">{d['jd']}</span>
                        <span style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#3a3a3a;">not found</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#2d6a4f; font-size:0.85rem;">No significant gaps detected.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature counts
            st.markdown(f"""
            <div class="card" style="margin-top:0;">
                <div class="section-label">Extraction Stats</div>
                <div class="match-row">
                    <span style="color:#555; font-size:0.85rem;">CV features (after filter)</span>
                    <span style="color:#e8e4dd; font-family:'DM Mono',monospace; font-size:0.8rem;">{len(cv_features)}</span>
                </div>
                <div class="match-row">
                    <span style="color:#555; font-size:0.85rem;">JD requirements (after filter)</span>
                    <span style="color:#e8e4dd; font-family:'DM Mono',monospace; font-size:0.8rem;">{len(jd_features)}</span>
                </div>
                <div class="match-row">
                    <span style="color:#555; font-size:0.85rem;">Requirements evaluated</span>
                    <span style="color:#e8e4dd; font-family:'DM Mono',monospace; font-size:0.8rem;">{len(details)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── NER Highlighting ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Named Entity Highlighting — Resume</div>', unsafe_allow_html=True)
        ner_options = {
            "ents": ["SKILL", "ORG", "GPE", "PERSON"],
            "colors": {
                "SKILL": "#2d4a1e",
                "ORG": "#1e2d4a",
                "GPE": "#3a2d1e",
                "PERSON": "#2d1e3a",
            }
        }
        ner_html = displacy.render(doc_cv, style="ent", options=ner_options, jupyter=False)
        st.markdown(
            f'<div class="ner-container">{ner_html}</div>',
            unsafe_allow_html=True
        )


def page_metrics():
    st.markdown('<p class="page-title">System Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Performance characteristics and design choices</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Model Selection Rationale</div>
        <div class="match-row">
            <span style="color:#e8e4dd; font-size:0.85rem;">Sentence Transformer</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#c8b560;">all-MiniLM-L6-v2</span>
        </div>
        <div class="match-row">
            <span style="color:#e8e4dd; font-size:0.85rem;">NER + Chunking</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#c8b560;">spaCy en_core_web_md</span>
        </div>
        <div class="match-row">
            <span style="color:#e8e4dd; font-size:0.85rem;">Similarity Metric</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#c8b560;">Cosine Similarity</span>
        </div>
        <div class="match-row">
            <span style="color:#e8e4dd; font-size:0.85rem;">Match Threshold</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#c8b560;">0.65 (65%)</span>
        </div>
        <div class="match-row">
            <span style="color:#e8e4dd; font-size:0.85rem;">Filter Threshold</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#c8b560;">0.30 technical similarity</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Noise Filtering Design</div>
        <p style="color:#7a7a7a; font-size:0.88rem; line-height:1.8; margin:0;">
            The semantic filter uses two anchor clusters — one representing
            <strong style="color:#e8e4dd;">technical competency</strong> and one representing
            <strong style="color:#e8e4dd;">administrative information</strong> — to evaluate
            each extracted entity. Entities that score above the threshold on the technical
            cluster AND score higher on technical than administrative anchors are retained.
            This dual-cluster approach is more robust than single-sided thresholding, because
            some administrative terms can appear moderately "technical" in isolation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-label">Known Limitations</div>
        <p style="color:#7a7a7a; font-size:0.88rem; line-height:1.8; margin-bottom:10px;">
            The system may still retain some borderline terms depending on the job description wording.
            Very short CVs or poorly structured PDFs may produce fewer features.
            The 0.65 cosine similarity threshold may need adjustment for niche technical domains
            where terminology differs significantly from general technical vocabulary.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. MAIN
# ==========================================

def main():
    apply_ui()
    nlp, sbert_model = load_models()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 24px 14px 16px 14px;">
            <p style="font-family:'Instrument Serif',serif; font-size:1.4rem; color:#e8e4dd; margin:0; line-height:1.1;">
                CV Analyzer
            </p>
            <p style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#333; letter-spacing:3px; text-transform:uppercase; margin:4px 0 0 0;">
                NLP Engine v3
            </p>
        </div>
        <hr style="margin:0 0 8px 0; border-color:#1a1a1a;">
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "",
            options=["Analyzer", "Methodology", "System Evaluation"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <hr style="margin-top:auto;">
        <div style="padding: 0 14px 24px 14px;">
            <p style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#2a2a2a; letter-spacing:1px; margin:0;">
                BINUS UNIVERSITY<br>COMP6885001 — NLP<br>2025/2026
            </p>
        </div>
        """, unsafe_allow_html=True)

    if page == "Analyzer":
        page_analyzer(nlp, sbert_model)
    elif page == "Methodology":
        page_about()
    elif page == "System Evaluation":
        page_metrics()


if __name__ == "__main__":
    main()