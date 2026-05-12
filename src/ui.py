import streamlit as st

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0d0d0d;
        color: #e8e4dd;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #1e1e1e;
    }
    [data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

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
    .card {
        background: #111111;
        border: 1px solid #1e1e1e;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .card-accent-green { border-left: 3px solid #2d6a4f; }
    .card-accent-red { border-left: 3px solid #6b2737; }

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
    .tfidf-score {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #444;
        margin-top: 6px;
    }
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
    .ner-container {
        background: #111111;
        border: 1px solid #1e1e1e;
        border-radius: 8px;
        padding: 24px;
        line-height: 2.4;
        font-size: 0.88rem;
        color: #a0998f;
    }
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
    .stButton > button:hover { opacity: 0.85 !important; transform: none !important; box-shadow: none !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: #111111 !important;
        border: 1px dashed #2a2a2a !important;
        border-radius: 8px !important;
    }
    textarea {
        background: #111111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 6px !important;
        color: #e8e4dd !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }
    textarea:focus { border-color: #c8b560 !important; box-shadow: none !important; }
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
    .nav-item:hover, .nav-item.active { background: #1a1a1a; color: #e8e4dd; }
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
    hr { border: none; border-top: 1px solid #1a1a1a; margin: 20px 0; }
    [data-testid="stSpinner"] { color: #c8b560 !important; }
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
    .badge-green {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        background: #1a2a1a;
        color: #5a9a5a;
        padding: 3px 10px;
        border-radius: 3px;
        letter-spacing: 1px;
    }
    .badge-amber {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        background: #2a2010;
        color: #c8a030;
        padding: 3px 10px;
        border-radius: 3px;
        letter-spacing: 1px;
    }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar(tfidf_available: bool) -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 24px 14px 16px 14px;">
            <p style="font-family:'Instrument Serif',serif; font-size:1.4rem; color:#e8e4dd; margin:0; line-height:1.1;">
                CV Analyzer
            </p>
            <p style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#333; letter-spacing:3px; text-transform:uppercase; margin:4px 0 0 0;">
                NLP Engine v4
            </p>
        </div>
        <hr style="margin:0 0 8px 0; border-color:#1a1a1a;">
        """, unsafe_allow_html=True)

        badge = (
            '<span class="badge-green">TF-IDF artifact loaded</span>'
            if tfidf_available
            else '<span class="badge-amber">Run notebook to export TF-IDF</span>'
        )
        st.markdown(f'<div style="padding: 0 14px 12px 14px;">{badge}</div>', unsafe_allow_html=True)

        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "",
            options=["Analyzer", "Methodology", "System Evaluation"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="margin-top:auto;">
        <div style="padding: 0 14px 24px 14px;">
            <p style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#2a2a2a; letter-spacing:1px; margin:0;">
                BINUS UNIVERSITY<br>COMP6885001 — NLP<br>2025/2026
            </p>
        </div>
        """, unsafe_allow_html=True)

    return page


def render_analyzer_inputs() -> tuple:
    st.markdown('<p class="page-title">CV Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a resume and enter job requirements</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4], gap="large")
    with col_left:
        st.markdown('<div class="section-label">Resume — PDF</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")
    with col_right:
        st.markdown('<div class="section-label">Job Requirements</div>', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Paste the job description or qualifications",
            height=140,
            placeholder="e.g. Required: Python, 3+ years experience in machine learning, proficiency in SQL...",
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Run Analysis")
    return uploaded_file, jd_text, run_clicked


def render_score(score: float, tfidf_score: float | None) -> None:
    label = "Strong Match" if score >= 70 else "Moderate Match" if score >= 45 else "Weak Match"
    tfidf_line = ""
    if tfidf_score is not None:
        tfidf_line = f'<p class="tfidf-score">TF-IDF Lexical Score: {tfidf_score:.3f}</p>'
    st.markdown(f"""
    <div class="score-block">
        <p class="score-number">{score:.1f}<span style="font-size:2rem; color:#555;">%</span></p>
        <p class="score-unit">Semantic Match Score — {label}</p>
        {tfidf_line}
    </div>
    """, unsafe_allow_html=True)


def render_match_breakdown(details: list, cv_features: set, jd_features: set) -> None:
    strong = [d for d in details if d["score"] > 0.75]
    moderate = [d for d in details if 0.50 <= d["score"] <= 0.75]
    gaps = [d for d in details if d["score"] < 0.50]

    col_match, col_gap = st.columns(2, gap="large")

    with col_match:
        st.markdown(f"""
        <div class="card card-accent-green">
            <div class="section-label">Strong Matches — {len(strong)}</div>
        """, unsafe_allow_html=True)
        if strong:
            for d in strong:
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


def render_ner_section(ner_html: str) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Named Entity Highlighting — Resume</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ner-container">{ner_html}</div>', unsafe_allow_html=True)


def render_methodology() -> None:
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


def render_evaluation() -> None:
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
    <div class="card">
        <div class="section-label">Noise Filtering Design</div>
        <p style="color:#7a7a7a; font-size:0.88rem; line-height:1.8; margin:0;">
            The semantic filter uses two anchor clusters — one representing
            <strong style="color:#e8e4dd;">technical competency</strong> and one representing
            <strong style="color:#e8e4dd;">administrative information</strong> — to evaluate
            each extracted entity. Entities that score above the threshold on the technical
            cluster AND score higher on technical than administrative anchors are retained.
            This dual-cluster approach is more robust than single-sided thresholding.
        </p>
    </div>
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
