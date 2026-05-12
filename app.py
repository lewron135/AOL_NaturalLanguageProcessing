import streamlit as st
import PyPDF2

from src.extraction.engine import analyze, load_nlp, load_sbert, load_tfidf
from src import ui

st.set_page_config(
    page_title="CV Analyzer — NLP Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    ui.inject_css()

    with st.spinner("Loading NLP models..."):
        nlp = load_nlp()
        sbert = load_sbert()
        tfidf = load_tfidf()

    page = ui.render_sidebar(tfidf_available=tfidf is not None)

    if page == "Analyzer":
        file, jd, run = ui.render_analyzer_inputs()
        if run and file and jd.strip():
            with st.spinner("Analyzing..."):
                raw = " ".join(p.extract_text() or "" for p in PyPDF2.PdfReader(file).pages)
                results = analyze(raw, jd, nlp, sbert, tfidf)
            ui.render_score(results["score"], results["tfidf_score"])
            ui.render_match_breakdown(results["details"], results["cv_features"], results["jd_features"])
            ui.render_ner_section(results["ner_html"])
        elif run:
            if not file:
                st.warning("Please upload a PDF resume.")
            elif not jd.strip():
                st.warning("Please enter the job requirements.")

    elif page == "Methodology":
        ui.render_methodology()

    elif page == "System Evaluation":
        ui.render_evaluation()


if __name__ == "__main__":
    main()
