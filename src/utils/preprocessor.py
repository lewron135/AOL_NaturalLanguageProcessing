# Must stay in sync with notebook Section 6-7
import re
import unicodedata
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import ftfy as _ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

# Common PDF encoding artifacts: mojibake sequences and Unicode oddities
# ordered longest-match first so replacements don't chain incorrectly
_PDF_ARTIFACT_MAP: list[tuple[str, str]] = [
    # Mojibake sequences (UTF-8 bytes mis-decoded as Latin-1)
    ("â", "'"),   # right single quotation mark
    ("â", "'"),   # left single quotation mark
    ("â", '"'),   # left double quotation mark
    ("â", '"'),   # right double quotation mark
    ("â", "-"),   # en dash
    ("â", "-"),   # em dash
    ("â¢", " "),   # bullet
    # Unicode typographic characters
    ("’", "'"), ("‘", "'"),
    ("“", '"'), ("”", '"'),
    ("–", "-"), ("—", "-"),
    ("•", " "), ("·", " "),
    # Ligatures common in PDF-extracted text
    ("ﬁ", "fi"), ("ﬂ", "fl"),
    ("ﬃ", "ffi"), ("ﬄ", "ffl"),
    # Non-breaking / zero-width spaces
    (" ", " "), ("​", ""), ("﻿", ""),
]

_MULTI_SPACE = re.compile(r" {2,}")

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")


def normalize_encoding(text: str) -> str:
    """Fix encoding artifacts, collapse whitespace, and lowercase.

    Uses ftfy when available; falls back to unicodedata NFC + regex mapping
    for common PDF extraction artifacts (mojibake, ligatures, smart quotes).
    """
    text = str(text)
    if _FTFY_AVAILABLE:
        text = _ftfy.fix_text(text, normalization="NFC")
    else:
        text = unicodedata.normalize("NFC", text)
        for bad, good in _PDF_ARTIFACT_MAP:
            if bad in text:
                text = text.replace(bad, good)
        # Strip remaining non-ASCII that wasn't mapped
        text = text.encode("ascii", errors="ignore").decode("ascii")
    text = _MULTI_SPACE.sub(" ", text).strip().lower()
    return text


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    tokens = [_LEMMATIZER.lemmatize(t) for t in text.split()]
    return " ".join(tokens)


def full_preprocess(text: str) -> str:
    return lemmatize_text(clean_text(normalize_encoding(text)))


def pdf_clean(text: str) -> str:
    text = normalize_encoding(text)
    text = re.sub(r"(?<=\b\w)\s(?=\w\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
