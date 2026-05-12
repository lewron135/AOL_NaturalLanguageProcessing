# Must stay in sync with notebook Section 6-7
import re
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")


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
    return lemmatize_text(clean_text(text))


def pdf_clean(text: str) -> str:
    text = re.sub(r"(?<=\b\w)\s(?=\w\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
