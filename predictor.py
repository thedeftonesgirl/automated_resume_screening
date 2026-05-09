"""
predictor.py  ─  Inference-only module for Resume Screening
───────────────────────────────────────────────────────────
Loads the pre-trained artefacts saved by the notebook
(best_model.pkl, tfidf_vectorizer.pkl, ranking_vectorizer.pkl)
and exposes a single public function:

    rank_resumes(job_description: str, resume_texts: list[str])
        → list[dict]   (sorted by similarity, highest first)

No Jupyter / nbconvert dependency is needed at runtime.
"""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import nltk

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "models"

_BEST_MODEL_PATH        = MODEL_DIR / "best_model.pkl"
_TFIDF_VECTORIZER_PATH  = MODEL_DIR / "tfidf_vectorizer.pkl"
_RANKING_VEC_PATH       = MODEL_DIR / "ranking_vectorizer.pkl"

# ── NLTK bootstrap (once per interpreter session) ──────────────────────────
_NLTK_PACKAGES = [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
]

def _ensure_nltk() -> None:
    for data_path, pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(data_path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()

from nltk.corpus   import stopwords as _sw_corpus
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

_STOP_WORDS  = set(_sw_corpus.words("english"))
_LEMMATIZER  = WordNetLemmatizer()


# ── text preprocessing (mirrors the notebook exactly) ─────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalise a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                # non-alphabetics
    text = re.sub(r"\s+", " ", text).strip()              # extra whitespace
    tokens = word_tokenize(text)
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens
              if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── lazy model loading ─────────────────────────────────────────────────────
_model            = None
_tfidf_vectorizer = None
_ranking_vec      = None


def _load_models() -> None:
    global _model, _tfidf_vectorizer, _ranking_vec
    if _model is None:
        with open(_BEST_MODEL_PATH,       "rb") as f:
            _model = pickle.load(f)
        with open(_TFIDF_VECTORIZER_PATH, "rb") as f:
            _tfidf_vectorizer = pickle.load(f)
        with open(_RANKING_VEC_PATH,      "rb") as f:
            _ranking_vec = pickle.load(f)


# ── public API ─────────────────────────────────────────────────────────────
def rank_resumes(job_description: str, resume_texts: list) -> list:
    """
    Rank a list of resume texts against a job description.

    Parameters
    ----------
    job_description : str   Raw job-description text (user input).
    resume_texts    : list  Each element is the raw text of one resume.

    Returns
    -------
    list of dict, sorted by similarity_score descending:
        {
          "rank":             int,
          "candidate_id":     int  (1-based index),
          "similarity_score": float,
          "match_percent":    float,
          "category":         str,
          "snippet":          str  (first 300 chars of cleaned text),
        }
    """
    _load_models()

    from sklearn.metrics.pairwise import cosine_similarity

    if not resume_texts:
        return []

    # 1. Preprocess every text
    cleaned_jd      = preprocess_text(job_description)
    cleaned_resumes = [preprocess_text(t) for t in resume_texts]

    # 2. TF-IDF features for classification
    tfidf_resumes = _tfidf_vectorizer.transform(cleaned_resumes)

    # 3. Predict category for each resume
    categories = _model.predict(tfidf_resumes)

    # 4. Ranking vectorizer features
    ranking_resumes = _ranking_vec.transform(cleaned_resumes)
    ranking_jd      = _ranking_vec.transform([cleaned_jd])

    # 5. Cosine similarities  →  shape (n_resumes,)
    sims = cosine_similarity(ranking_resumes, ranking_jd).flatten()

    # 6. Build results list
    results = []
    for idx, (sim, cat, cleaned) in enumerate(
            zip(sims, categories, cleaned_resumes)):
        results.append({
            "candidate_id":     idx + 1,
            "similarity_score": float(round(sim, 6)),
            "match_percent":    float(round(sim * 100, 2)),
            "category":         str(cat),
            "snippet":          cleaned[:300],
        })

    # 7. Sort highest first and add rank
    results.sort(key=lambda r: r["similarity_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results
