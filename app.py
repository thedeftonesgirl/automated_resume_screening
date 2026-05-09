"""
app.py  ─  Flask backend for the Automated Resume Screening System
"""

import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ── optional PDF support ────────────────────────────────────────────────────
try:
    import pdfminer.high_level as _pdf
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ── optional DOCX support ───────────────────────────────────────────────────
try:
    import docx as _docx
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


def _extract_text(file_storage) -> str:
    """Return plain text from an uploaded file (txt, pdf, or docx)."""
    filename = (file_storage.filename or "").lower()
    raw = file_storage.read()

    if filename.endswith(".pdf"):
        if not _PDF_OK:
            return ""
        import io
        return _pdf.extract_text(io.BytesIO(raw)) or ""

    if filename.endswith(".docx"):
        if not _DOCX_OK:
            return ""
        import io
        doc = _docx.Document(io.BytesIO(raw))
        return "\n".join(para.text for para in doc.paragraphs)

    # default: treat as UTF-8 / Latin-1 text (.txt etc.)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ── routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        job_description = request.form.get("job_description", "").strip()
        if not job_description:
            return jsonify({"error": "Job description is required."}), 400

        files = request.files.getlist("resumes")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "At least one resume file is required."}), 400

        resume_texts = []
        filenames    = []
        for f in files:
            if f.filename:
                text = _extract_text(f)
                resume_texts.append(text)
                filenames.append(f.filename)

        if not resume_texts:
            return jsonify({"error": "No readable resume content found."}), 400

        # Import here so the heavy model loads only on first request
        from predictor import rank_resumes
        results = rank_resumes(job_description, resume_texts)

        # Attach the original filename to each result
        for r in results:
            idx = r["candidate_id"] - 1      # 0-based
            if 0 <= idx < len(filenames):
                r["filename"] = filenames[idx]
            else:
                r["filename"] = f"Resume {r['candidate_id']}"

        return jsonify({"results": results})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
