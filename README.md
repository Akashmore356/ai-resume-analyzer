# AI-Based Resume Analyzer with Personalized Improvement Suggestions 🚀

An end-to-end Machine Learning and NLP application designed to evaluate candidate resumes against target job descriptions, compute ATS Match Scores using SentenceBERT embeddings, extract skills via spaCy taxonomy matching, and deliver tailored improvement feedback using Large Language Models (Gemini / OpenAI API).

---

## 📌 Features & Architecture

1. **Multi-Format Text Parsing**:
   - **PDF Parsing**: PyMuPDF (`fitz`) for fast text extraction.
   - **DOCX Parsing**: `python-docx` to extract text from paragraphs and layout tables.

2. **Feature & Skill Extraction**:
   - **spaCy NLP Pipeline**: Entity recognition and linguistic analysis.
   - **Rich Skill Taxonomy**: Pre-categorized technical and soft skills covering Programming, Frameworks, Cloud/DevOps, Databases, AI/ML, and Methodologies.
   - **Contact & Structure Extraction**: Email, phone, and standard section detection (Work Experience, Education, Projects, Skills, Certifications).

3. **Semantic Matching & ATS Scoring**:
   - **SentenceBERT Embeddings**: Uses `all-MiniLM-L6-v2` for dense vector text representation.
   - **Cosine Similarity**: Quantifies semantic alignment between resume content and target job requirements.
   - **Weighted ATS Score**:
     - `50%` Semantic Embedding Similarity
     - `40%` Keyword & Skill Coverage
     - `10%` Resume Structural Completeness

4. **Personalized Improvement Engine**:
   - Integrated with **Google Gemini API** (`gemini-2.5-flash` / `gemini-1.5-flash`) or **OpenAI API** (`gpt-3.5-turbo`).
   - Structured feedback report containing:
     - Match strengths
     - Critical missing skills & natural integration guidance
     - Structural & ATS layout formatting fixes
     - **STAR Method** rewritten bullet points with quantifiable metrics.
   - **Offline Rule-Based Fallback**: Generates structured feedback even without an active LLM API key.

5. **Interactive Streamlit Dashboard**:
   - Custom styled UI with Plotly radial gauge chart, metric badges, side-by-side skill pills, section completeness checklist, and downloadable Markdown reports.

---

## 📁 Directory Structure

```
.
├── requirements.txt      # Python package dependencies
├── parser.py            # PDF/DOCX text extraction, spaCy NLP & skill taxonomy parsing
├── matcher.py           # SentenceBERT embeddings, cosine similarity & composite ATS score engine
├── advisor.py           # LLM API client (Gemini/OpenAI) with structured prompt & offline fallback
├── app.py               # Interactive Streamlit UI dashboard
└── README.md            # Installation & setup guide
```

---

## ⚡ Terminal Setup & Execution Commands

### 1. Clone / Navigate to Project Directory
```bash
cd "/Users/akashmore/Documents/AI-Based Resume Analyzer with Personalized Improvement Suggestions,"
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download spaCy English NLP Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set Environment Variables (LLM API Key)
Set your Google Gemini API key or OpenAI API key as an environment variable (or enter it in the Streamlit UI sidebar):

**For macOS / Linux (zsh or bash):**
```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
# OR for OpenAI
export OPENAI_API_KEY="your_actual_openai_api_key_here"
```

### 6. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.
