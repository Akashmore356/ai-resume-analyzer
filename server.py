"""
server.py - FastAPI REST API & Web Server for AI Resume Analyzer.
Exposes endpoints for file uploads, semantic matching, ATS scoring, and AI feedback generation.
Serves a built-in modern single-page web app.
"""

import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Import core machine learning modules
from parser import parse_resume_content
from matcher import evaluate_resume_vs_job
from advisor import generate_improvement_suggestions

load_dotenv()

app = FastAPI(
    title="AI Resume Analyzer API",
    description="REST API for parsing resumes, computing ATS match scores, and generating LLM suggestions.",
    version="2.0.0"
)

# Enable CORS for React / Next.js / Vue frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_JOB_DESCRIPTIONS = {
    "Machine Learning Engineer": """We are looking for a Machine Learning Engineer with strong experience in Python, PyTorch/TensorFlow, and MLOps. 
Key Requirements:
- 2+ years experience building and deploying machine learning models in production.
- Expertise in NLP, Deep Learning, Scikit-learn, Pandas, and NumPy.
- Strong knowledge of Cloud platforms (AWS/GCP), Docker, Kubernetes, and CI/CD pipelines.
- Proficiency in SQL, REST APIs, Microservices, and System Design.
- Bachelor's or Master's degree in Computer Science or related quantitative field.""",

    "Data Scientist": """Seeking a Data Scientist to analyze complex datasets and build predictive models.
Key Requirements:
- Deep expertise in Python, R, SQL, Pandas, NumPy, and Scikit-learn.
- Strong foundation in Statistical Analysis, A/B Testing, Feature Engineering, and Data Visualization.
- Experience with Machine Learning algorithms (Random Forest, XGBoost, Clustering).
- Familiarity with Big Data tools like Spark/PySpark and Cloud environments (AWS/Snowflake).""",

    "Full Stack Software Engineer": """Looking for a Full Stack Engineer proficient in modern web development frameworks.
Key Requirements:
- Experience with React, TypeScript, Node.js, Express, and HTML5/CSS3.
- Database design using PostgreSQL, MySQL, or MongoDB.
- Knowledge of RESTful APIs, GraphQL, and microservice architecture.
- Hands-on experience with Git, Docker, AWS, and Agile/Scrum methodologies.""",

    "DevOps / Cloud Engineer": """Seeking a DevOps Engineer to automate and scale our cloud infrastructure.
Key Requirements:
- Strong skills in Linux, Bash, Python, Docker, and Kubernetes.
- Infrastructure as Code using Terraform or Ansible.
- CI/CD pipeline implementation with GitHub Actions or Jenkins.
- Expertise in Cloud architecture (AWS, Azure, or GCP), Monitoring (Prometheus/Grafana), and Nginx."""
}


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Resume Analyzer API", "version": "2.0.0"}


@app.get("/api/sample-jds")
def get_sample_jds():
    """Returns sample job descriptions for quick testing."""
    return {"sample_jds": SAMPLE_JOB_DESCRIPTIONS}


@app.post("/api/analyze")
async def analyze_resume_endpoint(
    file: UploadFile = File(...),
    job_role: str = Form(...),
    job_description: str = Form(...),
    api_key: Optional[str] = Form(None)
):
    """
    Main API endpoint for parsing resumes, evaluating ATS scores, and generating AI suggestions.
    """
    if not file.filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX files are supported.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    try:
        # Read file contents
        file_bytes = await file.read()
        
        # Create a mock file object with .name attribute
        class NamedBytes:
            def __init__(self, name: str, data: bytes):
                self.name = name
                self._data = data
            def read(self):
                return self._data
                
        file_obj = NamedBytes(file.filename, file_bytes)

        # Step 1: Parse Resume
        resume_data = parse_resume_content(file_obj)

        # Step 2: Semantic Matching & ATS Scoring
        evaluation = evaluate_resume_vs_job(resume_data, job_role, job_description)

        # Step 3: LLM Advisor Report Generation
        ai_report, advisor_source = generate_improvement_suggestions(
            resume_data, evaluation, job_role, job_description, custom_api_key=api_key
        )

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_type": resume_data.get("file_type"),
            "contact_info": resume_data.get("contact_info"),
            "word_count": resume_data.get("word_count"),
            "evaluation": evaluation,
            "ai_report": ai_report,
            "advisor_source": advisor_source
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")


# Serve static web frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>AI Resume Analyzer API Server</h1><p>Visit /docs for API documentation.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
