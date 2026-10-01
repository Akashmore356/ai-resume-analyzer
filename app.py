"""
app.py - Main Streamlit Dashboard for AI-Based Resume Analyzer.
Ties together parser.py, matcher.py, and advisor.py into an interactive, intuitive UI.
"""

import os
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import custom processing modules
from parser import parse_resume_content
from matcher import evaluate_resume_vs_job
from advisor import generate_improvement_suggestions

# Load environment variables (.env file support)
load_dotenv()

# Pre-defined sample Job Descriptions for convenient user testing
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
- Strong foundation in Statistical Analysis, A/B Testing, Feature Engineering, and Data Visualization (Tableau/PowerBI/Matplotlib).
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

def inject_custom_css():
    """Injects custom CSS styling for badges, cards, and overall dashboard aesthetics."""
    st.markdown("""
        <style>
        /* Global Page Adjustments */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        
        /* Modern Hero Banner */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
            padding: 2.5rem 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .main-header h1 {
            color: #ffffff;
            margin: 0;
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .main-header p {
            color: #94a3b8;
            margin-top: 0.5rem;
            font-size: 1.05rem;
            font-weight: 400;
        }
        
        /* Metric Card Container */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .metric-title {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.2rem;
        }

        /* Pill Badges */
        .skill-pill {
            display: inline-block;
            padding: 0.4rem 0.85rem;
            margin: 0.25rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
        }
        .pill-matched {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        .pill-missing {
            background-color: #ffe4e6;
            color: #9f1239;
            border: 1px solid #fca5a5;
        }

        /* Status Badge */
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1.25rem;
            border-radius: 12px;
            color: white;
            font-weight: 800;
            font-size: 1.1rem;
            letter-spacing: 0.02em;
        }
        </style>
    """, unsafe_allow_html=True)


def create_plotly_gauge(score: float):
    """Creates a responsive Plotly radial gauge chart for ATS Match Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ATS Match Score", 'font': {'size': 18, 'color': "#0f172a", 'weight': "bold"}},
        number={'suffix': "%", 'font': {'size': 38, 'weight': "bold"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#2563eb", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': "#fef2f2"},
                {'range': [50, 75], 'color': "#fffbeb"},
                {'range': [75, 100], 'color': "#f0fdf4"}
            ],
            'threshold': {
                'line': {'color': "#16a34a", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer & Improvement Advisor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.image("https://img.icons8.com/isometric/100/resume.png", width=70)
        st.title("Settings & Config")
        st.markdown("---")

        # Job Role Selector
        st.subheader("1. Target Role")
        selected_role = st.selectbox(
            "Select Target Job Profile:",
            options=list(SAMPLE_JOB_DESCRIPTIONS.keys()) + ["Custom Role"]
        )

        if selected_role != "Custom Role":
            default_jd = SAMPLE_JOB_DESCRIPTIONS[selected_role]
        else:
            default_jd = ""

        st.markdown("---")

        # LLM API Configuration
        st.subheader("2. AI Advisor Config")
        env_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        env_openai = os.getenv("OPENAI_API_KEY")

        if env_gemini:
            st.success("🟢 Gemini API Key detected (Environment)")
        elif env_openai:
            st.success("🟢 OpenAI API Key detected (Environment)")
        else:
            st.info("💡 Optional: Enter an API key for dynamic LLM feedback, or run in offline rule-based mode.")

        user_api_key = st.text_input(
            "API Key (Gemini or OpenAI):",
            type="password",
            help="Optional. Leave blank to use environment variable or offline rule engine."
        )

        st.markdown("---")
        st.markdown("### 🎓 Project Info")
        st.caption("**Title**: AI-Based Resume Analyzer with Personalized Improvement Suggestions")
        st.caption("**Tech Stack**: Streamlit, SentenceBERT, spaCy, FastAPI, Gemini/OpenAI API")

    # --- MAIN HEADER ---
    st.markdown("""
        <div class="main-header">
            <span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.3);">✨ AI POWERED ATS OPTIMIZER</span>
            <h1 style="margin-top: 10px;">📄 AI-Based Resume Analyzer</h1>
            <p>Score your resume against job descriptions using SentenceBERT vector embeddings, extract skill gaps, and generate STAR-method bullet rewrites with AI.</p>
        </div>
    """, unsafe_allow_html=True)

    # Feature Highlight Cards
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("⚡ **SentenceBERT Match**: Cosine vector similarity comparing full contextual meaning.")
    with f2:
        st.info("🔍 **spaCy Skill Extraction**: Identifies technical, framework, cloud & soft skill gaps.")
    with f3:
        st.info("🤖 **STAR Method Rewriter**: Rewrites bullet points into measurable achievements.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INPUT SECTION ---
    st.subheader("📋 Step 1: Upload Resume & Job Requirements")
    col_file, col_jd = st.columns([1, 1], gap="large")

    with col_file:
        st.markdown("#### Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file:",
            type=["pdf", "docx"],
            help="Supports standard PDF and Microsoft Word DOCX formats."
        )

    with col_jd:
        st.markdown("#### Target Job Description")
        job_description_text = st.text_area(
            "Paste target Job Description text:",
            value=default_jd,
            height=180,
            placeholder="Paste responsibilities, qualifications, and required skills here..."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Resume & Compute Match Score", type="primary", use_container_width=True)

    # --- PROCESSING & RESULTS ---
    if analyze_btn:
        if not uploaded_file:
            st.error("⚠️ Please upload a resume file (PDF or DOCX) before proceeding.")
            return

        if not job_description_text.strip():
            st.error("⚠️ Please provide a Target Job Description or select a preset job role.")
            return

        with st.spinner("⏳ Parsing document text, running spaCy NLP & SentenceBERT vector analysis..."):
            try:
                # Step 1: Parse resume file
                resume_data = parse_resume_content(uploaded_file)
                
                # Step 2: Evaluate matching & scores
                evaluation = evaluate_resume_vs_job(resume_data, selected_role, job_description_text)
                
                # Step 3: Generate AI Advisor report
                api_key_to_use = user_api_key if user_api_key.strip() else None
                ai_report, advisor_source = generate_improvement_suggestions(
                    resume_data, evaluation, selected_role, job_description_text, custom_api_key=api_key_to_use
                )
                
                st.session_state['analysis_done'] = True
                st.session_state['resume_data'] = resume_data
                st.session_state['evaluation'] = evaluation
                st.session_state['ai_report'] = ai_report
                st.session_state['advisor_source'] = advisor_source

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                return

    # Render results if available in session state
    if st.session_state.get('analysis_done', False):
        evaluation = st.session_state['evaluation']
        resume_data = st.session_state['resume_data']
        ai_report = st.session_state['ai_report']
        advisor_source = st.session_state['advisor_source']

        st.markdown("---")
        st.subheader("📊 ATS Evaluation Dashboard")

        # Row 1: Gauge chart + Metric summary cards
        c_gauge, c_metrics = st.columns([1.2, 1.8], gap="large")

        with c_gauge:
            fig_gauge = create_plotly_gauge(evaluation['overall_ats_score'])
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                f"<div style='text-align: center; margin-top: -10px;'>"
                f"<span class='status-badge' style='background-color: {evaluation['badge_color']};'>"
                f"{evaluation['badge_label']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with c_metrics:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Semantic Match</div>
                        <div class="metric-value">{evaluation['semantic_score']}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Skill Match</div>
                        <div class="metric-value">{evaluation['skill_match_score']}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Structure Score</div>
                        <div class="metric-value">{evaluation['section_score']}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Contact Info & Quick Stats
            st.markdown("##### 📌 Candidate Snapshot")
            c_info1, c_info2, c_info3 = st.columns(3)
            contact = resume_data.get("contact_info", {})
            c_info1.write(f"**Email**: {contact.get('email', 'N/A')}")
            c_info2.write(f"**Phone**: {contact.get('phone', 'N/A')}")
            c_info3.write(f"**Word Count**: {resume_data.get('word_count', 0)} words")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Side-by-side Skill Breakdown
        st.subheader("💡 Skills Comparison")
        col_matched, col_missing = st.columns(2, gap="medium")

        with col_matched:
            st.markdown("#### ✅ Matched Skills Found")
            matched = evaluation['matched_skills']
            if matched:
                pills_html = "".join([f"<span class='skill-pill pill-matched'>{s}</span>" for s in matched])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.info("No exact matching skills from taxonomy found.")

        with col_missing:
            st.markdown("#### ⚠️ Missing / Critical Skills to Add")
            missing = evaluation['missing_skills']
            if missing:
                pills_html = "".join([f"<span class='skill-pill pill-missing'>{s}</span>" for s in missing])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.success("Great job! All key required skills present in resume.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 3: Resume Section Completeness Checklist
        st.subheader("📑 Resume Structure Completeness")
        sections = evaluation['sections_found']
        s_cols = st.columns(len(sections))
        for idx, (sec_name, sec_present) in enumerate(sections.items()):
            with s_cols[idx]:
                if sec_present:
                    st.success(f"✔️ {sec_name}")
                else:
                    st.error(f"❌ {sec_name}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 4: AI Improvement Suggestions Report
        st.subheader(f"🤖 Personalized Improvement Feedback")
        st.caption(f"Source: **{advisor_source}**")

        st.markdown(ai_report)

        st.markdown("---")
        
        # Download Report Option
        st.download_button(
            label="📥 Download Complete Analysis Report (.md)",
            data=f"# Resume Analysis Report for {selected_role}\n\n"
                 f"**Overall ATS Score**: {evaluation['overall_ats_score']}%\n"
                 f"**Semantic Score**: {evaluation['semantic_score']}%\n"
                 f"**Skill Match Score**: {evaluation['skill_match_score']}%\n\n"
                 f"{ai_report}",
            file_name="resume_analysis_report.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
