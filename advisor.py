"""
advisor.py - AI-Powered Personalized Improvement Engine.
Generates targeted, actionable feedback and STAR-method bullet point rewrites using LLM APIs
(Google Gemini API or OpenAI API via environment variables) with intelligent rule-based fallback.
"""

import os
import json
from typing import Dict, Any, Optional

def get_llm_client_and_provider(custom_api_key: Optional[str] = None):
    """
    Detects available LLM credentials and returns provider type ('gemini', 'openai', or None).
    """
    api_key = custom_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return "gemini", api_key
        
    openai_key = custom_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key:
        return "openai", openai_key
        
    return None, None


def call_gemini_api(prompt: str, api_key: str) -> str:
    """Invokes Google Gemini API using official google-genai or google-generativeai package."""
    try:
        # Try new google-genai SDK first
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        if response and response.text:
            return response.text
    except Exception:
        pass

    try:
        # Fallback to google-generativeai SDK
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API execution error: {e}")
        
    raise RuntimeError("Could not retrieve response from Gemini API.")


def call_openai_api(prompt: str, api_key: str) -> str:
    """Invokes OpenAI API using openai v1+ SDK."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert ATS Resume Coach and Senior Technical Recruiter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"OpenAI API execution error: {e}")


def generate_offline_rule_based_report(evaluation: Dict[str, Any], job_role: str) -> str:
    """
    Generates a high-quality structured Markdown report when LLM API keys are not provided.
    Guarantees the application always provides immediate actionable value.
    """
    ats_score = evaluation["overall_ats_score"]
    missing_skills = evaluation["missing_skills"]
    matched_skills = evaluation["matched_skills"]
    sections = evaluation["sections_found"]
    
    missing_str = ", ".join(missing_skills[:8]) if missing_skills else "None identified! Good skill coverage."
    matched_str = ", ".join(matched_skills[:8]) if matched_skills else "Limited direct skill matches found."
    
    missing_sections = [k for k, v in sections.items() if not v]
    missing_sections_str = ", ".join(missing_sections) if missing_sections else "All core sections present."

    report = f"""### 🎯 AI Resume Analysis Report (Rule-Based Mode)

#### 🌟 1. Key Match Strengths
- **Role Alignment**: Your resume contains key background elements for **{job_role}**.
- **Matched Skills**: You explicitly mention {len(matched_skills)} relevant skills ({matched_str}).
- **Structure**: Core content sections parsed cleanly with readable text density.

---

#### ⚠️ 2. Missing & Critical Skills to Add
- **High Priority Skill Gaps**: `{missing_str}`
- **Actionable Advice**: Integrate these exact keyword terms naturally into your **Work Experience** bullet points or a dedicated **Technical Skills** section. Avoid keyword stuffing; pair each skill with context (e.g., *"Utilized {missing_skills[0] if missing_skills else 'Python'} to optimize data processing latency"*).

---

#### 🛠️ 3. Structural & Formatting Fixes
- **Missing Sections**: {missing_sections_str}
- **ATS Formatting Best Practices**:
  - Use standard single-column layouts. Avoid multi-column text boxes, graphics, or tables that confuse parser parsers.
  - Ensure clear section headers (`Work Experience`, `Education`, `Projects`, `Skills`).
  - Keep word count balanced (aim for 400 - 800 words for single/double page standard resumes).

---

#### ⚡ 4. Rewritten Bullet Points (STAR Method)
Below are tailored examples demonstrating how to convert passive descriptions into high-impact **STAR (Situation, Task, Action, Result)** statements:

1. **Original**: *"Worked on developing REST APIs and fixed backend bugs for client applications."*  
   👉 **STAR Rewrite**: *"Engineered 12+ scalable RESTful microservice APIs using Python and PostgreSQL, resolving 45+ critical bug tickets and reducing average response latency by 28%."*

2. **Original**: *"Responsible for building machine learning models to predict customer churn."*  
   👉 **STAR Rewrite**: *"Designed and deployed a Random Forest predictive ML model, processing 100K+ customer records to forecast churn with 89% accuracy, boosting retention strategy efficiency by 15%."*

3. **Original**: *"Helped team with cloud deployment and CI/CD pipelines."*  
   👉 **STAR Rewrite**: *"Automated multi-stage CI/CD deployment pipelines using GitHub Actions and Docker, accelerating production deployment frequency by 40% while maintaining 99.9% uptime."*

---
> 💡 *Tip: Set your Gemini or OpenAI API Key in the sidebar or environment variables (`GEMINI_API_KEY` / `OPENAI_API_KEY`) to generate dynamic, personalized LLM feedback!*
"""
    return report


def generate_improvement_suggestions(
    resume_data: Dict[str, Any],
    evaluation: Dict[str, Any],
    job_role: str,
    job_description: str,
    custom_api_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Main function to generate personalized improvement feedback.
    Returns (report_markdown, source_status).
    """
    provider, api_key = get_llm_client_and_provider(custom_api_key)
    
    if not provider or not api_key:
        report = generate_offline_rule_based_report(evaluation, job_role)
        return report, "Offline Rule Engine (No API Key provided)"
        
    prompt = f"""
You are a Senior Technical Recruiter and Career Coach. Analyze the following resume against the target job requirements and provide clear, actionable feedback.

TARGET JOB ROLE: {job_role}

TARGET JOB DESCRIPTION:
{job_description[:3000]}

CANDIDATE RESUME METRICS & EXTRACTED DATA:
- Overall ATS Match Score: {evaluation['overall_ats_score']}%
- Semantic Similarity Score: {evaluation['semantic_score']}%
- Matched Skills Found: {', '.join(evaluation['matched_skills'])}
- Missing / Missing Critical Skills: {', '.join(evaluation['missing_skills'])}
- Section Presence: {json.dumps(evaluation['sections_found'])}

RESUME RAW TEXT EXTRACT:
{resume_data['raw_text'][:4000]}

INSTRUCTIONS:
Provide a well-formatted Markdown report divided into exact section headers:

### 🌟 1. Key Match Strengths
Highlight 2-3 specific strengths of this candidate relevant to the target role.

### ⚠️ 2. Missing & Critical Skills to Add
Detail the top missing skills and explain specifically how the candidate should integrate them into their experience section.

### 🛠️ 3. Structural & Formatting Fixes
Highlight weak sections, layout suggestions, word choice improvements, and ATS formatting recommendations.

### ⚡ 4. Rewritten Bullet Points (STAR Method)
Select 2-3 specific bullet points or phrases from the candidate's actual resume text and provide before/after STAR method rewrites:
- **Original Bullet**: "...'
- 👉 **STAR Rewrite**: "..." (Include action verbs, technologies used, and quantifiable metrics/results like %, $, or numerical impact).

Keep the feedback highly professional, encouraging, precise, and practical.
"""

    try:
        if provider == "gemini":
            response_text = call_gemini_api(prompt, api_key)
            return response_text, "Powered by Gemini AI"
        elif provider == "openai":
            response_text = call_openai_api(prompt, api_key)
            return response_text, "Powered by OpenAI GPT"
    except Exception as e:
        # If LLM API call fails, fall back gracefully
        fallback_report = generate_offline_rule_based_report(evaluation, job_role)
        fallback_report += f"\n\n> ⚠️ Note: LLM API call encountered an issue ({str(e)}). Displaying rule-based feedback."
        return fallback_report, "Fallback Mode (API Error)"

    fallback_report = generate_offline_rule_based_report(evaluation, job_role)
    return fallback_report, "Fallback Mode"
