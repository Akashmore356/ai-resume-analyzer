"""
matcher.py - Semantic Matching & ATS Scoring Engine.
Uses SentenceBERT (all-MiniLM-L6-v2) and scikit-learn cosine_similarity to compare
resume content against job descriptions, calculate skill overlaps, and compute ATS match scores.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from parser import extract_skills_from_text

# Global lazy-loaded SentenceTransformer model
_transformer_model = None

def get_sentence_transformer():
    """Loads and caches all-MiniLM-L6-v2 SentenceTransformer model."""
    global _transformer_model
    if _transformer_model is None:
        try:
            _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model 'all-MiniLM-L6-v2': {e}")
    return _transformer_model


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two text snippets using SentenceBERT embeddings.
    Returns score as a percentage [0.0, 100.0].
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_sentence_transformer()
    
    # Generate embeddings
    embeddings = model.encode([text1, text2], show_progress_bar=False, normalize_embeddings=True)
    
    # Cosine similarity between embedding vectors
    sim_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    similarity_score = float(sim_matrix[0][0])
    
    # Clip and scale to percentage [0, 100]
    percentage_score = float(np.clip(similarity_score * 100.0, 0.0, 100.0))
    return round(percentage_score, 2)


def compare_skills(resume_skills: List[str], job_description_text: str) -> Dict[str, Any]:
    """
    Compares candidate skills against required skills extracted from Job Description.
    """
    # Extract skills present in Job Description using taxonomy
    jd_skill_data = extract_skills_from_text(job_description_text)
    jd_skills = jd_skill_data["flat_skills"]
    
    resume_skills_set = set(s.lower() for s in resume_skills)
    jd_skills_set = set(s.lower() for s in jd_skills)
    
    # Map back to canonical skill names
    skill_lookup = {s.lower(): s for s in (resume_skills + jd_skills)}
    
    matched_set = resume_skills_set.intersection(jd_skills_set)
    missing_set = jd_skills_set.difference(resume_skills_set)
    extra_set = resume_skills_set.difference(jd_skills_set)
    
    matched_skills = sorted([skill_lookup.get(s, s) for s in matched_set])
    missing_skills = sorted([skill_lookup.get(s, s) for s in missing_set])
    extra_skills = sorted([skill_lookup.get(s, s) for s in extra_set])
    
    if len(jd_skills) > 0:
        skill_match_percentage = (len(matched_skills) / len(jd_skills)) * 100.0
    else:
        # If no explicit skills found in JD text, default to standard semantic match
        skill_match_percentage = 70.0 if len(resume_skills) > 0 else 0.0
        
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "required_jd_skills": jd_skills,
        "skill_match_percentage": round(min(skill_match_percentage, 100.0), 2)
    }


def compute_section_score(sections: Dict[str, bool]) -> float:
    """
    Computes structure completeness score based on presence of essential sections.
    Key sections: Work Experience, Education, Projects, Skills (25% each).
    """
    essential_keys = ["Work Experience", "Education", "Projects", "Skills"]
    present_count = sum(1 for key in essential_keys if sections.get(key, False))
    score = (present_count / len(essential_keys)) * 100.0
    return round(score, 2)


def evaluate_resume_vs_job(resume_data: Dict[str, Any], job_role: str, job_description: str) -> Dict[str, Any]:
    """
    Main evaluation function tying together semantic similarity, skill comparison, and structural completeness.
    
    Weights for ATS Score calculation:
    - 50% Semantic BERT Similarity
    - 40% Keyword & Skill Coverage
    - 10% Resume Section Completeness
    """
    resume_text = resume_data["raw_text"]
    resume_skills = resume_data["skills"]
    sections = resume_data["sections"]
    
    # Combine Job Role and Job Description for contextual matching
    full_jd_text = f"Target Role: {job_role}\n\nJob Description:\n{job_description}"
    
    # 1. Semantic BERT Similarity
    semantic_score = compute_semantic_similarity(resume_text, full_jd_text)
    
    # 2. Skill Comparison
    skill_analysis = compare_skills(resume_skills, job_description)
    skill_match_score = skill_analysis["skill_match_percentage"]
    
    # 3. Structural Completeness Score
    section_score = compute_section_score(sections)
    
    # 4. Composite ATS Match Score Calculation
    ats_score = (0.50 * semantic_score) + (0.40 * skill_match_score) + (0.10 * section_score)
    ats_score = round(float(np.clip(ats_score, 0.0, 100.0)), 1)
    
    # Determine Score Rating Badge & Color
    if ats_score >= 80.0:
        badge_label = "Strong Match"
        badge_color = "#22c55e"  # Green
    elif ats_score >= 60.0:
        badge_label = "Moderate Match"
        badge_color = "#f59e0b"  # Amber
    else:
        badge_label = "Needs Improvement"
        badge_color = "#ef4444"  # Red

    return {
        "overall_ats_score": ats_score,
        "badge_label": badge_label,
        "badge_color": badge_color,
        "semantic_score": semantic_score,
        "skill_match_score": skill_match_score,
        "section_score": section_score,
        "matched_skills": skill_analysis["matched_skills"],
        "missing_skills": skill_analysis["missing_skills"],
        "extra_skills": skill_analysis["extra_skills"],
        "required_jd_skills": skill_analysis["required_jd_skills"],
        "sections_found": sections,
        "resume_word_count": resume_data.get("word_count", 0),
        "contact_info": resume_data.get("contact_info", {})
    }
