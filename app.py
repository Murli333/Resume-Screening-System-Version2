import streamlit as st
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import string
import html
from PyPDF2 import PdfReader
import pickle
import ast
from nltk.corpus import stopwords


# ------------------------------------------------------------------
# File loading
# ------------------------------------------------------------------
@st.cache_resource
def load_files():
    tf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    X = load_npz("job_matrix.npz")
    jobs = pd.read_csv("jobs_deploy.csv")
    jobs["skills"] = jobs["skills"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return tf, X, jobs

tf, X, jobs = load_files()


# ------------------------------------------------------------------
# Cleaning and preprocessing
# ------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+www\S+', '', text)
    text = re.sub(r'\S+@+\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text

stop = set(stopwords.words('english'))
stop.discard('not')
stop.discard('without')
stop.discard('with')
stop.discard('no')


# ------------------------------------------------------------------
# Skill list
# ------------------------------------------------------------------
skills = [
    # Programming Languages
    'python', 'java', 'c language', 'c++', 'c#', 'javascript', 'typescript',
    'php', 'ruby', 'go language', 'rust', 'kotlin', 'swift', 'r language', 'scala',

    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle',
    'redis', 'cassandra', 'firebase',

    # Data Science & ML
    'machine learning', 'deep learning', 'artificial intelligence',
    'data science', 'data analysis', 'data mining',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn',
    'pandas', 'numpy', 'matplotlib', 'seaborn',
    'opencv', 'nltk', 'spacy', 'xgboost', 'lightgbm',

    # Web Development
    'html', 'css', 'bootstrap', 'tailwind',
    'react', 'nextjs', 'vue', 'angular',
    'nodejs', 'express', 'django', 'flask',
    'fastapi', 'spring boot', 'laravel',

    # Cloud & DevOps
    'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'jenkins',
    'terraform', 'ansible',
    'ci/cd', 'github actions',

    # Version Control
    'git', 'github', 'gitlab', 'bitbucket',

    # Big Data
    'hadoop', 'spark', 'kafka', 'hive',

    # Operating Systems
    'linux', 'unix', 'windows',

    # APIs
    'rest api', 'graphql', 'microservices',

    # Mobile Development
    'android', 'ios', 'flutter', 'react native',

    # Testing
    'selenium', 'pytest', 'junit', 'postman',

    # BI Tools
    'power bi', 'tableau', 'excel',

    # CS Fundamentals
    'data structures', 'algorithms',
    'object oriented programming',
    'oop', 'operating systems',
    'computer networks', 'dbms'
]


def extract_skills(text):
    text = text.lower()
    found = []
    for skill in skills:
        if skill in text:
            found.append(skill)
    return found


# ------------------------------------------------------------------
# List of selectable job types, built from the jobs data itself
# ------------------------------------------------------------------
AUTO_DETECT_LABEL = "Auto-detect best match (search every job type)"


@st.cache_data
def get_job_categories(auto_label):
    titles = jobs['title'].dropna().unique().tolist()
    return [auto_label] + sorted(titles)


# ------------------------------------------------------------------
# Recommender system — supports an optional `category` filter. When a
# category is given, similarity is only computed against postings that
# match that exact job title.
# ------------------------------------------------------------------
def recommend_jobs(resume_text, category=None, top_n=10):
    cleaned_resume = clean_text(resume_text)
    cleaned_resume = ' '.join(
        word for word in cleaned_resume.split() if word not in stop
    )
    resume_vector = tf.transform([cleaned_resume])

    if category and category != AUTO_DETECT_LABEL:
        mask = (jobs['title'].values == category)
        candidate_positions = np.where(mask)[0]

        if len(candidate_positions) == 0:
            empty = jobs.iloc[0:0][
                ['title', 'company_name', 'location', 'formatted_work_type', 'description', 'skills']
            ].copy()
            empty['score'] = []
            return empty, np.array([], dtype=int)

        X_subset = X[candidate_positions]
        sims = cosine_similarity(resume_vector, X_subset)[0]
        order = sims.argsort()[::-1][:top_n]
        top_positions = candidate_positions[order]
        top_scores = sims[order]
    else:
        sims = cosine_similarity(resume_vector, X)[0]
        order = sims.argsort()[::-1][:top_n]
        top_positions = order
        top_scores = sims[order]

    result = jobs.iloc[top_positions][
        ['title', 'company_name', 'location', 'formatted_work_type', 'description', 'skills']
    ].copy()
    result['score'] = (top_scores * 100).round(2)

    return result, top_positions


# ------------------------------------------------------------------
# Roadmap for missing skills
# ------------------------------------------------------------------
roadmap = {
    'docker': 'Learn Docker and containerization.',
    'aws': 'Learn AWS (EC2, S3, IAM).',
    'kubernetes': 'Learn Kubernetes basics and deployments.',
    'react': 'Build frontend projects with React.',
    'sql': 'Practice SQL queries, joins and window functions.',
    'tensorflow': 'Build deep learning projects using TensorFlow.',
    'pytorch': 'Learn PyTorch and implement neural networks.',
    'git': 'Learn Git workflows and GitHub collaboration.',
    'linux': 'Learn Linux commands and shell scripting.',
    'mongodb': 'Learn NoSQL databases and MongoDB.'
}


# ------------------------------------------------------------------
# General ATS analysis — searches across ALL job postings.
# ------------------------------------------------------------------
def ats_analysis(resume_text):
    recommendations, positions = recommend_jobs(resume_text)

    top_pos = positions[0]
    top_job = jobs.iloc[top_pos]

    job_skills = top_job['skills']
    resume_skills = extract_skills(resume_text)

    matched_skills = list(set(job_skills) & set(resume_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    skill_score = round(len(matched_skills) / max(len(job_skills), 1) * 100, 2)
    similarity_score = round(recommendations.iloc[0]['score'], 2)
    ats_score = round(0.7 * similarity_score + 0.3 * skill_score, 2)

    learning_recommendations = [roadmap[s] for s in missing_skills if s in roadmap]

    return {
        'recommended_role': top_job['title'],
        'company': top_job['company_name'],
        'ats_score': ats_score,
        'similarity_score': similarity_score,
        'skill_match_score': skill_score,
        'resume_skills': resume_skills,
        'job_skills': job_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'learning_recommendations': learning_recommendations,
        'top_recommendations': recommendations
    }


# ------------------------------------------------------------------
# Skill requirements for a specific job category
# ------------------------------------------------------------------
def get_required_skills_for_category(category, min_frequency_pct=20, max_skills=15):
    category_jobs = jobs[jobs['title'] == category]
    total_postings = len(category_jobs)
    if total_postings == 0:
        return []

    skill_counts = {}
    for skill_list in category_jobs['skills']:
        if isinstance(skill_list, list):
            for s in set(skill_list):
                skill_counts[s] = skill_counts.get(s, 0) + 1

    ranked = sorted(skill_counts.items(), key=lambda kv: kv[1], reverse=True)

    threshold = max(1, int(total_postings * min_frequency_pct / 100))
    required = [s for s, cnt in ranked if cnt >= threshold]

    if not required:
        required = [s for s, _ in ranked[:max_skills]]

    return required[:max_skills]


# ------------------------------------------------------------------
# ATS analysis targeted at ONE job type the user picked
# ------------------------------------------------------------------
def ats_analysis_for_category(resume_text, category):
    resume_skills = extract_skills(resume_text)
    required_skills = get_required_skills_for_category(category)

    matched_skills = [s for s in required_skills if s in resume_skills]
    missing_skills = [s for s in required_skills if s not in resume_skills]

    skill_score = round(len(matched_skills) / max(len(required_skills), 1) * 100, 2)

    recommendations, positions = recommend_jobs(resume_text, category=category, top_n=10)

    if len(recommendations) > 0:
        similarity_score = round(recommendations.iloc[0]['score'], 2)
        top_job = jobs.iloc[positions[0]]
        recommended_role = top_job['title']
        company = top_job['company_name']
    else:
        similarity_score = 0.0
        recommended_role = category
        company = "N/A"

    ats_score = round(0.7 * similarity_score + 0.3 * skill_score, 2)
    learning_recommendations = [roadmap[s] for s in missing_skills if s in roadmap]

    return {
        'job_category': category,
        'recommended_role': recommended_role,
        'company': company,
        'ats_score': ats_score,
        'similarity_score': similarity_score,
        'skill_match_score': skill_score,
        'resume_skills': resume_skills,
        'required_skills': required_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'learning_recommendations': learning_recommendations,
        'top_recommendations': recommendations,
        'postings_found': len(jobs[jobs['title'] == category])
    }


# ------------------------------------------------------------------
# PDF text extraction
# ------------------------------------------------------------------
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# ==================================================================
# BRANDING — swap these three for your own details
# ==================================================================
APP_NAME = "Wavelength"
APP_TAGLINE = "Tune your resume to the frequency of the job you actually want."
DEVELOPER_NAME = "Murli Mishra"
GITHUB_URL = "https://github.com/Murli333"
LINKEDIN_URL = "https://www.linkedin.com/in/murli-mishra-ab0705332/"

GITHUB_ICON = """<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>"""

LINKEDIN_ICON = """<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M13.5 0h-11C1.12 0 0 1.12 0 2.5v11C0 14.88 1.12 16 2.5 16h11c1.38 0 2.5-1.12 2.5-2.5v-11C16 1.12 14.88 0 13.5 0zM4.98 13.5H2.5V6h2.48v7.5zM3.74 4.98c-.8 0-1.44-.65-1.44-1.44 0-.8.65-1.44 1.44-1.44.8 0 1.44.65 1.44 1.44 0 .8-.64 1.44-1.44 1.44zM13.5 13.5H11v-3.9c0-.93-.02-2.13-1.3-2.13-1.3 0-1.5 1.02-1.5 2.06v3.97H5.72V6h2.4v1.02h.03c.33-.63 1.15-1.3 2.37-1.3 2.53 0 3 1.67 3 3.83v3.95z"/></svg>"""


# ------------------------------------------------------------------
# Small formatting / scoring helpers
# ------------------------------------------------------------------
def score_band(score):
    if score >= 90:
        return "high"
    elif score >= 70:
        return "mid"
    return "low"


def fmt_pct(value):
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def safe_str(value, default="—"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


# ------------------------------------------------------------------
# Global stylesheet (kept as a plain string — it is full of literal
# `{ }` for CSS rules, so it must NOT be an f-string)
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --paper: #EDF1F5;
  --paper-raised: #FFFFFF;
  --ink: #10232B;
  --ink-soft: #55676D;
  --line: #CBD5D8;
  --teal: #1F7A6C;
  --teal-text: #14584D;
  --teal-tint: #E1F0EC;
  --amber: #E8A33D;
  --amber-text: #8A5A17;
  --amber-tint: #FBEEDA;
  --coral: #B0413E;
  --coral-text: #8F332F;
  --coral-tint: #F7E8E6;
  --info-text: #2B4A5E;
  --info-tint: #E7ECF3;
}

html, body, .stApp { background: var(--paper) !important; }
* { box-sizing: border-box; }

.stApp, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  color: var(--ink);
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* Native widget re-skin */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--paper-raised);
  border: 1.5px solid var(--line);
  border-radius: 10px;
  font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper-raised);
  border: 2px dashed var(--amber);
  border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] * {
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] small {
  color: var(--ink-soft) !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] * {
  color: var(--ink) !important;
}
[data-baseweb="popover"] [role="listbox"] {
  background: var(--paper-raised);
}
[data-baseweb="popover"] [role="listbox"] * {
  color: var(--ink) !important;
  font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stTextArea"] textarea {
  background: var(--paper-raised);
  border: 1px solid var(--line);
  border-radius: 10px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--ink-soft);
}
[data-testid="stCheckbox"] label p {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink-soft);
  font-size: 0.9rem;
}
[data-testid="stSelectbox"]:focus-within div[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"]:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(31,122,108,.15);
}

/* Eyebrow labels */
.wl-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  text-transform: uppercase;
  letter-spacing: .13em;
  font-size: .72rem;
  color: var(--teal-text);
  display: block;
  margin-bottom: .3rem;
}

/* Hero */
.wl-hero {
  padding: 2.25rem 2.5rem;
  background: var(--paper-raised);
  border: 1px solid var(--line);
  border-radius: 20px;
  margin-bottom: .75rem;
}
.wl-hero-title {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 2.75rem;
  letter-spacing: -0.01em;
  margin: 0;
  color: var(--ink);
  line-height: 1.05;
}
.wl-hero-tagline {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink-soft);
  font-size: 1.08rem;
  margin: .6rem 0 0 0;
  max-width: 52ch;
}
.wl-hero-stats {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .78rem;
  color: var(--ink-soft);
  margin-top: .9rem;
}

/* Wave divider */
.wl-wave { width: 100%; height: 20px; margin: 1.6rem 0; opacity: .55; }
.wl-wave svg { width: 100%; height: 100%; display: block; }

/* Section titles */
.wl-section-title {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 1.3rem;
  margin: .1rem 0 .8rem 0;
  color: var(--ink);
}
.wl-note { font-size: .9rem; color: var(--ink-soft); }
.wl-tip { font-size: .95rem; color: var(--ink); margin: .35rem 0; }
.wl-empty { font-size: .9rem; color: var(--ink-soft); font-style: italic; }

/* Status banners */
.wl-banner {
  display: flex;
  align-items: center;
  gap: .65rem;
  padding: .85rem 1.1rem;
  border-radius: 12px;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: .93rem;
  margin: .5rem 0 1rem 0;
  border: 1px solid transparent;
}
.wl-banner-good { background: var(--teal-tint); color: var(--teal-text); border-color: rgba(31,122,108,.25); }
.wl-banner-warn { background: var(--amber-tint); color: var(--amber-text); border-color: rgba(232,163,61,.35); }
.wl-banner-bad  { background: var(--coral-tint); color: var(--coral-text); border-color: rgba(176,65,62,.25); }
.wl-banner-info { background: var(--info-tint); color: var(--info-text); border-color: rgba(43,74,94,.2); }
.wl-banner-icon { font-weight: 700; flex-shrink: 0; }

/* Tuner gauge */
.wl-tuner { margin: .25rem 0 1.25rem 0; }
.wl-tuner-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: .6rem;
}
.wl-tuner-score {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2rem;
  font-weight: 600;
  color: var(--ink);
}
.wl-tuner-status {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .78rem;
  color: var(--ink-soft);
  margin-left: .6rem;
}
.wl-tuner-track {
  position: relative;
  height: 12px;
  border-radius: 8px;
  overflow: visible;
  background: linear-gradient(to right,
    var(--coral) 0%, var(--coral) 70%,
    var(--amber) 70%, var(--amber) 90%,
    var(--teal) 90%, var(--teal) 100%);
  opacity: .88;
}
.wl-tuner-needle {
  position: absolute;
  top: -5px;
  width: 3px;
  height: 22px;
  background: var(--ink);
  border-radius: 2px;
  transform: translateX(-50%);
  box-shadow: 0 0 0 4px var(--paper-raised), 0 1px 4px rgba(0,0,0,.35);
}
.wl-tuner-labels {
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .66rem;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-top: .5rem;
}

/* Skill chips */
.wl-chip-wrap { display: flex; flex-wrap: wrap; margin-top: .2rem; }
.wl-chip {
  display: inline-block;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: .85rem;
  padding: .35rem .8rem;
  margin: 0 .4rem .4rem 0;
  border-radius: 999px;
  border: 1px solid transparent;
}
.wl-chip-matched { background: var(--teal-tint); color: var(--teal-text); border-color: rgba(31,122,108,.25); }
.wl-chip-missing { background: var(--coral-tint); color: var(--coral-text); border-color: rgba(176,65,62,.25); }
.wl-chip-neutral { background: var(--info-tint); color: var(--info-text); border-color: rgba(43,74,94,.2); }

/* Job cards */
.wl-job-list { display: flex; flex-direction: column; gap: .6rem; margin-top: .2rem; }
.wl-job-card {
  display: flex;
  align-items: center;
  gap: 1.1rem;
  background: var(--paper-raised);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .9rem 1.15rem;
}
.wl-job-rank {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .8rem;
  color: var(--ink-soft);
  width: 1.6rem;
  flex-shrink: 0;
}
.wl-job-main { flex-grow: 1; min-width: 0; }
.wl-job-role {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 1.03rem;
  color: var(--ink);
}
.wl-job-meta {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: .82rem;
  color: var(--ink-soft);
  margin-top: .15rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.wl-job-score { flex-shrink: 0; }
.wl-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 600;
  font-size: .78rem;
  padding: .3rem .65rem;
  border-radius: 999px;
  white-space: nowrap;
}
.wl-badge-high { background: var(--teal-tint); color: var(--teal-text); }
.wl-badge-mid  { background: var(--amber-tint); color: var(--amber-text); }
.wl-badge-low  { background: var(--coral-tint); color: var(--coral-text); }

/* Footer */
.wl-footer {
  margin-top: 2.5rem;
  padding: 1.4rem 0 .4rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: .75rem;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: .85rem;
  color: var(--ink-soft);
  border-top: 1px solid var(--line);
}
.wl-footer-links { display: flex; gap: 1rem; align-items: center; }
.wl-footer-links a {
  color: var(--ink-soft);
  display: inline-flex;
  transition: color .15s ease;
}
.wl-footer-links a:hover { color: var(--teal-text); }
</style>
"""


# ------------------------------------------------------------------
# A small reusable sine-wave path, used as the recurring visual motif
# instead of plain horizontal rules.
# ------------------------------------------------------------------
def _build_wave_path(width=400, height=24, period=100, amplitude=12):
    mid = height / 2
    top = mid - amplitude / 2
    bottom = mid + amplitude / 2
    segments = [f"M0,{mid}"]
    x = 0
    while x < width:
        segments.append(f"C{x + period * 0.125},{top} {x + period * 0.375},{top} {x + period * 0.5},{mid}")
        segments.append(f"C{x + period * 0.625},{bottom} {x + period * 0.875},{bottom} {x + period},{mid}")
        x += period
    return " ".join(segments)

WAVE_PATH = _build_wave_path()


def render_html(content):
    """
    Render raw HTML through st.markdown safely.

    Streamlit's markdown parser treats any line indented 4+ spaces as a
    code block. Nicely-indented multi-line f-strings (readable in the
    Python source) trip this rule, and joining several HTML snippets in
    a loop can leave a whitespace-only line between them that closes the
    "this is real HTML" parsing early — the remainder then gets shown as
    literal, syntax-highlighted text instead of rendering. Stripping
    leading whitespace from every line has no visual effect on the
    rendered HTML/CSS (browsers collapse that whitespace anyway) but
    stops the parser from ever misreading it as code.
    """
    stripped = "\n".join(line.lstrip() for line in content.split("\n"))
    st.markdown(stripped, unsafe_allow_html=True)


def render_wave_divider(color="#1F7A6C"):
    render_html(f"""
    <div class="wl-wave">
      <svg viewBox="0 0 400 24" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <path d="{WAVE_PATH}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" />
      </svg>
    </div>
    """)


def render_hero():
    render_html(f"""
    <div class="wl-hero">
      <span class="wl-eyebrow">Resume &#8646; role matching</span>
      <h1 class="wl-hero-title">{html.escape(APP_NAME)}</h1>
      <p class="wl-hero-tagline">{html.escape(APP_TAGLINE)}</p>
      <p class="wl-hero-stats">{len(jobs):,} postings indexed &middot; {len(get_job_categories(AUTO_DETECT_LABEL)) - 1} job types tracked</p>
    </div>
    """)


def render_status_banner(message_html, level="info"):
    icons = {"good": "&#10003;", "warn": "!", "bad": "&#10005;", "info": "&#8594;"}
    render_html(f"""
    <div class="wl-banner wl-banner-{level}">
      <span class="wl-banner-icon">{icons.get(level, "&#8594;")}</span>
      <span>{message_html}</span>
    </div>
    """)


def render_tuner_gauge(score, label="FREQUENCY MATCH"):
    score = float(score)
    clamped = max(0.0, min(100.0, score))
    if score >= 90:
        status = "Locked on"
    elif score >= 70:
        status = "Tuning in"
    else:
        status = "Off frequency"
    render_html(f"""
    <div class="wl-tuner">
      <div class="wl-tuner-head">
        <span class="wl-eyebrow">{label}</span>
        <div>
          <span class="wl-tuner-score">{fmt_pct(score)}</span>
          <span class="wl-tuner-status">{status}</span>
        </div>
      </div>
      <div class="wl-tuner-track">
        <div class="wl-tuner-needle" style="left:{clamped}%;"></div>
      </div>
      <div class="wl-tuner-labels">
        <span>Off frequency</span><span>Tuning in</span><span>Locked on</span>
      </div>
    </div>
    """)


def render_skill_chips(skill_list, variant="neutral", empty_message="None found."):
    if not skill_list:
        render_html(f'<p class="wl-empty">{html.escape(empty_message)}</p>')
        return
    chips = "".join(
        f'<span class="wl-chip wl-chip-{variant}">{html.escape(str(s))}</span>' for s in skill_list
    )
    render_html(f'<div class="wl-chip-wrap">{chips}</div>')


def render_job_cards(df, max_rows=5):
    rows = df.head(max_rows).reset_index(drop=True)
    cards = ['<div class="wl-job-list">']
    for i, row in rows.iterrows():
        band = score_band(row['score'])
        title = html.escape(safe_str(row['title'], "Untitled role"))
        company = html.escape(safe_str(row['company_name'], "Unknown company"))
        location = html.escape(safe_str(row['location'], "Location n/a"))
        work_type = html.escape(safe_str(row['formatted_work_type'], "Type n/a"))
        cards.append(f"""<div class="wl-job-card">
          <div class="wl-job-rank">{i + 1:02d}</div>
          <div class="wl-job-main">
            <div class="wl-job-role">{title}</div>
            <div class="wl-job-meta">{company} &middot; {location} &middot; {work_type}</div>
          </div>
          <div class="wl-job-score"><span class="wl-badge wl-badge-{band}">{fmt_pct(row['score'])}</span></div>
        </div>""")
    cards.append('</div>')
    render_html("".join(cards))


def render_footer():
    render_html(f"""
    <div class="wl-footer">
      <span>Built by <strong>{html.escape(DEVELOPER_NAME)}</strong></span>
      <span class="wl-footer-links">
        <a href="{html.escape(GITHUB_URL)}" target="_blank" rel="noopener noreferrer" title="GitHub">{GITHUB_ICON}</a>
        <a href="{html.escape(LINKEDIN_URL)}" target="_blank" rel="noopener noreferrer" title="LinkedIn">{LINKEDIN_ICON}</a>
      </span>
    </div>
    """)


# ------------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} · ATS Resume Analyzer",
    page_icon="📡",
    layout="wide"
)

render_html(CUSTOM_CSS)
render_hero()
render_wave_divider("#1F7A6C")

# ------------------------------------------------------------------
# STEP 1 — choose the type of job to be screened against
# ------------------------------------------------------------------
st.markdown('<span class="wl-eyebrow">Step 1 &middot; Tune in</span>', unsafe_allow_html=True)
st.markdown('<h3 class="wl-section-title">Choose the job you\'re targeting</h3>', unsafe_allow_html=True)

job_categories = get_job_categories(AUTO_DETECT_LABEL)
selected_category = st.selectbox(
    "Job type",
    job_categories,
    label_visibility="collapsed"
)

show_general_matches = True
if selected_category != AUTO_DETECT_LABEL:
    show_general_matches = st.checkbox(
        "Also show my top matches across every job type",
        value=True
    )

render_wave_divider("#E8A33D")

# ------------------------------------------------------------------
# STEP 2 — upload resume
# ------------------------------------------------------------------
st.markdown('<span class="wl-eyebrow">Step 2 &middot; Transmit</span>', unsafe_allow_html=True)
st.markdown('<h3 class="wl-section-title">Upload your resume</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Resume",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    render_status_banner("Resume received — signal acquired.", "good")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="wl-eyebrow">Raw signal</span>', unsafe_allow_html=True)
        st.markdown('<h4 class="wl-section-title">Resume preview</h4>', unsafe_allow_html=True)
        st.text_area("Preview", resume_text[:300], height=220, label_visibility="collapsed")

    # ================================================================
    # MODE A — user picked a specific job type
    # ================================================================
    if selected_category != AUTO_DETECT_LABEL:
        report = ats_analysis_for_category(resume_text, selected_category)

        with col2:
            st.markdown('<span class="wl-eyebrow">Frequency match</span>', unsafe_allow_html=True)
            st.markdown(f'<h4 class="wl-section-title">{html.escape(selected_category)}</h4>', unsafe_allow_html=True)
            render_tuner_gauge(report['ats_score'])

        render_wave_divider("#1F7A6C")

        st.markdown(
            f'<p class="wl-note">Found <strong>{report["postings_found"]}</strong> posting(s) for '
            f'<strong>{html.escape(selected_category)}</strong> in the database.</p>',
            unsafe_allow_html=True
        )
        render_status_banner(
            f'Closest posting: <strong>{html.escape(safe_str(report["recommended_role"]))}</strong> at '
            f'{html.escape(safe_str(report["company"]))} — {fmt_pct(report["similarity_score"])} match',
            "info"
        )

        st.markdown('<span class="wl-eyebrow">Top channels</span>', unsafe_allow_html=True)
        st.markdown(
            f'<h4 class="wl-section-title">Best-matched {html.escape(selected_category)} postings</h4>',
            unsafe_allow_html=True
        )
        if len(report['top_recommendations']) > 0:
            render_job_cards(report['top_recommendations'])
        else:
            render_status_banner("No postings found for this job type yet.", "warn")

        render_wave_divider("#B0413E")

        st.markdown('<span class="wl-eyebrow">Requirements</span>', unsafe_allow_html=True)
        st.markdown(
            f'<h4 class="wl-section-title">Skills the market expects for {html.escape(selected_category)}</h4>',
            unsafe_allow_html=True
        )
        render_skill_chips(report['required_skills'], "neutral", "No skill data available for this job type.")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<span class="wl-eyebrow">In sync</span>', unsafe_allow_html=True)
            st.markdown('<h4 class="wl-section-title">Skills you already bring</h4>', unsafe_allow_html=True)
            render_skill_chips(report['matched_skills'], "matched", "No overlap found yet.")
        with col4:
            st.markdown('<span class="wl-eyebrow">Needs tuning</span>', unsafe_allow_html=True)
            st.markdown(
                f'<h4 class="wl-section-title">Skills to improve for {html.escape(selected_category)}</h4>',
                unsafe_allow_html=True
            )
            render_skill_chips(report['missing_skills'], "missing", "No gaps — nice work!")

        st.markdown('<span class="wl-eyebrow">Next steps</span>', unsafe_allow_html=True)
        st.markdown('<h4 class="wl-section-title">Where to focus</h4>', unsafe_allow_html=True)
        if report['learning_recommendations']:
            for rec in report['learning_recommendations']:
                st.markdown(f'<p class="wl-tip">&rarr; {html.escape(rec)}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="wl-note">No specific learning recommendations available.</p>', unsafe_allow_html=True)

        # ---- Separate, optional section: best matches everywhere ----
        if show_general_matches:
            render_wave_divider("#1F7A6C")
            st.markdown('<span class="wl-eyebrow">Wide scan</span>', unsafe_allow_html=True)
            st.markdown('<h3 class="wl-section-title">Top matches across every job type</h3>', unsafe_allow_html=True)
            general_report = ats_analysis(resume_text)
            render_status_banner(
                f'Best overall match: <strong>{html.escape(safe_str(general_report["recommended_role"]))}</strong> at '
                f'{html.escape(safe_str(general_report["company"]))} — ATS score {fmt_pct(general_report["ats_score"])}',
                "good"
            )
            render_job_cards(general_report['top_recommendations'])

    # ================================================================
    # MODE B — auto-detect, searches every job type
    # ================================================================
    else:
        report = ats_analysis(resume_text)

        with col2:
            st.markdown('<span class="wl-eyebrow">Frequency match</span>', unsafe_allow_html=True)
            st.markdown('<h4 class="wl-section-title">Best overall match</h4>', unsafe_allow_html=True)
            render_tuner_gauge(report['ats_score'])

        render_wave_divider("#1F7A6C")
        render_status_banner(
            f'Best match: <strong>{html.escape(safe_str(report["recommended_role"]))}</strong> at '
            f'{html.escape(safe_str(report["company"]))} — ATS score {fmt_pct(report["ats_score"])}',
            "good"
        )

        st.markdown('<span class="wl-eyebrow">Top channels</span>', unsafe_allow_html=True)
        st.markdown('<h4 class="wl-section-title">Top recommended jobs</h4>', unsafe_allow_html=True)
        render_job_cards(report['top_recommendations'])

        render_wave_divider("#B0413E")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<span class="wl-eyebrow">In sync</span>', unsafe_allow_html=True)
            st.markdown('<h4 class="wl-section-title">Matched skills</h4>', unsafe_allow_html=True)
            render_skill_chips(report['matched_skills'], "matched", "No matched skills found.")
        with col4:
            st.markdown('<span class="wl-eyebrow">Needs tuning</span>', unsafe_allow_html=True)
            st.markdown('<h4 class="wl-section-title">Missing skills</h4>', unsafe_allow_html=True)
            render_skill_chips(report['missing_skills'], "missing", "No missing skills!")

        st.markdown('<span class="wl-eyebrow">Next steps</span>', unsafe_allow_html=True)
        st.markdown('<h4 class="wl-section-title">Learning recommendations</h4>', unsafe_allow_html=True)
        if report['learning_recommendations']:
            for rec in report['learning_recommendations']:
                st.markdown(f'<p class="wl-tip">&rarr; {html.escape(rec)}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="wl-note">No specific learning recommendations available.</p>', unsafe_allow_html=True)

render_wave_divider("#CBD5D8")
render_footer()