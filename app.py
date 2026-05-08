import flet as ft
import threading
import json
import urllib.request
import os
import pickle
import re
ML_AVAILABLE    = False
_ml_models      = None   # {"Random Forest": model, "Logistic Regression": model, ...}
_ml_vectorizer  = None   # TfidfVectorizer

MODEL_CACHE_DIR  = "model_cache"
VECTORIZER_PATH  = os.path.join(MODEL_CACHE_DIR, "vectorizer.pkl")
MODELS_PATH      = os.path.join(MODEL_CACHE_DIR, "models.pkl")

try:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def _clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
        return " ".join(_ps.stem(w) for w in text.split())
    ML_AVAILABLE = True
except Exception:
    def _clean_text(text: str) -> str:
        return str(text).lower()


def load_cached_models() -> bool:
    global _ml_models, _ml_vectorizer
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODELS_PATH):
        try:
            with open(VECTORIZER_PATH, "rb") as f:
                _ml_vectorizer = pickle.load(f)
            with open(MODELS_PATH, "rb") as f:
                _ml_models = pickle.load(f)
            return True
        except Exception:
            pass
    return False


def save_models_to_cache(models: dict, vectorizer) -> None:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(models, f)


def train_from_csv(csv_path: str, status_cb=None) -> tuple:
    global _ml_models, _ml_vectorizer
    if not ML_AVAILABLE:
        return False, "scikit-learn / nltk не установлены"
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score

        if status_cb: status_cb("📂 Загружаем датасет...")
        df = pd.read_csv(csv_path)
        if df.empty:
            return False, "Датасет пустой"
        for col in ["job_role", "skills"]:
            if col not in df.columns:
                return False, f"Нет колонки '{col}' в CSV"

        if status_cb: status_cb("🔧 Препроцессинг...")
        if "certifications" not in df.columns:
            df["certifications"] = ""
        df["text"] = (df["skills"].astype(str) + " " + df["certifications"].astype(str)).apply(_clean_text)

        if status_cb: status_cb("📐 TF-IDF векторизация...")
        vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english",
            ngram_range=(1, 2), min_df=2, max_df=0.85, sublinear_tf=True
        )
        X = vectorizer.fit_transform(df["text"])
        y = df["job_role"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Baseline":            DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "Naive Bayes":         MultinomialNB(),
        }
        for name, model in models.items():
            if status_cb: status_cb(f"🤖 Обучаем {name}...")
            model.fit(X_tr, y_tr)

        accs = {n: accuracy_score(y_te, m.predict(X_te)) for n, m in models.items()}
        best_name = max(accs, key=accs.get)

        _ml_models     = models
        _ml_vectorizer = vectorizer
        save_models_to_cache(models, vectorizer)

        acc_lines = " | ".join(f"{n}: {a:.1%}" for n, a in accs.items())
        return True, f"OK  {acc_lines}  |  Best: {best_name} ({accs[best_name]:.1%})"

    except Exception as ex:
        return False, f"Ошибка: {ex}"


def predict_with_ml(skills_text: str):
    if not _ml_models or not _ml_vectorizer:
        return None
    try:
        vec = _ml_vectorizer.transform([_clean_text(skills_text)])
        per_model = {}
        for name, model in _ml_models.items():
            if name == "Baseline":
                continue
            probs   = model.predict_proba(vec)[0]
            classes = model.classes_
            per_model[name] = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]

        agg = {}
        for tops in per_model.values():
            for role, prob in tops:
                agg[role] = agg.get(role, 0.0) + prob / len(per_model)

        sorted_roles = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:5]
        max_s = sorted_roles[0][1] or 1.0

        def agreement(role):
            return sum(1 for tops in per_model.values() if tops[0][0] == role)

        predictions = [
            {"role": role, "score": round(score / max_s * 95),
             "reason": f"{agreement(role)}/{len(per_model)} models agree"}
            for role, score in sorted_roles
        ]
        breakdown = ", ".join(f"{n}: {tops[0][0]}" for n, tops in per_model.items())

        return {
            "top_role":       predictions[0]["role"],
            "confidence":     predictions[0]["score"],
            "summary":        f"Predicted by {len(per_model)} ML models — {breakdown}.",
            "predictions":    predictions,
            "missing_skills": [],
            "career_tip":     "Train on more data to improve confidence.",
            "source":         "ml",
        }
    except Exception:
        return None


#  CLAUDE API

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-5"

CLAUDE_ENRICH_PROMPT = """You are a career advisor enriching ML model predictions.

Respond ONLY with valid JSON (no markdown):
{
  "summary": "One sentence why these skills fit the top role.",
  "missing_skills": ["skill1", "skill2", "skill3"],
  "career_tip": "One actionable sentence."
}"""

CLAUDE_STANDALONE_PROMPT = """You are a career intelligence AI. Analyze skills and predict IT job roles.

Respond ONLY with valid JSON (no markdown):
{
  "top_role": "Role Name",
  "confidence": 87,
  "summary": "One sentence why.",
  "predictions": [
    {"role": "Role Name", "score": 87, "reason": "Short reason"},
    {"role": "Role Name", "score": 64, "reason": "Short reason"},
    {"role": "Role Name", "score": 41, "reason": "Short reason"},
    {"role": "Role Name", "score": 22, "reason": "Short reason"},
    {"role": "Role Name", "score": 11, "reason": "Short reason"}
  ],
  "missing_skills": ["skill1", "skill2"],
  "career_tip": "One actionable sentence."
}
Roles: Data Scientist, Data Analyst, Backend Developer, Frontend Developer,
DevOps Engineer, QA Engineer, ML Engineer, Project Manager, Full Stack Developer,
Cloud Architect, Security Engineer, Mobile Developer."""


def _claude_request(api_key: str, system: str, user_msg: str) -> dict:
    payload = json.dumps({
        "model": CLAUDE_MODEL, "max_tokens": 512,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode()
    req = urllib.request.Request(
        CLAUDE_API_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read())["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)


def enrich_with_claude(api_key: str, skills_text: str, ml_result: dict) -> dict:
    user_msg = (
        f"Skills: {skills_text}\n"
        f"Top role: {ml_result['top_role']} ({ml_result['confidence']}%)\n"
        f"Others: {', '.join(p['role'] for p in ml_result['predictions'][1:])}"
    )
    enrichment = _claude_request(api_key, CLAUDE_ENRICH_PROMPT, user_msg)
    return {**ml_result,
            "summary":        enrichment.get("summary",        ml_result["summary"]),
            "missing_skills": enrichment.get("missing_skills", []),
            "career_tip":     enrichment.get("career_tip",     ml_result["career_tip"]),
            "source":         "ml+claude"}


def predict_claude_only(api_key: str, skills_text: str) -> dict:
    result = _claude_request(api_key, CLAUDE_STANDALONE_PROMPT,
                              f"Analyze these skills: {skills_text}")
    result["source"] = "claude"
    return result


#  KEYWORD FALLBACK
SKILL_ROLE_MAP = {
    "python":       {"Data Scientist": 0.4, "Backend Developer": 0.35, "ML Engineer": 0.25},
    "sql":          {"Data Analyst": 0.6, "Data Scientist": 0.3, "Backend Developer": 0.1},
    "react":        {"Frontend Developer": 0.8, "Full Stack Developer": 0.2},
    "javascript":   {"Frontend Developer": 0.6, "Full Stack Developer": 0.3, "Backend Developer": 0.1},
    "docker":       {"DevOps Engineer": 0.6, "Backend Developer": 0.25, "ML Engineer": 0.15},
    "kubernetes":   {"DevOps Engineer": 0.7, "Cloud Architect": 0.2, "Backend Developer": 0.1},
    "tensorflow":   {"ML Engineer": 0.7, "Data Scientist": 0.3},
    "pytorch":      {"ML Engineer": 0.65, "Data Scientist": 0.35},
    "pandas":       {"Data Analyst": 0.45, "Data Scientist": 0.45, "ML Engineer": 0.1},
    "selenium":     {"QA Engineer": 0.85, "Backend Developer": 0.15},
    "java":         {"Backend Developer": 0.75, "Full Stack Developer": 0.25},
    "jira":         {"Project Manager": 0.7, "QA Engineer": 0.3},
    "agile":        {"Project Manager": 0.8, "QA Engineer": 0.2},
    "aws":          {"Cloud Architect": 0.5, "DevOps Engineer": 0.35, "Backend Developer": 0.15},
    "fastapi":      {"Backend Developer": 0.6, "ML Engineer": 0.4},
    "tableau":      {"Data Analyst": 0.9, "Data Scientist": 0.1},
    "go":           {"Backend Developer": 0.65, "DevOps Engineer": 0.35},
    "flutter":      {"Mobile Developer": 0.85, "Full Stack Developer": 0.15},
    "llm":          {"ML Engineer": 0.65, "Data Scientist": 0.35},
}
ALL_ROLES = [
    "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer",
    "DevOps Engineer", "QA Engineer", "ML Engineer", "Project Manager",
    "Full Stack Developer", "Cloud Architect", "Security Engineer", "Mobile Developer"
]

def fallback_predict(skills_text: str):
    skills = [s.strip().lower() for s in skills_text.replace(",", " ").split() if s.strip()]
    scores = {r: 0.0 for r in ALL_ROLES}
    for skill in skills:
        for kw, rw in SKILL_ROLE_MAP.items():
            if kw in skill or skill in kw:
                for role, w in rw.items():
                    scores[role] += w
    total = sum(scores.values())
    if not total:
        return None
    ranked = sorted([(r, s/total*100) for r, s in scores.items() if s], key=lambda x: x[1], reverse=True)[:5]
    top_role, top_pct = ranked[0]
    return {
        "top_role":       top_role,
        "confidence":     int(top_pct),
        "summary":        f"Keyword pattern matched your skills to {top_role}.",
        "predictions":    [{"role": r, "score": int(s), "reason": "Keyword match"} for r, s in ranked],
        "missing_skills": [],
        "career_tip":     "Train ML models on jobs.csv for better predictions.",
        "source":         "keyword",
    }

#  UNIFIED PREDICT


SOURCE_LABELS = {
    "ml+claude": ("🤖+✦", "ML Models + Claude AI", "#34d399"),
    "ml":        ("🤖",   "Your ML Models",         "#818cf8"),
    "claude":    ("✦",    "Claude AI",              "#f472b6"),
    "keyword":   ("🔑",   "Keyword Fallback",       "#fbbf24"),
}

def unified_predict(skills_text: str, api_key: str, status_cb=None) -> dict:
    ml_result = predict_with_ml(skills_text)

    if ml_result:
        if api_key:
            try:
                if status_cb: status_cb("✦ Обогащаем через Claude AI...")
                return enrich_with_claude(api_key, skills_text, ml_result)
            except Exception:
                pass
        return ml_result

    if api_key:
        try:
            if status_cb: status_cb("✦ Предсказываем через Claude AI...")
            return predict_claude_only(api_key, skills_text)
        except Exception:
            pass

    if status_cb: status_cb("🔑 Keyword fallback...")
    return fallback_predict(skills_text) or {
        "top_role": "Unknown", "confidence": 0,
        "summary": "No skills matched.", "predictions": [],
        "missing_skills": [], "career_tip": "Try adding specific technical skills.",
        "source": "keyword",
    }




BG      = "#09090f"
BG2     = "#13131f"
BG3     = "#1c1c2e"
BG4     = "#242438"
ACCENT  = "#7c6af7"
ACCENT2 = "#a89cf8"
ACCENT3 = "#c4bbff"
TEXT    = "#f0eeff"
TEXT2   = "#8b85b5"
TEXT3   = "#5a5480"
BORDER  = "#2a2842"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER  = "#f87171"

ROLE_PALETTE = {
    "Data Scientist":       ("#818cf8", "#1e1b4b"),
    "Data Analyst":         ("#38bdf8", "#082f49"),
    "Backend Developer":    ("#34d399", "#064e3b"),
    "Frontend Developer":   ("#fb923c", "#431407"),
    "DevOps Engineer":      ("#f87171", "#450a0a"),
    "QA Engineer":          ("#c084fc", "#3b0764"),
    "ML Engineer":          ("#f472b6", "#4a044e"),
    "Project Manager":      ("#2dd4bf", "#042f2e"),
    "Full Stack Developer": ("#facc15", "#422006"),
    "Cloud Architect":      ("#60a5fa", "#1e3a5f"),
    "Security Engineer":    ("#a78bfa", "#2e1065"),
    "Mobile Developer":     ("#4ade80", "#052e16"),
}
ROLE_ICONS = {
    "Data Scientist": "📊", "Data Analyst": "📈", "Backend Developer": "⚙️",
    "Frontend Developer": "🎨", "DevOps Engineer": "🔧", "QA Engineer": "🧪",
    "ML Engineer": "🤖", "Project Manager": "📋", "Full Stack Developer": "💻",
    "Cloud Architect": "☁️", "Security Engineer": "🛡️", "Mobile Developer": "📱",
}
SUGGESTED_SKILLS = [
    "Python", "SQL", "React", "Docker", "TensorFlow", "Selenium", "Jira",
    "Pandas", "AWS", "PyTorch", "Kubernetes", "FastAPI", "Tableau",
    "Scrum", "Vue", "TypeScript", "Go", "Flutter", "Next.js", "LLM",
]
TEAM = [
    {"name": "Aliya Yskak",   "role": "ML Engineer",    "emoji": "🤖", "color": "#f472b6"},
    {"name": "Meruyert Askar", "role": "Data Engineer",  "emoji": "📊", "color": "#38bdf8"},
    {"name": "Bekzhan Karim", "role": "Frontend & Demo","emoji": "🎨", "color": "#fb923c"},
]



def nav_bar(page: ft.Page, current: str):
    def btn(label, route):
        active = current == route
        return ft.TextButton(
            content=ft.Text(label, size=13,
                weight=ft.FontWeight.W_600 if active else ft.FontWeight.NORMAL,
                color=ACCENT3 if active else TEXT2),
            on_click=lambda e, r=route: page.go(r),
            style=ft.ButtonStyle(overlay_color=ft.colors.with_opacity(0.06, ACCENT),
                                 shape=ft.RoundedRectangleBorder(radius=8)),
        )
    return ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Text("Job", size=17, weight=ft.FontWeight.W_800, color=TEXT),
                ft.Text("Predictor", size=17, weight=ft.FontWeight.W_800, color=ACCENT2),
                ft.Container(
                    content=ft.Text("AI", size=10, color=ACCENT3, weight=ft.FontWeight.W_700),
                    bgcolor=ft.colors.with_opacity(0.18, ACCENT), border_radius=4,
                    padding=ft.padding.symmetric(horizontal=6, vertical=2), margin=ft.margin.only(left=4),
                ),
            ]),
            ft.Row([btn("Home", "/"), btn("Demo", "/demo"), btn("About", "/about")], spacing=2),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        bgcolor=ft.colors.with_opacity(0.92, BG),
        padding=ft.padding.symmetric(horizontal=36, vertical=14),
        border=ft.border.only(bottom=ft.BorderSide(1, BORDER)),
    )

def badge(text, color=ACCENT2):
    return ft.Container(
        content=ft.Text(text, size=12, color=color, weight=ft.FontWeight.W_600),
        bgcolor=ft.colors.with_opacity(0.12, ACCENT),
        border=ft.border.all(1, ft.colors.with_opacity(0.3, color)),
        border_radius=20, padding=ft.padding.symmetric(horizontal=14, vertical=5),
    )


#  HOME PAGE

def home_page(page: ft.Page):
    ml_loaded = load_cached_models()
    engine_status = (
        f"🤖 ML models ready ({len([n for n in (_ml_models or {}) if n != 'Baseline'])} models cached)"
        if ml_loaded else "🔑 No trained models — open Demo to train on jobs.csv"
    )
    features = [
        ("🤖", "Your ML Models",  "Random Forest, Logistic Regression & Naive Bayes from project.py"),
        ("✦",  "Claude AI Boost", "Optional enrichment: career tips & missing skills via Claude"),
        ("📈", "Confidence Bars", "Per-model agreement scores shown visually"),
        ("🔁", "Train Anytime",   "Upload jobs.csv and retrain models from the Demo page"),
    ]
    steps = [
        ("01", "Train Models",   "Load jobs.csv to train your sklearn models", "📂"),
        ("02", "Enter Skills",   "Type your technical skills in the predictor", "📝"),
        ("03", "Get Predictions","ML models rank job roles with confidence scores", "🎯"),
    ]

    return ft.Column([
        nav_bar(page, "/"),
        ft.Column([
            ft.Container(
                content=ft.Column([
                    badge("✦  INF375 · Skill-Based Career Intelligence"),
                    ft.Container(height=24),
                    ft.Text("Predict Your\nIT Career Path",
                            size=58, weight=ft.FontWeight.W_900, color=TEXT,
                            text_align=ft.TextAlign.CENTER),
                    ft.Text("powered by your own ML models",
                            size=36, weight=ft.FontWeight.W_900, color=ACCENT2,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(height=18),
                    ft.Text("Train Random Forest, Logistic Regression & Naive Bayes\n"
                            "on real job data — then predict roles from any skill set.",
                            size=15, color=TEXT2, text_align=ft.TextAlign.CENTER),
                    ft.Container(height=14),
                    ft.Container(
                        content=ft.Text(engine_status, size=13,
                                        color=SUCCESS if ml_loaded else WARNING),
                        bgcolor=ft.colors.with_opacity(0.1, SUCCESS if ml_loaded else WARNING),
                        border=ft.border.all(1, ft.colors.with_opacity(0.3, SUCCESS if ml_loaded else WARNING)),
                        border_radius=10, padding=ft.padding.symmetric(horizontal=16, vertical=8),
                    ),
                    ft.Container(height=30),
                    ft.Row([
                        ft.ElevatedButton("Try the Demo →",
                            on_click=lambda e: page.go("/demo"),
                            style=ft.ButtonStyle(bgcolor=ACCENT, color=TEXT,
                                shape=ft.RoundedRectangleBorder(radius=12),
                                padding=ft.padding.symmetric(horizontal=32, vertical=18), elevation=0)),
                        ft.OutlinedButton("About the Project",
                            on_click=lambda e: page.go("/about"),
                            style=ft.ButtonStyle(color=TEXT2, side=ft.BorderSide(1, BORDER),
                                shape=ft.RoundedRectangleBorder(radius=12),
                                padding=ft.padding.symmetric(horizontal=32, vertical=18),
                                overlay_color=ft.colors.with_opacity(0.05, ACCENT))),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=14),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                padding=ft.padding.symmetric(vertical=80, horizontal=40),
                alignment=ft.alignment.center,
            ),

            ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Text(v, size=28, weight=ft.FontWeight.W_900, color=ACCENT2),
                            ft.Text(l, size=12, color=TEXT3),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                        expand=True, alignment=ft.alignment.center,
                    ) for v, l in [("3", "ML Models"), ("50+", "Skills"), ("ML+Claude", "Engine"), ("INF375", "Course")]
                ]),
                bgcolor=BG2, border=ft.border.all(1, BORDER), border_radius=16,
                padding=ft.padding.symmetric(vertical=26, horizontal=40),
                margin=ft.margin.symmetric(horizontal=40),
            ),

            ft.Container(height=40),
            ft.Container(
                content=ft.Column([
                    ft.Text("How It Works", size=32, weight=ft.FontWeight.W_800, color=TEXT,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(height=28),
                    ft.Row([
                        ft.Container(
                            content=ft.Column([
                                ft.Row([ft.Text(num, size=12, color=ACCENT2, weight=ft.FontWeight.W_700),
                                        ft.Text(ico, size=18)], spacing=6),
                                ft.Container(height=10),
                                ft.Text(title, size=15, weight=ft.FontWeight.W_700, color=TEXT),
                                ft.Container(height=4),
                                ft.Text(desc, size=12, color=TEXT2),
                            ], spacing=2),
                            bgcolor=BG2, border=ft.border.all(1, BORDER),
                            border_radius=14, padding=22, expand=True,
                        ) for num, title, desc, ico in steps], spacing=14,
                    ),
                ]),
                padding=ft.padding.symmetric(vertical=0, horizontal=40),
            ),

            ft.Container(height=40),
            ft.Container(
                content=ft.Column([
                    ft.Text("Why JobPredictor AI?", size=32, weight=ft.FontWeight.W_800, color=TEXT,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(height=28),
                    ft.Row([
                        ft.Container(
                            content=ft.Column([
                                ft.Text(emoji, size=24),
                                ft.Container(height=8),
                                ft.Text(title, size=14, weight=ft.FontWeight.W_700, color=TEXT),
                                ft.Container(height=4),
                                ft.Text(desc, size=12, color=TEXT2, text_align=ft.TextAlign.CENTER),
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                            bgcolor=BG2, border=ft.border.all(1, BORDER),
                            border_radius=14, padding=20, expand=True,
                        ) for emoji, title, desc in features], spacing=14,
                    ),
                ]),
                padding=ft.padding.symmetric(vertical=0, horizontal=40),
            ),
            ft.Container(height=60),
        ], scroll=ft.ScrollMode.AUTO, expand=True),
    ], expand=True, spacing=0)



#  DEMO PAGE

def demo_page(page: ft.Page):
    load_cached_models()

    api_key_field = ft.TextField(
        hint_text="sk-ant-api03-... (опционально — добавляет советы через Claude)",
        bgcolor=BG3, border_color=BORDER, focused_border_color=ft.colors.with_opacity(0.6, ACCENT),
        color=TEXT2, hint_style=ft.TextStyle(color=TEXT3),
        border_radius=10, password=True, can_reveal_password=True, text_size=13,
    )
    skills_field = ft.TextField(
        hint_text="e.g.  Python  SQL  TensorFlow  Docker  FastAPI",
        bgcolor=BG3, border_color=BORDER, focused_border_color=ft.colors.with_opacity(0.7, ACCENT),
        color=TEXT, hint_style=ft.TextStyle(color=TEXT3),
        border_radius=12, min_lines=3, max_lines=6, text_size=15,
    )
    csv_field = ft.TextField(
        hint_text="jobs.csv",
        value="jobs.csv",
        bgcolor=BG3, border_color=BORDER, focused_border_color=ft.colors.with_opacity(0.6, ACCENT),
        color=TEXT2, hint_style=ft.TextStyle(color=TEXT3),
        border_radius=10, text_size=13, expand=True,
    )

    status_text  = ft.Text("", size=12, color=TEXT3)
    error_text   = ft.Text("", color=DANGER, size=13)
    train_status = ft.Text("", size=12, color=TEXT2)

    top_card     = ft.Container(visible=False)
    source_badge = ft.Container(visible=False)
    bars_col     = ft.Column([], spacing=10)
    tip_card     = ft.Container(visible=False)
    missing_card = ft.Container(visible=False)
    results_area = ft.Column([
        source_badge, ft.Container(height=6), top_card,
        ft.Container(height=8), bars_col,
        ft.Container(height=8), tip_card, missing_card,
    ], visible=False)

    def add_skill(s):
        cur = skills_field.value or ""
        if s.lower() not in cur.lower():
            skills_field.value = (cur + " " + s).strip()
            page.update()

    chips_row = ft.Row(
        [ft.Container(
            content=ft.Text(s, size=12, color=TEXT2),
            bgcolor=BG3, border=ft.border.all(1, BORDER), border_radius=8,
            padding=ft.padding.symmetric(horizontal=10, vertical=5),
            on_click=lambda e, s=s: add_skill(s), ink=True,
        ) for s in SUGGESTED_SKILLS],
        wrap=True, spacing=6, run_spacing=6,
    )

    has_ml = bool(_ml_models)
    ml_indicator = ft.Container(
        content=ft.Row([
            ft.Icon(ft.icons.CIRCLE, size=8, color=SUCCESS if has_ml else WARNING),
            ft.Text(
                f"ML models ready ({len([n for n in (_ml_models or {}) if n != 'Baseline'])} loaded)" if has_ml
                else "No ML models — train first or use fallback",
                size=12, color=SUCCESS if has_ml else WARNING,
            ),
        ], spacing=6),
        bgcolor=ft.colors.with_opacity(0.08, SUCCESS if has_ml else WARNING),
        border=ft.border.all(1, ft.colors.with_opacity(0.2, SUCCESS if has_ml else WARNING)),
        border_radius=8, padding=ft.padding.symmetric(horizontal=12, vertical=6),
    )

    def refresh_indicator():
        has = bool(_ml_models)
        count = len([n for n in (_ml_models or {}) if n != "Baseline"])
        ml_indicator.content.controls[0].color = SUCCESS if has else WARNING
        ml_indicator.content.controls[1].value = (
            f"ML models ready ({count} loaded)" if has
            else "No ML models — train first or use fallback"
        )
        ml_indicator.content.controls[1].color = SUCCESS if has else WARNING
        ml_indicator.bgcolor = ft.colors.with_opacity(0.08, SUCCESS if has else WARNING)
        ml_indicator.border  = ft.border.all(1, ft.colors.with_opacity(0.2, SUCCESS if has else WARNING))

    def render_results(data: dict):
        src = data.get("source", "keyword")
        ico, label, col = SOURCE_LABELS.get(src, ("🔑", "Fallback", WARNING))

        source_badge.content = ft.Container(
            content=ft.Row([ft.Text(ico, size=14),
                             ft.Text(f"Predicted by: {label}", size=12, color=col,
                                     weight=ft.FontWeight.W_600)], spacing=8),
            bgcolor=ft.colors.with_opacity(0.1, col),
            border=ft.border.all(1, ft.colors.with_opacity(0.25, col)),
            border_radius=8, padding=ft.padding.symmetric(horizontal=14, vertical=7),
        )
        source_badge.visible = True

        fg, _ = ROLE_PALETTE.get(data["top_role"], (ACCENT, BG3))
        top_card.content = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("✦ Best Match", size=11,
                            color=ft.colors.with_opacity(0.7, fg), weight=ft.FontWeight.W_600),
                    ft.Container(height=4),
                    ft.Text(ROLE_ICONS.get(data["top_role"], "💼") + "  " + data["top_role"],
                            size=26, weight=ft.FontWeight.W_800, color=TEXT),
                    ft.Container(height=6),
                    ft.Text(data.get("summary", ""), size=13, color=TEXT2),
                ], expand=True, spacing=0),
                ft.Column([
                    ft.Text(f"{data['confidence']}%", size=40, weight=ft.FontWeight.W_900,
                            color=fg, text_align=ft.TextAlign.CENTER),
                    ft.Text("confidence", size=11, color=TEXT3, text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
            ], spacing=24),
            bgcolor=ft.colors.with_opacity(0.1, fg),
            border=ft.border.all(1, ft.colors.with_opacity(0.3, fg)),
            border_radius=16, padding=24,
        )
        top_card.visible = True

        bars_col.controls.clear()
        max_s = max((p["score"] for p in data["predictions"]), default=1)
        for pred in data["predictions"]:
            role, score, reason = pred["role"], pred["score"], pred.get("reason", "")
            fg2, _ = ROLE_PALETTE.get(role, (ACCENT, BG3))
            bar_w = max(4, int(score / max_s * 260))
            bars_col.controls.append(ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(ROLE_ICONS.get(role, "💼"), size=14),
                        ft.Text(role, size=13, color=TEXT, weight=ft.FontWeight.W_600, expand=True),
                        ft.Text(f"{score}%", size=13, color=fg2, weight=ft.FontWeight.W_700),
                    ]),
                    ft.Container(
                        content=ft.Container(width=bar_w, height=5, bgcolor=fg2, border_radius=3),
                        bgcolor=BG4, border_radius=3, height=5,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    ),
                    ft.Text(reason, size=11, color=TEXT3),
                ], spacing=6),
                bgcolor=BG2, border=ft.border.all(1, BORDER),
                border_radius=12, padding=ft.padding.symmetric(horizontal=16, vertical=12),
            ))

        tip = data.get("career_tip", "")
        if tip:
            tip_card.content = ft.Container(
                content=ft.Row([ft.Text("💡", size=18),
                                 ft.Text(tip, size=13, color=TEXT2, expand=True)], spacing=12),
                bgcolor=ft.colors.with_opacity(0.08, WARNING),
                border=ft.border.all(1, ft.colors.with_opacity(0.25, WARNING)),
                border_radius=12, padding=16,
            )
            tip_card.visible = True

        missing = data.get("missing_skills", [])
        if missing:
            missing_card.content = ft.Container(
                content=ft.Column([
                    ft.Text("Skills to learn next:", size=12, color=TEXT3, weight=ft.FontWeight.W_600),
                    ft.Container(height=8),
                    ft.Row([
                        ft.Container(
                            content=ft.Text(sk, size=12, color=ACCENT2),
                            bgcolor=ft.colors.with_opacity(0.1, ACCENT),
                            border=ft.border.all(1, ft.colors.with_opacity(0.2, ACCENT2)),
                            border_radius=8, padding=ft.padding.symmetric(horizontal=10, vertical=5),
                        ) for sk in missing
                    ], wrap=True, spacing=6, run_spacing=6),
                ]),
                bgcolor=BG2, border=ft.border.all(1, BORDER),
                border_radius=12, padding=16,
            )
            missing_card.visible = True

        results_area.visible = True
        page.update()

    predict_btn = ft.ElevatedButton(
        "✦  Predict Role",
        style=ft.ButtonStyle(bgcolor=ACCENT, color=TEXT,
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=ft.padding.symmetric(horizontal=28, vertical=15), elevation=0),
    )
    clear_btn = ft.TextButton("Clear", style=ft.ButtonStyle(color=TEXT3))
    train_btn = ft.OutlinedButton(
        "🤖 Train Models",
        style=ft.ButtonStyle(color=ACCENT2,
            side=ft.BorderSide(1, ft.colors.with_opacity(0.4, ACCENT2)),
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=ft.padding.symmetric(horizontal=18, vertical=12)),
    )

    def do_predict(_):
        text = (skills_field.value or "").strip()
        if not text:
            error_text.value = "Введи хотя бы один навык."
            page.update()
            return
        error_text.value = ""
        results_area.visible = False
        top_card.visible = source_badge.visible = tip_card.visible = missing_card.visible = False
        bars_col.controls.clear()
        predict_btn.disabled = True
        page.update()

        def worker():
            api_key = (api_key_field.value or "").strip()
            data = unified_predict(text, api_key,
                                   lambda s: (setattr(status_text, "value", s), page.update()))
            status_text.value = ""
            predict_btn.disabled = False
            render_results(data)

        threading.Thread(target=worker, daemon=True).start()

    def do_clear(_):
        skills_field.value = error_text.value = status_text.value = ""
        results_area.visible = top_card.visible = source_badge.visible = False
        tip_card.visible = missing_card.visible = False
        bars_col.controls.clear()
        predict_btn.disabled = False
        page.update()

    def do_train(_):
        csv_path = (csv_field.value or "jobs.csv").strip()
        train_btn.disabled = True
        train_status.value = "⏳ Начинаем обучение..."
        page.update()

        def worker():
            ok, msg = train_from_csv(csv_path,
                                     lambda s: (setattr(train_status, "value", s), page.update()))
            train_status.value = ("✓ " if ok else "✗ ") + msg
            train_btn.disabled = False
            refresh_indicator()
            page.update()

        threading.Thread(target=worker, daemon=True).start()

    predict_btn.on_click = do_predict
    clear_btn.on_click   = do_clear
    train_btn.on_click   = do_train

    return ft.Column([
        nav_bar(page, "/demo"),
        ft.Column([
            ft.Container(
                content=ft.Column([
                    ft.Text("Role Predictor", size=36, weight=ft.FontWeight.W_900, color=TEXT),
                    ft.Text("Your ML models + optional Claude AI enrichment",
                            size=14, color=TEXT2),
                    ft.Container(height=22),

                    ml_indicator,
                    ft.Container(height=18),

                    # Train panel
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.icons.PSYCHOLOGY_OUTLINED, size=14, color=TEXT3),
                                ft.Text("Обучение моделей", size=13, color=TEXT2,
                                        weight=ft.FontWeight.W_600),
                            ], spacing=6),
                            ft.Container(height=8),
                            ft.Row([csv_field, train_btn], spacing=10,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Container(height=4),
                            train_status,
                        ]),
                        bgcolor=BG3, border=ft.border.all(1, BORDER),
                        border_radius=12, padding=16,
                    ),
                    ft.Container(height=14),

                    # API key panel
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.icons.KEY_OUTLINED, size=14, color=TEXT3),
                                ft.Text("Claude API Key", size=12, color=TEXT3,
                                        weight=ft.FontWeight.W_600),
                                ft.Text("(опционально)", size=11, color=TEXT3),
                            ], spacing=6),
                            ft.Container(height=6),
                            api_key_field,
                        ]),
                        bgcolor=BG3, border=ft.border.all(1, BORDER),
                        border_radius=12, padding=16,
                    ),
                    ft.Container(height=18),

                    ft.Text("Твои навыки", size=14, color=TEXT2, weight=ft.FontWeight.W_600),
                    ft.Container(height=6),
                    skills_field,
                    ft.Container(height=10),
                    ft.Text("Быстрое добавление:", size=12, color=TEXT3),
                    ft.Container(height=4),
                    chips_row,
                    ft.Container(height=6),
                    error_text,
                    ft.Container(height=16),

                    ft.Row([predict_btn, clear_btn,
                            ft.Container(expand=True), status_text],
                           spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=24),
                    results_area,
                ]),
                bgcolor=BG2, border=ft.border.all(1, BORDER),
                border_radius=20, padding=36, margin=ft.margin.all(32),
            ),
        ], scroll=ft.ScrollMode.AUTO, expand=True),
    ], expand=True, spacing=0)


#  ABOUT PAGE

def about_page(page: ft.Page):
    tech_stack = [
        ("🐍", "Python 3.11",   "Core language"),
        ("🌲", "Random Forest", "sklearn ensemble"),
        ("📊", "Logistic Reg.", "sklearn linear"),
        ("🅱️",  "Naive Bayes",  "sklearn NB"),
        ("📐", "TF-IDF",        "Feature extraction"),
        ("✦",  "Claude AI",     "Enrichment layer"),
        ("🎯", "Flet",          "UI Framework"),
    ]
    pipeline = [
        ("jobs.csv",       "Raw dataset",         BG3,                                BORDER),
        ("preprocess()",   "Clean + TF-IDF",      ft.colors.with_opacity(0.1, ACCENT),   ft.colors.with_opacity(0.3, ACCENT2)),
        ("train_models()", "3 sklearn models",    ft.colors.with_opacity(0.1, SUCCESS),  ft.colors.with_opacity(0.3, SUCCESS)),
        ("predict()",      "Aggregate scores",    ft.colors.with_opacity(0.1, WARNING),  ft.colors.with_opacity(0.3, WARNING)),
        ("Claude enrich",  "Tips + missing",      ft.colors.with_opacity(0.1, "#f472b6"),ft.colors.with_opacity(0.3, "#f472b6")),
    ]

    return ft.Column([
        nav_bar(page, "/about"),
        ft.Column([
            ft.Container(
                content=ft.Column([
                    badge("INF375 · Artificial Intelligence · Final Project"),
                    ft.Container(height=16),
                    ft.Text("About the Project", size=40, weight=ft.FontWeight.W_900, color=TEXT),
                    ft.Container(height=10),
                    ft.Text(
                        "Trains Random Forest, Logistic Regression and Naive Bayes on real "
                        "job listings from hh.ru and tgaijobs.ru. Claude AI optionally enriches "
                        "predictions with career tips and missing skill suggestions.",
                        size=14, color=TEXT2,
                    ),

                    ft.Container(height=36),
                    ft.Text("ML Pipeline", size=22, weight=ft.FontWeight.W_800, color=TEXT),
                    ft.Container(height=14),
                    ft.Row([
                        ft.Container(
                            content=ft.Column([
                                ft.Text(name, size=13, weight=ft.FontWeight.W_700, color=TEXT),
                                ft.Text(desc, size=11, color=TEXT2),
                            ], spacing=4),
                            bgcolor=bg, border=ft.border.all(1, bord),
                            border_radius=10, padding=14, expand=True,
                        ) for name, desc, bg, bord in pipeline], spacing=8,
                    ),

                    ft.Container(height=32),
                    ft.Text("Tech Stack", size=22, weight=ft.FontWeight.W_800, color=TEXT),
                    ft.Container(height=14),
                    ft.Row([
                        ft.Container(
                            content=ft.Column([
                                ft.Text(emoji, size=22),
                                ft.Container(height=6),
                                ft.Text(name, size=12, weight=ft.FontWeight.W_700, color=TEXT),
                                ft.Text(desc, size=11, color=TEXT2),
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                            bgcolor=BG3, border=ft.border.all(1, BORDER),
                            border_radius=12, padding=16, expand=True, alignment=ft.alignment.center,
                        ) for emoji, name, desc in tech_stack],
                        wrap=True, spacing=10, run_spacing=10,
                    ),

                    ft.Container(height=32),
                    ft.Text("Team", size=22, weight=ft.FontWeight.W_800, color=TEXT),
                    ft.Container(height=14),
                    ft.Column([
                        ft.Container(
                            content=ft.Row([
                                ft.Container(content=ft.Text(m["emoji"], size=22),
                                    bgcolor=ft.colors.with_opacity(0.12, m["color"]),
                                    border_radius=50, width=48, height=48,
                                    alignment=ft.alignment.center),
                                ft.Column([
                                    ft.Text(m["name"], size=14, weight=ft.FontWeight.W_700, color=TEXT),
                                    ft.Text(m["role"], size=12, color=TEXT2),
                                ], spacing=2, expand=True),
                            ], spacing=14),
                            bgcolor=BG3, border=ft.border.all(1, BORDER),
                            border_radius=12, padding=ft.padding.symmetric(horizontal=18, vertical=14),
                        ) for m in TEAM], spacing=8,
                    ),

                    ft.Container(height=32),
                    ft.Text("Dataset", size=22, weight=ft.FontWeight.W_800, color=TEXT),
                    ft.Container(height=14),
                    ft.Container(
                        content=ft.Row([
                            ft.Container(
                                content=ft.Column([
                                    ft.Text(ico, size=18),
                                    ft.Container(height=4),
                                    ft.Text(lbl, size=11, color=TEXT3, weight=ft.FontWeight.W_600),
                                    ft.Text(val, size=13, color=TEXT),
                                ], spacing=2),
                                bgcolor=BG4, border=ft.border.all(1, BORDER),
                                border_radius=10, padding=14, expand=True,
                            ) for ico, lbl, val in [
                                ("📡", "Source",     "hh.ru + tgaijobs.ru"),
                                ("🔧", "Collection", "requests + Selenium"),
                                ("📂", "Fields",     "title, skills, exp, salary"),
                                ("🎓", "Purpose",    "INF375 supervised ML"),
                            ]], spacing=10,
                        ),
                        bgcolor=BG2, border=ft.border.all(1, BORDER),
                        border_radius=14, padding=20,
                    ),
                    ft.Container(height=50),
                ], spacing=0),
                padding=ft.padding.symmetric(horizontal=40, vertical=44),
            ),
        ], scroll=ft.ScrollMode.AUTO, expand=True),
    ], expand=True, spacing=0)


#  MAIN

def main(page: ft.Page):
    page.title       = "JobPredictor AI — INF375"
    page.theme_mode  = ft.ThemeMode.DARK
    page.bgcolor     = BG
    page.padding     = 0
    page.window.width      = 1160
    page.window.height     = 800
    page.window.min_width  = 860
    page.window.min_height = 620

    load_cached_models()

    def route_change(e):
        page.views.clear()
        route = page.route
        if route == "/demo":
            v = ft.View("/demo",  [demo_page(page)],  bgcolor=BG, padding=0)
        elif route == "/about":
            v = ft.View("/about", [about_page(page)], bgcolor=BG, padding=0)
        else:
            v = ft.View("/",      [home_page(page)],  bgcolor=BG, padding=0)
        page.views.append(v)
        page.update()

    page.on_route_change = route_change
    page.go("/")


ft.app(target=main)