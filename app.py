"""
נץ תקומה – עוזר נתונים בית ספרי
Phase 1: File Upload · Validation · Dashboard · Summary Cards · Charts
Phase 2: Hebrew Chatbot · Intent Classification · Predefined Analysis Functions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from intent_classifier import classify_with_context, INTENT_LABELS
from chat_engine import (
    ChatContext, ChatResponse, handle_intent,
    _extract_grade, _extract_class, _extract_mikbatz,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="נץ תקומה – עוזר נתונים בית ספרי",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_FILES = {
    "status": {
        "label": "FAKE_perut_status_miluy.xlsx",
        "description": "סטטוס מילוי שאלון",
        "required_cols": ["ת.ז תלמיד", "שם תלמיד", "שכבה", "מקבילה", "מחנך", "סטטוס מילוי שאלון"],
        "icon": "📋",
    },
    "tosot": {
        "label": "FAKE_perut_tosot_mipuy.xlsx",
        "description": "תוצאות מיפוי (שאלון מלא)",
        "required_cols": ["מס' זהות תלמיד", "שם תלמיד", "שכבה", "מקבילה", "מקבץ", "מיומנות", "היגד", "תשובה"],
        "icon": "📊",
    },
    "tochnit": {
        "label": "FAKE_perut_shiuch_tochnit.xlsx",
        "description": "שיוך תוכניות חינוכיות",
        "required_cols": ["מס' זהות תלמיד", "שם תלמיד", "שכבה", "מקבילה", "מקבץ", "שם תוכנית חינוכית"],
        "icon": "🎓",
    },
}

GRADE_ORDER = ["א", "ב", "ג", "ד", "ה", "ו"]

# ── Pastel palette (Ministry-of-Education style) ────────────────────────────
PALETTE = {
    "navy":   "#4F7EA8",
    "teal":   "#6DB7A3",
    "green":  "#A5B879",
    "orange": "#F3BE72",
    "purple": "#9A7AA0",
    "sky":    "#9EC3D7",
    "red":    "#D95F59",
    "grey":   "#B0BCCC",
}
CHART_BG = "#ffffff"

# Explicit light-theme defaults applied to every Plotly figure.
# Prevents dark-mode browser/OS theme from bleeding into chart fonts and axes.
_CHART_FONT_COLOR = "#1a2340"
_CHART_GRID_COLOR = "#e8ecf2"

def _apply_light_theme(fig, height=None):
    """Apply explicit light-theme colours to every Plotly figure.

    Sets font, axis, legend, title, and bar-label colours explicitly so that
    no colour is left transparent or inherited — which breaks in dark mode.
    Call this on every chart before returning it.
    """
    font_cfg = dict(color=_CHART_FONT_COLOR,
                    family="Segoe UI, Noto Sans Hebrew, Arial, sans-serif")
    tick_cfg = dict(color=_CHART_FONT_COLOR)

    updates = dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=font_cfg,
    )
    if height is not None:
        updates["height"] = height

    fig.update_layout(**updates)

    # Style the title font ONLY when the figure already has a non-empty title.
    # Never call update_layout(title=...) or title_font=... on a figure with
    # no title — doing so creates a Title object whose text Plotly may serialise
    # as "undefined" in the SVG.
    existing_title = ""
    if fig.layout.title and fig.layout.title.text is not None:
        existing_title = str(fig.layout.title.text).strip()

    # Defensive cleanup: erase any spurious "undefined" title
    if existing_title.lower() in ("undefined", "none"):
        fig.update_layout(title_text="")
        existing_title = ""

    if existing_title:
        # Only add font styling when there is real title text
        fig.update_layout(title_font=dict(color=_CHART_FONT_COLOR))

    fig.update_xaxes(
        tickfont=tick_cfg,
        title_font=dict(color=_CHART_FONT_COLOR),
        gridcolor=_CHART_GRID_COLOR,
        zerolinecolor=_CHART_GRID_COLOR,
    )
    fig.update_yaxes(
        tickfont=tick_cfg,
        title_font=dict(color=_CHART_FONT_COLOR),
        gridcolor=_CHART_GRID_COLOR,
        zerolinecolor=_CHART_GRID_COLOR,
    )
    fig.update_layout(
        legend=dict(font=dict(color=_CHART_FONT_COLOR)),
    )
    # Force bar/scatter text label colour for every trace.
    # Traces with textposition="inside" that already set their own textfont
    # (e.g. white-on-coloured bars) are left alone; we only add colour where
    # textfont.color is not yet set explicitly on that trace.
    for trace in fig.data:
        if hasattr(trace, "textfont") and trace.textfont is not None:
            existing = trace.textfont.color
            if existing is None or existing == "":
                trace.textfont = dict(**(trace.textfont.to_plotly_json() or {}),
                                      color=_CHART_FONT_COLOR)
        elif hasattr(trace, "textposition") and trace.textposition not in (None, "none"):
            try:
                trace.textfont = dict(color=_CHART_FONT_COLOR)
            except Exception:
                pass
    return fig


MIKBATZ_COLORS = {
    "מתקדמים בכל":           PALETTE["teal"],
    "תקינים בכל":            PALETTE["navy"],
    "מתקשים לימודית":        PALETTE["orange"],
    "מתקשים חברתית ורגשית": PALETTE["purple"],
    "מתקשים בכל":            PALETTE["red"],
}

STATUS_COLORS = {
    "הסתיים": PALETTE["teal"],
    "חלקי":   PALETTE["orange"],
    "חסר":    PALETTE["red"],
}

DIFFICULTY_ANSWERS = ["מתקשה", "מתקשה מאוד"]

# ══════════════════════════════════════════════════════════════════════════════
# PERMANENT BUSINESS-LOGIC RULE — DO NOT BYPASS
# ══════════════════════════════════════════════════════════════════════════════
# Rows in FAKE_perut_tosot_mipuy where מיומנות == "-" are background /
# questionnaire-metadata rows.  They are NOT real skill or domain rows.
#
#   ✅ Keep them in the raw uploaded DataFrame (do not drop on load).
#   ❌ NEVER include them in any skill / domain / difficulty-by-skill
#      analysis, chart, summary, or chatbot response — in any phase.
#
# All code that touches מיומנות data MUST call skill_rows(df) below instead
# of filtering inline.  This makes the rule impossible to accidentally skip.
# ══════════════════════════════════════════════════════════════════════════════
NON_SKILL_VALUE = "-"


def skill_rows(df_tosot: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that represent a real skill/domain assessment.

    This is the single mandatory gateway for every skill-level analysis.
    Never query מיומנות data without calling this first.

    RULE: rows where מיומנות == NON_SKILL_VALUE ("-") are metadata rows,
    not skill rows, and must always be excluded.
    """
    filtered = df_tosot[df_tosot["מיומנות"] != NON_SKILL_VALUE]
    # Hard assertion: the sentinel value must never appear in downstream work.
    assert NON_SKILL_VALUE not in filtered["מיומנות"].values, (
        "skill_rows() filter failed — NON_SKILL_VALUE leaked into result"
    )
    return filtered

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ════════════════════════════════════════════════════════════════════
     FORCED LIGHT THEME
     Every major container gets an explicit background-color.
     Never transparent — overrides browser/OS dark mode.
  ════════════════════════════════════════════════════════════════════ */
  :root { color-scheme: light only; }

  html, body {
    background-color: #F8FAFC !important;
    color: #1a2340 !important;
  }
  [data-testid="stAppViewContainer"], .stApp {
    background-color: #F8FAFC !important;
    color: #1a2340 !important;
  }
  [data-testid="stAppViewContainer"] > .main, .main {
    background-color: #F8FAFC !important;
  }
  .main .block-container {
    background-color: #F8FAFC !important;
    padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;
  }
  /* Tab panels */
  [data-testid="stTabPanel"], .stTabs [data-baseweb="tab-panel"] {
    background-color: #F8FAFC !important;
  }
  /* Expanders */
  [data-testid="stExpander"] > div {
    background-color: #ffffff !important;
    border: 1px solid #e4e8ef; border-radius: 8px;
  }
  /* Dataframes */
  [data-testid="stDataFrame"] { background-color: #ffffff !important; }

  /* RTL + font */
  html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Segoe UI', 'Noto Sans Hebrew', Arial, sans-serif;
  }

  /* ── Sidebar — force light even in browser dark mode ── */
  section[data-testid="stSidebar"],
  [data-testid="stSidebarContent"] {
    background-color: #ffffff !important;
    border-left: 1px solid #e4e8ef !important;
    direction: rtl;
  }
  section[data-testid="stSidebar"] * { direction: rtl; text-align: right; }

  /* All text inside sidebar */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div,
  section[data-testid="stSidebar"] small,
  section[data-testid="stSidebar"] button {
    color: #1a2340 !important;
  }

  /* ── File uploader — drag-and-drop zone ── */
  [data-testid="stFileUploader"],
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileUploaderDropzoneInput"] {
    background-color: #f8fafc !important;
    border: 1.5px dashed #b0bccc !important;
    border-radius: 8px !important;
    color: #1a2340 !important;
  }

  /* The inner section/label text inside the dropzone */
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploaderDropzone"] span,
  [data-testid="stFileUploaderDropzone"] p,
  [data-testid="stFileUploaderDropzone"] small {
    color: #5a6a82 !important;
    background-color: transparent !important;
  }

  /* Browse-files / upload button inside the dropzone */
  [data-testid="stFileUploaderDropzone"] button,
  [data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    border: 1.5px solid #4F7EA8 !important;
    border-radius: 6px !important;
  }
  [data-testid="stFileUploaderDropzone"] button:hover,
  [data-testid="baseButton-secondary"]:hover {
    background-color: #eef3f8 !important;
    color: #1e3a5f !important;
  }

  /* ── Uploaded file pill (the card that appears after upload) ── */
  [data-testid="stFileUploaderFile"],
  [data-testid="stUploadedFile"],
  .uploadedFile {
    background-color: #ffffff !important;
    border: 1px solid #dde2ea !important;
    border-radius: 6px !important;
    color: #1a2340 !important;
  }

  /* File name text */
  [data-testid="stFileUploaderFile"] span,
  [data-testid="stUploadedFile"] span,
  [data-testid="stFileUploaderFileName"],
  .uploadedFileName {
    color: #1a2340 !important;
    background-color: transparent !important;
  }

  /* File size text */
  [data-testid="stFileUploaderFileData"] small,
  [data-testid="stFileUploaderFileData"] span,
  [data-testid="stUploadedFileData"] small,
  [data-testid="stUploadedFileData"] span {
    color: #5a6a82 !important;
  }

  /* Remove-file (×) icon button */
  [data-testid="stFileUploaderDeleteBtn"] button,
  [data-testid="stUploadedFileDeleteBtn"] button,
  [aria-label="Remove file"] {
    background-color: transparent !important;
    color: #8a9ab2 !important;
    border: none !important;
  }
  [data-testid="stFileUploaderDeleteBtn"] button:hover,
  [data-testid="stUploadedFileDeleteBtn"] button:hover {
    color: #D95F59 !important;
    background-color: #fde8e8 !important;
  }

  /* ── Header ── */
  .main-header {
    background: #1e3a5f; border-radius: 12px;
    padding: 1.6rem 2rem; margin-bottom: 1.2rem;
    color: #ffffff !important;
  }
  .main-header * { color: #ffffff !important; }
  .main-header h1 {
    font-size:2rem; margin:0; font-weight:700;
    direction:rtl; text-align:right;
    color:#ffffff !important;
  }
  .main-header p {
    font-size:0.98rem; margin:0.3rem 0 0;
    direction:rtl; text-align:right;
    color:rgba(255,255,255,0.88) !important;
  }

  /* ── KPI cards ── */
  .kpi-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem; }
  .kpi-card {
    background: #ffffff !important;
    border-radius:10px; padding:1rem 1.2rem;
    box-shadow:0 1px 5px rgba(0,0,0,0.06);
    border-top:3px solid var(--accent, #4F7EA8);
    text-align:right; direction:rtl;
    flex:1 1 0; min-width:130px; min-height:110px;
    display:flex; flex-direction:column; justify-content:center;
    color:#1a2340 !important;
  }
  .kpi-card:hover { box-shadow:0 3px 12px rgba(0,0,0,0.10); }
  .kpi-icon  { font-size:1.2rem; margin-bottom:0.25rem; }
  .kpi-value { font-size:2rem; font-weight:800; color:var(--accent, #4F7EA8); line-height:1.1; }
  .kpi-label { font-size:0.82rem; color:#5a6a82; margin-top:0.2rem; }
  .kpi-sub   { font-size:0.72rem; color:#8a9ab2; margin-top:0.1rem; min-height:0.9rem; }

  /* ── Section headers ── */
  .section-title {
    font-size:1.15rem; font-weight:700; color:#1e3a5f;
    margin:1.2rem 0 0.6rem; padding-bottom:0.35rem;
    border-bottom:2px solid #dde2ea;
    direction:rtl; text-align:right; background:transparent;
  }

  /* ── White content cards ── */
  .chart-box {
    background: #ffffff !important;
    border-radius:10px; padding:0.8rem 1rem;
    box-shadow:0 1px 5px rgba(0,0,0,0.05);
    margin-bottom:0.8rem;
    border:1px solid #e8ecf2;
    direction:rtl; color:#1a2340;
  }
  .chart-box p, .chart-box caption, .chart-box small {
    direction:rtl; text-align:right; color:#1a2340;
  }

  /* ── Badges ── */
  .badge-ok   { background:#e6f7ef; color:#1d7a4e; border-radius:6px; padding:2px 9px; font-size:0.8rem; font-weight:600; }
  .badge-err  { background:#fde8e8; color:#b83535; border-radius:6px; padding:2px 9px; font-size:0.8rem; font-weight:600; }
  .badge-wait { background:#eef1f5; color:#5a6a82; border-radius:6px; padding:2px 9px; font-size:0.8rem; font-weight:600; }

  /* ── Onboarding ── */
  .onboard-box {
    background:#ffffff !important; border:1px solid #dde2ea;
    border-radius:12px; padding:2rem 1.8rem;
    text-align:center; margin:1.5rem auto; max-width:600px;
    direction:rtl; color:#1a2340;
  }
  .onboard-box h2 { color:#1e3a5f; font-size:1.5rem; margin-bottom:0.4rem; }
  .onboard-box p  { color:#475569; font-size:0.95rem; }

  /* ── Validation ── */
  .val-ok  { color:#1d7a4e; font-weight:600; }
  .val-err { color:#b83535; font-weight:600; }

  /* ── RTL markdown / captions ── */
  .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
  .stMarkdown h3, .stMarkdown li, .stMarkdown ul {
    direction:rtl; text-align:right; color:#1a2340;
  }
  .stCaption, div[data-testid="stCaptionContainer"] {
    direction:rtl; text-align:right; color:#5a6a82;
  }
  .stTabs [data-baseweb="tab"] { direction:rtl; }
  div[data-testid="metric-container"] { direction:rtl; text-align:right; }

  /* ── Chat ── */
  .intent-badge {
    font-size:0.68rem; background:#eef1f5; color:#5a6a82;
    border-radius:5px; padding:1px 6px; margin-right:4px; font-family:monospace;
  }

  /* ── Chat tab outer columns ── */
  [data-testid="stVerticalBlock"] { background-color: transparent !important; }

  /* ── Example questions panel (the left column) ── */
  .chat-examples-panel,
  .chat-examples-panel * {
    background-color: #ffffff !important;
    color: #1a2340 !important;
  }

  /* Buttons inside the examples panel */
  .chat-examples-panel button,
  .chat-examples-panel [data-testid="baseButton-secondary"] {
    background-color: #f8fafc !important;
    color: #1e3a5f !important;
    border: 1px solid #dde2ea !important;
    border-radius: 7px !important;
  }
  .chat-examples-panel button:hover {
    background-color: #eef3f8 !important;
    color: #1e3a5f !important;
    border-color: #4F7EA8 !important;
  }

  /* ── All Streamlit buttons in the chat area — light forced ── */
  [data-testid="stButton"] > button {
    background-color: #f8fafc !important;
    color: #1e3a5f !important;
    border: 1px solid #dde2ea !important;
  }
  [data-testid="stButton"] > button:hover {
    background-color: #eef3f8 !important;
    border-color: #4F7EA8 !important;
  }

  /* ── Chat message containers (user + assistant) ── */
  div[data-testid="stChatMessage"] {
    direction: rtl;
    background-color: #ffffff !important;
    border: 1px solid #e8ecf2 !important;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    color: #1a2340 !important;
  }
  /* All text/elements inside any message bubble */
  div[data-testid="stChatMessage"] *,
  div[data-testid="stChatMessage"] p,
  div[data-testid="stChatMessage"] span,
  div[data-testid="stChatMessage"] div,
  div[data-testid="stChatMessage"] li {
    color: #1a2340 !important;
    background-color: transparent !important;
  }
  /* Avatar column */
  div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
  div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background-color: #eef3f8 !important;
    color: #1e3a5f !important;
  }
  /* User message: slightly tinted background to distinguish from assistant */
  div[data-testid="stChatMessage"][data-testid*="user"],
  div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #eef3f8 !important;
  }

  /* ── Chat input box ── */
  /* Target every wrapper layer Streamlit puts around the input */
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInput"],
  .stChatInputContainer,
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div,
  [data-testid="stChatInputContainer"] > div,
  section.main [data-testid="stBottom"],
  section.main [data-testid="stBottom"] > * {
    background-color: #ffffff !important;
    border-color: #c8d2de !important;
    color: #1a2340 !important;
  }
  /* The visible input border wrapper */
  [data-testid="stChatInputContainer"] {
    border: 1.5px solid #c8d2de !important;
    border-radius: 10px !important;
  }
  /* Textarea */
  [data-testid="stChatInputContainer"] textarea,
  [data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1a2340 !important;
    caret-color: #1e3a5f !important;
  }
  [data-testid="stChatInputContainer"] textarea::placeholder,
  [data-testid="stChatInput"] textarea::placeholder {
    color: #8a9ab2 !important;
  }
  /* Send button */
  [data-testid="stChatInputContainer"] button,
  [data-testid="stChatInput"] button {
    background-color: #4F7EA8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
  }
  [data-testid="stChatInputContainer"] button:hover,
  [data-testid="stChatInput"] button:hover {
    background-color: #3d6a8f !important;
  }
  /* SVG icon inside send button */
  [data-testid="stChatInputContainer"] button svg,
  [data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
  }

  /* ════════════════════════════════════════════════════════════
     DARK-MODE ROBUSTNESS — force explicit colours on Streamlit
     widget chrome that can pick up OS/browser dark theme.
  ════════════════════════════════════════════════════════════ */

  /* Tab bar background and labels */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #F8FAFC !important;
    border-bottom: 2px solid #dde2ea !important;
  }
  .stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #1e3a5f !important;
    font-weight: 600;
    direction: rtl;
  }
  .stTabs [aria-selected="true"] {
    color: #1e3a5f !important;
    border-bottom: 3px solid #4F7EA8 !important;
    background-color: #ffffff !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #4F7EA8 !important;
    background-color: #eef3f8 !important;
  }

  /* Selectbox / radio / checkbox labels */
  .stSelectbox label, .stRadio label, .stCheckbox label,
  .stTextInput label, .stFileUploader label,
  [data-testid="stWidgetLabel"] {
    color: #1a2340 !important;
  }

  /* Expander header text */
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span {
    color: #1a2340 !important;
    background-color: #ffffff !important;
  }

  /* Metrics */
  [data-testid="metric-container"] label,
  [data-testid="metric-container"] div {
    color: #1a2340 !important;
  }

  /* Plotly chart containers — ensure the SVG sits on white */
  [data-testid="stPlotlyChart"],
  .js-plotly-plot,
  .plot-container {
    background-color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), dtype=str)


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def coerce_types(df_status: pd.DataFrame, df_tosot: pd.DataFrame, df_tochnit: pd.DataFrame):
    """Strip whitespace and fix numeric columns."""
    for df in [df_status, df_tosot, df_tochnit]:
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

    # Pupil IDs as string (already str but strip again)
    df_status["ת.ז תלמיד"] = df_status["ת.ז תלמיד"].astype(str).str.strip()
    for df in [df_tosot, df_tochnit]:
        df["מס' זהות תלמיד"] = df["מס' זהות תלמיד"].astype(str).str.strip()

    # מקבילה as string (keeps "1", "2", …)
    for df in [df_status, df_tosot, df_tochnit]:
        if "מקבילה" in df.columns:
            df["מקבילה"] = df["מקבילה"].astype(str).str.strip()

    return df_status, df_tosot, df_tochnit


# ══════════════════════════════════════════════════════════════════════════════
# Analysis helpers (pure Pandas – reused by Phase 2 chat)
# ══════════════════════════════════════════════════════════════════════════════

def get_school_name(df_tochnit: pd.DataFrame) -> str:
    if "שם מוסד" in df_tochnit.columns:
        vals = df_tochnit["שם מוסד"].dropna().unique()
        if len(vals):
            return vals[0]
    return "בית הספר"


def summary_kpis(df_status, df_tosot, df_tochnit):
    total = len(df_status)
    completed  = (df_status["סטטוס מילוי שאלון"] == "הסתיים").sum()
    partial    = (df_status["סטטוס מילוי שאלון"] == "חלקי").sum()
    missing    = (df_status["סטטוס מילוי שאלון"] == "חסר").sum()
    with_prog  = df_tochnit["מס' זהות תלמיד"].nunique()
    without_prog = total - with_prog

    difficult_mkbtz = ["מתקשים לימודית", "מתקשים חברתית ורגשית", "מתקשים בכל"]
    if df_tosot is not None and not df_tosot.empty:
        difficult = df_tosot[df_tosot["מקבץ"].isin(difficult_mkbtz)]["מס' זהות תלמיד"].nunique()
    else:
        difficult = 0

    return {
        "total": total,
        "completed": completed,
        "partial": partial,
        "missing": missing,
        "with_prog": with_prog,
        "without_prog": without_prog,
        "difficult": difficult,
        "pct_completed": round(completed / total * 100, 1) if total else 0,
    }


def questionnaire_status_dist(df_status):
    counts = df_status["סטטוס מילוי שאלון"].value_counts().reset_index()
    counts.columns = ["סטטוס", "מספר תלמידים"]
    return counts


def questionnaire_by_grade(df_status):
    grade_status = (
        df_status.groupby(["שכבה", "סטטוס מילוי שאלון"])
        .size()
        .reset_index(name="מספר")
    )
    # Add % within grade
    totals = df_status.groupby("שכבה").size().rename("סה\"כ")
    grade_status = grade_status.merge(totals, on="שכבה")
    grade_status["אחוז"] = (grade_status["מספר"] / grade_status["סה\"כ"] * 100).round(1)
    # Sort grades
    grade_status["שכבה"] = pd.Categorical(grade_status["שכבה"], categories=GRADE_ORDER, ordered=True)
    return grade_status.sort_values(["שכבה", "סטטוס מילוי שאלון"])


def questionnaire_by_class(df_status):
    df_status = df_status.copy()
    df_status["כיתה"] = df_status["שכבה"] + df_status["מקבילה"]
    class_status = (
        df_status.groupby(["כיתה", "שכבה", "מקבילה", "סטטוס מילוי שאלון"])
        .size()
        .reset_index(name="מספר")
    )
    totals = df_status.groupby("כיתה").size().rename("סה\"כ")
    class_status = class_status.merge(totals, on="כיתה")
    class_status["אחוז"] = (class_status["מספר"] / class_status["סה\"כ"] * 100).round(1)
    class_status["שכבה_ord"] = pd.Categorical(class_status["שכבה"], categories=GRADE_ORDER, ordered=True)
    return class_status.sort_values(["שכבה_ord", "מקבילה"])


def mikbatz_dist(df_tosot):
    counts = df_tosot.drop_duplicates("מס' זהות תלמיד")["מקבץ"].value_counts().reset_index()
    counts.columns = ["מקבץ", "מספר תלמידים"]
    return counts


def mikbatz_by_grade(df_tosot):
    unique = df_tosot.drop_duplicates("מס' זהות תלמיד")[["מס' זהות תלמיד", "שכבה", "מקבץ"]]
    grp = unique.groupby(["שכבה", "מקבץ"]).size().reset_index(name="מספר")
    grp["שכבה"] = pd.Categorical(grp["שכבה"], categories=GRADE_ORDER, ordered=True)
    return grp.sort_values(["שכבה", "מקבץ"])


def programs_dist(df_tochnit, top_n=12):
    counts = df_tochnit["שם תוכנית חינוכית"].value_counts().head(top_n).reset_index()
    counts.columns = ["תוכנית", "מספר שיוכים"]
    return counts


def pupils_without_programs(df_status, df_tosot, df_tochnit):
    """Return a DataFrame of pupils from Table 1 who have no row in Table 3.

    Enriches each row with מקבץ from Table 2 where available.
    Reusable by Phase 2 chatbot intent: pupils_without_program.
    """
    assigned_ids = set(df_tochnit["מס' זהות תלמיד"].unique())
    no_prog = df_status[~df_status["ת.ז תלמיד"].isin(assigned_ids)].copy()

    # Enrich with מקבץ from tosot (one row per pupil – take first occurrence)
    if df_tosot is not None and not df_tosot.empty:
        mkbatz_map = (
            df_tosot.drop_duplicates("מס' זהות תלמיד")
            .set_index("מס' זהות תלמיד")["מקבץ"]
        )
        no_prog["מקבץ"] = no_prog["ת.ז תלמיד"].map(mkbatz_map).fillna("לא ידוע")
    else:
        no_prog["מקבץ"] = "לא ידוע"

    # Friendly class label
    no_prog["כיתה"] = no_prog["שכבה"] + no_prog["מקבילה"]

    # Sort by grade order then class
    no_prog["שכבה_ord"] = pd.Categorical(no_prog["שכבה"], categories=GRADE_ORDER, ordered=True)
    no_prog = no_prog.sort_values(["שכבה_ord", "מקבילה", "שם תלמיד"]).drop(columns="שכבה_ord")

    return no_prog


def no_prog_by_grade(df_status, df_tochnit):
    """Grade-level breakdown: total pupils vs. those without a program."""
    assigned_ids = set(df_tochnit["מס' זהות תלמיד"].unique())
    df = df_status.copy()
    df["ללא תוכנית"] = ~df["ת.ז תלמיד"].isin(assigned_ids)
    grp = df.groupby("שכבה").agg(
        סה_כל=("ת.ז תלמיד", "count"),
        ללא_תוכנית=("ללא תוכנית", "sum"),
    ).reset_index()
    grp["עם תוכנית"] = grp["סה_כל"] - grp["ללא_תוכנית"]
    grp["אחוז ללא תוכנית"] = (grp["ללא_תוכנית"] / grp["סה_כל"] * 100).round(1)
    grp["שכבה"] = pd.Categorical(grp["שכבה"], categories=GRADE_ORDER, ordered=True)
    return grp.sort_values("שכבה")


def difficulty_by_skill(df_tosot):
    """Count distinct pupils with a difficulty answer, per real skill domain.

    Business rule:
    - Count UNIQUE pupils (not rows) per מיומנות.
    - A pupil counts once per מיומנות if they have at least one row where
      תשובה is "מתקשה" or "מתקשה מאוד".
    - Each pupil has many rows per skill (one per היגד statement); nunique()
      on pupil ID is the correct aggregation — do NOT count rows.
    - Uses skill_rows() to enforce the NON_SKILL_VALUE exclusion rule.

    Returns a DataFrame with columns:
      מיומנות | תלמידים מתקשים | סה"כ תלמידים בשאלון | % מתקשים
    The denominator (סה"כ) is unique pupils who have ANY skill row, i.e.
    completed the questionnaire — giving the principal a meaningful rate.
    """
    sr = skill_rows(df_tosot)
    # Denominator: unique pupils who appear in skill rows (completed questionnaire)
    total_per_skill = (
        sr.groupby("מיומנות")["מס' זהות תלמיד"]
        .nunique()
        .rename("סהכ תלמידים בשאלון")
    )
    # Numerator: unique pupils with at least one difficulty answer per skill
    diff = sr[sr["תשובה"].isin(DIFFICULTY_ANSWERS)]
    skill_counts = (
        diff.groupby("מיומנות")["מס' זהות תלמיד"]
        .nunique()
        .reset_index(name="תלמידים מתקשים")
    )
    skill_counts = skill_counts.merge(total_per_skill, on="מיומנות")
    skill_counts["% מתקשים"] = (
        skill_counts["תלמידים מתקשים"] / skill_counts["סהכ תלמידים בשאלון"] * 100
    ).round(1)
    return skill_counts.sort_values("תלמידים מתקשים", ascending=False)


# Order in which תשובה levels appear in the grouped bar chart
ANSWER_LEVEL_ORDER = ["מתקשה מאוד", "מתקשה", "תקין", "מתקדם"]

# Pastel colours per answer level (ordered to match ANSWER_LEVEL_ORDER)
ANSWER_LEVEL_COLORS = {
    "מתקשה מאוד": PALETTE["red"],
    "מתקשה":      PALETTE["orange"],
    "תקין":        PALETTE["navy"],
    "מתקדם":       PALETTE["teal"],
}


def skill_level_by_skill(df_tosot):
    """Count unique pupils per (מיומנות, תשובה) pair.

    Business rules:
    - Excludes rows where מיומנות = "-" (via skill_rows())
    - Only includes תשובה values in ANSWER_LEVEL_ORDER
    - Counts UNIQUE pupils per cell — a pupil with many היגד rows in the same
      (מיומנות, תשובה) combination is counted once.
    """
    sr = skill_rows(df_tosot)
    relevant = sr[sr["תשובה"].isin(ANSWER_LEVEL_ORDER)]
    grp = (
        relevant.groupby(["מיומנות", "תשובה"])["מס' זהות תלמיד"]
        .nunique()
        .reset_index(name="תלמידים")
    )
    return grp



def completion_pct_by_class(df_status):
    df = df_status.copy()
    df["כיתה"] = df["שכבה"] + df["מקבילה"]
    grp = df.groupby("כיתה").agg(
        סה_כל=("ת.ז תלמיד", "count"),
        הסתיים=("סטטוס מילוי שאלון", lambda x: (x == "הסתיים").sum()),
    ).reset_index()
    grp["אחוז מילוי"] = (grp["הסתיים"] / grp["סה_כל"] * 100).round(1)
    return grp.sort_values("אחוז מילוי")


# ══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════════

def chart_status_donut(df_status):
    counts = questionnaire_status_dist(df_status)
    colors = [STATUS_COLORS.get(s, PALETTE["grey"]) for s in counts["סטטוס"]]
    fig = go.Figure(go.Pie(
        labels=counts["סטטוס"],
        values=counts["מספר תלמידים"],
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=13,
        direction="clockwise",
        sort=False,
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=20, b=40, l=10, r=10),
        height=300,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    _apply_light_theme(fig)
    return fig


def chart_status_by_grade(df_status):
    data = questionnaire_by_grade(df_status)
    statuses = ["הסתיים", "חלקי", "חסר"]
    colors   = [STATUS_COLORS[s] for s in statuses]

    fig = go.Figure()
    for status, color in zip(statuses, colors):
        sub = data[data["סטטוס מילוי שאלון"] == status]
        fig.add_trace(go.Bar(
            name=status,
            x=sub["שכבה"],
            y=sub["מספר"],
            marker_color=color,
            text=sub["מספר"],
            textposition="inside",
            insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=GRADE_ORDER, title="שכבה"),
        yaxis_title="מספר תלמידים",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=10),
        height=320,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    _apply_light_theme(fig)
    return fig


def chart_completion_pct_by_class(df_status):
    grp = completion_pct_by_class(df_status)
    colors = [PALETTE["teal"] if p >= 75 else PALETTE["orange"] if p >= 50 else PALETTE["red"]
              for p in grp["אחוז מילוי"]]
    fig = go.Figure(go.Bar(
        x=grp["כיתה"],
        y=grp["אחוז מילוי"],
        marker_color=colors,
        text=grp["אחוז מילוי"].apply(lambda v: f"{v}%"),
        textposition="outside",
        # customdata carries [pupils_completed, total] for the hover template
        customdata=grp[["הסתיים", "סה_כל"]].values,
        hovertemplate=(
            "<b>כיתה %{x}</b><br>"
            "הגישו שאלון: %{customdata[0]} מתוך %{customdata[1]} תלמידים<br>"
            "אחוז מילוי: %{y}%"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=75, line_dash="dash", line_color=PALETTE["grey"],
                  annotation_text="75%", annotation_position="left")
    fig.update_layout(
        # Push yaxis ceiling high enough so "outside" text labels are never clipped
        yaxis=dict(range=[0, 125], title="אחוז מילוי שאלון (%)"),
        # Extra bottom margin so the x-axis title clears the tick labels
        margin=dict(t=30, b=55, l=40, r=10),
        height=360,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(
        showgrid=False,
        title=dict(text="כיתה", standoff=12),   # standoff keeps title away from ticks
        tickangle=0,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    _apply_light_theme(fig)
    return fig


def chart_mikbatz(df_tosot):
    data = mikbatz_dist(df_tosot)
    colors = [MIKBATZ_COLORS.get(m, PALETTE["grey"]) for m in data["מקבץ"]]
    fig = go.Figure(go.Bar(
        x=data["מקבץ"],
        y=data["מספר תלמידים"],
        marker_color=colors,
        text=data["מספר תלמידים"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="מקבץ",
        yaxis_title="מספר תלמידים",
        yaxis=dict(range=[0, data["מספר תלמידים"].max() * 1.18]),  # headroom for outside labels
        margin=dict(t=20, b=110, l=30, r=10),   # tall bottom margin for angled tick labels
        height=340,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=-35,          # angle long Hebrew labels so they don't collide
        automargin=True,        # let Plotly expand margin further if still needed
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    _apply_light_theme(fig)
    return fig


def chart_mikbatz_by_grade(df_tosot):
    data = mikbatz_by_grade(df_tosot)
    mikbatzot = list(MIKBATZ_COLORS.keys())
    fig = go.Figure()
    for m in mikbatzot:
        sub = data[data["מקבץ"] == m]
        fig.add_trace(go.Bar(
            name=m,
            x=sub["שכבה"].astype(str),
            y=sub["מספר"],
            marker_color=MIKBATZ_COLORS[m],
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=GRADE_ORDER, title="שכבה"),
        yaxis_title="מספר תלמידים",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(t=20, b=80, l=30, r=10),
        height=350,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    _apply_light_theme(fig)
    return fig


def chart_programs(df_tochnit):
    data = programs_dist(df_tochnit)
    fig = go.Figure(go.Bar(
        y=data["תוכנית"],
        x=data["מספר שיוכים"],
        orientation="h",
        marker_color=PALETTE["navy"],
        text=data["מספר שיוכים"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="מספר שיוכים",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=20, b=20, l=200, r=50),
        height=max(280, len(data) * 30),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=False)
    _apply_light_theme(fig)
    return fig


def chart_no_prog_by_grade(df_status, df_tochnit):
    grp = no_prog_by_grade(df_status, df_tochnit)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ללא תוכנית",
        x=grp["שכבה"].astype(str),
        y=grp["ללא_תוכנית"],
        marker_color=PALETTE["red"],
        text=grp["ללא_תוכנית"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        name="עם תוכנית",
        x=grp["שכבה"].astype(str),
        y=grp["עם תוכנית"],
        marker_color=PALETTE["teal"],
        text=grp["עם תוכנית"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    # Percentage annotations above each bar group
    for _, row in grp.iterrows():
        fig.add_annotation(
            x=str(row["שכבה"]),
            y=row["סה_כל"] + 1,
            text=f'{row["אחוז ללא תוכנית"]}% ללא',
            showarrow=False,
            font=dict(size=11, color=PALETTE["red"]),
            yanchor="bottom",
        )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=GRADE_ORDER, title="שכבה"),
        yaxis_title="מספר תלמידים",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=30, l=30, r=10),
        height=340,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    _apply_light_theme(fig)
    return fig


def chart_difficulty_by_skill(df_tosot):
    # difficulty_by_skill() already excludes NON_SKILL_VALUE rows upstream.
    # Each bar = unique pupils with ≥1 difficulty answer in that מיומנות.
    # Labels are placed INSIDE the bar end so they are never clipped by the right margin.
    data = difficulty_by_skill(df_tosot).head(10)
    bar_labels = [
        f'{row["תלמידים מתקשים"]} ({row["% מתקשים"]}%)'
        for _, row in data.iterrows()
    ]
    x_max = data["תלמידים מתקשים"].max()
    fig = go.Figure(go.Bar(
        y=data["מיומנות"],
        x=data["תלמידים מתקשים"],
        orientation="h",
        marker_color=PALETTE["orange"],
        text=bar_labels,
        textposition="inside",
        insidetextanchor="end",     # anchor to the right tip of each bar — always visible
        textfont=dict(color="white", size=12),
        customdata=data[["% מתקשים", "סהכ תלמידים בשאלון"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "תלמידים מתקשים: %{x}<br>"
            "מתוך: %{customdata[1]} שהגישו שאלון<br>"
            "שיעור: %{customdata[0]}%"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis=dict(
            title="תלמידים ייחודיים עם קושי (ספירה ולא שורות)",
            range=[0, x_max * 1.05],   # tiny breathing room; labels live inside bars
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=50, l=220, r=20),   # right margin can be small now
        height=max(300, len(data) * 42),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=False)
    _apply_light_theme(fig)
    return fig



def chart_skill_levels_by_skill(df_tosot):
    """Grouped bar chart: unique pupils per תשובה level for each מיומנות.

    x-axis: מיומנות (skill domain)
    grouped bars: one bar per תשובה level in ANSWER_LEVEL_ORDER
    y-axis: unique pupils
    """
    data = skill_level_by_skill(df_tosot)
    if data.empty:
        return go.Figure()

    # Sort skills alphabetically for a stable x-axis order
    skills_sorted = sorted(data["מיומנות"].unique())

    fig = go.Figure()
    for level in ANSWER_LEVEL_ORDER:
        subset = data[data["תשובה"] == level]
        # Align to full skill list (fill 0 for missing cells)
        counts = {row["מיומנות"]: row["תלמידים"] for _, row in subset.iterrows()}
        y_vals = [counts.get(s, 0) for s in skills_sorted]
        fig.add_trace(go.Bar(
            name=level,
            x=skills_sorted,
            y=y_vals,
            marker_color=ANSWER_LEVEL_COLORS[level],
            text=[str(v) if v > 0 else "" for v in y_vals],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            tickangle=-30,
            automargin=True,
        ),
        yaxis=dict(range=[0, data["תלמידים"].max() * 1.22]),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(t=40, b=100, l=40, r=10),
        height=420,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=_CHART_GRID_COLOR)
    _apply_light_theme(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar – file upload
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple:
    st.sidebar.markdown("""
    <div style='text-align:right; padding: 0.5rem 0 1rem;'>
      <span style='font-size:1.8rem;'>🦅</span>
      <span style='font-size:1.05rem; font-weight:700; color:#1e3a5f; margin-right:8px;'>נץ תקומה</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 📂 העלאת קבצים")
    st.sidebar.caption("העלה את שלושת קבצי האקסל של בית הספר")

    dfs = {}
    errors = {}

    for key, meta in REQUIRED_FILES.items():
        st.sidebar.markdown(f"**{meta['icon']} {meta['description']}**")
        file = st.sidebar.file_uploader(
            meta["label"],
            type=["xlsx"],
            key=f"upload_{key}",
            label_visibility="collapsed",
        )
        if file:
            try:
                df = load_excel(file.read(), file.name)
                ok, missing = validate_dataframe(df, meta["required_cols"])
                if ok:
                    dfs[key] = df
                    st.sidebar.markdown(f'<span class="badge-ok">✓ הועלה בהצלחה · {len(df):,} שורות</span>', unsafe_allow_html=True)
                else:
                    errors[key] = f"עמודות חסרות: {', '.join(missing)}"
                    st.sidebar.markdown(f'<span class="badge-err">✗ שגיאת מבנה</span>', unsafe_allow_html=True)
                    st.sidebar.error(errors[key])
            except Exception as e:
                errors[key] = str(e)
                st.sidebar.markdown(f'<span class="badge-err">✗ שגיאה בטעינה</span>', unsafe_allow_html=True)
                st.sidebar.error(str(e))
        else:
            st.sidebar.markdown(f'<span class="badge-wait">⟳ ממתין להעלאה</span>', unsafe_allow_html=True)

        st.sidebar.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

    # Validation summary
    if dfs:
        st.sidebar.markdown("---")
        uploaded = len(dfs)
        total_req = len(REQUIRED_FILES)
        if uploaded == total_req:
            st.sidebar.success(f"✅ כל {total_req} הקבצים הועלו בהצלחה")
        else:
            st.sidebar.warning(f"⚠️ {uploaded} מתוך {total_req} קבצים הועלו")

    st.sidebar.markdown("---")
    st.sidebar.caption("Phase 1 · פאזה ראשונה בלבד – לוח מחוונים")
    return dfs, errors


# ══════════════════════════════════════════════════════════════════════════════
# Onboarding screen
# ══════════════════════════════════════════════════════════════════════════════

def render_onboarding():
    st.markdown("""
    <div class="onboard-box">
      <div style="font-size:3.5rem; margin-bottom:0.5rem;">🦅</div>
      <h2>ברוכים הבאים לנץ תקומה</h2>
      <p>עוזר הנתונים החכם של המנהל/ת</p>
      <hr style="border:none; border-top:1px solid #e2e8f0; margin:1rem 0;">
      <p style="font-size:0.95rem;">
        כדי להציג את לוח המחוונים, יש להעלות את שלושת קבצי האקסל
        מהסרגל הצדדי מימין.
      </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    file_info = [
        ("📋", "סטטוס מילוי שאלון", "FAKE_perut_status_miluy.xlsx",
         "טבלת אב של תלמידים עם סטטוס מילוי השאלון לכל תלמיד"),
        ("📊", "תוצאות מיפוי", "FAKE_perut_tosot_mipuy.xlsx",
         "פירוט תשובות השאלון עבור תלמידים שסיימו מילוי"),
        ("🎓", "שיוך תוכניות", "FAKE_perut_shiuch_tochnit.xlsx",
         "תוכניות חינוכיות המשויכות לתלמידים"),
    ]
    for col, (icon, title, fname, desc) in zip(cols, file_info):
        with col:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:1.2rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.07); text-align:right; height:100%;">
              <div style="font-size:2rem; margin-bottom:0.4rem;">{icon}</div>
              <div style="font-weight:700; color:#1e3a5f; margin-bottom:0.3rem;">{title}</div>
              <div style="font-size:0.78rem; color:#4F7EA8; font-family:monospace; margin-bottom:0.5rem;">{fname}</div>
              <div style="font-size:0.85rem; color:#64748b;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI Cards row
# ══════════════════════════════════════════════════════════════════════════════

def render_kpi_cards(kpis):
    card_data = [
        (PALETTE["navy"], "👨‍🎓", f"{kpis['total']:,}",       "סך כל התלמידים",              ""),
        (PALETTE["teal"], "✅",   f"{kpis['completed']:,}",   "שאלון הושלם",                 f"{kpis['pct_completed']}% מהכלל"),
        (PALETTE["orange"], "⚠️",  f"{kpis['partial']:,}",     "שאלון חלקי",                  ""),
        (PALETTE["red"], "❌",   f"{kpis['missing']:,}",     "שאלון חסר",                   ""),
        (PALETTE["purple"], "📚",  f"{kpis['with_prog']:,}",   "עם תוכנית חינוכית",            f"{kpis['without_prog']:,} ללא תוכנית"),
        (PALETTE["sky"], "⚡",  f"{kpis['difficult']:,}",   "תלמידים מתקשים",              "לפי שיוך מקבץ"),
    ]
    # Render as a single HTML flex row — guarantees uniform height across all cards
    cards_html = '<div class="kpi-row">'
    for accent, icon, value, label, sub in card_data:
        cards_html += f"""
        <div class="kpi-card" style="--accent:{accent};">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-sub">{sub}&nbsp;</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard – Questionnaire section
# ══════════════════════════════════════════════════════════════════════════════

def render_questionnaire_section(df_status):
    st.markdown('<div class="section-title">📋 מצב מילוי שאלון</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.caption("התפלגות סטטוס – כלל בית הספר")
        st.plotly_chart(chart_status_donut(df_status), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.caption("סטטוס מילוי לפי שכבה")
        st.plotly_chart(chart_status_by_grade(df_status), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.caption("אחוז מילוי שאלון לפי כיתה")
    st.plotly_chart(chart_completion_pct_by_class(df_status), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom classes table
    with st.expander("📉 כיתות עם שיעור מילוי נמוך (עד 60%)", expanded=False):
        grp = completion_pct_by_class(df_status)
        low = grp[grp["אחוז מילוי"] < 60].sort_values("אחוז מילוי")
        if low.empty:
            st.info("אין כיתות עם שיעור מילוי נמוך מ-60% 🎉")
        else:
            st.dataframe(
                low[["כיתה", "סה_כל", "הסתיים", "אחוז מילוי"]].rename(columns={
                    "סה_כל": "סה\"כ תלמידים", "הסתיים": "שאלון הושלם"
                }),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard – Mikbatz section
# ══════════════════════════════════════════════════════════════════════════════

def render_mikbatz_section(df_tosot):
    st.markdown('<div class="section-title">🔷 פרופיל מקבץ תלמידים</div>', unsafe_allow_html=True)
    st.caption("מוצג רק עבור תלמידים שמולא עליהם שאלון מלא (הסתיים)")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.caption("התפלגות מקבץ – כלל בית הספר")
        st.plotly_chart(chart_mikbatz(df_tosot), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.caption("מקבץ לפי שכבה")
        st.plotly_chart(chart_mikbatz_by_grade(df_tosot), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Difficulty by skill
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.caption("תלמידים ייחודיים מתקשים לפי מיומנות — ספירת תלמידים (לא שורות). תלמיד נספר פעם אחת לכל מיומנות אם ענה 'מתקשה' או 'מתקשה מאוד' על לפחות היגד אחד. האחוז הוא מתוך תלמידים שהגישו שאלון.")
    st.plotly_chart(chart_difficulty_by_skill(df_tosot), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # NEW — Grouped bar: pupil level per skill
    st.markdown('<div class="section-title" style="margin-top:1.4rem;">📊 התפלגות רמות תפקוד לפי מיומנות</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.caption("מספר תלמידים ייחודיים בכל רמת תשובה, לפי מיומנות — כל תלמיד נספר פעם אחת לכל צירוף (מיומנות, תשובה). אינו כולל שורות עם מיומנות '-'.")
    st.plotly_chart(chart_skill_levels_by_skill(df_tosot), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard – Programs section
# ══════════════════════════════════════════════════════════════════════════════

def _render_no_program_expander(df_status, df_tosot, df_tochnit):
    """Expandable section: pupils from Table 1 with no entry in Table 3."""
    no_prog = pupils_without_programs(df_status, df_tosot, df_tochnit)
    total       = len(df_status)
    count       = len(no_prog)
    pct         = round(count / total * 100, 1) if total else 0

    with st.expander(f"🔴 תלמידים ללא תוכנית חינוכית  ({count:,} תלמידים · {pct}% מבית הספר)", expanded=False):

        # ── Summary metric cards ───────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""
        <div class="kpi-card" style="--accent:#D95F59;">
          <div style="font-size:1.3rem;">🔴</div>
          <div class="kpi-value">{count:,}</div>
          <div class="kpi-label">תלמידים ללא תוכנית</div>
          <div class="kpi-sub">מתוך {total:,} תלמידים בבית הספר</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="kpi-card" style="--accent:#F3BE72;">
          <div style="font-size:1.3rem;">📊</div>
          <div class="kpi-value">{pct}%</div>
          <div class="kpi-label">אחוז מכלל התלמידים</div>
          <div class="kpi-sub">&nbsp;</div>
        </div>
        """, unsafe_allow_html=True)

        # Count of difficult pupils (by מקבץ) among those without a program
        difficult_mkbtz = {"מתקשים לימודית", "מתקשים חברתית ורגשית", "מתקשים בכל"}
        difficult_no_prog = no_prog[no_prog["מקבץ"].isin(difficult_mkbtz)]
        m3.markdown(f"""
        <div class="kpi-card" style="--accent:#9A7AA0;">
          <div style="font-size:1.3rem;">⚡</div>
          <div class="kpi-value">{len(difficult_no_prog):,}</div>
          <div class="kpi-label">מתוכם – תלמידים מתקשים</div>
          <div class="kpi-sub">ללא שיוך לאף תוכנית</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # ── Breakdown chart by grade ───────────────────────────────────────
        st.markdown("**התפלגות לפי שכבה**")
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(
            chart_no_prog_by_grade(df_status, df_tochnit),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Pupil table ────────────────────────────────────────────────────
        st.markdown("**רשימת תלמידים ללא תוכנית חינוכית**")

        display_cols = {
            "שם תלמיד":            "שם תלמיד",
            "שכבה":                "שכבה",
            "כיתה":                "כיתה",
            "סטטוס מילוי שאלון":   "סטטוס שאלון",
            "מקבץ":                "מקבץ",
        }
        table = no_prog[list(display_cols.keys())].rename(columns=display_cols)

        # Colour-code the מקבץ column via a styled dataframe
        def style_row(row):
            color = MIKBATZ_COLORS.get(row["מקבץ"], PALETTE["grey"])
            return [""] * (len(row) - 1) + [f"color: {color}; font-weight: 600"]

        styled = (
            table.style
            .apply(style_row, axis=1)
            .set_properties(**{"text-align": "right"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(400, 38 + len(table) * 35))

        if not difficult_no_prog.empty:
            st.caption(
                f"💡 {len(difficult_no_prog):,} מהתלמידים ברשימה זו משויכים למקבץ מתקשים "
                f"ואינם משויכים לאף תוכנית חינוכית – מומלץ לבחון שיוך מענה."
            )


def render_programs_section(df_tochnit, df_status=None, df_tosot=None):
    st.markdown('<div class="section-title">🎓 תוכניות חינוכיות</div>', unsafe_allow_html=True)

    total_pupils_with_prog = df_tochnit["מס' זהות תלמיד"].nunique()
    total_unique_progs     = df_tochnit["שם תוכנית חינוכית"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("תלמידים עם לפחות תוכנית אחת", f"{total_pupils_with_prog:,}")
    c2.metric("תוכניות חינוכיות שונות", f"{total_unique_progs:,}")

    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.caption("תוכניות נפוצות ביותר (לפי מספר שיוכים)")
    st.plotly_chart(chart_programs(df_tochnit), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Programs by mikbatz
    with st.expander("📊 תוכניות לפי מקבץ", expanded=False):
        grp = (
            df_tochnit.groupby(["מקבץ", "שם תוכנית חינוכית"])
            .size()
            .reset_index(name="מספר")
            .sort_values(["מקבץ", "מספר"], ascending=[True, False])
        )
        # Sort מקבץ by the defined order; put "-" (no questionnaire) last
        mkbatz_order = list(MIKBATZ_COLORS.keys()) + ["-"]
        ordered_mkbtz = sorted(
            grp["מקבץ"].unique(),
            key=lambda m: mkbatz_order.index(m) if m in mkbatz_order else 999,
        )
        for mkb in ordered_mkbtz:
            sub = grp[grp["מקבץ"] == mkb].head(5)
            color = MIKBATZ_COLORS.get(mkb, PALETTE["grey"])
            if mkb == "-":
                label = "ללא שיוך מקבץ (לא הגישו שאלון)"
                icon  = "⚪"
            else:
                label = mkb
                icon  = "●"
            st.markdown(
                f'<span style="color:{color}; font-weight:700;">{icon} {label}</span>',
                unsafe_allow_html=True,
            )
            st.dataframe(sub[["שם תוכנית חינוכית", "מספר"]], use_container_width=True, hide_index=True)

    # ── No-program pupils ──────────────────────────────────────────────────
    if df_status is not None:
        _render_no_program_expander(df_status, df_tosot, df_tochnit)


# ══════════════════════════════════════════════════════════════════════════════
# Validation detail expander
# ══════════════════════════════════════════════════════════════════════════════

def render_validation_detail(dfs):
    with st.expander("🔍 פרטי אימות קבצים", expanded=False):
        for key, meta in REQUIRED_FILES.items():
            if key in dfs:
                df = dfs[key]
                st.markdown(f"**{meta['icon']} {meta['description']}** &nbsp; `{meta['label']}`")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("שורות", f"{len(df):,}")
                col_b.metric("עמודות", len(df.columns))
                if key == "status":
                    col_c.metric("תלמידים ייחודיים", df["ת.ז תלמיד"].nunique())
                elif key == "tosot":
                    col_c.metric("תלמידים ייחודיים", df["מס' זהות תלמיד"].nunique())
                elif key == "tochnit":
                    col_c.metric("תלמידים ייחודיים", df["מס' זהות תלמיד"].nunique())

                # Check for expected columns
                ok_cols, missing = validate_dataframe(df, meta["required_cols"])
                if ok_cols:
                    st.markdown('<span class="val-ok">✓ כל העמודות הנדרשות קיימות</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="val-err">✗ חסרות עמודות: {", ".join(missing)}</span>', unsafe_allow_html=True)
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Chat UI
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_QUESTIONS = [
    "לכמה תלמידים אין שאלון?",
    "מה המצב בשכבה ג?",
    "מי המתקשים בכל שאינם בתוכנית?",
    "מי מתקשה באנגלית?",
    "הצג תלמידים עם שאלון חלקי",
    "כמה תלמידים יש בכל מקבץ?",
    "סיכום בית הספר",
    "סיכום כיתה ד2",
]

GREETING_TEXT = "שלום ניר, איך אוכל לעזור לך היום? 😊"


def _init_session_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "bot", "content": GREETING_TEXT, "response": None}
        ]
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = ChatContext()


def _render_message(idx: int, role: str, content: str,
                    response: "ChatResponse | None" = None):
    """Render one chat turn using st.chat_message for proper RTL bubble layout."""
    avatar = "🦅" if role == "bot" else "👤"
    with st.chat_message(name=role, avatar=avatar):
        st.markdown(
            f'<div style="direction:rtl; text-align:right;">{content}</div>',
            unsafe_allow_html=True,
        )
        if response:
            if response.chart is not None:
                st.plotly_chart(
                    response.chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"chat_chart_{idx}",   # unique key prevents duplicate-element crash
                )
            if response.table is not None:
                st.dataframe(
                    response.table,
                    use_container_width=True,
                    hide_index=True,
                    height=min(380, 38 + len(response.table) * 35),
                )


def render_chat_tab(df_status, df_tosot, df_tochnit):
    _init_session_state()

    # ── Two-column layout: examples panel (left) + chat (right) ─────────────
    col_examples, col_chat = st.columns([1, 3], gap="medium")

    with col_examples:
        st.markdown(
            '<div class="chat-examples-panel" style="background:#ffffff !important;'
            ' border-radius:12px; padding:1rem 0.8rem;'
            ' border:1px solid #dde2ea; direction:rtl;">'
            '<div style="font-weight:700; color:#1e3a5f !important; margin-bottom:0.6rem;'
            ' font-size:0.92rem;">💡 שאלות לדוגמה</div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                _process_query(q, df_status, df_tosot, df_tochnit)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chat:
        # ── Message history ──────────────────────────────────────────────────
        for idx, msg in enumerate(st.session_state.chat_messages):
            _render_message(idx, msg["role"], msg["content"], msg.get("response"))

        # ── Input box ────────────────────────────────────────────────────────
        user_input = st.chat_input("שאל שאלה...")
        if user_input and user_input.strip():
            _process_query(user_input.strip(), df_status, df_tosot, df_tochnit)
            st.rerun()


def _process_query(query: str, df_status, df_tosot, df_tochnit):
    """Classify → dispatch → append both turns to session state."""
    ctx: ChatContext = st.session_state.chat_context

    st.session_state.chat_messages.append(
        {"role": "user", "content": query, "response": None}
    )

    intent = classify_with_context(query, ctx)

    with st.spinner("מעבד..."):
        response: ChatResponse = handle_intent(
            intent=intent,
            query=query,
            ctx=ctx,
            df_status=df_status,
            df_tosot=df_tosot,
            df_tochnit=df_tochnit,
        )

    if response.context:
        st.session_state.chat_context = response.context

    intent_label = INTENT_LABELS.get(intent, intent)
    annotated_text = (
        response.text
        + f'\n\n<span class="intent-badge">{intent_label}</span>'
    )
    st.session_state.chat_messages.append(
        {"role": "bot", "content": annotated_text, "response": response}
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dfs, errors = render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
      <h1 style="color:#ffffff !important;">🦅 נץ תקומה</h1>
      <p style="color:rgba(255,255,255,0.88) !important;">עוזר נתונים בית ספרי · העלה את קבצי בית הספר וקבל תמונת מצב מיידית</p>
    </div>
    """, unsafe_allow_html=True)

    all_uploaded = len(dfs) == len(REQUIRED_FILES)

    if not dfs:
        render_onboarding()
        return

    # Coerce & merge
    df_status  = dfs.get("status")
    df_tosot   = dfs.get("tosot")
    df_tochnit = dfs.get("tochnit")

    if df_status is not None and df_tosot is not None and df_tochnit is not None:
        df_status, df_tosot, df_tochnit = coerce_types(df_status, df_tosot, df_tochnit)

    # School name & info banner
    school_name = get_school_name(df_tochnit) if df_tochnit is not None else "בית הספר"
    school_year = (df_tochnit["שנת לימודים"].iloc[0]
                   if df_tochnit is not None and "שנת לימודים" in df_tochnit.columns
                   else "")
    school_id   = (df_tochnit["סמל מוסד"].iloc[0]
                   if df_tochnit is not None and "סמל מוסד" in df_tochnit.columns
                   else "")

    st.markdown(f"""
    <div style="background:#f0f7ff; border-radius:10px; padding:0.8rem 1.2rem;
                display:flex; justify-content:space-between; align-items:center;
                margin-bottom:1rem; direction:rtl;">
      <div>
        <span style="font-size:1.1rem; font-weight:700; color:#1e3a5f;">🏫 {school_name}</span>
        &nbsp;&nbsp;<span style="color:#64748b; font-size:0.9rem;">סמל מוסד: {school_id}</span>
      </div>
      <div style="color:#64748b; font-size:0.9rem;">שנת לימודים: {school_year}</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload status bar
    uploaded_count = len(dfs)
    total_count    = len(REQUIRED_FILES)
    if not all_uploaded:
        st.warning(f"⚠️ הועלו {uploaded_count} מתוך {total_count} קבצים – חלק מהלוח לא יוצג")

    # Validation detail
    render_validation_detail(dfs)

    # ── KPI Cards ──────────────────────────────────────────────────────────
    if df_status is not None and df_tochnit is not None:
        kpis = summary_kpis(df_status, df_tosot, df_tochnit)
        render_kpi_cards(kpis)
    elif df_status is not None:
        kpis = summary_kpis(
            df_status,
            df_tosot,
            pd.DataFrame(columns=["מס' זהות תלמיד", "שם תוכנית חינוכית"]),
        )
        render_kpi_cards(kpis)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────
    tabs = []
    tab_labels = []

    if df_status is not None:
        tab_labels.append("📋 מצב מילוי שאלון")
    if df_tosot is not None:
        tab_labels.append("🔷 מקבץ ומיומנויות")
    if df_tochnit is not None:
        tab_labels.append("🎓 תוכניות חינוכיות")
    # Chat tab always available once at least one file is uploaded
    tab_labels.append("💬 עוזר חכם")

    if tab_labels:
        tab_objects = st.tabs(tab_labels)
        idx = 0
        if df_status is not None:
            with tab_objects[idx]:
                render_questionnaire_section(df_status)
            idx += 1
        if df_tosot is not None:
            with tab_objects[idx]:
                render_mikbatz_section(df_tosot)
            idx += 1
        if df_tochnit is not None:
            with tab_objects[idx]:
                render_programs_section(df_tochnit, df_status=df_status, df_tosot=df_tosot)
            idx += 1
        # Chat tab is always last
        with tab_objects[idx]:
            render_chat_tab(df_status, df_tosot, df_tochnit)


if __name__ == "__main__":
    main()
