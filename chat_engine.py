"""
chat_engine.py  –  Phase 2 chatbot: deterministic analysis + Hebrew responses.

Context rules (authoritative):
  • ONLY explicit anaphora tokens (מתוכם / מהם / מתוך אלה / אותם תלמידים /
    בכיתה הזו / בשכבה הזו) cause a query to inherit the previous result set.
  • Every other query is treated as fresh / school-wide, regardless of previous ctx.
  • Explicit whole-school tokens override everything.
  • skill_rows() MUST gate every מיומנות analysis (מיומנות="-" excluded).
  • Empty results → graceful Hebrew message, never a fallback school summary.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

GRADE_ORDER        = ["א", "ב", "ג", "ד", "ה", "ו"]
DIFFICULTY_ANSWERS = ["מתקשה", "מתקשה מאוד"]
NON_SKILL_VALUE    = "-"
DIFFICULT_MIKBATZ  = {"מתקשים לימודית", "מתקשים חברתית ורגשית", "מתקשים בכל"}
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

MIKBATZ_COLORS = {
    "מתקדמים בכל":           PALETTE["teal"],
    "תקינים בכל":            PALETTE["navy"],
    "מתקשים לימודית":        PALETTE["orange"],
    "מתקשים חברתית ורגשית": PALETTE["purple"],
    "מתקשים בכל":            PALETTE["red"],
}
STATUS_COLORS = {"הסתיים": PALETTE["teal"], "חלקי": PALETTE["orange"], "חסר": PALETTE["red"]}


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class ChatContext:
    last_intent:    Optional[str]          = None
    last_result_df: Optional[pd.DataFrame] = None  # last pupil-level result
    last_grade:     Optional[str]          = None
    last_class:     Optional[str]          = None
    last_mikbatz:   Optional[str]          = None
    last_skill:     Optional[str]          = None  # last מיומנות mentioned
    # Pending state: stored when clarification is asked
    pending_intent: Optional[str]          = None  # intent waiting for scope answer
    pending_mikbatz: Optional[str]         = None  # mikbatz for the pending question

@dataclass
class ChatResponse:
    text:    str
    table:   Optional[pd.DataFrame] = None
    chart:   Optional[go.Figure]    = None
    context: Optional[ChatContext]   = None


# ── Mandatory gateway ────────────────────────────────────────────────────────

def skill_rows(df_tosot: pd.DataFrame) -> pd.DataFrame:
    """MANDATORY: exclude מיומנות='-' metadata rows before any skill analysis."""
    filtered = df_tosot[df_tosot["מיומנות"] != NON_SKILL_VALUE]
    assert NON_SKILL_VALUE not in filtered["מיומנות"].values
    return filtered


# ── Token sets ──────────────────────────────────────────────────────────────

# Tokens that EXPLICITLY mean "use previous filtered result set".
# Must be unambiguous — no token that can appear as a substring of common words.
# "אלו" / "אלה" removed because they appear inside "שאלון" / "שאלוהם" etc.
_FOLLOWUP_TOKENS = [
    "מתוכם", "מהם",
    "מתוך אלה", "מתוך התלמידים האלו",
    "אותם תלמידים",
    "בכיתה הזו", "בשכבה הזו",
    "מתוך אותם",
]

# Tokens that explicitly override context → always answer school-wide
_WHOLE_SCHOOL_TOKENS = [
    "מתוך כלל תלמידי בית הספר",
    "בכל בית הספר",
    "כלל בית הספר",
    "כל בית הספר",
    "בכלל בית הספר",
    "הכוונה לכלל",          # answer to clarification question
    "בבית הספר כולו",
    "לכלל תלמידי",          # "לכלל תלמידי בית הספר"
    "לכלל בית הספר",
]

# Tokens that mean "same condition, whole school"
_SAME_CONDITION_WHOLE_SCHOOL = [
    "כמה כאלה", "כמה מהם בכלל", "כמה יש בכלל",
    "בכלל בית הספר",        # without "כלל" — shorter variant
]


def _is_followup(query: str) -> bool:
    """True ONLY for explicit multi-char anaphora tokens — no short ambiguous ones."""
    return any(t in query for t in _FOLLOWUP_TOKENS)


def _is_whole_school(query: str) -> bool:
    return any(t in query for t in _WHOLE_SCHOOL_TOKENS)


def _is_same_condition_whole_school(query: str, ctx: ChatContext) -> bool:
    """'כמה כאלה יש בכלל בית הספר?' — keep active mikbatz/condition, drop scope."""
    return (any(t in query for t in _SAME_CONDITION_WHOLE_SCHOOL)
            and ctx.last_mikbatz is not None)


def _is_clarification_answer(query: str) -> bool:
    """User is answering our scope-clarification question with a whole-school choice."""
    return _is_whole_school(query) or any(t in query for t in [
        "הכוונה לכלל", "כלל בית הספר", "הכוונה לבית הספר", "כל בית הספר",
        "לכלל תלמידי", "לכלל בית הספר", "לכלל",
    ])


# ── Parsing helpers ──────────────────────────────────────────────────────────

_CLASS_WORD = r"(?:כית[הת]?|כית[בב]|כת[הת]?|בכית[הת]?|בכית[בב]|בכת[הת]?)\s*"


def _extract_class(query: str) -> Optional[str]:
    m = re.search(_CLASS_WORD + r"([א-ו])\s*(\d)", query)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r"([א-ו])\s*(\d)", query)
    if m:
        return m.group(1) + m.group(2)
    return None


def _extract_grade(query: str) -> Optional[str]:
    """Extract grade letter for whole-grade queries; None when a class digit is present."""
    if re.search(r"[א-ו]\s*\d", query):
        return None
    m = re.search(r"שכב[הת]\s*([א-ו])(?:\s|[?!.,]|$)", query)
    if m:
        return m.group(1)
    for g in GRADE_ORDER:
        if re.search(rf"(?:^|\s){re.escape(g)}(?:\s|[?!.,]|$)", query):
            return g
    return None


def _extract_mikbatz(query: str) -> Optional[str]:
    """Return the specific מקבץ value mentioned in the query; normalises בהכל → בכל.

    Returns None when query contains only a generic difficulty word (e.g. 'מתקשים')
    without a specific מקבץ qualifier — those are handled by _is_general_difficulty.
    """
    normalised = query.replace("בהכל", "בכל")
    for mkb in sorted(MIKBATZ_COLORS.keys(), key=len, reverse=True):
        if mkb in normalised:
            return mkb
    return None


def _is_general_difficulty(query: str) -> bool:
    """True when the query asks about 'מתקשים' in general (not a specific מקבץ or skill).

    Business rule:
    - "המתקשים בכל" → specific מקבץ = "מתקשים בכל"  → _extract_mikbatz handles it
    - "מתקשה באנגלית" → skill difficulty                → _handle_pupils_with_skill_difficulty
    - "המתקשים" / "תלמידים מתקשים" (no qualifier) →
      means ALL difficulty מקבץ groups: מתקשים בכל + מתקשים חברתית ורגשית + מתקשים לימודית
      → this function returns True so handler uses DIFFICULT_MIKBATZ
    """
    GENERAL_DIFFICULTY_TOKENS = [
        "מתקשים", "מתקשות", "מתקשת", "מתקשה",
        "תלמידים מתקשים", "תלמידות מתקשות",
    ]
    # Not general if a specific מקבץ is named
    if _extract_mikbatz(query) is not None:
        return False
    # Not general if a specific skill domain is named (those go to skill_difficulty)
    if _extract_skill(query) is not None:
        return False
    return any(t in query for t in GENERAL_DIFFICULTY_TOKENS)


# Canonical skill names as they appear in the מיומנות column, paired with
# Hebrew surface-form aliases used in natural queries.
_SKILL_ALIASES: list[tuple[str, list[str]]] = [
    ("היבטים התנהגותיים", ["היבטים התנהגותיים", "התנהגות", "התנהגותי", "התנהגותית"]),
    ("היבטים חברתיים",    ["היבטים חברתיים", "חברתי", "חברתית", "חברתיים", "חברתיות"]),
    ("היבטים רגשיים",     ["היבטים רגשיים", "רגשי", "רגשית", "רגשיים", "רגשיות"]),
    ("מוטיבציה והרגלי למידה", ["מוטיבציה", "הרגלי למידה", "הרגלי לימוד"]),
    ("קשב ופעלתנות יתר",  ["קשב", "פעלתנות", "ריכוז", "קשב ופעלתנות"]),
    ("חושי תנועתי",       ["חושי תנועתי", "חושי", "תנועתי"]),
    ("שפת אם",            ["שפת אם", "שפה", "קריאה", "כתיבה", "עברית"]),
    ("מתמטיקה",           ["מתמטיקה", "מתמט", "חשבון", "מספרים"]),
    ("אנגלית",            ["אנגלית", "אנגלי", "אנגליה"]),
]
# Longest-match: sort by alias length descending so "היבטים חברתיים" beats "חברתי"
_SKILL_ALIAS_FLAT: list[tuple[str, str]] = sorted(
    [(alias, canonical)
     for canonical, aliases in _SKILL_ALIASES
     for alias in aliases],
    key=lambda x: len(x[0]), reverse=True,
)


def _extract_skill(query: str) -> Optional[str]:
    """Return the canonical מיומנות name mentioned in the query, or None.

    Uses longest-match so 'היבטים חברתיים' wins over bare 'חברתי'.
    """
    for alias, canonical in _SKILL_ALIAS_FLAT:
        if alias in query:
            return canonical
    return None


def _followup_base(query: str, ctx: ChatContext,
                   df_status: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return previous result df ONLY for explicit anaphora; None otherwise.

    Rule: if the query contains any _FOLLOWUP_TOKEN → use ctx.last_result_df.
          All other queries (including example buttons) → None (fresh / school-wide).
    """
    if _is_whole_school(query):
        return None   # explicit override
    if _is_followup(query) and ctx.last_result_df is not None:
        prev = ctx.last_result_df
        if "ת.ז תלמיד" not in prev.columns and "מס' זהות תלמיד" in prev.columns:
            if df_status is not None:
                ids = set(prev["מס' זהות תלמיד"].unique())
                return df_status[df_status["ת.ז תלמיד"].isin(ids)].copy()
        return prev
    return None   # fresh query — never inherit previous result


def _is_asking_with_program(query: str) -> bool:
    return any(t in query for t in [
        "משויכים לתוכנית", "יש להם תוכנית", "עם תוכנית",
        "שיש להם תוכנית", "משויך לתוכנית",
    ])


def _no_results(label: str, ctx: ChatContext) -> ChatResponse:
    """Graceful empty-result response — never falls back to school summary."""
    return ChatResponse(
        text=f"לא מצאתי תלמידים שעונים לתנאים האלו {label}.",
        context=ctx,
    )


# ── Table formatter ──────────────────────────────────────────────────────────

def _pupil_display_cols(df: pd.DataFrame, include_mikbatz: bool = True) -> pd.DataFrame:
    df = df.copy()
    if "כיתה" not in df.columns and "שכבה" in df.columns and "מקבילה" in df.columns:
        df["כיתה"] = df["שכבה"] + df["מקבילה"]
    keep = {}
    for src, dst in [("שם תלמיד","שם תלמיד"),("שכבה","שכבה"),("כיתה","כיתה"),
                     ("סטטוס מילוי שאלון","סטטוס שאלון"),("מקבץ","מקבץ")]:
        if src in df.columns and (src != "מקבץ" or include_mikbatz):
            keep[src] = dst
    return df[list(keep.keys())].rename(columns=keep)


# ── Micro chart ──────────────────────────────────────────────────────────────

def _bar_chart(labels, values, colors, title="", height=260, h=False):
    if h:
        fig = go.Figure(go.Bar(y=labels, x=values, orientation="h",
                               marker_color=colors, text=values, textposition="outside"))
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          xaxis=dict(range=[0, max(values)*1.2] if values else [0,1]))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                               text=values, textposition="outside"))
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=labels),
            yaxis=dict(range=[0, max(values)*1.2] if values else [0,1]))

    layout_kwargs = dict(height=height, margin=dict(t=30,b=40,l=10,r=80),
                         paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                         showlegend=False)
    # Only set title when there is actual text — passing title="" or title=None
    # causes Plotly to create a Title object with text="" which can render as
    # "undefined" in the SVG when font styling is applied afterwards.
    if title:
        layout_kwargs["title_text"] = title
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=not h, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=h,     gridcolor="#f1f5f9")
    return fig


# ── Intent handlers ──────────────────────────────────────────────────────────

def _handle_greeting(ctx, **_) -> ChatResponse:
    return ChatResponse(text="שלום ניר, איך אוכל לעזור לך היום? 😊", context=ctx)


def _handle_school_summary(df_status, df_tosot, df_tochnit, ctx, **_) -> ChatResponse:
    total     = len(df_status)
    completed = (df_status["סטטוס מילוי שאלון"]=="הסתיים").sum()
    partial   = (df_status["סטטוס מילוי שאלון"]=="חלקי").sum()
    missing   = (df_status["סטטוס מילוי שאלון"]=="חסר").sum()
    pct       = round(completed/total*100,1) if total else 0
    with_prog = df_tochnit["מס' זהות תלמיד"].nunique() if df_tochnit is not None else 0
    difficult = 0
    if df_tosot is not None and not df_tosot.empty:
        dedup = df_tosot.drop_duplicates("מס' זהות תלמיד")
        difficult = dedup[dedup["מקבץ"].isin(DIFFICULT_MIKBATZ)].shape[0]
    # Reset context — whole-school result clears previous filters
    ctx.last_intent = "school_summary"
    ctx.last_result_df = None; ctx.last_grade = None; ctx.last_class = None
    ctx.pending_intent = None; ctx.pending_mikbatz = None
    return ChatResponse(
        text=(f"**סיכום בית הספר**\n\n"
              f"- סך תלמידים: **{total:,}**\n"
              f"- שאלון הושלם: **{completed:,}** ({pct}%)\n"
              f"- שאלון חלקי: **{partial:,}**\n"
              f"- שאלון חסר: **{missing:,}**\n"
              f"- עם תוכנית חינוכית: **{with_prog:,}** | ללא: **{total-with_prog:,}**\n"
              f"- תלמידים מתקשים (לפי מקבץ): **{difficult:,}**"),
        context=ctx)


def _handle_questionnaire_status_summary(df_status, ctx, **_) -> ChatResponse:
    total=len(df_status); completed=(df_status["סטטוס מילוי שאלון"]=="הסתיים").sum()
    partial=(df_status["סטטוס מילוי שאלון"]=="חלקי").sum()
    missing=(df_status["סטטוס מילוי שאלון"]=="חסר").sum()
    pct=round(completed/total*100,1) if total else 0
    chart = _bar_chart(["הסתיים","חלקי","חסר"],[int(completed),int(partial),int(missing)],
                       [STATUS_COLORS[s] for s in ["הסתיים","חלקי","חסר"]],"סטטוס שאלון")
    ctx.last_intent="questionnaire_status_summary"; ctx.last_result_df=None
    return ChatResponse(
        text=(f"**סטטוס מילוי שאלון – כלל בית הספר**\n\n"
              f"- סה\"כ: **{total:,}** | ✅ הושלם: **{completed:,}** ({pct}%)"
              f" | ⚠️ חלקי: **{partial:,}** | ❌ חסר: **{missing:,}**"),
        chart=chart, context=ctx)


def _handle_questionnaire_status_by_grade(df_status, query, ctx, **_) -> ChatResponse:
    grade = _extract_grade(query)   # never inherit from ctx — fresh query
    grp = df_status.groupby(["שכבה","סטטוס מילוי שאלון"]).size().reset_index(name="מספר")
    totals = df_status.groupby("שכבה").size().rename("סהכ")
    grp = grp.merge(totals, on="שכבה")
    grp["שכבה"] = pd.Categorical(grp["שכבה"], categories=GRADE_ORDER, ordered=True)
    grp = grp.sort_values(["שכבה","סטטוס מילוי שאלון"])
    if grade:
        grp = grp[grp["שכבה"]==grade]; ctx.last_grade = grade
    if grp.empty:
        return _no_results(f"לשכבה {grade}" if grade else "", ctx)
    fig = go.Figure()
    for status in ["הסתיים","חלקי","חסר"]:
        sub = grp[grp["סטטוס מילוי שאלון"]==status]
        fig.add_trace(go.Bar(name=status, x=sub["שכבה"].astype(str), y=sub["מספר"],
                             marker_color=STATUS_COLORS[status],
                             text=sub["מספר"], textposition="inside", insidetextanchor="middle"))
    fig.update_layout(barmode="stack", height=280, margin=dict(t=10,b=30,l=10,r=10),
                      legend=dict(orientation="h",y=-0.25),
                      paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
    fig.update_xaxes(categoryorder="array", categoryarray=GRADE_ORDER, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    rows = []
    for g in grp["שכבה"].cat.categories if hasattr(grp["שכבה"],"cat") else sorted(grp["שכבה"].unique()):
        sub=grp[grp["שכבה"]==g]
        if sub.empty: continue
        t=sub["סהכ"].iloc[0]; d=sub.loc[sub["סטטוס מילוי שאלון"]=="הסתיים","מספר"].sum()
        rows.append(f"שכבה {g}: {d}/{t} הושלמו ({round(d/t*100,1) if t else 0}%)")
    label = f"שכבה {grade}" if grade else "לפי שכבה"
    ctx.last_intent="questionnaire_status_by_grade"
    return ChatResponse(text=f"**סטטוס שאלון {label}**\n\n"+"\n".join(f"- {r}" for r in rows),
                        chart=fig, context=ctx)


def _handle_pupils_missing_questionnaire(df_status, query, ctx, **_) -> ChatResponse:
    # Only inherit previous result for explicit anaphora; otherwise school-wide
    base  = _followup_base(query, ctx, df_status)
    src   = base if base is not None else df_status
    grade = _extract_grade(query)   # only from query, never from ctx implicitly
    df    = src[src["סטטוס מילוי שאלון"]=="חסר"].copy()
    if grade:
        df = df[df["שכבה"]==grade]; ctx.last_grade = grade; label = f"בשכבה {grade}"
    else:
        label = "בכלל בית הספר"
    if df.empty:
        return _no_results(label, ctx)
    df["כיתה"] = df["שכבה"] + df["מקבילה"]
    table = _pupil_display_cols(df, include_mikbatz=False)
    ctx.last_intent="pupils_missing_questionnaire"; ctx.last_result_df=df
    return ChatResponse(
        text=f"**תלמידים שלא הגישו שאלון {label}**\n\nנמצאו **{len(df):,}** תלמידים.",
        table=table, context=ctx)


def _handle_pupils_partial_questionnaire(df_status, query, ctx, **_) -> ChatResponse:
    base  = _followup_base(query, ctx, df_status)
    src   = base if base is not None else df_status
    grade = _extract_grade(query)   # only from query
    df    = src[src["סטטוס מילוי שאלון"]=="חלקי"].copy()
    if grade:
        df = df[df["שכבה"]==grade]; ctx.last_grade = grade; label = f"בשכבה {grade}"
    else:
        label = "בכלל בית הספר"
    if df.empty:
        return _no_results(label, ctx)
    df["כיתה"] = df["שכבה"] + df["מקבילה"]
    table = _pupil_display_cols(df, include_mikbatz=False)
    ctx.last_intent="pupils_partial_questionnaire"; ctx.last_result_df=df
    return ChatResponse(
        text=f"**תלמידים עם שאלון חלקי {label}**\n\nנמצאו **{len(df):,}** תלמידים.",
        table=table, context=ctx)


def _handle_pupils_by_mikbatz(df_status, df_tosot, query, ctx, **_) -> ChatResponse:
    if df_tosot is None or df_tosot.empty:
        return ChatResponse(text="אין נתוני שאלון זמינים.", context=ctx)

    whole_school = _is_whole_school(query)
    same_cond_ws = _is_same_condition_whole_school(query, ctx)

    # "כמה כאלה יש בכלל בית הספר?" — keep ctx.last_mikbatz, apply school-wide.
    # Must be checked BEFORE the simple whole_school branch, because the query
    # may contain both a whole-school token AND a same-condition signal.
    if same_cond_ws:
        mkb  = ctx.last_mikbatz
        all_ = df_tosot.drop_duplicates("מס' זהות תלמיד")
        sub  = all_[all_["מקבץ"]==mkb].copy()
        sub["כיתה"] = sub["שכבה"] + sub["מקבילה"]
        if sub.empty:
            return _no_results(f"במקבץ '{mkb}' בכלל בית הספר", ctx)
        ctx.last_intent = "pupils_by_mikbatz"; ctx.last_result_df = sub
        ctx.last_class = None; ctx.last_grade = None
        return ChatResponse(
            text=f"ישנם **{len(sub):,}** תלמידים במקבץ '{mkb}' בכלל בית הספר.",
            table=sub[["שם תלמיד","שכבה","כיתה"]],
            context=ctx)

    base = None if whole_school else _followup_base(query, ctx, df_status)

    specific_mkb = _extract_mikbatz(query)
    class_label  = _extract_class(query)
    grade        = None if class_label else _extract_grade(query)

    # Clarification answer: user answered previous ambiguous question
    # Read pending_mikbatz regardless of whether _is_clarification_answer matches,
    # since routing here already means classify_with_context resolved pending_intent.
    if ctx.pending_intent == "pupils_by_mikbatz" and ctx.pending_mikbatz:
        if not specific_mkb:
            specific_mkb = ctx.pending_mikbatz
        whole_school = True
        base = None
        ctx.pending_intent = None; ctx.pending_mikbatz = None
    elif _is_clarification_answer(query) and ctx.pending_intent == "pupils_by_mikbatz":
        specific_mkb = ctx.pending_mikbatz or specific_mkb
        whole_school = True; base = None
        ctx.pending_intent = None; ctx.pending_mikbatz = None

    # Ambiguity: specific mikbatz, no explicit scope, previous context exists
    if (not whole_school and base is None and not class_label and grade is None
            and not _is_followup(query) and ctx.last_result_df is not None
            and ctx.last_intent is not None and specific_mkb is not None):
        ctx.pending_intent  = "pupils_by_mikbatz"
        ctx.pending_mikbatz = specific_mkb
        ctx.last_intent = "unclear"
        return ChatResponse(
            text="האם הכוונה היא לתלמידים שעליהם דיברנו קודם, או לכלל תלמידי בית הספר?",
            context=ctx)

    unique = df_tosot.drop_duplicates("מס' זהות תלמיד")[
        ["מס' זהות תלמיד","שם תלמיד","שכבה","מקבילה","מקבץ"]].copy()
    unique["כיתה"] = unique["שכבה"] + unique["מקבילה"]

    if base is not None:
        id_col   = "ת.ז תלמיד" if "ת.ז תלמיד" in base.columns else "מס' זהות תלמיד"
        prev_ids = set(base[id_col].unique())
        unique   = unique[unique["מס' זהות תלמיד"].isin(prev_ids)]

    if class_label:
        g, p = class_label[0], class_label[1]
        unique = unique[(unique["שכבה"]==g) & (unique["מקבילה"]==p)]
        ctx.last_class = class_label; ctx.last_grade = g
        scope = f"כיתה {class_label}"
        class_total = len(df_status[(df_status["שכבה"]==g) & (df_status["מקבילה"]==p)])
    elif grade:
        unique = unique[unique["שכבה"]==grade]; ctx.last_grade = grade
        scope = f"שכבה {grade}"
        class_total = len(df_status[df_status["שכבה"]==grade])
    else:
        scope = "כלל בית הספר" if base is None else "התוצאות הקודמות"
        class_total = None

    if specific_mkb:
        subset = unique[unique["מקבץ"]==specific_mkb]
        ctx.last_mikbatz = specific_mkb
        if subset.empty:
            ctx.last_intent = "pupils_by_mikbatz"
            return _no_results(f"במקבץ '{specific_mkb}' – {scope}", ctx)
        if class_total is not None:
            scope_type = "כיתה" if class_label else "שכבה"
            scope_name = class_label if class_label else grade
            lead = (f"ישנם **{class_total}** תלמידים ב{scope_type} {scope_name}, "
                    f"מתוכם **{len(subset)}** תלמידים במקבץ '{specific_mkb}'.")
        else:
            lead = f"ישנם **{len(subset):,}** תלמידים במקבץ '{specific_mkb}' – {scope}."
        table = subset[["שם תלמיד","שכבה","כיתה"]]
        ctx.last_intent = "pupils_by_mikbatz"; ctx.last_result_df = subset
        return ChatResponse(text=lead, table=table, context=ctx)

    # General difficulty: query says "מתקשים" without a specific מקבץ.
    # Business rule: this refers to ALL difficulty groups combined —
    #   "מתקשים בכל", "מתקשים חברתית ורגשית", "מתקשים לימודית"
    # It does NOT include "תקינים בכל" or "מתקדמים בכל".
    if _is_general_difficulty(query):
        subset = unique[unique["מקבץ"].isin(DIFFICULT_MIKBATZ)]
        if subset.empty:
            ctx.last_intent = "pupils_by_mikbatz"
            return _no_results(f"מתקשים – {scope}", ctx)
        # Show breakdown by מקבץ within the difficulty groups
        dist = subset["מקבץ"].value_counts().reset_index()
        dist.columns = ["מקבץ","מספר תלמידים"]
        colors = [MIKBATZ_COLORS.get(m,"#94a3b8") for m in dist["מקבץ"]]
        chart  = _bar_chart(dist["מקבץ"].tolist(), dist["מספר תלמידים"].tolist(), colors,
                            f"תלמידים מתקשים – {scope}", height=240)
        lines  = [f"- {r['מקבץ']}: **{r['מספר תלמידים']:,}**" for _,r in dist.iterrows()]
        scope_label = f" {scope}" if scope != "כלל בית הספר" else ""
        text   = (f"**תלמידים מתקשים{scope_label}** (כלל קבוצות הקושי)\n\n"
                  + "\n".join(lines)
                  + f"\n\n**סה\"כ: {len(subset):,} תלמידים מתקשים**"
                  + "\n\nניתן לצמצם לקבוצה ספציפית, למשל: 'מי הם המתקשים בכל?'")
        ctx.last_intent = "pupils_by_mikbatz"; ctx.last_result_df = subset
        return ChatResponse(text=text, chart=chart, context=ctx)

    # Full distribution (no difficulty signal — show all מקבצים)
    dist = unique["מקבץ"].value_counts().reset_index()
    dist.columns = ["מקבץ","מספר תלמידים"]
    if dist.empty:
        return _no_results(f"– {scope}", ctx)
    colors = [MIKBATZ_COLORS.get(m,"#94a3b8") for m in dist["מקבץ"]]
    chart  = _bar_chart(dist["מקבץ"].tolist(), dist["מספר תלמידים"].tolist(), colors,
                        f"התפלגות מקבץ – {scope}", height=280)
    lines  = [f"- {r['מקבץ']}: **{r['מספר תלמידים']:,}**" for _,r in dist.iterrows()]
    text   = (f"**התפלגות לפי מקבץ – {scope}**\n\n"+"\n".join(lines)
              +"\n\nניתן לשאול על מקבץ ספציפי, למשל: 'מי הם המתקשים בכל?'")
    ctx.last_intent = "pupils_by_mikbatz"
    return ChatResponse(text=text, chart=chart, context=ctx)


def _handle_pupils_with_skill_difficulty(df_tosot, query, ctx, df_status=None, **_) -> ChatResponse:
    """Return pupils who struggle in a specific מיומנות (any skill domain).

    BUSINESS RULE: skill_rows() is mandatory — מיומנות='-' rows are excluded.
    Counts are unique pupils, never rows.
    Supports optional שכבה or כיתה filter.
    Returns a bar chart by grade.
    """
    if df_tosot is None or df_tosot.empty:
        return ChatResponse(text="אין נתוני שאלון זמינים.", context=ctx)

    skill = _extract_skill(query) or ctx.last_skill
    if not skill:
        return ChatResponse(
            text="לא זיהיתי איזו מיומנות נשאלת. נסה לציין במפורש, למשל: 'מתקשים באנגלית' או 'קשב'.",
            context=ctx)

    class_label = _extract_class(query)
    grade       = None if class_label else _extract_grade(query)

    sr   = skill_rows(df_tosot)  # mandatory — excludes מיומנות="-"
    diff = sr[(sr["מיומנות"] == skill) & (sr["תשובה"].isin(DIFFICULTY_ANSWERS))]

    if class_label:
        g, p = class_label[0], class_label[1]
        diff = diff[(diff["שכבה"] == g) & (diff["מקבילה"] == p)]
        ctx.last_class = class_label; ctx.last_grade = g
        label = f"בכיתה {class_label}"
    elif grade:
        diff = diff[diff["שכבה"] == grade]
        ctx.last_grade = grade
        label = f"בשכבה {grade}"
    else:
        label = "בכלל בית הספר"

    unique = diff.drop_duplicates("מס' זהות תלמיד")[
        ["מס' זהות תלמיד", "שם תלמיד", "שכבה", "מקבילה", "מקבץ"]
    ].copy()
    unique["כיתה"] = unique["שכבה"] + unique["מקבילה"]

    if unique.empty:
        ctx.last_skill = skill; ctx.last_intent = "pupils_with_skill_difficulty"
        return _no_results(f"במיומנות '{skill}' {label}", ctx)

    # Bar chart by grade
    grade_counts = (
        unique.groupby("שכבה").size()
        .reindex(GRADE_ORDER, fill_value=0)
        .reset_index()
    )
    grade_counts.columns = ["שכבה", "מספר"]
    chart = _bar_chart(
        grade_counts["שכבה"].tolist(), grade_counts["מספר"].tolist(),
        [PALETTE["navy"]] * len(grade_counts),
        f"מתקשים ב'{skill}' לפי שכבה", height=260,
    )

    ctx.last_skill = skill; ctx.last_intent = "pupils_with_skill_difficulty"
    ctx.last_result_df = unique
    return ChatResponse(
        text=(f"נמצאו **{len(unique):,}** תלמידים ייחודיים המתקשים ב'{skill}' {label}.\n\n"
              f"_ספירה: תלמידים ייחודיים, לא שורות._"),
        table=unique[["שם תלמיד", "שכבה", "כיתה", "מקבץ"]],
        chart=chart,
        context=ctx)


def _handle_pupils_without_program(df_status, df_tosot, df_tochnit, query, ctx, **_) -> ChatResponse:
    if df_tochnit is None:
        return ChatResponse(text="אין נתוני תוכניות זמינים.", context=ctx)

    whole_school = _is_whole_school(query)
    base         = None if whole_school else _followup_base(query, ctx, df_status)
    working      = base if base is not None else df_status

    if base is not None:
        class_label = None; grade = None
    else:
        class_label = _extract_class(query)
        grade       = None if class_label else _extract_grade(query)

    assigned = set(df_tochnit["מס' זהות תלמיד"].unique())

    def _enrich_mkb(df):
        df = df.copy()
        if df_tosot is not None and not df_tosot.empty:
            mkb_map = df_tosot.drop_duplicates("מס' זהות תלמיד").set_index("מס' זהות תלמיד")["מקבץ"]
            df["מקבץ"] = df["ת.ז תלמיד"].map(mkb_map).fillna("לא ידוע")
        else:
            df["מקבץ"] = "לא ידוע"
        if "כיתה" not in df.columns and "שכבה" in df.columns and "מקבילה" in df.columns:
            df["כיתה"] = df["שכבה"] + df["מקבילה"]
        return df

    asking_with = _is_asking_with_program(query)

    if asking_with and base is not None:
        df = _enrich_mkb(working)
        id_col  = "ת.ז תלמיד" if "ת.ז תלמיד" in df.columns else "מס' זהות תלמיד"
        with_df = df[df[id_col].isin(assigned)].copy()
        prog_info = (
            df_tochnit.groupby("מס' זהות תלמיד")["שם תוכנית חינוכית"]
            .apply(lambda x: ", ".join(sorted(x.unique())))
            .reset_index()
            .rename(columns={"מס' זהות תלמיד":"ת.ז תלמיד","שם תוכנית חינוכית":"תוכניות"})
        )
        if not with_df.empty:
            with_df = with_df.merge(prog_info, on="ת.ז תלמיד", how="left")
            with_df["תוכניות"] = with_df["תוכניות"].fillna("—")
        without_df = df[~df[id_col].isin(assigned)]
        table_cols = [c for c in ["שם תלמיד","שכבה","כיתה","מקבץ","תוכניות"] if c in with_df.columns]
        ctx.last_intent="pupils_without_program"; ctx.last_result_df=with_df
        return ChatResponse(
            text=(f"מתוך {len(df):,} התלמידים הקודמים:\n\n"
                  f"- ✅ **{len(with_df):,}** משויכים לתוכנית חינוכית\n"
                  f"- ❌ **{len(without_df):,}** אינם משויכים לאף תוכנית"),
            table=with_df[table_cols] if not with_df.empty else None,
            context=ctx)

    df = _enrich_mkb(working)
    df = df[~df["ת.ז תלמיד"].isin(assigned)].copy()

    if class_label:
        g, p = class_label[0], class_label[1]
        df = df[(df["שכבה"]==g) & (df["מקבילה"]==p)]
        ctx.last_class=class_label; ctx.last_grade=g; label=f"בכיתה {class_label}"
        denom = len(working[(working["שכבה"]==g) & (working["מקבילה"]==p)])
    elif grade:
        df = df[df["שכבה"]==grade]; ctx.last_grade=grade; label=f"בשכבה {grade}"
        denom = len(working[working["שכבה"]==grade])
    else:
        label = "בתוצאות הקודמות" if base is not None else "בכלל בית הספר"
        denom = len(working)

    if df.empty:
        return _no_results(label, ctx)
    pct = round(len(df)/denom*100,1) if denom else 0
    table = df[["שם תלמיד","שכבה","כיתה","סטטוס מילוי שאלון","מקבץ"]].rename(
        columns={"סטטוס מילוי שאלון":"סטטוס שאלון"})
    ctx.last_intent="pupils_without_program"; ctx.last_result_df=df
    return ChatResponse(
        text=f"נמצאו **{len(df):,}** תלמידים ({pct}%) ללא תוכנית חינוכית {label}.",
        table=table, context=ctx)


def _handle_skill_difficulty_no_program(df_status, df_tosot, df_tochnit, query, ctx, **_) -> ChatResponse:
    """Pupils who struggle in a specific מיומנות AND have no educational program.

    BUSINESS RULE: skill_rows() mandatory — מיומנות='-' excluded.
    Counts unique pupils, not rows.
    """
    if df_tosot is None or df_tosot.empty:
        return ChatResponse(text="אין נתוני שאלון זמינים.", context=ctx)
    if df_tochnit is None:
        return ChatResponse(text="אין נתוני תוכניות זמינים.", context=ctx)

    skill = _extract_skill(query) or ctx.last_skill
    if not skill:
        return ChatResponse(
            text="לא זיהיתי איזו מיומנות נשאלת. נסה: 'מתקשים באנגלית שאינם בתוכנית'.",
            context=ctx)

    class_label = _extract_class(query)
    grade       = None if class_label else _extract_grade(query)

    sr   = skill_rows(df_tosot)
    diff = sr[(sr["מיומנות"] == skill) & (sr["תשובה"].isin(DIFFICULTY_ANSWERS))]

    if class_label:
        g, p = class_label[0], class_label[1]
        diff = diff[(diff["שכבה"] == g) & (diff["מקבילה"] == p)]
        ctx.last_class = class_label; ctx.last_grade = g
        label = f"בכיתה {class_label}"
    elif grade:
        diff = diff[diff["שכבה"] == grade]; ctx.last_grade = grade
        label = f"בשכבה {grade}"
    else:
        label = "בכלל בית הספר"

    unique = diff.drop_duplicates("מס' זהות תלמיד")[
        ["מס' זהות תלמיד", "שם תלמיד", "שכבה", "מקבילה", "מקבץ"]
    ].copy()
    unique["כיתה"] = unique["שכבה"] + unique["מקבילה"]

    # Filter to those without a program
    assigned = set(df_tochnit["מס' זהות תלמיד"].unique())
    no_prog  = unique[~unique["מס' זהות תלמיד"].isin(assigned)].copy()

    if no_prog.empty:
        ctx.last_skill = skill; ctx.last_intent = "skill_difficulty_no_program"
        return _no_results(f"מתקשים ב'{skill}' ללא תוכנית {label}", ctx)

    grade_counts = (
        no_prog.groupby("שכבה").size()
        .reindex(GRADE_ORDER, fill_value=0).reset_index()
    )
    grade_counts.columns = ["שכבה", "מספר"]
    chart = _bar_chart(
        grade_counts["שכבה"].tolist(), grade_counts["מספר"].tolist(),
        [PALETTE["red"]] * len(grade_counts),
        f"מתקשים ב'{skill}' ללא תוכנית – {label}", height=260)

    table = no_prog[["שם תלמיד", "שכבה", "כיתה", "מקבץ"]].sort_values(["שכבה"])
    ctx.last_skill = skill; ctx.last_intent = "skill_difficulty_no_program"
    ctx.last_result_df = no_prog
    return ChatResponse(
        text=(f"נמצאו **{len(no_prog):,}** תלמידים המתקשים ב'{skill}' {label} "
              f"ואינם משויכים לאף תוכנית חינוכית.\n\n"
              f"_מתוך {len(unique):,} מתקשים ב'{skill}' {label}._"),
        table=table, chart=chart, context=ctx)


def _handle_difficult_pupils_without_program(df_status, df_tosot, df_tochnit, query, ctx, **_) -> ChatResponse:
    """Pupils in difficulty מקבצים who have no educational program.

    If a specific מקבץ (e.g. 'מתקשים בכל') is named in the query, filter to
    that group only.  If only the general 'מתקשים' is used, include all three
    DIFFICULT_MIKBATZ groups.
    """
    if df_tochnit is None:
        return ChatResponse(text="אין נתוני תוכניות זמינים.", context=ctx)
    whole_school = _is_whole_school(query)
    base         = None if whole_school else _followup_base(query, ctx, df_status)
    working      = base if base is not None else df_status
    class_label  = _extract_class(query)
    grade        = None if class_label else _extract_grade(query)

    # Determine which מקבץ groups to use
    specific_mkb = _extract_mikbatz(query)
    if specific_mkb and specific_mkb in DIFFICULT_MIKBATZ:
        # User asked specifically about one difficulty group
        mkbatz_filter = {specific_mkb}
        mkb_label = f"מקבץ '{specific_mkb}'"
    else:
        # General "מתקשים" — use all three difficulty groups
        mkbatz_filter = DIFFICULT_MIKBATZ
        mkb_label = "כלל קבוצות הקושי"

    assigned = set(df_tochnit["מס' זהות תלמיד"].unique())
    df = working[~working["ת.ז תלמיד"].isin(assigned)].copy()
    if df_tosot is not None and not df_tosot.empty:
        mkb_map = df_tosot.drop_duplicates("מס' זהות תלמיד").set_index("מס' זהות תלמיד")["מקבץ"]
        df["מקבץ"] = df["ת.ז תלמיד"].map(mkb_map).fillna("לא ידוע")
    else:
        df["מקבץ"] = "לא ידוע"
    df["כיתה"] = df["שכבה"] + df["מקבילה"]
    df = df[df["מקבץ"].isin(mkbatz_filter)]

    if class_label:
        g, p = class_label[0], class_label[1]
        df = df[(df["שכבה"] == g) & (df["מקבילה"] == p)]
        ctx.last_class = class_label; ctx.last_grade = g; label = f"בכיתה {class_label}"
    elif grade:
        df = df[df["שכבה"] == grade]; ctx.last_grade = grade; label = f"בשכבה {grade}"
    else:
        label = "בכלל בית הספר" if base is None else "בתוצאות הקודמות"

    if df.empty:
        return _no_results(f"{mkb_label} ללא תוכנית {label}", ctx)

    gc = df.groupby("שכבה").size().reindex(GRADE_ORDER, fill_value=0).reset_index()
    gc.columns = ["שכבה", "מספר"]
    chart = _bar_chart(gc["שכבה"].tolist(), gc["מספר"].tolist(),
                       [PALETTE["red"]] * len(gc),
                       f"{mkb_label} ללא תוכנית – {label}", height=260)

    table = df[["שם תלמיד", "שכבה", "כיתה", "מקבץ"]].sort_values(["שכבה", "מקבץ"])

    # Build text — only mention sub-groups when showing all groups
    if mkbatz_filter == DIFFICULT_MIKBATZ:
        mkbatzim = ", ".join(sorted(df["מקבץ"].unique()))
        extra = f"\n\nמקבצים: {mkbatzim}."
    else:
        extra = ""  # user asked for a specific group — don't confuse with list

    ctx.last_intent = "difficult_pupils_without_program"; ctx.last_result_df = df
    return ChatResponse(
        text=f"נמצאו **{len(df):,}** תלמידים ({mkb_label}) ללא תוכנית חינוכית {label}.{extra}",
        table=table, chart=chart, context=ctx)


def _handle_class_summary(df_status, df_tosot, query, ctx, **_) -> ChatResponse:
    class_label = _extract_class(query)
    if class_label is None and _is_followup(query):
        class_label = ctx.last_class
    grade = None
    if class_label is None:
        grade = _extract_grade(query)
        if grade is None and _is_followup(query):
            grade = ctx.last_grade

    if class_label:
        g, p = class_label[0], class_label[1]
        sub = df_status[(df_status["שכבה"]==g) & (df_status["מקבילה"]==p)]
        if sub.empty:
            return ChatResponse(text=f"לא נמצאה כיתה {class_label}.", context=ctx)
        total=len(sub); completed=(sub["סטטוס מילוי שאלון"]=="הסתיים").sum()
        partial=(sub["סטטוס מילוי שאלון"]=="חלקי").sum()
        missing=(sub["סטטוס מילוי שאלון"]=="חסר").sum()
        pct=round(completed/total*100,1)
        mkb_info=""
        if df_tosot is not None and not df_tosot.empty:
            pupil_mkb = (df_tosot[(df_tosot["שכבה"]==g) & (df_tosot["מקבילה"]==p)]
                         .drop_duplicates("מס' זהות תלמיד")["מקבץ"].value_counts())
            if not pupil_mkb.empty:
                mkb_info = "\n\n**פרופיל מקבץ:**\n"+"\n".join(f"  - {k}: {v}" for k,v in pupil_mkb.items())
        ctx.last_class=class_label; ctx.last_grade=g
        ctx.last_intent="class_summary"; ctx.last_result_df=sub
        return ChatResponse(
            text=(f"ישנם **{total}** תלמידים בכיתה {class_label}.\n\n"
                  f"- ✅ שאלון הושלם: **{completed}** ({pct}%)\n"
                  f"- ⚠️ שאלון חלקי: **{partial}**\n"
                  f"- ❌ שאלון חסר: **{missing}**"+mkb_info),
            context=ctx)

    elif grade:
        df2 = df_status[df_status["שכבה"]==grade].copy()
        df2["כיתה"] = df2["שכבה"] + df2["מקבילה"]
        grp = df2.groupby("כיתה").agg(
            סהכ=("ת.ז תלמיד","count"),
            הושלם=("סטטוס מילוי שאלון", lambda x:(x=="הסתיים").sum()),
            חלקי=("סטטוס מילוי שאלון",  lambda x:(x=="חלקי").sum()),
            חסר=("סטטוס מילוי שאלון",   lambda x:(x=="חסר").sum()),
        ).reset_index()
        grp["% מילוי"] = (grp["הושלם"]/grp["סהכ"]*100).round(1)
        ctx.last_grade=grade; ctx.last_intent="class_summary"; ctx.last_result_df=df2
        return ChatResponse(
            text=f"ישנם **{grp['סהכ'].sum()}** תלמידים בשכבה {grade}, ב-{len(grp)} כיתות.",
            table=grp, context=ctx)

    else:
        df2=df_status.copy(); df2["כיתה"]=df2["שכבה"]+df2["מקבילה"]
        grp=df2.groupby(["שכבה","כיתה"]).agg(
            סהכ=("ת.ז תלמיד","count"),
            הושלם=("סטטוס מילוי שאלון", lambda x:(x=="הסתיים").sum())).reset_index()
        grp["% מילוי"]=(grp["הושלם"]/grp["סהכ"]*100).round(1)
        grp["_ord"]=pd.Categorical(grp["שכבה"],categories=GRADE_ORDER,ordered=True)
        grp=grp.sort_values(["_ord","כיתה"]).drop(columns="_ord")
        ctx.last_intent="class_summary"
        return ChatResponse(
            text=f"**סיכום כל הכיתות**\n\n{len(grp)} כיתות. שאל על כיתה ספציפית, למשל: 'מה המצב בכיתה ג2?'",
            table=grp[["כיתה","שכבה","סהכ","הושלם","% מילוי"]], context=ctx)


def _handle_unclear(query, ctx, **_) -> ChatResponse:
    return ChatResponse(
        text=("לא הצלחתי להבין את השאלה. 🤔\n\nנסה:\n"
              "- _לכמה תלמידים אין שאלון?_\n- _מה המצב בשכבה ג?_\n"
              "- _מי המתקשים בכל שאינם בתוכנית?_\n"
              "- _מי מתקשה באנגלית?_\n- _כמה תלמידים מתקשים בקשב?_"),
        context=ctx)


# ── Dispatcher ───────────────────────────────────────────────────────────────

HANDLER_MAP = {
    "greeting":                         _handle_greeting,
    "school_summary":                   _handle_school_summary,
    "questionnaire_status_summary":     _handle_questionnaire_status_summary,
    "questionnaire_status_by_grade":    _handle_questionnaire_status_by_grade,
    "pupils_missing_questionnaire":     _handle_pupils_missing_questionnaire,
    "pupils_partial_questionnaire":     _handle_pupils_partial_questionnaire,
    "pupils_by_mikbatz":                _handle_pupils_by_mikbatz,
    "pupils_with_skill_difficulty":     _handle_pupils_with_skill_difficulty,
    "skill_difficulty_no_program":      _handle_skill_difficulty_no_program,
    "pupils_without_program":           _handle_pupils_without_program,
    "difficult_pupils_without_program": _handle_difficult_pupils_without_program,
    "class_summary":                    _handle_class_summary,
    "unclear":                          _handle_unclear,
}


def handle_intent(intent, query, ctx, df_status=None, df_tosot=None, df_tochnit=None) -> ChatResponse:
    handler = HANDLER_MAP.get(intent, _handle_unclear)
    try:
        return handler(query=query, ctx=ctx,
                       df_status=df_status, df_tosot=df_tosot, df_tochnit=df_tochnit)
    except Exception as exc:
        return ChatResponse(text=f"אירעה שגיאה: `{exc}`\nאנא נסה שאלה אחרת.", context=ctx)
