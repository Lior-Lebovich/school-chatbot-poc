"""
intent_classifier.py
────────────────────
Deterministic, keyword-based Hebrew intent classifier.
No LLM, no embeddings, no external calls.

Each intent is defined by:
  - required: ALL of these token-sets must match (AND logic)
  - any_of:   at least one token-set must match (OR logic within each group)
  - exclude:  if any token in this list matches → skip this intent

Tokens are matched as substrings of the normalised query.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ── Supported intents ──────────────────────────────────────────────────────────
INTENTS = [
    "greeting",
    "school_summary",
    "questionnaire_status_summary",
    "questionnaire_status_by_grade",
    "pupils_missing_questionnaire",
    "pupils_partial_questionnaire",
    "pupils_by_mikbatz",
    "pupils_with_skill_difficulty",
    "pupils_without_program",
    "difficult_pupils_without_program",
    "class_summary",
    "unclear",
]


@dataclass
class IntentRule:
    intent: str
    # Every sub-list is an OR group; ALL groups must fire (AND across groups)
    required: list[list[str]] = field(default_factory=list)
    # Presence of ANY token here blocks this intent
    exclude: list[str] = field(default_factory=list)
    # Priority – higher wins when two intents match
    priority: int = 0


# ── Rules (highest priority listed first) ─────────────────────────────────────
RULES: list[IntentRule] = [

    # ── Greeting ──────────────────────────────────────────────────────────────
    IntentRule(
        intent="greeting",
        required=[[
            "שלום", "היי", "בוקר טוב", "ערב טוב", "מה שלום",
            "מה נשמע", "אהלן", "שלומך",
        ]],
        priority=90,
    ),

    # ── School summary ────────────────────────────────────────────────────────
    IntentRule(
        intent="school_summary",
        required=[[
            "סיכום", "תמצית", "תמונת מצב", "סקירה",
            "מה המצב", "סה\"כ", "סהכ", "בית הספר", "כללי",
        ]],
        exclude=["שכבה", "כיתה", "שכבת", "כיתת", "בשכבה", "בכיתה",
                 "מתקדמים", "תקינים", "מתקשים", "מקבץ",
                 "כאלה",        # "כמה כאלה בכלל בית הספר" → pupils_by_mikbatz
                 "הכוונה",      # clarification answer → pupils_by_mikbatz
                 "לכלל תלמידי", # "לכלל תלמידי בית הספר" → pending intent
                 ],
        priority=50,
    ),

    # ── Class/grade summary — only when grade or class entity is present ──────
    # Split into two rules so mikbatz queries that also mention a class
    # (e.g. "כמה תלמידים בכיתה ג2 מתקשים בכל?") fall through to pupils_by_mikbatz.
    IntentRule(
        intent="class_summary",
        required=[
            # Must mention a grade/class location word
            ["שכבה", "כיתה", "שכבת", "כיתת", "בשכבה", "בכיתה"],
            # AND a count/summary keyword (not just any question)
            ["כמה", "מה המצב", "סיכום", "כמה יש", "מה יש", "מצב"],
        ],
        # Exclude when a mikbatz term is also present → pupils_by_mikbatz handles it
        exclude=["מתקדמים", "תקינים", "מתקשים", "מקבץ", "מקבצים"],
        priority=65,
    ),

    # ── Pupils by mikbatz ─────────────────────────────────────────────────────
    # Priority 72: beats class_summary so "כמה בכיתה ג2 מתקשים בכל" goes here
    IntentRule(
        intent="pupils_by_mikbatz",
        required=[[
            "מקבץ", "מקבצים", "מתקדמים", "תקינים", "מתקשים",
            "קבוצת", "קבוצות", "פרופיל",
            "כמה כאלה",   # "כמה כאלה בכלל בית הספר?" — same condition, wider scope
            "הכוונה",     # clarification answer: "הכוונה לכלל תלמידי בית הספר"
        ]],
        exclude=["תוכנית", "ללא תוכנית", "תוכניות",
                 "שאלון", "מילוי"],
        priority=72,
    ),

    # ── Questionnaire status – by grade ───────────────────────────────────────
    IntentRule(
        intent="questionnaire_status_by_grade",
        required=[
            ["שאלון", "מילוי", "שאלונים"],
            ["שכבה", "כיתה", "שכבות", "כיתות", "לפי שכבה", "לפי כיתה"],
        ],
        priority=70,
    ),

    # ── Questionnaire status – general ────────────────────────────────────────
    IntentRule(
        intent="questionnaire_status_summary",
        required=[[
            "שאלון", "מילוי", "שאלונים", "סטטוס",
        ]],
        exclude=["שכבה", "כיתה", "חסר", "חסרים", "חלקי", "חלקיים"],
        priority=60,
    ),

    # ── Missing questionnaire ─────────────────────────────────────────────────
    IntentRule(
        intent="pupils_missing_questionnaire",
        required=[
            ["שאלון", "מילוי", "שאלונים"],
            ["חסר", "חסרים", "לא מילא", "לא מילאו", "לא הגיש", "לא הגישו",
             "ללא שאלון", "לא התחיל", "לא התחילו", "אין שאלון", "אין להם שאלון"],
        ],
        priority=75,
    ),

    # ── Partial questionnaire ─────────────────────────────────────────────────
    IntentRule(
        intent="pupils_partial_questionnaire",
        required=[
            ["שאלון", "מילוי", "שאלונים"],
            ["חלקי", "חלקיים", "לא סיים", "לא סיימו", "התחיל", "התחילו",
             "שאלון חלקי"],
        ],
        priority=75,
    ),

    # ── Skill difficulty (any מיומנות, not just math) ────────────────────────
    # Triggers on difficulty keywords combined with ANY known skill domain or
    # the generic "מיומנות" word. Priority 80 — above mikbatz (72) so
    # "מתקשים במתמטיקה" doesn't accidentally fire pupils_by_mikbatz.
    IntentRule(
        intent="pupils_with_skill_difficulty",
        required=[
            # Must mention a difficulty concept
            ["מתקשה", "מתקשים", "קשיים", "קושי", "מתקשות", "מתקשת"],
            # AND a skill domain OR generic skill word
            [
                "מתמטיקה", "מתמט", "חשבון",          # math
                "אנגלית", "אנגלי",                    # English
                "שפת אם", "קריאה", "כתיבה",           # Hebrew literacy
                "היבטים חברתיים", "חברתי", "חברתית",  # social
                "היבטים רגשיים", "רגשי", "רגשית",     # emotional
                "היבטים התנהגותיים", "התנהגות",        # behavioural
                "מוטיבציה", "הרגלי למידה",             # motivation / study habits
                "קשב", "פעלתנות",                     # attention / hyperactivity
                "חושי תנועתי", "חושי",                 # sensorimotor
                "מיומנות", "מיומנויות",               # generic
            ],
        ],
        exclude=["תוכנית", "ללא תוכנית", "מקבץ"],
        priority=80,
    ),

    # ── Difficult pupils without program ─────────────────────────────────────
    IntentRule(
        intent="difficult_pupils_without_program",
        required=[
            ["מתקשים", "קשיים", "קושי"],
            ["ללא תוכנית", "אין תוכנית", "לא משויך", "לא משויכים",
             "ללא שיוך", "לא שויך", "אין להם תוכנית", "שאין להם",
             "שאינם בתוכנית", "שאינם משויכים",
             "ואינם בתוכנית", "ואין להם תוכנית",
             "אינם בתוכנית", "אינם משויכים"],
        ],
        priority=85,
    ),

    # ── All pupils without any program ───────────────────────────────────────
    IntentRule(
        intent="pupils_without_program",
        required=[[
            "ללא תוכנית", "אין תוכנית", "לא משויך", "לא משויכים",
            "ללא שיוך", "לא שויך", "לא קיבל", "לא קיבלו",
            # Follow-up phrasing: "כמה מתוכם משויכים לתוכנית"
            "משויכים לתוכנית", "משויך לתוכנית", "יש להם תוכנית",
            "עם תוכנית חינוכית", "שיש להם תוכנית",
        ]],
        exclude=["מתקשים", "קשיים", "קושי"],
        priority=80,
    ),
]

# Sort rules by priority descending so highest-priority intent wins
RULES.sort(key=lambda r: r.priority, reverse=True)


# ── Normalisation ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    text = text.strip().lower()
    text = re.sub(r'[״״""\'`]', '"', text)   # unify quote chars
    text = re.sub(r'[,;:.!?–—]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _token_match(norm_query: str, tokens: list[str]) -> bool:
    """True if any token in the list appears as a substring in the query."""
    return any(t in norm_query for t in tokens)


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(query: str) -> str:
    """Return the best-matching intent name for a Hebrew query string."""
    norm = _normalise(query)

    for rule in RULES:
        # Check exclusions first
        if rule.exclude and _token_match(norm, rule.exclude):
            continue
        # All required groups must have at least one match
        if all(_token_match(norm, group) for group in rule.required):
            return rule.intent

    return "unclear"


def classify_with_context(query: str, context: "ChatContext") -> str:
    """Classify with awareness of anaphoric follow-up phrases and pending clarifications.

    Priority order:
    1. If a pending clarification is waiting (context.pending_intent is set),
       and the query is a scope-confirmation answer, route to pending_intent.
    2. If query has explicit classifiable content, classify normally.
    3. If query contains ONLY anaphora tokens with no other signal,
       repeat the previous intent so the handler can filter by last_result_df.
    4. Otherwise return 'unclear'.
    """
    # ── Priority 1: resolve pending clarification ──────────────────────────────
    # Scope-confirmation tokens: user is answering our clarification question.
    _SCOPE_CONFIRM_TOKENS = [
        "לכלל", "כלל בית הספר", "כל בית הספר", "בית הספר כולו",
        "הכוונה לכלל", "לכלל תלמידי", "בכלל בית הספר",
    ]
    if (context.pending_intent
            and any(t in query for t in _SCOPE_CONFIRM_TOKENS)):
        return context.pending_intent   # handler reads pending_mikbatz / pending_filters

    # ── Priority 2: normal classification ─────────────────────────────────────
    normal_intent = classify(query)
    if normal_intent != "unclear":
        return normal_intent

    # ── Priority 3: bare follow-up with anaphora token ────────────────────────
    norm = _normalise(query)
    FOLLOWUP_TOKENS = ["מתוכם", "מהם", "אותם תלמידים",
                       "השכבה הזו", "הכיתה הזו",
                       "מתוך אלה", "מתוך אותם"]
    if any(t in norm for t in FOLLOWUP_TOKENS) and context.last_intent:
        return context.last_intent

    return "unclear"


# ── Convenience: human-readable intent label ──────────────────────────────────

INTENT_LABELS: dict[str, str] = {
    "greeting":                        "ברכה",
    "school_summary":                  "סיכום בית הספר",
    "questionnaire_status_summary":    "סטטוס שאלון",
    "questionnaire_status_by_grade":   "סטטוס שאלון לפי שכבה",
    "pupils_missing_questionnaire":    "תלמידים ללא שאלון",
    "pupils_partial_questionnaire":    "תלמידים עם שאלון חלקי",
    "pupils_by_mikbatz":               "תלמידים לפי מקבץ",
    "pupils_with_skill_difficulty":    "מתקשים במיומנות",
    "pupils_without_program":          "ללא תוכנית חינוכית",
    "difficult_pupils_without_program":"מתקשים ללא תוכנית",
    "class_summary":                   "סיכום כיתה",
    "unclear":                         "לא ברור",
}
