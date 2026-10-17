import os
import re
import sqlite3
import html
import textwrap

import joblib
import pandas as pd
import streamlit as st

from langdetect import detect

from preprocess import clean_text
from rag import retrieve
from semantic_sentiment import semantic_predict


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Political Sentiment Analyzer",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sentiment_model.pkl",
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.pkl",
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "predictions.db",
)

CSS_PATH = os.path.join(
    BASE_DIR,
    "style.css",
)


# ============================================================
# HTML RENDERING
# ============================================================

def render_html(markup):

    cleaned_markup = textwrap.dedent(
        markup
    ).strip()

    if hasattr(st, "html"):
        st.html(cleaned_markup)
    else:
        st.markdown(
            re.sub(r">\s+<", "><", cleaned_markup),
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD CSS
# ============================================================

if os.path.exists(CSS_PATH):

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8",
    ) as css_file:

        css_content = css_file.read()

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )

else:

    st.error(
        "style.css not found. "
        "Make sure style.css is in the same folder as app.py."
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            f"Model file not found:\n{MODEL_PATH}"
        )

        st.stop()

    if not os.path.exists(
        VECTORIZER_PATH
    ):

        st.error(
            f"Vectorizer file not found:\n{VECTORIZER_PATH}"
        )

        st.stop()

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return (
        model,
        vectorizer,
    )


model, vectorizer = load_model()


# ============================================================
# DATABASE
# ============================================================

def init_database():

    os.makedirs(
        os.path.dirname(
            DATABASE_PATH
        ),
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            input_text TEXT,

            language TEXT,

            translated_text TEXT,

            ml_prediction TEXT,

            semantic_prediction TEXT,

            final_sentiment TEXT,

            confidence REAL,

            semantic_confidence REAL,

            decision_method TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    columns = {
        row[1]
        for row in cursor.fetchall()
    }

    if "translated_text" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN translated_text TEXT
            """
        )

    if "semantic_prediction" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN semantic_prediction TEXT
            """
        )

    if "semantic_confidence" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN semantic_confidence REAL
            """
        )

    connection.commit()

    connection.close()


init_database()


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    input_text,
    language,
    translated_text,
    ml_prediction,
    semantic_prediction,
    final_sentiment,
    confidence,
    semantic_confidence,
    decision_method,
):

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )

    connection.execute(
        """
        INSERT INTO predictions (

            input_text,
            language,
            translated_text,
            ml_prediction,
            semantic_prediction,
            final_sentiment,
            confidence,
            semantic_confidence,
            decision_method

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            input_text,
            language,
            translated_text,
            ml_prediction,
            semantic_prediction,
            final_sentiment,
            confidence,
            semantic_confidence,
            decision_method,
        ),
    )

    connection.commit()

    connection.close()


# ============================================================
# CUSTOM HTML TABLE
# ============================================================

def render_html_table(
    dataframe,
    table_class="custom-table",
    escape=False,
):

    if (
        dataframe is None
        or dataframe.empty
    ):

        st.info(
            "No data available."
        )

        return

    table_html = dataframe.to_html(
        index=False,
        escape=escape,
        classes=table_class,
        border=0,
    )

    render_html(
        f"""
        <div class="table-wrapper">
            {table_html}
        </div>
        """
    )


# ============================================================
# SENTIMENT RULES
# ============================================================

POSITIVE_RULES = {

    "good",
    "great",
    "excellent",
    "successful",
    "success",
    "effective",
    "improving",
    "improved",
    "progress",
    "welcomed",
    "welcome",
    "appreciated",
    "highly_appreciated",
    "great_job",
    "doing_great_job",
    "doing_good_work",
    "excellent_work",
    "excellent_job",
    "strong_performance",
    "promising",
    "benefit",
    "benefits",
    "meaningful",
    "achievement",
    "achievements",
    "better",
    "growth",
    "prosperity",
    "step_in_the_right_direction",

}


NEGATIVE_RULES = {

    "bad",
    "poor",
    "failed",
    "failure",
    "unsuccessful",
    "ineffective",
    "disappointing",
    "disappointed",
    "very_disappointed",
    "frustrated",
    "extremely_frustrated",
    "concerned",
    "concern",
    "worried",
    "worry",
    "inflation",
    "rising_prices",
    "price_rise",
    "price_increase",
    "high_inflation",
    "unemployment",
    "unemployed",
    "jobs_hard_find",
    "lack_jobs",
    "lack_employment",
    "failed_deliver",
    "failed_create_jobs",
    "poor_governance",
    "disaster",
    "terrible",
    "problem",
    "problems",
    "serious",
    "struggling",
    "lack",
    "little",
    "hard",
    "difficult",
    "unresolved",
    "unfulfilled",
    "burden",
    "crisis",
    "worse",
    "uneven",
    "unevenly",
    "complain",
    "complaints",
    "questionable",
    "uncertain",

}


# ============================================================
# PHRASE MATCHING
# ============================================================

def _phrase_matches(
    text,
    phrases,
):

    matches = []

    normalized = re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()

    for phrase in phrases:

        search_phrase = phrase.replace(
            "_",
            " ",
        )

        pattern = (
            r"(?<!\w)"
            + re.escape(search_phrase)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized,
        ):

            matches.append(
                search_phrase
            )

    return list(
        dict.fromkeys(matches)
    )


def detect_sentiment_evidence(
    text
):

    positive = _phrase_matches(
        text,
        POSITIVE_RULES,
    )

    negative = _phrase_matches(
        text,
        NEGATIVE_RULES,
    )

    has_mixed = bool(
        positive
        and negative
    )

    return (
        positive,
        negative,
        has_mixed,
    )


# ============================================================
# MIXED CONTEXT
# ============================================================

def detect_mixed_context(
    text,
    positive_matches,
    negative_matches,
):

    if (
        positive_matches
        and negative_matches
    ):

        return True

    connectors = [

        "even though",
        "nevertheless",
        "nonetheless",
        "although",
        "however",
        "despite",
        "whereas",
        "though",
        "while",
        "but",
        "yet",

    ]

    text_lower = text.lower()

    return any(

        re.search(
            r"(?<!\w)"
            + re.escape(connector)
            + r"(?!\w)",
            text_lower,
        )

        for connector in connectors

    ) and bool(
        positive_matches
        or negative_matches
    )


# ============================================================
# CONTRAST ANALYSIS
# ============================================================

def analyze_contrast_structure(
    text
):

    text_lower = re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()

    connectors = [

        "even though",
        "nevertheless",
        "nonetheless",
        "although",
        "however",
        "whereas",
        "despite",
        "though",
        "while",
        "but",
        "yet",

    ]

    connector_match = None
    connector_position = None

    for connector in connectors:

        match = re.search(
            r"(?<!\w)"
            + re.escape(connector)
            + r"(?!\w)",
            text_lower,
        )

        if match:

            if (
                connector_position is None
                or match.start()
                < connector_position
            ):

                connector_position = (
                    match.start()
                )

                connector_match = (
                    match.group(0)
                )

    if connector_match is None:

        if re.search(
            r"\bon the one hand\b.*"
            r"\bon the other hand\b",
            text_lower,
            re.DOTALL,
        ):

            return {

                "detected": True,
                "direction": "MIXED",
                "connector":
                    "contrast structure",

            }

        if re.search(
            r"\bneither\b.*\bnor\b",
            text_lower,
            re.DOTALL,
        ):

            return {

                "detected": True,
                "direction": "MIXED",
                "connector":
                    "neither/nor",

            }

        return {

            "detected": False,
            "direction": None,
            "connector": None,

        }

    first_clause = text_lower[
        :connector_position
    ].strip(" ,;:")

    second_clause = text_lower[
        connector_position
        + len(connector_match):
    ].strip(" ,;:")

    positive_terms = {

        "good",
        "great",
        "excellent",
        "successful",
        "success",
        "effective",
        "improving",
        "improved",
        "progress",
        "benefit",
        "benefits",
        "welcomed",
        "welcome",
        "appreciated",
        "promising",
        "strong",
        "positive",
        "reassuring",
        "meaningful",
        "confident",
        "advantage",
        "advantages",
        "achievement",
        "achievements",
        "better",
        "growth",
        "prosperity",
        "sensible",
        "reasonable",

    }

    negative_terms = {

        "bad",
        "poor",
        "failed",
        "failure",
        "unsuccessful",
        "ineffective",
        "disappointing",
        "disappointed",
        "frustrated",
        "concerned",
        "concern",
        "worried",
        "inflation",
        "unemployment",
        "unemployed",
        "problem",
        "problems",
        "serious",
        "struggling",
        "lack",
        "little",
        "hard",
        "difficult",
        "terrible",
        "disastrous",
        "exaggerated",
        "unresolved",
        "without",
        "complain",
        "complaints",
        "burden",
        "crisis",
        "worse",
        "unevenly",
        "uneven",
        "unfulfilled",
        "questionable",
        "uncertain",
        "unrealistic",
        "shortcomings",
        "criticism",
        "decline",
        "declining",
        "unaffordable",
        "protesting",
        "unsupported",
        "premature",

    }

    def clause_score(
        clause
    ):

        words = re.findall(
            r"\b[\w']+\b",
            clause,
        )

        positive = sum(
            word in positive_terms
            for word in words
        )

        negative = sum(
            word in negative_terms
            for word in words
        )

        negation_patterns = [

            r"\bnot\s+(?:good|great|successful|effective|positive|improving|better|meaningful|substantial)\b",

            r"\bno\s+(?:improvement|benefit|progress|success|evidence)\b",

            r"\bwithout\s+(?:meaningful|substantial|clear|real)\b",

            r"\bnot\s+without\b",

            r"\byet\s+to\b",

            r"\bfailed\s+to\b",

            r"\bfar\s+from\s+(?:certain|successful|effective|satisfactory)\b",

            r"\bnot\s+(?:materialize|materialized)\b",

            r"\bremain\s+(?:unresolved|questionable|uncertain)\b",

            r"\blittle\s+evidence\b",

            r"\blittle\s+improvement\b",

            r"\blittle\s+benefit\b",

            r"\blittle\s+progress\b",

            r"\blittle\s+comfort\b",

            r"\bnot\s+translated\s+into\b",

            r"\bnot\s+been\s+translated\s+into\b",

            r"\buncertain\s+funding\b",

            r"\bunrealistic\s+timeline\b",

        ]

        for pattern in negation_patterns:

            if re.search(
                pattern,
                clause,
            ):

                negative += 2

        return (
            positive
            - negative
        )

    first_score = clause_score(
        first_clause
    )

    second_score = clause_score(
        second_clause
    )

    if second_score <= -1:

        direction = "NEGATIVE"

    elif second_score >= 1:

        direction = "POSITIVE"

    elif (
        first_score > 0
        and second_score < 0
    ):

        direction = "NEGATIVE"

    elif (
        first_score < 0
        and second_score > 0
    ):

        direction = "POSITIVE"

    else:

        direction = "MIXED"

    if re.search(
        r"\bneither\b.*\bnor\b",
        text_lower,
        re.DOTALL,
    ):

        direction = "MIXED"

    if re.search(
        r"\bnot\b.*\bbut\b",
        text_lower,
        re.DOTALL,
    ):

        if (
            first_score < 0
            and second_score > 0
        ):

            direction = "POSITIVE"

        elif (
            first_score > 0
            and second_score < 0
        ):

            direction = "NEGATIVE"

    return {

        "detected": True,

        "direction":
            direction,

        "connector":
            connector_match,

        "first_clause":
            first_clause,

        "second_clause":
            second_clause,

        "first_score":
            first_score,

        "second_score":
            second_score,

    }


# ============================================================
# SARCASM
# ============================================================

def detect_sarcasm_context(
    text
):

    text_lower = text.lower()

    sarcasm_patterns = [

        "apparently everything",
        "apparently unemployment",
        "oh wonderful",
        "wonderful, another",
        "what an impressive achievement",
        "if this is what",
        "so-called success",
        "so-called successful",
        "supposedly",
        "conveniently avoiding",
        "conveniently ignores",
        "conveniently ignoring",
        "apparently",

    ]

    negative_context_words = [

        "protesting",
        "unemployment",
        "inflation",
        "poor services",
        "poor public services",
        "struggling",
        "rising prices",
        "unaffordable",
        "failure",
        "failed",
        "problem",
        "problems",
        "crisis",
        "little evidence",
        "little improvement",
        "no improvement",
        "not achieved",
        "not delivered",
        "remain unresolved",
        "housing less accessible",

    ]

    sarcasm_detected = any(
        pattern in text_lower
        for pattern in sarcasm_patterns
    )

    negative_context = any(
        phrase in text_lower
        for phrase in negative_context_words
    )

    return (
        sarcasm_detected
        and negative_context
    )


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(
    text
):

    try:

        return detect(
            text
        )

    except Exception:

        return "en"


# ============================================================
# LANGUAGE VALIDATION (ENGLISH ONLY)
# ============================================================

def validate_english_input(
    text,
    language,
):

    if language != "en":
        render_html(
            """
            <div class="result-card" style="border-left: 5px solid var(--purple); text-align: center; padding: 24px;">
                <div class="result-topline" style="color: var(--purple);">LANGUAGE NOTICE</div>
                <div style="font-size: 18px; font-weight: 800; color: var(--text-dark); margin-bottom: 8px;">
                    ⚠️ English Only
                </div>
                <div style="font-size: 14px; color: var(--text); line-height: 1.6; max-width: 600px; margin: 0 auto;">
                    Please enter your political statement in English.<br>
                    <span style="color: var(--text-muted); font-size: 13px;">
                        This sentiment analyzer currently supports English text only.
                    </span>
                </div>
            </div>
            """
        )
        st.stop()

    return text


# ============================================================
# HYBRID SENTIMENT ENGINE
# ============================================================

def hybrid_sentiment_decision(

    ml_prediction,
    ml_confidence,

    semantic_prediction,
    semantic_confidence,

    positive_matches,
    negative_matches,

    mixed_context,

    contrast_structure,

    contrast_analysis=None,

    sarcasm_detected=False,

):

    ml_prediction = str(
        ml_prediction
    ).upper()

    semantic_prediction = str(
        semantic_prediction
    ).upper()

    has_positive = bool(
        positive_matches
    )

    has_negative = bool(
        negative_matches
    )

    # 1. Contrast has highest priority.
    if (
        contrast_structure
        and contrast_analysis
        and contrast_analysis.get(
            "detected"
        )
    ):

        direction = (
            contrast_analysis.get(
                "direction"
            )
        )

        if direction == "NEGATIVE":

            return {

                "final_sentiment":
                    "NEGATIVE",

                "decision_method":
                    "Contrast-Aware Negative Decision",

            }

        if direction == "POSITIVE":

            return {

                "final_sentiment":
                    "POSITIVE",

                "decision_method":
                    "Contrast-Aware Positive Decision",

            }

        if direction == "MIXED":

            return {

                "final_sentiment":
                    "MIXED",

                "decision_method":
                    "Contrast-Aware Mixed Decision",

            }

    # 2. Sarcasm.
    if sarcasm_detected:

        return {

            "final_sentiment":
                "NEGATIVE",

            "decision_method":
                "Sarcasm-Aware Negative Decision",

        }

    # 3. Explicit mixed evidence.
    if (
        has_positive
        and has_negative
    ):

        return {

            "final_sentiment":
                "MIXED",

            "decision_method":
                "Mixed Evidence Detection",

        }

    # 4. Mixed context.
    if mixed_context:

        return {

            "final_sentiment":
                "MIXED",

            "decision_method":
                "Mixed Context Detection",

        }

    # 5. Model agreement.
    if (
        ml_prediction
        == semantic_prediction
    ):

        return {

            "final_sentiment":
                ml_prediction,

            "decision_method":
                "ML + Transformer Agreement",

        }

    # 6. Strong transformer.
    if semantic_confidence >= 75:

        return {

            "final_sentiment":
                semantic_prediction,

            "decision_method":
                "Transformer Semantic Model",

        }

    # 7. Positive evidence.
    if (
        has_positive
        and not has_negative
    ):

        return {

            "final_sentiment":
                "POSITIVE",

            "decision_method":
                "Semantic + Positive Evidence",

        }

    # 8. Negative evidence.
    if (
        has_negative
        and not has_positive
    ):

        return {

            "final_sentiment":
                "NEGATIVE",

            "decision_method":
                "Semantic + Negative Evidence",

        }

    # 9. Confidence fallback.
    if (
        ml_confidence
        >= semantic_confidence
    ):

        return {

            "final_sentiment":
                ml_prediction,

            "decision_method":
                "TF-IDF ML Model",

        }

    return {

        "final_sentiment":
            semantic_prediction,

        "decision_method":
            "Transformer Semantic Model",

    }


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_sentiment(
    text
):

    cleaned_text = clean_text(
        text
    )

    vector = vectorizer.transform(
        [cleaned_text]
    )

    ml_prediction = str(
        model.predict(
            vector
        )[0]
    ).upper()

    probabilities = (
        model.predict_proba(
            vector
        )[0]
    )

    probability_dict = {

        str(cls).upper():
            float(prob)

        for cls, prob in zip(
            model.classes_,
            probabilities,
        )

    }

    ml_confidence = (
        max(probabilities)
        * 100
    )

    (
        positive_matches,
        negative_matches,
        has_mixed,
    ) = detect_sentiment_evidence(
        cleaned_text
    )

    mixed_context = (
        detect_mixed_context(
            cleaned_text,
            positive_matches,
            negative_matches,
        )
    )

    contrast_analysis = (
        analyze_contrast_structure(
            cleaned_text
        )
    )

    contrast_structure = (
        contrast_analysis[
            "detected"
        ]
    )

    sarcasm_detected = (
        detect_sarcasm_context(
            cleaned_text
        )
    )

    semantic_result = (
        semantic_predict(
            text
        )
    )

    semantic_prediction = str(
        semantic_result[
            "label"
        ]
    ).upper()

    semantic_confidence = float(
        semantic_result[
            "confidence"
        ]
    )

    decision = (
        hybrid_sentiment_decision(

            ml_prediction=
                ml_prediction,

            ml_confidence=
                ml_confidence,

            semantic_prediction=
                semantic_prediction,

            semantic_confidence=
                semantic_confidence,

            positive_matches=
                positive_matches,

            negative_matches=
                negative_matches,

            mixed_context=
                mixed_context,

            contrast_structure=
                contrast_structure,

            contrast_analysis=
                contrast_analysis,

            sarcasm_detected=
                sarcasm_detected,

        )
    )

    return {

        "final_sentiment":
            decision[
                "final_sentiment"
            ],

        "ml_prediction":
            ml_prediction,

        "ml_confidence":
            ml_confidence,

        "semantic_prediction":
            semantic_prediction,

        "semantic_confidence":
            semantic_confidence,

        "probabilities":
            probability_dict,

        "positive_matches":
            positive_matches,

        "negative_matches":
            negative_matches,

        "has_mixed":
            has_mixed,

        "mixed_context":
            mixed_context,

        "contrast_structure":
            contrast_structure,

        "contrast_direction":
            contrast_analysis.get(
                "direction"
            ),

        "contrast_connector":
            contrast_analysis.get(
                "connector"
            ),

        "contrast_first_clause":
            contrast_analysis.get(
                "first_clause",
                "",
            ),

        "contrast_second_clause":
            contrast_analysis.get(
                "second_clause",
                "",
            ),

        "first_clause_score":
            contrast_analysis.get(
                "first_score",
                0,
            ),

        "second_clause_score":
            contrast_analysis.get(
                "second_score",
                0,
            ),

        "sarcasm_detected":
            sarcasm_detected,

        "decision_method":
            decision[
                "decision_method"
            ],

    }


# ============================================================
# AI INSIGHT
# ============================================================

def generate_ai_insight(
    result
):

    final = result[
        "final_sentiment"
    ]

    ml = result[
        "ml_prediction"
    ]

    ml_conf = result[
        "ml_confidence"
    ]

    semantic = result[
        "semantic_prediction"
    ]

    semantic_conf = result[
        "semantic_confidence"
    ]

    method = result[
        "decision_method"
    ]

    connector = result.get(
        "contrast_connector"
    )

    if (
        final == "NEGATIVE"
        and result.get(
            "contrast_structure"
        )
    ):

        connector_text = (

            f"'{connector}'"

            if connector

            else
            "a contrast connector"

        )

        return (

            "The statement contains a "
            "contrast structure using "
            f"{connector_text}. "

            "Although the first clause may "
            "contain positive information, "
            "the post-contrast clause contains "
            "the stronger negative signal. "

            "The second clause therefore has "
            "greater influence on the final "
            "decision. "

            f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Final decision: {method}."

        )

    if result.get(
        "sarcasm_detected"
    ):

        return (

            "The statement contains language "
            "consistent with sarcasm or ironic "
            "criticism. Positive surface wording "
            "is interpreted together with the "
            "surrounding negative context. "

            f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Final decision: {method}."

        )

    if final == "MIXED":

        positive = (

            ", ".join(
                result[
                    "positive_matches"
                ]
            )

            or
            "contextual positive signals"

        )

        negative = (

            ", ".join(
                result[
                    "negative_matches"
                ]
            )

            or
            "contextual negative signals"

        )

        return (

            "The statement contains "
            "contrasting political sentiment. "

            f"Positive evidence: {positive}. "

            f"Negative evidence: {negative}. "

            f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Final decision: {method}."

        )

    if final == "POSITIVE":

        return (

            "The statement contains "
            "predominantly positive political "
            "sentiment. "

            f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Final decision: {method}."

        )

    if final == "NEGATIVE":

        return (

            "The statement contains "
            "predominantly negative political "
            "sentiment. "

            f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Final decision: {method}."

        )

    return (

        "The statement appears mainly "
        "neutral or informational. "

        f"TF-IDF predicted {ml} "
        f"({ml_conf:.2f}%), while the "
        f"Transformer predicted {semantic} "
        f"({semantic_conf:.2f}%). "

        f"Final decision: {method}."

    )


# ============================================================
# DECISION EXPLANATION
# ============================================================

def generate_decision_explanation(
    result
):

    final = result[
        "final_sentiment"
    ]

    ml = result[
        "ml_prediction"
    ]

    ml_conf = result[
        "ml_confidence"
    ]

    semantic = result[
        "semantic_prediction"
    ]

    semantic_conf = result[
        "semantic_confidence"
    ]

    method = result[
        "decision_method"
    ]

    if final == "MIXED":

        details = []

        if result[
            "positive_matches"
        ]:

            details.append(
                "positive evidence: "
                + ", ".join(
                    result[
                        "positive_matches"
                    ]
                )
            )

        if result[
            "negative_matches"
        ]:

            details.append(
                "negative evidence: "
                + ", ".join(
                    result[
                        "negative_matches"
                    ]
                )
            )

        if result.get(
            "contrast_direction"
        ):

            detail = (

                "contrast direction: "
                + result[
                    "contrast_direction"
                ]

            )

            if result.get(
                "contrast_connector"
            ):

                detail += (

                    " using '"
                    + result[
                        "contrast_connector"
                    ]
                    + "'"

                )

            details.append(
                detail
            )

        if not details:

            details.append(
                "contrasting linguistic structure"
            )

        return (

            "The statement contains "
            "conflicting or contrasting "
            "sentiment. "

            + ". ".join(
                details
            )

            + ". "

            + f"TF-IDF predicted {ml} "
            f"({ml_conf:.2f}%), while the "
            f"Transformer predicted {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Decision: {method}."

        )

    if (
        final == "NEGATIVE"
        and result.get(
            "contrast_structure"
        )
    ):

        return (

            "A contrast structure was detected. "
            "The second clause contains the "
            "stronger negative signal, so it "
            "receives greater weight in the "
            "final decision. "

            f"TF-IDF: {ml} "
            f"({ml_conf:.2f}%). "

            f"Transformer: {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Decision: {method}."

        )

    if result.get(
        "sarcasm_detected"
    ):

        return (

            "The statement contains sarcastic "
            "or ironic wording combined with "
            "negative context. The system "
            "therefore interprets the underlying "
            "sentiment as negative. "

            f"TF-IDF: {ml} "
            f"({ml_conf:.2f}%). "

            f"Transformer: {semantic} "
            f"({semantic_conf:.2f}%). "

            f"Decision: {method}."

        )

    return (

        f"The final result is {final}. "

        f"TF-IDF predicted {ml} with "
        f"{ml_conf:.2f}% confidence. "

        f"The Transformer predicted "
        f"{semantic} with "
        f"{semantic_conf:.2f}% confidence. "

        f"Decision method: {method}."

    )


# ============================================================
# SIDEBAR (FULLY INDENTED INSIDE WITH ST.SIDEBAR)
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">
                🗳️
            </div>
            <div class="brand-copy">
                <div class="brand-title">
                    Political Sentiment
                </div>
                <div class="brand-subtitle">
                    AI-Powered Analysis
                </div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-section-title">
            MAIN MENU
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "🔍 Analysis",
            "📊 CSV Analysis",
            "📜 Prediction History",
        ],
        label_visibility="collapsed",
    )

    render_html(
        """
        <div class="sidebar-section-title">
            SYSTEM
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-system-card">
            <div class="system-chip">
                <span class="system-chip-tag">PRIMARY MODEL</span>
                <span class="system-chip-val">TF-IDF + Logistic Regression</span>
            </div>
            <div class="system-chip">
                <span class="system-chip-tag">SEMANTIC MODEL</span>
                <span class="system-chip-val">Twitter-RoBERTa</span>
            </div>
            <div class="system-chip">
                <span class="system-chip-tag">DECISION ENGINE</span>
                <span class="system-chip-val">Contrast-Aware Hybrid</span>
            </div>
            <div class="system-chip">
                <span class="system-chip-tag">CONTEXT ENGINE</span>
                <span class="system-chip-val">SentenceTransformer + FAISS</span>
            </div>
        </div>
        """
    )


# ============================================================
# MAIN DASHBOARD HEADER
# ============================================================

render_html(
    """
    <div class="dashboard-header">
        <div class="dashboard-header-content">
            <div class="page-title">
                🗳️ Political Sentiment Analyzer
            </div>
            <div class="dashboard-subtitle">
                Analyze political statements using
                Machine Learning, Transformer NLP
                and Retrieval-Augmented Generation.
            </div>
        </div>
    </div>
    """
)


# ============================================================
# ANALYSIS PAGE
# ============================================================

if page == "🔍 Analysis":

    render_html(
        """
        <div class="page-section">
            <div class="section-heading">
                <span class="section-icon">
                    ✎
                </span>
                <div>
                    <div class="section-title">
                        Analyze Political Statement
                    </div>
                    <div class="section-description">
                        Enter a political statement and
                        evaluate it using the hybrid NLP
                        sentiment pipeline.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    user_text = st.text_area(
        "Political Statement",

        placeholder=(
            "Example: The government has made "
            "progress, although several "
            "problems remain."
        ),

        height=130,

        label_visibility="collapsed",

        key="political_statement",
    )

    analyze_col = st.columns(
        [1, 0.28, 1]
    )[1]

    with analyze_col:

        analyze_clicked = st.button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:

        if not user_text.strip():

            st.warning(
                "Please enter a political statement."
            )

            st.stop()

        language = detect_language(
            user_text
        )

        validated_text = validate_english_input(
            user_text,
            language,
        )

        with st.spinner(
            "Analyzing statement..."
        ):

            result = predict_sentiment(
                validated_text
            )

        save_prediction(

            user_text,

            language,

            validated_text,

            result[
                "ml_prediction"
            ],

            result[
                "semantic_prediction"
            ],

            result[
                "final_sentiment"
            ],

            result[
                "ml_confidence"
            ],

            result[
                "semantic_confidence"
            ],

            result[
                "decision_method"
            ],

        )

        final = result[
            "final_sentiment"
        ]

        sentiment_class = {

            "POSITIVE":
                "sentiment-positive",

            "NEGATIVE":
                "sentiment-negative",

            "MIXED":
                "sentiment-mixed",

            "NEUTRAL":
                "sentiment-neutral",

        }.get(
            final,
            "sentiment-neutral",
        )

        message = {

            "POSITIVE":
                "Positive political sentiment detected.",

            "NEGATIVE":
                "Negative political sentiment detected.",

            "MIXED":
                "Conflicting positive and negative sentiment detected.",

            "NEUTRAL":
                "Neutral or informational statement detected.",

        }.get(
            final,
            "Sentiment detected.",
        )

        render_html(
            f"""
            <div class="result-card">
                <div class="result-topline">
                    FINAL SENTIMENT
                </div>
                <div class="sentiment-result {sentiment_class}">
                    <div class="sentiment-result-icon">
                        ●
                    </div>
                    <div class="sentiment-result-content">
                        <div class="sentiment-label">
                            {html.escape(final)}
                        </div>
                        <div class="sentiment-message">
                            {html.escape(message)}
                        </div>
                    </div>
                </div>
                <div class="result-metrics">
                    <div class="result-metric">
                        <div class="result-metric-label">
                            TF-IDF MODEL
                        </div>
                        <div class="result-metric-value">
                            {html.escape(
                                result["ml_prediction"]
                            )}
                            <span>·</span>
                            {result["ml_confidence"]:.2f}%
                        </div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-label">
                            SEMANTIC MODEL
                        </div>
                        <div class="result-metric-value">
                            {html.escape(
                                result["semantic_prediction"]
                            )}
                            <span>·</span>
                            {result["semantic_confidence"]:.2f}%
                        </div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-label">
                            DECISION ENGINE
                        </div>
                        <div class="result-metric-value result-method">
                            {html.escape(
                                result["decision_method"]
                            )}
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="section-heading section-heading-spaced">
                <span class="section-icon">
                    ◈
                </span>
                <div>
                    <div class="section-title">
                        Analysis Overview
                    </div>
                </div>
            </div>
            """
        )

        positive_matches = result.get(
            "positive_matches",
            [],
        )

        negative_matches = result.get(
            "negative_matches",
            [],
        )

        evidence = (

            "MIXED"

            if positive_matches
            and negative_matches

            else "POSITIVE"

            if positive_matches

            else "NEGATIVE"

            if negative_matches

            else "NONE"

        )

        overview_df = pd.DataFrame(
            [

                {
                    "Component":
                        "TF-IDF + Logistic Regression",

                    "Prediction":
                        result[
                            "ml_prediction"
                        ],

                    "Confidence":
                        f"{result['ml_confidence']:.2f}%",

                },

                {
                    "Component":
                        "Twitter-RoBERTa",

                    "Prediction":
                        result[
                            "semantic_prediction"
                        ],

                    "Confidence":
                        f"{result['semantic_confidence']:.2f}%",

                },

                {
                    "Component":
                        "Rule-Based Evidence",

                    "Prediction":
                        evidence,

                    "Confidence":
                        "—",

                },

                {
                    "Component":
                        "Final Hybrid Decision",

                    "Prediction":
                        final,

                    "Confidence":
                        "—",

                },

            ]
        )

        render_html_table(
            overview_df,
            "custom-table analysis-summary-table",
            escape=False,
        )

        # --------------------------------------------------------
        # AI INSIGHT
        # --------------------------------------------------------

        render_html(
            """
            <div class="section-heading section-heading-spaced">
                <span class="section-icon">
                    ✦
                </span>
                <div>
                    <div class="section-title">
                        AI Insight
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            f"""
            <div class="insight-box">
                <div class="insight-label">
                    MODEL INTERPRETATION
                </div>
                <div class="insight-text">
                    {html.escape(
                        generate_ai_insight(
                            result
                        )
                    )}
                </div>
            </div>
            """
        )

        # --------------------------------------------------------
        # EVIDENCE
        # --------------------------------------------------------

        evidence_col_1, evidence_col_2 = (
            st.columns(2)
        )

        with evidence_col_1:

            positive_items = ""

            if positive_matches:

                positive_items = "".join(

                    f"""
                    <div class="evidence-line">
                        <span class="evidence-bullet">
                            ✓
                        </span>
                        {html.escape(
                            item.title()
                        )}
                    </div>
                    """

                    for item
                    in positive_matches

                )

            else:

                positive_items = """
                <div class="evidence-empty">
                    No explicit positive evidence
                    detected.
                </div>
                """

            render_html(
                f"""
                <div class="evidence-card evidence-positive">
                    <div class="evidence-title">
                        Positive Evidence
                    </div>
                    {positive_items}
                </div>
                """
            )

        with evidence_col_2:

            negative_items = ""

            if negative_matches:

                negative_items = "".join(

                    f"""
                    <div class="evidence-line">
                        <span class="evidence-bullet">
                            ✓
                        </span>
                        {html.escape(
                            item.title()
                        )}
                    </div>
                    """

                    for item
                    in negative_matches

                )

            else:

                negative_items = """
                <div class="evidence-empty">
                    No explicit negative evidence
                    detected.
                </div>
                """

            render_html(
                f"""
                <div class="evidence-card evidence-negative">
                    <div class="evidence-title">
                        Negative Evidence
                    </div>
                    {negative_items}
                </div>
                """
            )

        # --------------------------------------------------------
        # DECISION EXPLANATION (COLOR-CODED)
        # --------------------------------------------------------

        render_html(
            """
            <div class="section-heading section-heading-spaced">
                <span class="section-icon">
                    ↔
                </span>
                <div>
                    <div class="section-title">
                        Decision Explanation
                    </div>
                </div>
            </div>
            """
        )

        decision_class = {
            "POSITIVE": "decision-positive",
            "NEGATIVE": "decision-negative",
            "MIXED": "decision-mixed",
            "NEUTRAL": "decision-neutral",
        }.get(final, "decision-neutral")

        render_html(
            f"""
            <div class="decision-box {decision_class}">
                <div class="decision-label">
                    HYBRID DECISION
                </div>
                <div class="decision-text">
                    {html.escape(
                        generate_decision_explanation(
                            result
                        )
                    )}
                </div>
            </div>
            """
        )

        # --------------------------------------------------------
        # RELATED CONTEXT
        # --------------------------------------------------------

        render_html(
            """
            <div class="section-heading section-heading-spaced">
                <span class="section-icon">
                    ⌁
                </span>
                <div>
                    <div class="section-title">
                        Related Political Context
                    </div>
                    <div class="section-description">
                        Retrieved context from the
                        historical knowledge base.
                    </div>
                </div>
            </div>
            """
        )

        contexts = retrieve(
            validated_text,
            top_k=3,
        )

        if contexts:

            context_cards = ""

            for index, context in enumerate(
                contexts,
                start=1,
            ):

                context_cards += (

                    f"""
                    <div class="context-card">
                        <div class="context-number">
                            {index}
                        </div>
                        <div class="context-content">
                            <div class="context-label">
                                RELATED CONTEXT
                            </div>
                            <div class="context-text">
                                {html.escape(
                                    str(context)
                                )}
                            </div>
                        </div>
                    </div>
                    """

                )

            render_html(
                f"""
                <div class="context-grid">
                    {context_cards}
                </div>
                """
            )

        else:

            st.info(
                "No related political context found."
            )


# ============================================================
# CSV ANALYSIS
# ============================================================

elif page == "📊 CSV Analysis":

    render_html(
        """
        <div class="page-section">
            <div class="section-heading">
                <span class="section-icon">
                    ▦
                </span>
                <div>
                    <div class="section-title">
                        CSV Sentiment Analysis
                    </div>
                    <div class="section-description">
                        Upload a CSV containing political
                        statements and analyze the selected
                        text column.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        dataframe = pd.read_csv(
            uploaded_file
        )

        render_html_table(
            dataframe.head(10),
            "custom-table csv-preview-table",
            escape=False,
        )

        text_column = st.selectbox(
            "Select text column",
            dataframe.columns,
        )

        analyze_csv = st.button(
            "Analyze CSV",
            type="primary",
        )

        if analyze_csv:

            results = []

            progress = st.progress(
                0
            )

            total_rows = max(
                len(dataframe),
                1,
            )

            for index, value in enumerate(
                dataframe[
                    text_column
                ].tolist()
            ):

                text_value = (

                    ""

                    if pd.isna(value)

                    else str(value)

                )

                row = {

                    "Final Sentiment":
                        "UNKNOWN",

                    "TF-IDF Prediction":
                        "UNKNOWN",

                    "TF-IDF Confidence":
                        0,

                    "Transformer Prediction":
                        "UNKNOWN",

                    "Transformer Confidence":
                        0,

                    "Decision":
                        "No Text",

                    "Positive Evidence":
                        "",

                    "Negative Evidence":
                        "",

                    "Contrast Direction":
                        "",

                    "Contrast Connector":
                        "",

                }

                if text_value.strip():

                    try:

                        result = (
                            predict_sentiment(
                                text_value
                            )
                        )

                        row = {

                            "Final Sentiment":
                                result[
                                    "final_sentiment"
                                ],

                            "TF-IDF Prediction":
                                result[
                                    "ml_prediction"
                                ],

                            "TF-IDF Confidence":
                                round(
                                    result[
                                        "ml_confidence"
                                    ],
                                    2,
                                ),

                            "Transformer Prediction":
                                result[
                                    "semantic_prediction"
                                ],

                            "Transformer Confidence":
                                round(
                                    result[
                                        "semantic_confidence"
                                    ],
                                    2,
                                ),

                            "Decision":
                                result[
                                    "decision_method"
                                ],

                            "Positive Evidence":
                                ", ".join(
                                    result[
                                        "positive_matches"
                                    ]
                                ),

                            "Negative Evidence":
                                ", ".join(
                                    result[
                                        "negative_matches"
                                    ]
                                ),

                            "Contrast Direction":
                                result.get(
                                    "contrast_direction"
                                )
                                or "",

                            "Contrast Connector":
                                result.get(
                                    "contrast_connector"
                                )
                                or "",

                        }

                    except Exception as exc:

                        row[
                            "Final Sentiment"
                        ] = "ERROR"

                        row[
                            "Decision"
                        ] = str(exc)

                results.append(
                    row
                )

                progress.progress(
                    int(
                        (
                            (index + 1)
                            / total_rows
                        )
                        * 100
                    )
                )

            result_df = pd.concat(
                [
                    dataframe.reset_index(
                        drop=True
                    ),

                    pd.DataFrame(
                        results
                    ),
                ],
                axis=1,
            )

            st.success(
                "CSV analysis completed successfully."
            )

            render_html_table(
                result_df,
                "custom-table csv-results-table",
                escape=False,
            )

            st.download_button(
                "Download Results",

                data=result_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                ),

                file_name=(
                    "political_sentiment_results.csv"
                ),

                mime="text/csv",

            )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    render_html(
        """
        <div class="page-section">
            <div class="section-heading">
                <span class="section-icon">
                    ◷
                </span>
                <div>
                    <div class="section-title">
                        Prediction History
                    </div>
                    <div class="section-description">
                        Review previously analyzed political
                        statements and model decisions.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )

    history_df = pd.read_sql_query(
        """
        SELECT
            id,
            input_text,
            language,
            ml_prediction,
            semantic_prediction,
            final_sentiment,
            confidence,
            semantic_confidence,
            decision_method,
            created_at
        FROM predictions
        ORDER BY id DESC
        """,
        connection,
    )

    connection.close()

    if history_df.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        history_df["confidence"] = history_df["confidence"].round(2)
        history_df["semantic_confidence"] = history_df["semantic_confidence"].round(2)

        total_predictions = len(history_df)

        positive_count = int((history_df["final_sentiment"] == "POSITIVE").sum())
        negative_count = int((history_df["final_sentiment"] == "NEGATIVE").sum())
        mixed_count = int((history_df["final_sentiment"] == "MIXED").sum())

        render_html(
            f"""
            <div class="dashboard-stats history-stats">
                <div class="stat-card">
                    <div class="stat-icon">
                        #
                    </div>
                    <div class="stat-content">
                        <span class="stat-label">
                            TOTAL PREDICTIONS
                        </span>
                        <strong>
                            {total_predictions}
                        </strong>
                    </div>
                </div>
                <div class="stat-card sentiment-stat-positive">
                    <div class="stat-icon">
                        ●
                    </div>
                    <div class="stat-content">
                        <span class="stat-label">
                            POSITIVE
                        </span>
                        <strong>
                            {positive_count}
                        </strong>
                    </div>
                </div>
                <div class="stat-card sentiment-stat-negative">
                    <div class="stat-icon">
                        ●
                    </div>
                    <div class="stat-content">
                        <span class="stat-label">
                            NEGATIVE
                        </span>
                        <strong>
                            {negative_count}
                        </strong>
                    </div>
                </div>
                <div class="stat-card sentiment-stat-mixed">
                    <div class="stat-icon">
                        ●
                    </div>
                    <div class="stat-content">
                        <span class="stat-label">
                            MIXED
                        </span>
                        <strong>
                            {mixed_count}
                        </strong>
                    </div>
                </div>
            </div>
            """
        )

        def badge_sentiment(val):
            val_str = str(val).upper()
            css_map = {
                "POSITIVE": "sentiment-positive",
                "NEGATIVE": "sentiment-negative",
                "MIXED": "sentiment-mixed",
                "NEUTRAL": "sentiment-neutral",
            }
            cls = css_map.get(val_str, "sentiment-neutral")
            return f'<span class="table-badge {cls}">{val_str}</span>'

        display_history_df = history_df.copy()
        display_history_df["final_sentiment"] = display_history_df["final_sentiment"].apply(badge_sentiment)

        display_history_df = display_history_df.rename(
            columns={
                "id": "ID",
                "input_text": "Political Statement",
                "language": "Lang",
                "ml_prediction": "TF-IDF",
                "semantic_prediction": "RoBERTa",
                "final_sentiment": "Decision",
                "confidence": "TF-IDF Conf (%)",
                "semantic_confidence": "RoBERTa Conf (%)",
                "decision_method": "Decision Method",
                "created_at": "Timestamp",
            }
        )

        render_html_table(
            display_history_df,
            "custom-table history-table",
            escape=False,
        )

        chart_data = (
            history_df["final_sentiment"]
            .value_counts()
        )

        st.bar_chart(chart_data)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        <span>
            Political Sentiment Analyzer
        </span>
        <span class="footer-separator">
            •
        </span>
        <span>
            TF-IDF + Logistic Regression
        </span>
        <span class="footer-separator">
            •
        </span>
        <span>
            Transformer NLP
        </span>
        <span class="footer-separator">
            •
        </span>
        <span>
            Contrast-Aware Hybrid Decision Engine
        </span>
        <span class="footer-separator">
            •
        </span>
        <span>
            RAG
        </span>
    </div>
    """
)