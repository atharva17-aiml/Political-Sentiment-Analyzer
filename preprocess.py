import re


def clean_text(text):
    """
    Common preprocessing used during both
    training and prediction.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # --------------------------------------------------
    # IMPORTANT NEGATION PHRASES
    # --------------------------------------------------

    negations = {
        "not good": "not_good",
        "not great": "not_great",
        "not excellent": "not_excellent",
        "not effective": "not_effective",
        "not successful": "not_successful",
        "not working": "not_working",
        "not acceptable": "not_acceptable",
        "not satisfied": "not_satisfied",
        "not happy": "not_happy",
        "never good": "never_good",
        "never successful": "never_successful"
    }

    for old, new in negations.items():
        text = text.replace(old, new)

    # --------------------------------------------------
    # POLITICAL SENTIMENT PHRASES
    # --------------------------------------------------

    replacements = {
        "highly appreciated": "highly_appreciated",
        "great job": "great_job",
        "doing a great job": "doing_great_job",
        "doing good work": "doing_good_work",
        "excellent work": "excellent_work",
        "excellent job": "excellent_job",

        "employment opportunities": "employment_opportunities",

        "jobs are hard to find": "jobs_hard_find",
        "lack of jobs": "lack_jobs",
        "lack of employment": "lack_employment",

        "failed to deliver": "failed_deliver",
        "failed to create jobs": "failed_create_jobs",

        "poor governance": "poor_governance",
        "strong performance": "strong_performance",

        "rising prices": "rising_prices",
        "price rise": "price_rise",
        "price increase": "price_increase",

        "high inflation": "high_inflation",

        "jobless": "unemployed",
        "unemployment": "unemployment",

        "very disappointed": "very_disappointed",
        "extremely frustrated": "extremely_frustrated"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # --------------------------------------------------
    # NORMALIZE COMMON WORDS
    # --------------------------------------------------

    word_replacements = {
        "inflation": "inflation",
        "frustrated": "frustrated",
        "frustration": "frustrated",
        "disappointed": "disappointed",
        "disappointment": "disappointed",
        "corruption": "corruption",
        "corrupt": "corrupt",
        "successful": "successful",
        "success": "success",
        "effective": "effective",
        "efficient": "efficient",
        "improved": "improved",
        "improvement": "improvement"
    }

    for old, new in word_replacements.items():
        text = text.replace(old, new)

    # --------------------------------------------------
    # REMOVE URLS
    # --------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # --------------------------------------------------
    # REMOVE MENTIONS
    # --------------------------------------------------

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # --------------------------------------------------
    # REMOVE HASHTAGS BUT KEEP WORD
    # --------------------------------------------------

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # --------------------------------------------------
    # KEEP WORDS / UNDERSCORES
    # --------------------------------------------------

    text = re.sub(
        r"[^a-zA-Z_\s]",
        " ",
        text
    )

    # --------------------------------------------------
    # REMOVE EXTRA SPACES
    # --------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text