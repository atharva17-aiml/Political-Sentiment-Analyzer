import re
import nltk
from nltk.corpus import stopwords

# ---------------- LOAD STOPWORDS SAFELY ----------------
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

stop_words = load_stopwords()

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    """
    Cleans input text by:
    - Lowercasing
    - Removing URLs
    - Removing special characters
    - Removing extra spaces
    - Removing stopwords
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions & hashtags (NEW 🔥)
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize & remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)
