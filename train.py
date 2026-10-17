import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocess import clean_text


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 65)
print("🗳️ POLITICAL SENTIMENT ANALYSIS - TF-IDF MODEL TRAINING")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/dataset.csv"
)

df.columns = df.columns.str.strip()


print("\n📊 Dataset Preview:")
print(
    df.head()
)


print("\n📊 Dataset Columns:")
print(
    df.columns.tolist()
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "text",
    "label"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"❌ Missing required column: {column}"
        )


# ============================================================
# CLEAN TEXT
# ============================================================

print("\n🧹 Cleaning dataset...")


df["text"] = (
    df["text"]
    .astype(str)
    .apply(clean_text)
)


# Remove empty text
df = df[
    df["text"].str.strip() != ""
]


# Remove duplicates
df = df.drop_duplicates(
    subset="text"
)


# Normalize labels
df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# Keep valid labels
valid_labels = [
    "positive",
    "negative",
    "neutral"
]


df = df[
    df["label"].isin(
        valid_labels
    )
]


df = df.reset_index(
    drop=True
)


print(
    "\n📊 Dataset size:",
    len(df)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\n📊 Class Distribution:"
)

print(
    df["label"].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

X_text = df["text"]

y = df["label"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_text,
    y,

    test_size=0.25,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "\n📚 Training samples:",
    len(X_train)
)

print(
    "🧪 Testing samples:",
    len(X_test)
)


# ============================================================
# TF-IDF
# ============================================================

print(
    "\n🔢 Training TF-IDF vectorizer..."
)


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.95,

    sublinear_tf=True,

    max_features=5000,

    strip_accents="unicode"
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "✅ TF-IDF completed"
)


print(
    "Feature matrix:",
    X_train_tfidf.shape
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print(
    "\n🤖 Training Logistic Regression..."
)


model = LogisticRegression(

    max_iter=3000,

    C=2.0,

    class_weight="balanced",

    random_state=RANDOM_STATE
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "✅ Model trained successfully"
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("📊 MODEL EVALUATION")
print("=" * 65)


y_pred = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\n🎯 Accuracy: {accuracy * 100:.2f}%"
)


print(
    "\n📋 Classification Report:"
)


print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print(
    "\n🔲 Confusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# RULE DETECTION
# ============================================================

def rule_boost(text):

    text = text.lower()


    # Strong negative phrases

    negative_phrases = [

        "jobs are hard to find",

        "lack of jobs",

        "lack of employment",

        "unemployment",

        "rising prices",

        "high inflation",

        "price increase",

        "price rise",

        "poor governance",

        "failed to deliver",

        "failed to create jobs",

        "corruption",

        "corrupt government",

        "very disappointed",

        "extremely frustrated",

        "people are struggling",

        "not good",

        "not effective",

        "not successful"
    ]


    # Strong positive phrases

    positive_phrases = [

        "highly appreciated",

        "well appreciated",

        "great job",

        "doing a great job",

        "doing good work",

        "excellent work",

        "excellent job",

        "strong performance",

        "successful policy",

        "excellent policy",

        "significant improvement",

        "great progress",

        "very effective",

        "highly effective"
    ]


    for phrase in negative_phrases:

        if phrase in text:

            return "negative"


    for phrase in positive_phrases:

        if phrase in text:

            return "positive"


    return None


# ============================================================
# REAL WORLD TESTING
# ============================================================

print("\n" + "=" * 65)
print("🔍 REAL-WORLD TESTING")
print("=" * 65)


test_sentences = [

    "People are frustrated with inflation",

    "Government is doing a great job",

    "The policy was announced yesterday",

    "Jobs are hard to find",

    "This decision is not good",

    "The government's efforts to create more employment opportunities are highly appreciated.",

    "The government has failed to control inflation.",

    "The parliament discussed the new policy.",

    "The administration is successful",

    "The government is doing good work"
]


for text in test_sentences:

    cleaned = clean_text(
        text
    )


    vector = vectorizer.transform(
        [cleaned]
    )


    model_prediction = model.predict(
        vector
    )[0]


    probabilities = model.predict_proba(
        vector
    )[0]


    classes = model.classes_


    model_confidence = (
        max(probabilities) * 100
    )


    rule = rule_boost(
        text
    )


    if rule:

        final_prediction = rule

        method = "Rule + ML"

    else:

        final_prediction = model_prediction

        method = "ML Model"


    print(
        f"\nText: {text}"
    )

    print(
        f"ML Prediction: {model_prediction}"
    )

    print(
        f"Final Prediction: {final_prediction}"
    )

    print(
        f"ML Confidence: {model_confidence:.2f}%"
    )

    print(
        f"Method: {method}"
    )

    print(
        "Probabilities:"
    )


    for label, probability in zip(
        classes,
        probabilities
    ):

        print(
            f"   {label}: "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


with open(
    "model/sentiment_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


with open(
    "model/tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )


print("\n" + "=" * 65)

print(
    "💾 Model saved successfully!"
)

print(
    "📁 model/sentiment_model.pkl"
)

print(
    "📁 model/tfidf_vectorizer.pkl"
)

print("=" * 65)