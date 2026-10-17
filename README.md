# 🗳️ Political Sentiment Analyzer

An AI-powered **Political Sentiment Analysis System** built with Machine Learning and NLP to analyze political statements and classify them as **Positive, Negative, Neutral, or Mixed**.

The system combines a traditional TF-IDF machine learning model, a Transformer-based semantic model, rule-based linguistic analysis, contrast detection, and Retrieval-Augmented Generation (RAG) to produce more explainable sentiment decisions.

---

## 🚀 Features

### 🧠 Hybrid Sentiment Analysis
- TF-IDF feature extraction
- Logistic Regression classification
- Twitter-RoBERTa Transformer-based semantic analysis
- Hybrid decision engine combining multiple signals
- Sentiment confidence scores

### 🔍 Explainable AI
- Positive evidence detection
- Negative evidence detection
- Contrast-aware sentiment analysis
- Negation and linguistic pattern detection
- Sarcasm-aware analysis
- Explanation of the final sentiment decision

### 🎯 Sentiment Categories
- 🟢 Positive
- 🔴 Negative
- 🟣 Mixed
- ⚪ Neutral

### 🌐 English-Only Analysis
The current version supports **English political statements only**.

Hindi, Marathi, Hinglish, and other non-English inputs are detected and rejected instead of being automatically translated.

---

## 📊 CSV Sentiment Analysis

Upload a CSV file containing political statements and analyze an entire dataset.

Features include:

- CSV file upload
- Text-column selection
- Batch sentiment analysis
- TF-IDF prediction
- TF-IDF confidence
- Transformer prediction
- Transformer confidence
- Final sentiment
- Positive evidence
- Negative evidence
- Contrast direction
- Contrast connector
- Downloadable CSV results

The application provides a progress indicator while processing the dataset. :chatgpt-content-reference{index="2"}

---

## 📜 Prediction History

The application stores analyzed statements in an SQLite database.

Prediction History provides:

- Previously analyzed political statements
- TF-IDF predictions
- Transformer predictions
- Final sentiment
- Model confidence scores
- Decision method
- Timestamp
- Sentiment distribution visualization

:chatgpt-content-reference{index="3"}

---

## 🧠 Retrieval-Augmented Generation (RAG)

The system retrieves relevant political context from a historical knowledge base.

### RAG Components

- SentenceTransformer
- FAISS vector similarity search
- Historical political context
- Top relevant context retrieval

The application displays **Related Political Context** after sentiment analysis. :chatgpt-content-reference{index="4"}

---

## ⚙️ Sentiment Decision Pipeline

```text
Political Statement
        ↓
English Language Validation
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression
        ↓
Transformer Semantic Analysis
        ↓
Positive / Negative Evidence
        ↓
Negation & Contrast Analysis
        ↓
Sarcasm Detection
        ↓
Contrast-Aware Hybrid Decision Engine
        ↓
Final Sentiment
        ↓
RAG Historical Context
        ↓
Explanation & Visualization