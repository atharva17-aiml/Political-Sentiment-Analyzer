from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import random

# ---------------- LOAD DATA ----------------
df = pd.read_csv("rag_data.csv")

# ---------------- METADATA ----------------
countries = ["India", "USA"]
years = [2023, 2024]

# ✅ Make metadata consistent (only assign once)
if "country" not in df.columns:
    df["country"] = [random.choice(countries) for _ in range(len(df))]

if "year" not in df.columns:
    df["year"] = [random.choice(years) for _ in range(len(df))]

# Normalize for safe filtering
df["country"] = df["country"].astype(str)
df["year"] = df["year"].astype(str)

# ---------------- MODEL ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- EMBEDDINGS ----------------
texts_all = df['text'].tolist()
embeddings_all = model.encode(texts_all)

index = faiss.IndexFlatL2(384)
index.add(np.array(embeddings_all))


# ---------------- RETRIEVE ----------------
def retrieve(query, country=None, year=None):

    filtered_df = df.copy()

    # ✅ SAFE FILTERING
    if country:
        filtered_df = filtered_df[
            filtered_df['country'].str.lower() == str(country).lower()
        ]

    if year:
        filtered_df = filtered_df[
            filtered_df['year'] == str(year)
        ]

    # ✅ FALLBACK
    if filtered_df.empty:
        filtered_df = df.copy()

    texts = filtered_df['text'].tolist()

    # ✅ Use subset embeddings (efficient)
    emb = model.encode(texts)

    temp_index = faiss.IndexFlatL2(384)
    temp_index.add(np.array(emb))

    query_emb = model.encode([query])

    k = min(3, len(texts))
    D, I = temp_index.search(query_emb, k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append((texts[idx], dist))

    # Sort by similarity
    results = sorted(results, key=lambda x: x[1])

    return [r[0] for r in results]