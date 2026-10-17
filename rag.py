import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "rag_data.csv"
)


df.columns = df.columns.str.strip()


if "text" not in df.columns:

    raise ValueError(
        "❌ rag_data.csv must contain a 'text' column."
    )


# ============================================================
# CLEAN TEXT
# ============================================================

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)


df = df[
    df["text"] != ""
].reset_index(
    drop=True
)


# ============================================================
# METADATA
# ============================================================

if "country" not in df.columns:

    df["country"] = "Unknown"


if "year" not in df.columns:

    df["year"] = "Unknown"


df["country"] = (
    df["country"]
    .astype(str)
)


df["year"] = (
    df["year"]
    .astype(str)
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print(
    "🧠 Loading Sentence Transformer..."
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "✅ Embedding model loaded"
)


# ============================================================
# EMBEDDINGS
# ============================================================

texts = (
    df["text"]
    .tolist()
)


embeddings = embedding_model.encode(

    texts,

    convert_to_numpy=True,

    normalize_embeddings=True
)


# ============================================================
# FAISS
# ============================================================

dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings.astype(
        np.float32
    )
)


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    country=None,
    year=None,
    top_k=3
):

    if not query:

        return []


    filtered_df = df.copy()


    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    if country:

        filtered_df = filtered_df[
            filtered_df["country"]
            .str.lower()
            ==
            str(country).lower()
        ]


    # --------------------------------------------------------
    # YEAR FILTER
    # --------------------------------------------------------

    if year:

        filtered_df = filtered_df[
            filtered_df["year"]
            ==
            str(year)
        ]


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if filtered_df.empty:

        filtered_df = df.copy()


    texts = (
        filtered_df["text"]
        .tolist()
    )


    if not texts:

        return []


    # --------------------------------------------------------
    # EMBED FILTERED DATA
    # --------------------------------------------------------

    filtered_embeddings = (
        embedding_model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True
        )
    )


    temp_index = faiss.IndexFlatIP(
        filtered_embeddings.shape[1]
    )


    temp_index.add(
        filtered_embeddings.astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(

            [query],

            convert_to_numpy=True,

            normalize_embeddings=True
        )
    )


    k = min(
        top_k,
        len(texts)
    )


    scores, indices = (
        temp_index.search(
            query_embedding.astype(
                np.float32
            ),
            k
        )
    )


    results = []


    for idx, score in zip(
        indices[0],
        scores[0]
    ):

        if idx < 0:
            continue


        results.append(
            (
                texts[idx],
                float(score)
            )
        )


    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return [
        text
        for text, score in results
    ]