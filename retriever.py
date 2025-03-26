from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve(query, chunks, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    chunk_embs = np.array([chunk["embedding"] for chunk in chunks])
    scores = cosine_similarity(query_emb, chunk_embs)[0]

    scored_chunks = [
        {**chunk, "score": score} for chunk, score in zip(chunks, scores)
    ]
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]