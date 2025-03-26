from sentence_transformers import SentenceTransformer
import pickle
import copy

def compute_embeddings(chunks, use_precomputed = False):
    if (use_precomputed):
        with open("embeddings.pkl", "rb") as f:
            return pickle.load(f)

    chunks_with_embeddings = copy.deepcopy(chunks)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([chunk["text"] for chunk in chunks_with_embeddings])
    for chunk, emb in zip(chunks_with_embeddings, embeddings):
        chunk["embedding"] = emb
    return chunks_with_embeddings

if __name__ == "__main__":
    from chunks import get_chunks

    chunks = get_chunks('./mocks')
    chunks_with_embeddings = compute_embeddings(chunks)
    
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(chunks_with_embeddings, f)
    