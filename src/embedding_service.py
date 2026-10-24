import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts: str | list[str]):
    """
    Convert a string or a list of strings into embedding vectors.
    """
    return model.encode(texts)

def calculate_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    """
    return util.cos_sim(vec1, vec2)

if __name__ == "__main__":
    sample_texts = [
        "How to reset a password",
        "Steps to recover user login",
        "How to cook italian pasta",
    ]

    embeddings = get_embeddings(sample_texts)
    print("Shape:", embeddings.shape)

    score_0_1 = calculate_similarity(embeddings[0], embeddings[1])
    score_0_2 = calculate_similarity(embeddings[0], embeddings[2])

    print(f"Similarity of phrase 0 and 1: {score_0_1[0][0]:.4f}")
    print(f"Similarity of phrase 0 and 2: {score_0_2[0][0]:.4f}")