import os
import numpy as np
import pandas as pd

from .channel import get_project_paths
from .evaluate import get_snr_list, get_reconstructed_path_for_snr, l2_normalize


def load_data_for_snr(snr_db: float):
    """
    Load:
      - original sentences
      - clean MiniLM embeddings
      - reconstructed embeddings for a given SNR
    """
    paths = get_project_paths()

    # Sentences
    sentences_csv_path = os.path.join(paths["data_dir"], "minilm_sentences.csv")
    df = pd.read_csv(sentences_csv_path)
    sentences = df["sentence"].tolist()

    # Clean embeddings
    clean_embeddings = np.load(paths["embeddings"])

    # Reconstructed embeddings for the chosen SNR
    rec_path = get_reconstructed_path_for_snr(snr_db)
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Reconstructed embeddings for {snr_db} dB not found at {rec_path}. "
                                f"Run evaluate.py first to generate them.")
    rec_embeddings = np.load(rec_path)

    return sentences, clean_embeddings, rec_embeddings


def nearest_neighbour_indices(clean_embeddings: np.ndarray,
                              query_embeddings: np.ndarray,
                              top_k: int = 1) -> np.ndarray:
    """
    For each query embedding, find indices of top_k nearest neighbours
    in clean_embeddings using cosine similarity.
    Returns array of shape (num_queries, top_k) with indices.
    """
    # L2-normalize
    clean_norm = l2_normalize(clean_embeddings)
    query_norm = l2_normalize(query_embeddings)

    # Cosine similarity = dot product of normalized vectors
    sims = np.matmul(query_norm, clean_norm.T)  # (num_queries, N_clean)

    # argsort descending
    topk_indices = np.argsort(-sims, axis=1)[:, :top_k]
    return topk_indices, sims


def show_examples_for_snr(snr_db: float, num_examples: int = 5, top_k: int = 1):
    """
    Print a few original vs decoded sentences for the given SNR.
    """
    sentences, clean_embeddings, rec_embeddings = load_data_for_snr(snr_db)

    # Limit examples to available sentences
    num_sentences = len(sentences)
    num_examples = min(num_examples, num_sentences)

    # Take first num_examples reconstructed embeddings as queries
    query_emb = rec_embeddings[:num_examples]

    topk_indices, sims = nearest_neighbour_indices(clean_embeddings, query_emb, top_k=top_k)

    print("=" * 80)
    print(f"Nearest-neighbour decoding examples at SNR = {snr_db} dB")
    print("=" * 80)

    for i in range(num_examples):
        print(f"\nExample {i+1}:")
        print(f"  Original sentence: {sentences[i]}")

        for k in range(top_k):
            idx = topk_indices[i, k]
            sim_val = float(sims[i, idx])
            print(f"  Decoded (top-{k+1}) sentence: {sentences[idx]}")
            print(f"    Cosine similarity: {sim_val:.4f}")


def main():
    # Choose SNR here (must be one of the values generated in evaluate.py)
    snr_values = get_snr_list()
    print(f"Available SNR values (dB): {snr_values}")

    # For now, just hard-code one SNR. Change as needed.
    target_snr = 0  # e.g., 0, 5, 10, 15, 20
    print(f"\nUsing SNR = {target_snr} dB for decoding examples.\n")

    show_examples_for_snr(target_snr, num_examples=5, top_k=3)


if __name__ == "__main__":
    main()
