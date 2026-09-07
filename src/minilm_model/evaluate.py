import os
import numpy as np

from .channel import get_project_paths, run_channel_for_snr


def get_snr_list():
    """List of SNR values (dB) to evaluate."""
    return list(range(0, 21, 2))  # 0, 2, 4, ..., 20 dB


def get_reconstructed_path_for_snr(snr_db: float) -> str:
    """
    Build a unique filename for reconstructed embeddings for each SNR.
    Example: data/minilm_embeddings_rec_snr_10.npy
    """
    paths = get_project_paths()
    data_dir = paths["data_dir"]
    fname = f"minilm_embeddings_rec_snr_{int(snr_db)}.npy"
    return os.path.join(data_dir, fname)


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize along given axis."""
    norm = np.sqrt((x ** 2).sum(axis=axis, keepdims=True)) + eps
    return x / norm


def run_channel_and_save_by_snr():
    """
    For each SNR, call run_channel_for_snr and save reconstructed embeddings
    to a unique file per SNR.
    """
    snr_values = get_snr_list()
    paths = get_project_paths()
    embeddings = np.load(paths["embeddings"])  # original clean embeddings
    print(f"Original embeddings shape: {embeddings.shape}")

    for snr in snr_values:
        print("=" * 60)
        print(f"Running channel + saving reconstructed embeddings for SNR = {snr} dB")

        # Use existing function to produce reconstructed embeddings in memory
        _, rec_embeddings = run_channel_for_snr(snr)

        # Save per-SNR reconstructed embeddings
        rec_path = get_reconstructed_path_for_snr(snr)
        np.save(rec_path, rec_embeddings)
        print(f"Saved reconstructed embeddings for {snr} dB to: {rec_path}")


def evaluate_cosine_similarity():
    """
    Load original embeddings and per-SNR reconstructed embeddings,
    compute average cosine similarity, and print results.
    """
    snr_values = get_snr_list()
    paths = get_project_paths()
    embeddings = np.load(paths["embeddings"])
    print(f"Loaded original embeddings: {embeddings.shape}")

    orig_norm = l2_normalize(embeddings)

    avg_similarities = []

    for snr in snr_values:
        rec_path = get_reconstructed_path_for_snr(snr)
        if not os.path.exists(rec_path):
            print(f"[WARN] Reconstructed file for {snr} dB not found at {rec_path}. Skipping.")
            avg_similarities.append(None)
            continue

        rec_embeddings = np.load(rec_path)
        print(f"SNR {snr} dB: reconstructed embeddings shape: {rec_embeddings.shape}")

        rec_norm = l2_normalize(rec_embeddings)

        # Cosine similarity per sample = dot product of corresponding rows
        cos_per_sample = (orig_norm * rec_norm).sum(axis=1)  # shape (N,)
        avg_cos = float(cos_per_sample.mean())
        avg_similarities.append(avg_cos)

        print(f"Average cosine similarity at {snr} dB: {avg_cos:.4f}")

    print("\n=== Summary: Average cosine similarity vs SNR ===")
    for snr, sim in zip(snr_values, avg_similarities):
        if sim is None:
            print(f"SNR {snr} dB: (missing data)")
        else:
            print(f"SNR {snr} dB: {sim:.4f}")


def main():
    # 1) Run channel and save reconstructed embeddings per SNR
    run_channel_and_save_by_snr()

    # 2) Evaluate cosine similarity vs SNR
    evaluate_cosine_similarity()


if __name__ == "__main__":
    main()
