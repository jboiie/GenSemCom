import os
from pathlib import Path

import numpy as np
import pandas as pd

from .channel import get_project_paths  # same as in evaluate.py


K = 5  # choose 3 or 5 and keep fixed


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norm = np.sqrt((x ** 2).sum(axis=axis, keepdims=True)) + eps
    return x / norm


def get_snr_list():
    # Use same dense SNRs as in evaluate.py
    return list(range(0, 21, 2))  # 0, 2, 4, ..., 20 dB


def get_reconstructed_path_for_snr(snr_db: float) -> str:
    """
    Reconstructed embeddings file for given SNR:
    data/minilm_embeddings_rec_snr_X.npy
    """
    paths = get_project_paths()
    data_dir = paths["data_dir"]
    fname = f"minilm_embeddings_rec_snr_{int(snr_db)}.npy"
    return os.path.join(data_dir, fname)


def evaluate_snr(snr: int, k: int = K):
    """
    For a single SNR:
    - average cosine(clean, reconstructed)
    - top-1 and top-k retrieval accuracy over ALL sentences
    """
    paths = get_project_paths()
    orig = np.load(paths["embeddings"])  # (N, D), clean embeddings
    N, D = orig.shape
    orig_norm = l2_normalize(orig)

    rec_path = get_reconstructed_path_for_snr(snr)
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Reconstructed file for {snr} dB not found at {rec_path}")

    rec = np.load(rec_path)  # (N, D)
    rec_norm = l2_normalize(rec)

    # 1) average cosine similarity between clean and reconstructed embeddings
    # (per-sentence cosine, then mean)
    cos_per_sentence = (orig_norm * rec_norm).sum(axis=1)  # (N,)
    avg_cosine = float(cos_per_sentence.mean())

    # 2) retrieval: for each reconstructed embedding, find rank of its original
    top1_hits = 0
    topk_hits = 0

    # precompute for speed: (N, D)
    clean_matrix = orig_norm

    for i in range(N):
        q = rec_norm[i : i + 1]           # (1, D)
        sims = (q @ clean_matrix.T).ravel()   # (N,)
        # descending order: highest cosine first
        sorted_idx = np.argsort(-sims)

        if sorted_idx[0] == i:
            top1_hits += 1
        if i in sorted_idx[:k]:
            topk_hits += 1

    top1_acc = top1_hits / N
    topk_acc = topk_hits / N

    return {
        "snr_db": snr,
        "avg_cosine": avg_cosine,
        "top1_acc": top1_acc,
        f"top{k}_acc": topk_acc,
        "num_sentences": N,
    }


def main():
    results = []
    for snr in get_snr_list():
        print(f"Evaluating retrieval for SNR = {snr} dB ...")
        stats = evaluate_snr(snr)
        print(
            f"SNR {snr} dB -> "
            f"avg_cosine={stats['avg_cosine']:.4f}, "
            f"top1={stats['top1_acc']:.4f}, "
            f"top{K}={stats[f'top{K}_acc']:.4f}"
        )
        results.append(stats)

    # Save to CSV for the paper
    paths = get_project_paths()
    data_dir = Path(paths["data_dir"])
    out_path = data_dir / "minilm_retrieval_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved retrieval results to {out_path}")


if __name__ == "__main__":
    main()
