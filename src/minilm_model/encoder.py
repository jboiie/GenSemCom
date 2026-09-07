import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Corpus location. Defaults to the copy shipped with this repository; override
# with the GENSEMCOM_CORPUS environment variable to point at your own file.
_DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "10k_sentences.json"
)
DATASET_PATH = os.environ.get("GENSEMCOM_CORPUS", str(_DEFAULT_CORPUS))


def load_sentences_from_json(path: str):
    """
    JSON format:
    {
        "sentences": [
            "sentence 1",
            "sentence 2",
            ...
        ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]
    return sentences


def main():
    # Project root = three levels up from this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    sentences_csv_path = os.path.join(data_dir, "minilm_sentences.csv")
    embeddings_npy_path = os.path.join(data_dir, "minilm_embeddings.npy")

    print(f"Loading sentences from: {DATASET_PATH}")
    sentences = load_sentences_from_json(DATASET_PATH)
    print(f"Number of sentences loaded: {len(sentences)}")

    print("Loading MiniLM model (all-MiniLM-L6-v2) on CPU...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding sentences to embeddings...")
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=True
    )
    embeddings = np.asarray(embeddings)  # shape: (N, 384)

    print(f"Embeddings shape: {embeddings.shape}")

    # Save sentences and embeddings
    df = pd.DataFrame({"sentence": sentences})
    df.to_csv(sentences_csv_path, index=False, encoding="utf-8")

    np.save(embeddings_npy_path, embeddings)

    print(f"Saved sentences to:   {sentences_csv_path}")
    print(f"Saved embeddings to:  {embeddings_npy_path}")
    print("MiniLM encoder step completed.")


if __name__ == "__main__":
    main()
