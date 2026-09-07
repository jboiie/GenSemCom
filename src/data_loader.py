from pathlib import Path
from typing import List, Optional
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_sentences_json(
    filename: str = "10k_sentences.json",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Load sentences from a JSON file in data/processed with structure:
    {
        "sentences": ["...", "...", ...]
    }
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "sentences" not in data:
        raise KeyError("JSON must contain a 'sentences' key.")

    sentences = list(map(str, data["sentences"]))

    if max_samples is not None:
        sentences = sentences[:max_samples]

    return sentences
