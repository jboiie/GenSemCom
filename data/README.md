# Data

| File | Contents |
|---|---|
| `processed/10k_sentences.json` | 10,000 English sentences, `{"sentences": [...]}`. The diagnostic corpus of the paper (Sec. IV-A) |
| `raw/1k_sentences_random.json` | 1,000-sentence random subset used for the BLEU / ROUGE-L / BERTScore evaluation (Table II) |

Sentences were sampled from a Wikipedia scrape and filtered to lengths between
4 and 64 characters to avoid padding artifacts.

## Licensing

The **code** in this repository is MIT licensed (see `LICENSE`). The **sentence
data** is derived from Wikipedia and is therefore licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), not MIT.
Attribution: Wikipedia contributors. Redistribution of this data or derivatives
must preserve CC BY-SA terms.

Running the pipeline writes `.npy`, `.npz` and `.csv` intermediates into this
directory. Those are gitignored. Regenerate them rather than committing them.
