# Data provenance

## Source

- Dataset: `mteb/amazon_reviews_multi` from the Hugging Face Hub.
- Recorded source revision: `c379a6705fec24a2493fa68e011692605f44e119`.
- Languages: `en`, `fr`, and `zh`.
- Raw JSONL records expose `id`, `text`, integer `label` (0–4), and `label_text`; the integer
  `label` is treated as the star rating. See `docs/ASSUMPTIONS.md` for the loader adaptation.

## Mapping and sampling

The three classes are `Negative=0`, `Neutral=1`, and `Positive=2`. Star ratings map as
`0→0`, `1→0`, `2→1`, `3→2`, and `4→2`.

The existing benchmark was validated and reused. Its manifest records seed `20261021`, sampled
per language at 600 train, 200 validation, and 500 test examples. The recorded aggregate class
counts are:

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 713 | 363 | 724 | 1,800 |
| Validation | 253 | 118 | 229 | 600 |
| Test | 610 | 306 | 584 | 1,500 |

Sampling uses per-language/split reservoir sampling. Exact-text duplicates are excluded across
splits in test, validation, train priority order. Files are shuffled with the seeded RNG. The
manifest includes file SHA-256 hashes and language/split/class counts.

## Verification

Run `python -m sentiment validate-data` from the project root. It verifies manifest hashes,
record shape and labels, class/split counts, language coverage, and split overlap. Do not replace
an existing benchmark without regenerating all run comparisons against the new manifest.
