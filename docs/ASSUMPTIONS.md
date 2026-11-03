# Assumptions and verified adaptations

- C03: `datasets.load_dataset("mteb/amazon_reviews_multi", "en")` was attempted on the installed
  `datasets` version and failed because the repository still contains a dataset loading script,
  which this version no longer executes. The repository's raw JSONL files are used as the fallback.
- C03 schema probe: current raw records expose `id`, `text`, `label` (integer 0–4), and
  `label_text`; `label` is used as the canonical `stars` value. The schema detector also recognizes
  legacy `review_body` plus `stars` records if they satisfy the specified 0–4 range.
- C04 real benchmark: test support is Negative 610, Neutral 306, Positive 584. This differs from
  the earlier recorded support (609/310/581); the newly sampled manifest and files are authoritative
  for this rebuild, and no counts were changed to force agreement.
- C04 real benchmark class counts, train / validation / test:
  - Negative: 713 / 253 / 610
  - Neutral: 363 / 118 / 306
  - Positive: 724 / 229 / 584
- C01 checks ran on the available Python 3.13 environment. The project declares Python 3.10+;
  Python 3.10 compatibility is maintained in code, but was not separately exercised here.
