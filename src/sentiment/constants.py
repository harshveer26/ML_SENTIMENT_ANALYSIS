"""Shared benchmark labels, languages, and default parameters."""

LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
STAR_TO_LABEL = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
LANGUAGES = ["en", "fr", "zh"]
DEFAULT_SEED_BENCHMARK = 20261021
DEFAULT_SEED_TRAIN = 42
SPLIT_SIZES = {"train": 600, "validation": 200, "test": 500}
DATASET_NAME = "mteb/amazon_reviews_multi"
