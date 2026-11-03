"""Shared local-only tokenizer fixture."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Build a tiny word-level tokenizer without network access."""
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "[LANG=en]": 5,
        "[LANG=fr]": 6,
        "[LANG=zh]": 7,
        "[LANG=und]": 8,
        "good": 9,
        "bad": 10,
        "review": 11,
        "movie": 12,
        "very": 13,
        "long": 14,
        "text": 15,
        "nice": 16,
        "film": 17,
        "bon": 18,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[LANG=en]", "[LANG=fr]", "[LANG=zh]", "[LANG=und]"]}
    )
    return tokenizer
