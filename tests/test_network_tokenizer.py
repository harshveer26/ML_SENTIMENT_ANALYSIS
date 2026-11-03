"""Optional smoke test for the real mDeBERTa tokenizer (tokenizer files only)."""

import os

import pytest

pytestmark = pytest.mark.network


@pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network tests are opt-in")
def test_real_mdeberta_tokenizer_handles_french_and_chinese() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=True)
    for sentence in ("Ce produit fonctionne très bien.", "这个产品非常好。"):
        encoded = tokenizer(sentence, add_special_tokens=True)
        assert len(encoded["input_ids"]) > 2
        assert encoded["input_ids"][0] == tokenizer.cls_token_id
        assert encoded["input_ids"][-1] == tokenizer.sep_token_id
