"""Lossless byte-level tokenization and public package contracts."""
from __future__ import annotations

import pytest

import apedp_mmlm
from apedp_mmlm import BPEMMLMTokenizer


@pytest.mark.parametrize("text", ["", "Mixed CASE  \t\n", "\u03a8 \U0001f600 caf\u00e9", "<bos>\x00<0xFF>"])
@pytest.mark.parametrize("trained", [False, True])
def test_round_trip_preserves_exact_utf8(text, trained):
    tokenizer = BPEMMLMTokenizer(max_vocab_size=280)
    if trained:
        tokenizer.fit(["repeated words repeated words"] * 2)
    for special in (False, True):
        ids = tokenizer.encode(text, add_special_tokens=special)
        assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)
        assert tokenizer.decode(ids) == text


def test_merges_compress_and_are_deterministic_under_corpus_reordering():
    corpus = ["banana banana", "bandana banana"] * 3
    first = BPEMMLMTokenizer(max_vocab_size=280).fit(corpus)
    second = BPEMMLMTokenizer(max_vocab_size=280).fit(list(reversed(corpus)))
    assert first.merges == second.merges
    assert first.token_to_id == second.token_to_id
    assert len(first.encode(corpus[0], add_special_tokens=False)) < len(corpus[0].encode())
    assert first.vocab_size <= 280
    assert first.decode(first.encode(corpus[0])) == corpus[0]


def test_refitting_resets_the_learned_vocabulary():
    tokenizer = BPEMMLMTokenizer(max_vocab_size=270).fit(["aaaaaa"] * 3)
    assert tokenizer.merges
    tokenizer.fit([])
    assert tokenizer.merges == []
    assert tokenizer.vocab_size == 260
    assert tokenizer.decode(tokenizer.encode("aaaaaa")) == "aaaaaa"


def test_minimum_vocabulary_and_frequency_are_enforced():
    with pytest.raises(ValueError, match="260"):
        BPEMMLMTokenizer(max_vocab_size=259)
    with pytest.raises(ValueError, match="positive"):
        BPEMMLMTokenizer(min_frequency=0)
    tokenizer = BPEMMLMTokenizer(max_vocab_size=260).fit(["aaaaaa"])
    assert tokenizer.merges == []
    assert tokenizer.vocab_size == 260
    assert tokenizer.decode(tokenizer.encode("unseen \u03a8")) == "unseen \u03a8"


def test_frequency_threshold_and_special_token_decoding():
    tokenizer = BPEMMLMTokenizer(min_frequency=100).fit(["abc"])
    assert tokenizer.merges == []
    assert tokenizer.decode(tokenizer.encode("abc"), skip_special_tokens=False) == "<bos>abc<eos>"
    with pytest.raises(ValueError, match="Unknown token ID"):
        tokenizer.decode([tokenizer.vocab_size])


def test_all_declared_package_exports_resolve():
    assert all(hasattr(apedp_mmlm, name) for name in apedp_mmlm.__all__)
    for name in ("ZXDiagram", "mmlm_zx_t_count", "optimize_clifford_plus_t_zx", "BPEMMLMTokenizer"):
        assert name in apedp_mmlm.__all__
