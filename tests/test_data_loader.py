"""tests for the data loading pipeline.

these tests use a mock tokenizer and mock document stream to verify
the batching/chunking logic without hitting HuggingFace or needing a GPU.
"""

from unittest.mock import patch

import torch

from phloem.data_loader.redpajama import stream_token_batches


class FakeTokenizer:
    """minimal tokenizer stub for testing."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 1 token per character, token ID = ord(char)
        return [ord(c) for c in text]


FAKE_DOCS = ["abcdefgh", "ijklmnop", "qrstuvwx", "yzabcdef"]


def _fake_stream_documents(snapshot="2023-06", language="en", max_docs=None):
    docs = FAKE_DOCS if max_docs is None else FAKE_DOCS[:max_docs]
    yield from docs


@patch("phloem.data_loader.redpajama.stream_documents", _fake_stream_documents)
def test_stream_token_batches_shape():
    """batches should be (batch_size, seq_len) tensors of token IDs."""
    tokenizer = FakeTokenizer()
    batches = list(stream_token_batches(tokenizer, batch_size=2, seq_len=4, max_tokens=100))
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (2, 4)
        assert batch.dtype == torch.long


@patch("phloem.data_loader.redpajama.stream_documents", _fake_stream_documents)
def test_stream_token_batches_max_tokens():
    """should stop yielding after max_tokens is reached."""
    tokenizer = FakeTokenizer()
    batches = list(stream_token_batches(tokenizer, batch_size=1, seq_len=4, max_tokens=8))
    total_tokens = sum(b.numel() for b in batches)
    assert total_tokens >= 8
    # should not massively overshoot (at most one extra batch)
    assert total_tokens <= 12


@patch("phloem.data_loader.redpajama.stream_documents", _fake_stream_documents)
def test_stream_token_batches_content():
    """tokens should be actual character ordinals from the fake docs."""
    tokenizer = FakeTokenizer()
    batches = list(stream_token_batches(tokenizer, batch_size=1, seq_len=4, max_tokens=100))
    first_batch = batches[0]
    # first 4 tokens should be ord('a'), ord('b'), ord('c'), ord('d')
    expected = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("d")]], dtype=torch.long)
    assert torch.equal(first_batch, expected)


@patch("phloem.data_loader.redpajama.stream_documents", _fake_stream_documents)
def test_stream_token_batches_eos_inserted():
    """EOS tokens (id=0) should appear between documents in the token stream."""
    tokenizer = FakeTokenizer()
    batches = list(stream_token_batches(tokenizer, batch_size=1, seq_len=32, max_tokens=100))
    all_tokens = torch.cat(batches, dim=0).flatten().tolist()
    # EOS token (0) should appear in the stream between docs
    assert 0 in all_tokens
