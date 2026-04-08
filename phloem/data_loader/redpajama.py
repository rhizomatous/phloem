"""stream tokenized batches from RedPajama v2."""

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def stream_documents(
    snapshot: str = "2023-06",
    language: str = "en",
    max_docs: int | None = None,
) -> Iterator[str]:
    """stream raw text documents from RedPajama v2."""
    ds = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="default",
        partition="head_middle",
        snapshots=[snapshot],
        languages=[language],
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    for i, doc in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = doc.get("raw_content", "")
        if text:
            yield text


def stream_token_batches(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    seq_len: int = 1024,
    max_tokens: int | None = None,
    snapshot: str = "2023-06",
    language: str = "en",
    max_docs: int | None = None,
) -> Iterator[torch.Tensor]:
    """yield batches of token IDs (batch_size, seq_len) from RedPajama v2.

    concatenates documents with EOS separation into a single token stream,
    then chunks into fixed-length sequences and groups into batches.
    """
    tokens_yielded = 0
    buffer: list[int] = []
    batch: list[list[int]] = []
    eos_id = tokenizer.eos_token_id

    for text in stream_documents(
        snapshot=snapshot, language=language, max_docs=max_docs
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(token_ids)
        if eos_id is not None:
            buffer.append(eos_id)

        while len(buffer) >= seq_len:
            batch.append(buffer[:seq_len])
            buffer = buffer[seq_len:]

            if len(batch) == batch_size:
                tensor = torch.tensor(batch, dtype=torch.long)
                batch = []
                tokens_yielded += tensor.numel()
                yield tensor

                if max_tokens is not None and tokens_yielded >= max_tokens:
                    return
