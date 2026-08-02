"""Encode SDXL prompts in tokenizer-sized chunks."""

import logging
from dataclasses import dataclass
from typing import Any

import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptConditioning:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


@torch.no_grad()
def encode_sdxl_prompt(pipe: Any, prompt: str, device: str) -> PromptConditioning:
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    prompt_tokens = _tokenize_prompt(tokenizer, prompt)
    prompt_tokens_2 = _tokenize_prompt(tokenizer_2, prompt)
    chunk_count = max(
        _required_chunk_count(tokenizer, prompt_tokens),
        _required_chunk_count(tokenizer_2, prompt_tokens_2),
    )

    first_chunks = _build_token_chunks(tokenizer, prompt_tokens, chunk_count)
    second_chunks = _build_token_chunks(tokenizer_2, prompt_tokens_2, chunk_count)
    first_prompt_embeds = _encode_token_chunks(
        pipe.text_encoder,
        first_chunks,
        device,
    )
    second_prompt_embeds, pooled_prompt_embeds = _encode_token_chunks(
        pipe.text_encoder_2,
        second_chunks,
        device,
        return_pooled=True,
    )

    if pooled_prompt_embeds is None:
        raise RuntimeError("SDXL text encoders did not return pooled prompt embeddings")

    dtype = pipe.text_encoder_2.dtype
    conditioning = PromptConditioning(
        prompt_embeds=torch.cat((first_prompt_embeds, second_prompt_embeds), dim=-1).to(device=device, dtype=dtype),
        pooled_prompt_embeds=pooled_prompt_embeds.to(device=device, dtype=dtype),
    )
    log_level = logging.INFO if chunk_count > 1 else logging.DEBUG
    logger.log(
        log_level,
        "Encoded SDXL prompt: token_counts=%s model_max_lengths=%s chunk_count=%d "
        "prompt_embeds_shape=%s pooled_prompt_embeds_shape=%s device=%s",
        (len(prompt_tokens), len(prompt_tokens_2)),
        (tokenizer.model_max_length, tokenizer_2.model_max_length),
        chunk_count,
        tuple(conditioning.prompt_embeds.shape),
        tuple(conditioning.pooled_prompt_embeds.shape),
        device,
    )
    return conditioning


def _tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
    return tokenized["input_ids"]


def _required_chunk_count(tokenizer: Any, token_ids: list[int]) -> int:
    payload_length = _chunk_payload_length(tokenizer)
    return max(1, (len(token_ids) + payload_length - 1) // payload_length)


def _chunk_payload_length(tokenizer: Any) -> int:
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    payload_length = tokenizer.model_max_length - special_token_count
    if payload_length < 1:
        raise ValueError("Tokenizer model maximum length leaves no room for prompt tokens")
    return payload_length


def _build_token_chunks(
    tokenizer: Any,
    token_ids: list[int],
    chunk_count: int,
) -> list[list[int]]:
    payload_length = _chunk_payload_length(tokenizer)
    payloads = [token_ids[offset : offset + payload_length] for offset in range(0, len(token_ids), payload_length)]
    if not payloads:
        payloads.append([])
    payloads.extend([[] for _ in range(chunk_count - len(payloads))])

    return [
        tokenizer.prepare_for_model(
            payload,
            add_special_tokens=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
        )["input_ids"]
        for payload in payloads
    ]


def _encode_token_chunks(
    text_encoder: Any,
    chunks: list[list[int]],
    device: str,
    return_pooled: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    input_ids = torch.tensor(chunks, dtype=torch.long, device=device)
    encoder_output = text_encoder(input_ids, output_hidden_states=True)
    hidden_states = encoder_output.hidden_states[-2]
    prompt_embeds = hidden_states.reshape(
        1,
        hidden_states.shape[0] * hidden_states.shape[1],
        hidden_states.shape[2],
    )
    if not return_pooled:
        return prompt_embeds

    # SDXL uses the pooled output from the final text encoder.
    pooled_prompt_embeds = encoder_output[0][:1] if encoder_output[0].ndim == 2 else None
    return prompt_embeds, pooled_prompt_embeds
