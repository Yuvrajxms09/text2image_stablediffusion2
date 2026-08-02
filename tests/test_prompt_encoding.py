"""Tests for SDXL long-prompt encoding."""

import unittest

import torch

from ex_app.lib.prompt_encoding import encode_sdxl_prompt


class _FakeTokenizer:
    def __init__(self, model_max_length=6, extra_token=None):
        self.model_max_length = model_max_length
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.pad_token_id = 0
        self.extra_token = extra_token
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        input_ids = [ord(character) for character in prompt]
        if self.extra_token is not None:
            input_ids.append(self.extra_token)
        return {"input_ids": input_ids}

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.bos_token_id, *token_ids, self.eos_token_id]

    def num_special_tokens_to_add(self, pair):
        if pair:
            raise ValueError("Pair tokenization is not supported by this test tokenizer")
        return 2

    def prepare_for_model(
        self,
        token_ids,
        *,
        add_special_tokens,
        max_length,
        padding,
        truncation,
        return_attention_mask,
    ):
        if not add_special_tokens or padding != "max_length" or not truncation:
            raise ValueError("Unexpected token preparation options")
        if return_attention_mask:
            raise ValueError("Attention masks are not expected")

        prepared = self.build_inputs_with_special_tokens(token_ids)
        prepared.extend([self.pad_token_id] * (max_length - len(prepared)))
        return {"input_ids": prepared}


class _EncoderOutput:
    def __init__(self, hidden_states, first_output):
        self.hidden_states = hidden_states
        self.first_output = first_output

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.first_output


class _FakeTextEncoder:
    def __init__(self, embedding_size, returns_pooled_output=False):
        self.embedding_size = embedding_size
        self.returns_pooled_output = returns_pooled_output
        self.dtype = torch.float32
        self.input_ids = []

    def __call__(self, input_ids, output_hidden_states):
        self.input_ids.extend(input_ids.detach().cpu().tolist())
        hidden_state = input_ids.to(torch.float32).unsqueeze(-1)
        hidden_state = hidden_state.repeat(1, 1, self.embedding_size)
        first_output = hidden_state.sum(dim=1) if self.returns_pooled_output else hidden_state
        return _EncoderOutput(
            hidden_states=(hidden_state - 1, hidden_state, hidden_state + 1),
            first_output=first_output,
        )


class _FakePipeline:
    def __init__(self, tokenizer_2=None):
        self.tokenizer = _FakeTokenizer()
        self.tokenizer_2 = tokenizer_2 or _FakeTokenizer()
        self.text_encoder = _FakeTextEncoder(embedding_size=2)
        self.text_encoder_2 = _FakeTextEncoder(
            embedding_size=3,
            returns_pooled_output=True,
        )


class EncodeSdxlPromptTests(unittest.TestCase):
    def test_short_prompt_uses_one_window_for_both_encoders(self):
        pipe = _FakePipeline()

        conditioning = encode_sdxl_prompt(pipe, "ab", "cpu")

        expected_tokens = [101, ord("a"), ord("b"), 102, 0, 0]
        self.assertEqual(pipe.text_encoder.input_ids, [expected_tokens])
        self.assertEqual(pipe.text_encoder_2.input_ids, [expected_tokens])
        self.assertEqual(conditioning.prompt_embeds.shape, (1, 6, 5))
        self.assertEqual(conditioning.pooled_prompt_embeds.shape, (1, 3))

    def test_long_prompt_encodes_every_token_across_multiple_windows(self):
        pipe = _FakePipeline()

        conditioning = encode_sdxl_prompt(pipe, "abcdef", "cpu")

        self.assertEqual(
            pipe.text_encoder.input_ids,
            [
                [101, ord("a"), ord("b"), ord("c"), ord("d"), 102],
                [101, ord("e"), ord("f"), 102, 0, 0],
            ],
        )
        self.assertEqual(conditioning.prompt_embeds.shape, (1, 12, 5))

    def test_shorter_tokenization_is_padded_with_an_empty_window(self):
        pipe = _FakePipeline(tokenizer_2=_FakeTokenizer(extra_token=999))

        conditioning = encode_sdxl_prompt(pipe, "abcd", "cpu")

        self.assertEqual(
            pipe.text_encoder.input_ids,
            [
                [101, ord("a"), ord("b"), ord("c"), ord("d"), 102],
                [101, 102, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(len(pipe.text_encoder_2.input_ids), 2)
        self.assertEqual(conditioning.prompt_embeds.shape, (1, 12, 5))

    def test_prompt_is_passed_to_both_tokenizers_unchanged(self):
        pipe = _FakePipeline()
        prompt = "a + (b-c)"

        encode_sdxl_prompt(pipe, prompt, "cpu")

        self.assertEqual(pipe.tokenizer.prompts[0][0], prompt)
        self.assertEqual(pipe.tokenizer_2.prompts[0][0], prompt)
        expected_options = {
            "add_special_tokens": False,
            "truncation": False,
            "verbose": False,
        }
        self.assertEqual(pipe.tokenizer.prompts[0][1], expected_options)
        self.assertEqual(pipe.tokenizer_2.prompts[0][1], expected_options)

    def test_pooled_embeddings_come_from_the_first_window(self):
        pipe = _FakePipeline()

        conditioning = encode_sdxl_prompt(pipe, "abcdef", "cpu")

        first_window = torch.tensor(pipe.text_encoder_2.input_ids[0])
        expected = first_window.sum().repeat(3).reshape(1, 3).to(torch.float32)
        torch.testing.assert_close(conditioning.pooled_prompt_embeds, expected)

    def test_long_prompt_logs_conditioning_dimensions(self):
        pipe = _FakePipeline()

        with self.assertLogs("ex_app.lib.prompt_encoding", level="INFO") as captured:
            encode_sdxl_prompt(pipe, "abcdef", "cpu")

        self.assertIn("token_counts=(6, 6)", captured.output[0])
        self.assertIn("model_max_lengths=(6, 6)", captured.output[0])
        self.assertIn("chunk_count=2", captured.output[0])
        self.assertIn("prompt_embeds_shape=(1, 12, 5)", captured.output[0])
        self.assertIn("pooled_prompt_embeds_shape=(1, 3)", captured.output[0])
        self.assertIn("device=cpu", captured.output[0])


if __name__ == "__main__":
    unittest.main()
