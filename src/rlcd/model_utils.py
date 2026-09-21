"""Model loading and batched chat generation, shared by eval.py and
generate_preferences.py.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal-LM generation

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    conversations: list[list[dict]],
    max_new_tokens: int = 40,
    do_sample: bool = False,
    temperature: float = 1.0,
    batch_size: int = 16,
) -> list[str]:
    """conversations: list of chat-message lists (system+user). Returns the
    decoded completion text (assistant turn only) for each conversation.
    """
    outputs: list[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = model.generate(**enc, **gen_kwargs)
        new_tokens = out_ids[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs
