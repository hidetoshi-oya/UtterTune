from __future__ import annotations
from typing import Callable, Any

import torch


def apply_flash_attention_patch():
    """Patch Qwen2Encoder to use SDPA for faster inference.

    PyTorch 2.0+ SDPA automatically selects the best backend:
    - Flash Attention (if available)
    - Memory Efficient Attention
    - Math fallback

    Note: torch.compile is NOT applied here because CosyVoice loads weights
    via load_state_dict() AFTER __init__, and torch.compile changes the
    model's state_dict keys (adds _orig_mod prefix), causing key mismatches.

    No additional packages required.
    """
    try:
        from cosyvoice.llm import llm
        from transformers import Qwen2ForCausalLM
    except Exception as e:
        raise RuntimeError(f"[patch] failed to import cosyvoice llm: {e}")

    if getattr(llm.Qwen2Encoder, "_flash_attention_patched", False):
        return

    def patched_init(self, pretrain_path):
        torch.nn.Module.__init__(self)
        self.model = Qwen2ForCausalLM.from_pretrained(
            pretrain_path,
            attn_implementation="sdpa",
        )

    llm.Qwen2Encoder.__init__ = patched_init
    llm.Qwen2Encoder._flash_attention_patched = True
    print("[patch] Patched Qwen2Encoder to use SDPA")


def apply_patch():
    try:
        from cosyvoice.utils import frontend_utils as fu
    except Exception as e:
        raise RuntimeError(f"[patch] failed to import cosyvoice frontend_utils: {e}")

    if getattr(fu, "_split_paragraph_patched", False):
        return

    original: Callable[..., Any] = fu.split_paragraph

    def split_paragraph_patched(
        text,
        tokenizer_encode,
        lang,
        *,
        token_max_n=80,
        token_min_n=60,
        merge_len=20,
        comma_split=False,
    ):
        if lang in ("zh", "ja"):
            token_max_n = 300
            merge_len = 300
        return original(
            text,
            tokenizer_encode,
            lang,
            token_max_n=token_max_n,
            token_min_n=token_min_n,
            merge_len=merge_len,
            comma_split=comma_split,
        )

    fu.split_paragraph = split_paragraph_patched
    fu._split_paragraph_patched = True
    print(
        "[patch] Patched cosyvoice.utils.frontend_utils.split_paragraph for zh/ja "
        "(token_max_n=300, merge_len=300)"
    )

    # Apply Flash Attention 2 patch
    apply_flash_attention_patch()
