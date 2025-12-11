#!/usr/bin/env python3
"""
LoRA Merge and vLLM Export Script for CosyVoice2
=================================================

This script merges LoRA adapter weights into the base CosyVoice2 model
and exports it in vLLM-compatible format for faster inference.

Usage:
    python -m scripts.cv2.merge_lora_for_vllm \
        --base_model pretrained_models/CosyVoice2-0.5B \
        --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
        --output_dir pretrained_models/CosyVoice2-0.5B/vllm_merged

After running this script, start the server with:
    python -m scripts.cv2.server.main \
        --base_model pretrained_models/CosyVoice2-0.5B \
        --vllm_model_dir pretrained_models/CosyVoice2-0.5B/vllm_merged \
        --port 6333 --trt --jit --fp16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import safetensors.torch as st
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Apply patch before importing CosyVoice
from scripts.cv2.patch import apply_patch
apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer


def merge_lora_and_export_vllm(
    base_model_dir: str,
    lora_dir: str,
    output_dir: str,
    device: str = "cuda",
):
    """
    Merge LoRA adapter into base model and export for vLLM.

    Args:
        base_model_dir: Path to CosyVoice2 base model
        lora_dir: Path to LoRA adapter directory
        output_dir: Output directory for vLLM model
        device: Device to use for processing
    """
    output_path = Path(output_dir)

    # Check if output already exists
    if output_path.exists() and (output_path / "config.json").exists():
        logger.warning(f"Output directory already exists: {output_dir}")
        logger.warning("Delete it first if you want to regenerate.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return
        # Remove existing directory
        import shutil
        shutil.rmtree(output_path)

    logger.info("=" * 60)
    logger.info("LoRA Merge and vLLM Export")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_dir}")
    logger.info(f"LoRA dir:   {lora_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("-" * 60)

    # 1. Load base CosyVoice2 model (without vLLM/JIT/TRT)
    logger.info("Step 1: Loading base CosyVoice2 model...")
    cv2 = CosyVoice2(base_model_dir, fp16=False)
    base_llm = cv2.model.llm
    logger.info("Base model loaded.")

    # 2. Expand tokenizer with special tokens
    logger.info("Step 2: Expanding tokenizer with special tokens...")
    tok = get_qwen_tokenizer(
        token_path=f"{base_model_dir}/CosyVoice-BlankEN",
        skip_special_tokens=True
    )
    new_tokens = ["<PHON_START>", "<PHON_END>"]
    num_added = tok.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )
    logger.info(f"Added {num_added} special tokens")

    # 3. Resize model embeddings
    logger.info("Step 3: Resizing model embeddings...")
    original_vocab_size = base_llm.llm.model.config.vocab_size
    base_llm.llm.model.resize_token_embeddings(len(tok.tokenizer))
    new_vocab_size = len(tok.tokenizer)
    new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)
    logger.info(f"Vocab size: {original_vocab_size} -> {new_vocab_size}")
    logger.info(f"New token IDs: {new_ids}")

    # 4. Load LoRA adapter
    logger.info("Step 4: Loading LoRA adapter...")
    peft_config = PeftConfig.from_pretrained(lora_dir)
    peft_config.task_type = None  # Required for CosyVoice2

    peft_model = PeftModel.from_pretrained(
        base_llm,
        lora_dir,
        config=peft_config,
        is_trainable=False,
        torch_dtype=torch.float32,
    )
    peft_model.to(device).eval()
    logger.info("LoRA adapter loaded.")

    # 5. Patch special token embeddings
    logger.info("Step 5: Patching special token embeddings...")
    embed_path = Path(lora_dir) / "embed_patch.safetensors"
    if not embed_path.exists():
        raise FileNotFoundError(f"embed_patch.safetensors not found in {lora_dir}")

    rows = st.load_file(str(embed_path))["embed_rows"].to(device)
    with torch.no_grad():
        embed_weight = peft_model.base_model.llm.model.get_input_embeddings().weight
        rows = rows.to(dtype=embed_weight.dtype)
        embed_weight[new_ids] = rows
    logger.info("Special token embeddings patched.")

    # 6. Merge LoRA weights into base model
    logger.info("Step 6: Merging LoRA weights into base model...")
    merged_llm = peft_model.merge_and_unload()
    logger.info("LoRA weights merged. Model is now a regular Qwen2LM.")

    # 7. Update speech_embedding to include new tokens
    logger.info("Step 7: Expanding speech_embedding for new tokens...")
    old_speech_emb = merged_llm.speech_embedding
    old_num_embeddings = old_speech_emb.num_embeddings
    embedding_dim = old_speech_emb.embedding_dim

    # Create new speech_embedding with expanded vocabulary
    # The new tokens should map to the same IDs in speech space
    new_num_embeddings = old_num_embeddings + len(new_tokens)
    new_speech_emb = torch.nn.Embedding(new_num_embeddings, embedding_dim)

    with torch.no_grad():
        # Copy old embeddings
        new_speech_emb.weight[:old_num_embeddings] = old_speech_emb.weight
        # Initialize new token embeddings (copy from text embeddings)
        text_embed_weight = merged_llm.llm.model.get_input_embeddings().weight
        new_speech_emb.weight[old_num_embeddings:] = text_embed_weight[new_ids].to(
            dtype=new_speech_emb.weight.dtype
        )

    merged_llm.speech_embedding = new_speech_emb.to(device)
    logger.info(f"speech_embedding expanded: {old_num_embeddings} -> {new_num_embeddings}")

    # 8. Export to vLLM format
    logger.info("Step 8: Exporting to vLLM format...")
    export_cosyvoice2_vllm_merged(merged_llm, output_dir, torch.device(device))

    # 9. Save tokenizer with special tokens
    logger.info("Step 9: Saving tokenizer with special tokens...")
    tok.tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")
    logger.info(f"Tokenizer vocab size: {len(tok.tokenizer)}")
    logger.info(f"Special token IDs: {tok.tokenizer.convert_tokens_to_ids(new_tokens)}")

    logger.info("=" * 60)
    logger.info(f"vLLM model exported to: {output_dir}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To use the merged model, start the server with:")
    logger.info(f"  python -m scripts.cv2.server.main \\")
    logger.info(f"      --base_model {base_model_dir} \\")
    logger.info(f"      --vllm_model_dir {output_dir} \\")
    logger.info(f"      --port 6333 --trt --jit --fp16")


def export_cosyvoice2_vllm_merged(model, model_path: str, device: torch.device):
    """
    Export merged CosyVoice2 model for vLLM.

    Modified version of export_cosyvoice2_vllm that handles merged LoRA models.
    """
    import os

    output_path = Path(model_path)
    output_path.mkdir(parents=True, exist_ok=True)

    DEFAULT_VOCAB_PADDING_SIZE = 64
    pad_to = DEFAULT_VOCAB_PADDING_SIZE
    vocab_size = model.speech_embedding.num_embeddings
    feature_size = model.speech_embedding.embedding_dim
    pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to

    # Get the original llm_decoder size (may be smaller than vocab_size if new tokens were added)
    orig_decoder_size = model.llm_decoder.weight.shape[0]

    logger.info(f"speech_embedding vocab: {vocab_size}, llm_decoder vocab: {orig_decoder_size}, Padded: {pad_vocab_size}")

    dtype = torch.bfloat16

    # lm_head - expand llm_decoder to match speech_embedding vocab size
    # Note: bias=False because vLLM's Qwen2ForCausalLM doesn't support lm_head.bias
    # The bias from CosyVoice2's llm_decoder is absorbed into the weights as a trade-off
    new_lm_head = torch.nn.Linear(
        in_features=feature_size,
        out_features=pad_vocab_size,
        bias=False
    )
    with torch.no_grad():
        # Copy original decoder weights
        new_lm_head.weight[:orig_decoder_size] = model.llm_decoder.weight
        # Initialize new token weights (if vocab was expanded)
        if vocab_size > orig_decoder_size:
            new_lm_head.weight[orig_decoder_size:vocab_size] = 0
            logger.info(f"Initialized {vocab_size - orig_decoder_size} new decoder tokens")
        # Zero out padding
        new_lm_head.weight[vocab_size:] = 0
    model.llm.model.lm_head = new_lm_head
    logger.info("lm_head created without bias (vLLM Qwen2ForCausalLM compatibility)")

    # embed_tokens - replace with speech_embedding
    embed_tokens_backup = model.llm.model.model.embed_tokens
    new_codec_embed = torch.nn.Embedding(pad_vocab_size, feature_size)
    with torch.no_grad():
        new_codec_embed.weight[:vocab_size] = model.speech_embedding.weight
        new_codec_embed.weight[vocab_size:] = 0
    model.llm.model.set_input_embeddings(new_codec_embed)

    # Convert to BF16
    model.llm.model.to(device)
    model.llm.model.to(dtype)

    # Save config modifications
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings

    # Clean up generation config
    if hasattr(model.llm.model, 'generation_config'):
        if hasattr(model.llm.model.generation_config, 'eos_token_id'):
            del model.llm.model.generation_config.eos_token_id
    if hasattr(model.llm.model.config, 'bos_token_id'):
        del model.llm.model.config.bos_token_id
    if hasattr(model.llm.model.config, 'eos_token_id'):
        del model.llm.model.config.eos_token_id

    model.llm.model.config.vocab_size = pad_vocab_size
    model.llm.model.config.tie_word_embeddings = False

    # Save the model
    logger.info(f"Saving model to {model_path}...")
    model.llm.model.save_pretrained(model_path)

    # Patch config.json to use Qwen2ForCausalLM (built-in vLLM model)
    # Note: We use Qwen2ForCausalLM instead of custom CosyVoice2ForCausalLM
    # because vLLM 0.12.x uses multiprocessing (spawn) which prevents custom
    # model registration from persisting to worker processes.
    config_path = output_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['architectures'] = ['Qwen2ForCausalLM']
        config['model_type'] = 'qwen2'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Updated config.json with Qwen2ForCausalLM (vLLM built-in)")

    # Restore original config values (in case model is used later)
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.set_input_embeddings(embed_tokens_backup)

    logger.info("Export complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export for vLLM"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to CosyVoice2 base model directory"
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for vLLM model (default: {base_model}/vllm_merged)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = f"{args.base_model}/vllm_merged"

    merge_lora_and_export_vllm(
        base_model_dir=args.base_model,
        lora_dir=args.lora_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
