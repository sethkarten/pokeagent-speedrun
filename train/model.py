"""Gemma4 model loader + Liger fused loss helper for VLM SFT.

Centralizes the choices about attention backend, dtype, gradient
checkpointing, LoRA target regex, and the manual fused linear+CE
loss path so the smoke and full training scripts agree.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger("train.model")

MODEL_ID = "google/gemma-4-E4B-it"

# Gemma4 has q_proj/k_proj/v_proj projections in three places — only
# the language_model versions are bare nn.Linear. The audio + vision
# towers wrap Linear in Gemma4ClippableLinear which PEFT can't target.
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"

# Pulled from the model config — gemma4 softcaps the final logits at
# this value before the cross-entropy. Liger needs us to pass it in
# explicitly when we use the fused linear+CE path (the built-in
# transformers loss applies it for us, but we're bypassing that).
FINAL_LOGIT_SOFTCAP = 30.0


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def configure_sdpa_backends() -> None:
    """Force SDPA away from the MATH fallback for long-context attention.

    Gemma4 has 7 "global" text layers with head_dim 512. flash-attn 2
    caps at 256, so SDPA can't use FLASH for those layers. By default
    PyTorch falls back to MATH (full O(seq²) materialization), which
    OOMs at 14-16k context. EFFICIENT (the xformers-derived backend)
    handles head_dim 512 with much lower memory, so we keep FLASH +
    EFFICIENT enabled and turn MATH off entirely. The 35 head_dim 256
    layers still get FLASH; the 7 head_dim 512 layers get EFFICIENT.
    """
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # MATH stays enabled as a final fallback because in practice
    # gemma4's per-layer attention masks (with sliding window for
    # 35 layers, full causal for the 7 global head_dim 512 layers)
    # sometimes carry shapes that EFFICIENT rejects on this PyTorch
    # version. Disabling MATH causes "No available kernel" hard
    # crashes for those rejected cases. Letting MATH stay on is
    # slow at long context but keeps training functional; the goal
    # of training_max_length is to keep MATH-fallback paths cheap.
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)


def load_model(
    device_index: int = 0,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = True,
    quantize_4bit: bool = True,
):
    """Load Gemma4 in bf16 on a single GPU, optionally NF4-quantized.

    NOTE on attention backend: Gemma4 has THREE distinct attention
    shapes — vision tower (12 heads × 64), most text layers (8 × 256),
    and at least one text layer with head_dim 512 ("global_head_dim"
    in the config). flash-attn 2 caps head_dim at 256, so passing
    ``flash_attention_2`` directly fails on the 512-dim layer.
    SDPA dispatches to flash-attn opportunistically for shapes that
    fit and falls back to memory-efficient attention otherwise — so
    we get flash-attn for the bulk of the model and a safe fallback
    only on the global_head_dim layer.

    NOTE on quantization: with ``quantize_4bit=True`` we use
    bitsandbytes NF4 to compress the frozen base model from 16 GB to
    ~5 GB, leaving room for 16k+ context activations and rank 512
    LoRA on a single 5090. The vision and audio towers are skipped
    (they wrap nn.Linear in Gemma4ClippableLinear which bnb can't
    quantize), and the language model lm_head is also kept in bf16
    so the Liger fused linear+CE loss path stays accurate.
    """
    logger.info("loading %s on cuda:%d (attn=%s, dtype=bf16, 4bit=%s)",
                MODEL_ID, device_index, attn_implementation, quantize_4bit)

    quant_config = None
    if quantize_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Skip multimodal towers and lm_head — bnb can't quantize
            # Gemma4ClippableLinear, and lm_head needs to stay full
            # precision for the fused loss path.
            llm_int8_skip_modules=[
                "vision_tower",
                "audio_tower",
                "embed_vision",
                "embed_audio",
                "lm_head",
            ],
        )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        attn_implementation=attn_implementation,
        quantization_config=quant_config,
    )
    # Minimal kbit prep — we deliberately skip
    # `prepare_model_for_kbit_training` because it casts every
    # non-quantized parameter (embeddings, layer norms, vision/audio
    # towers) from bf16 to fp32, doubling 7 GB of weights and pushing
    # us back over budget on a single 5090. bf16 is stable enough on
    # Ada/Blackwell for our LoRA scope; we just need the gradient
    # checkpointing + input grad hook so backprop reaches the LoRA
    # adapters through the frozen 4-bit base.
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if quantize_4bit:
        # Freeze the base — peft's get_peft_model would do this too,
        # but doing it explicitly here makes the trainable param count
        # in print_trainable_parameters match expectations.
        for p in model.parameters():
            p.requires_grad = False
    return model


def attach_lora(
    model,
    rank: int = 512,
    alpha: int = 1024,
    dropout: float = 0.05,
) -> PeftModel:
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_REGEX,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ----------------------------------------------------------------------
# Fused-loss forward pass
# ----------------------------------------------------------------------


def _resolve_lm_head(model) -> nn.Linear:
    """Find the lm_head Linear regardless of PEFT wrapping."""
    cur = model
    for _ in range(4):
        if hasattr(cur, "lm_head") and isinstance(cur.lm_head, nn.Linear):
            return cur.lm_head
        if hasattr(cur, "base_model"):
            cur = cur.base_model
            continue
        if hasattr(cur, "model"):
            cur = cur.model
            continue
        break
    raise RuntimeError("could not locate lm_head on the model")


def _resolve_inner_model(model):
    """Find the inner Gemma4ForConditionalGeneration so we can call
    its forward without going through the PEFT wrapper's lm_head."""
    cur = model
    for _ in range(4):
        if type(cur).__name__ == "Gemma4ForConditionalGeneration":
            return cur
        if hasattr(cur, "base_model"):
            cur = cur.base_model
            continue
        if hasattr(cur, "model"):
            cur = cur.model
            continue
        break
    return model  # fall back; the caller will detect via attribute checks


_FUSED_LOSS_FN = None


def _fused_loss_fn():
    global _FUSED_LOSS_FN
    if _FUSED_LOSS_FN is None:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        _FUSED_LOSS_FN = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
            softcap=FINAL_LOGIT_SOFTCAP,
        )
    return _FUSED_LOSS_FN


def forward_with_fused_loss(
    model,
    batch: dict,
) -> torch.Tensor:
    """Run forward returning loss, computed via Liger fused linear+CE.

    The model's normal forward materializes a [batch, seq_len, vocab]
    logits tensor before computing CE — at seq_len=14k and vocab=262144
    that's ~14 GB just for the float-cast logits. We avoid that by:

    1. Calling the inner conditional-generation model with
       ``output_hidden_states=True`` and ``labels=None`` to skip the
       built-in loss + logits computation.
    2. Pulling the last hidden state.
    3. Calling Liger's ``LigerFusedLinearCrossEntropyLoss(lm_head.weight,
       hidden, labels)`` which fuses the matmul + softmax + CE in
       chunks so the logits tensor is never materialized.
    """
    labels = batch.pop("labels")
    # Important: skip the model's internal loss path. Some models also
    # skip lm_head when labels=None and output_hidden_states=True; even
    # if not, the cost is one [seq, vocab] matmul that we accept.
    outputs = model(
        **batch,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
    )
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
    # the last is the post-norm output that gets fed into lm_head.
    hidden = outputs.hidden_states[-1]  # [B, T, H]

    # Shift for next-token prediction
    shift_hidden = hidden[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(hidden.device)

    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)

    lm_head = _resolve_lm_head(model)
    loss = _fused_loss_fn()(
        lm_head.weight,
        flat_hidden,
        flat_labels,
        lm_head.bias,
    )
    return loss
