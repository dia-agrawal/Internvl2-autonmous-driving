# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from internvl.model.internlm2.modeling_internlm2 import (
    INTERNLM2_ATTENTION_CLASSES, InternLM2FlashAttention2,
    apply_rotary_pos_emb)

# Try to import flash_attn, but don't hard-fail if missing
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None


# Modified from internvl.model.internlm2.modeling_internlm2.InternLM2FlashAttention2
class InternLM2FlashAttention2ForPackedTraining(InternLM2FlashAttention2):

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        """

        # If flash_attn isn't available, just use the parent implementation (eager attention)
        if not HAS_FLASH_ATTN:
            return super()._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                dropout,
                softmax_scale,
            )

        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)

        with torch.no_grad():
            max_seqlen = max(
                [
                    cu_seqlens[idx + 1] - cu_seqlens[idx]
                    for idx in range(cu_seqlens.size(0) - 1)
                ]
            ).item()

        causal = self.is_causal and query_length != 1

        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        # restore batch dim for the caller
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output


def replace_internlm2_attention_class():
    INTERNLM2_ATTENTION_CLASSES["flash_attention_2"] = InternLM2FlashAttention2ForPackedTraining
    print("Replace INTERNLM2_ATTENTION_CLASSES to support packed training!!")
