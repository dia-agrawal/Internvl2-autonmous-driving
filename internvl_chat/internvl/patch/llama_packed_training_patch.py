# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
# Try to import flash_attn, but don't hard-fail if missing
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None

from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaFlashAttention2,
)


# Modified from transformers.models.llama.modeling_llama.LlamaFlashAttention2
class LlamaFlashAttention2ForPackedTraining(LlamaFlashAttention2):

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """Calls the forward method of Flash Attention.

        If `flash_attn` is not available, fall back to the parent (eager) implementation.
        """
        # If flash_attn isn't available, use the parent implementation (eager attention)
        if not HAS_FLASH_ATTN:
            return super()._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                dropout,
                softmax_scale,
                use_sliding_windows,
            )

        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0)

        with torch.no_grad():
            max_seqlen = max([
                cu_seqlens[idx+1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]).item()

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        if not use_sliding_windows:
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
        else:
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
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output


def replace_llama_attention_class():
    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaFlashAttention2ForPackedTraining
    print('Replace LLAMA_ATTENTION_CLASSES to support packed training!!')
