# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from .internlm2_packed_training_patch import replace_internlm2_attention_class

from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit
#from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import (concat_pad_data_collator,
                                dpo_concat_pad_data_collator,
                                pad_data_collator)
from .phi3_packed_training_patch import replace_phi3_attention_class
from .qwen2_packed_training_patch import replace_qwen2_attention_class
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

# Optional flash-attn support
try:
    import flash_attn  # noqa: F401
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    
if _HAS_FLASH_ATTN:
    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
    from .llama_packed_training_patch import replace_llama_attention_class
else:
    def replace_llama2_attn_with_flash_attn(*args, **kwargs):
        print("flash_attn not found: skipping LLaMA2 flash-attn monkey patch (using eager attention).")

    def replace_llama_attention_class(*args, **kwargs):
        print("flash_attn not found: skipping LLaMA packed training patch (using eager attention).")


__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm',
           'replace_llama2_attn_with_flash_attn',
           'replace_train_sampler',
           'replace_train_dataloader',
           'replace_internlm2_attention_class',
           'replace_qwen2_attention_class',
           'replace_phi3_attention_class',
           'replace_llama_attention_class',
           'pad_data_collator',
           'dpo_concat_pad_data_collator',
           'concat_pad_data_collator',
           'apply_liger_kernel_to_internvit']
