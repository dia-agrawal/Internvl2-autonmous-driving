# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from internvl.model.phi3.configuration_phi3 import Phi3Config
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):
        super().__init__(**kwargs)

        # Vision config
        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info(
                'vision_config is None. Initializing the InternVisionConfig with default values.'
            )

        # LLM config
        if llm_config is None:
            # Default to Qwen2ForCausalLM for InternVL3-1B
            llm_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info(
                'llm_config is None. Initializing the Qwen2Config config with default values (`Qwen2ForCausalLM`).'
            )

        self.vision_config = InternVisionConfig(**vision_config)

        # Robust architecture dispatch
        arch_list = None
        if isinstance(llm_config, dict):
            arch_list = llm_config.get('architectures')
        arch = arch_list[0] if arch_list and len(arch_list) > 0 else None

        if arch == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif arch == 'InternLM2ForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif arch == 'Phi3ForCausalLM':
            self.llm_config = Phi3Config(**llm_config)
        elif arch == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            # architectures missing or unknown: attempt to instantiate known config classes
            instanced = False
            last_exc = None
            for cfg_cls in (LlamaConfig, InternLM2Config, Phi3Config, Qwen2Config):
                try:
                    self.llm_config = cfg_cls(**llm_config)
                    instanced = True
                    break
                except Exception as e:
                    last_exc = e

            if not instanced:
                raise ValueError(
                    f"Unsupported or malformed LLM config architectures: {arch!r}. "
                    f"Tried known config classes and failed. Last error: {last_exc}"
                )

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
