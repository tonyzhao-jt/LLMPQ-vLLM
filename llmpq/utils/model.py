from typing import Tuple

from transformers import AutoModelForCausalLM  # old
from transformers import (AutoConfig, AutoTokenizer, BloomConfig, LlamaConfig,
                          OPTConfig, PretrainedConfig)


def get_h1_h2_from_config(model_config: PretrainedConfig) -> Tuple[int, int]:
    if isinstance(model_config, OPTConfig):
        return model_config.hidden_size, model_config.ffn_dim
    elif isinstance(model_config, BloomConfig):
        return model_config.hidden_size, model_config.hidden_size * 4
    else:
        return model_config.hidden_size, model_config.intermediate_size


def save_ckpt_dummy(model_id: str, save_path: str):
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # save
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
