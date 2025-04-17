from typing import Tuple

from transformers import (BloomConfig, LlamaConfig, OPTConfig,  # old
                          PretrainedConfig)


def get_h1_h2_from_config(model_config: PretrainedConfig) -> Tuple[int, int]:
    if isinstance(model_config, OPTConfig):
        return model_config.hidden_size, model_config.ffn_dim
    elif isinstance(model_config, BloomConfig):
        return model_config.hidden_size, model_config.hidden_size * 4
    elif isinstance(model_config, LlamaConfig):
        return model_config.hidden_size, model_config.intermediate_size
    else:
        raise NotImplementedError
