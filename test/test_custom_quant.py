from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
import torch 
from typing import Optional, Dict, Any, List, Union


def layer_to_bitwidth_method(prefix: 'str', dynamic: Dict[str, Dict[str, Union[int, bool]]]):
    # prefix here is the layer name 
    bit_scheme = {
        # with GPTQMarlin24Config W4A16
        '4':{
            'weight_bits': 4,
            'group_size': 128,
        },
        # with CompressedTensorsConfig W8A8
        '8':{
            "target_scheme_map": {},
            "quant_format": str,
        }
    }
    q_cls = None 
    return q_cls.get_quant_method() # return the q cls based on their quant method.
    
@register_quantization_config("llmpq")
class PQQuantConfig(QuantizationConfig):

    def __init__(
        self,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
    ) -> None:
        '''
            PQ fully relies on the config to provide 
        '''
        super().__init__()
        self.dynamic = dynamic
    
    
    @classmethod
    def get_name(cls) -> str:
        return "llmpq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        cls(dynamic)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        assert self.dynamic, "PQ requires dynamic to be set"
        layer_to_bitwidth_method(prefix,
                                self.dynamic)
        