from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig)
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override)
from vllm.logger import init_logger

import torch 
from typing import Optional, Dict, Any, List, Union

logger = init_logger(__name__)

def layer_to_bitwidth_method(dynamic_value: Dict[str, Dict[str, Union[int, bool]]]):
    # prefix here is the layer name 
    bit_scheme = {
        # with GPTQMarlin24Config W4A16
        '4':{
            'weight_bits': 4,
            'group_size': 128,
        },
        # with CompressedTensorsConfig W8A8
        '8':{
            "ignore": ['lm_head'],
            "sparsity_scheme_map": {},
            "sparsity_ignore_list": [],
            'quant_format': 'int-quantized',
            "target_scheme_map": 
            {
                'Linear': {
                    'weights': QuantizationArgs(
                        num_bits=8, 
                        type='int', 
                        symmetric=True, 
                        group_size=None, 
                        trategy='channel', 
                        block_structure=None, 
                        dynamic=False, 
                        actorder=None, 
                        observer='minmax', 
                        observer_kwargs={}
                    ), 
                  'input_activations': QuantizationArgs(
                        num_bits=8, 
                        type='int', 
                        symmetric=True, 
                        group_size=None, 
                        strategy='token', 
                        block_structure=None, 
                        dynamic=True, 
                        actorder=None, 
                        observer=None, 
                        observer_kwargs={}
                    )
                }
            }
        }
    }
    q_cls = None 
    bit = 16
    if 'bit' in dynamic_value:
        bit = dynamic_value['bit']
    
    if bit == 16:
        q_cls = UnquantizedLinearMethod()
    elif bit == 8:
        q_cls = CompressedTensorsConfig(**bit_scheme["8"])
    elif bit == 4:
        q_cls = GPTQMarlin24Config(**bit_scheme["4"])
    else:
        raise ValueError(f"Unsupported bitwidth {bit}")
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
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantization_config.json"]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        return cls(dynamic)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        assert self.dynamic, "PQ requires dynamic to be set"
        dynamic_value = get_dynamic_override(self, prefix)
        logger.info(f'assign layer {prefix}, dynamic_value: {dynamic_value}')
        return layer_to_bitwidth_method(dynamic_value)


if __name__ == '__main__':
    # SPDX-License-Identifier: Apache-2.0
    from vllm import LLM, SamplingParams
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    llm = LLM(model="/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded_pq", quantization='llmpq') # try cpu offload
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        tkn_num = len(output.outputs[0].token_ids)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}")