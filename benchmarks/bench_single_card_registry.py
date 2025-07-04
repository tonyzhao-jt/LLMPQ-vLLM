from typing import Any, Dict, List, Optional, Union

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)  # noqa: E501
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)


def layer_to_config(
    prefix: str,
    dynamic_value: Dict[str, Dict[str, Union[int, bool]]],
) -> Optional[QuantizationConfig]:
    # prefix here is the layer name
    bit_scheme = {
        # with GPTQMarlin24Config W4A16
        "4": {
            "weight_bits": 4,
            "group_size": 128,
            # for n GPTQ24
            "desc_act": True,
            "is_sym": True,
            "lm_head_quantized": False,
            "dynamic": {},
            "full_config": {},
        },
        "8": {
            "weight_bits": 8,
            "group_size": 128,
            # for n GPTQ24
            "desc_act": True,
            "is_sym": True,
            "lm_head_quantized": False,
            "dynamic": {},
            "full_config": {},
        },
        # with CompressedTensorsConfig W8A8
        "8-tc": {
            "ignore": ["lm_head"],
            "sparsity_scheme_map": {},
            "sparsity_ignore_list": [],
            "quant_format": "int-quantized",
            "target_scheme_map": {
                "Linear": {
                    "weights": QuantizationArgs(
                        num_bits=8,
                        type="int",
                        symmetric=True,
                        group_size=None,
                        trategy="channel",
                        block_structure=None,
                        dynamic=False,
                        actorder=None,
                        observer="minmax",
                        observer_kwargs={},
                    ),
                    "input_activations": QuantizationArgs(
                        num_bits=8,
                        type="int",
                        symmetric=True,
                        group_size=None,
                        strategy="token",
                        block_structure=None,
                        dynamic=True,
                        actorder=None,
                        observer=None,
                        observer_kwargs={},
                    ),
                }
            },
        },
    }
    q_cls = None
    bit = 16
    if "bits" in dynamic_value:
        bit = dynamic_value["bits"]

    from vllm.platforms import current_platform

    capability_tuple = current_platform.get_device_capability()
    cap = capability_tuple.to_int()
    scheme = bit_scheme[str(bit)]
    gptq_cls = GPTQConfig
    if bit in [4, 8]:
        if cap >= 80:
            gptq_cls = GPTQMarlinConfig
        else:
            scheme.pop("is_sym")
            scheme.pop("full_config")
            # scheme['desc_act'] = False

    if bit == 16:
        return None
    elif bit == "8-tc":
        q_cls = CompressedTensorsConfig(**scheme)
    elif bit == 8:
        q_cls = gptq_cls(**scheme)
    elif bit == 4:
        q_cls = gptq_cls(**scheme)
    else:
        raise ValueError(f"Unsupported bitwidth {bit}")

    return q_cls


@register_quantization_config("llmpq")
class PQQuantConfig(QuantizationConfig):
    def __init__(
        self,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        prepost_bit: int,
    ) -> None:
        """
        PQ fully relies on the config to provide
        """
        super().__init__()
        self.dynamic = dynamic
        self.prepost_bit = prepost_bit

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
        prepost_bit = cls.get_from_keys_or(config, ["prepost_bit"], 16)
        return cls(dynamic, prepost_bit)

    # from compressed-tensors
    def get_cache_scale(self, name: str) -> Optional[str]:
        q_cls = self.layer_to_qconfig(name)
        print(f"18 {name}, {q_cls}, {q_cls.get_cache_scale(name)}")
        return q_cls.get_cache_scale(name) if q_cls else None

    def layer_to_qconfig(self, layer_name: str):
        if (
            get_dynamic_override(  # noqa: E712
                self, layer_name=layer_name  # noqa: E712
            )
            == False
        ):  # noqa: E712
            return None

        if layer_name:
            if "lm_head" in layer_name:
                return None
            dynamic_value = get_dynamic_override(self, layer_name)
            if not dynamic_value:
                return None
            logger.info(f"check layer {layer_name} {dynamic_value}")
            return layer_to_config(layer_name, dynamic_value)
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        assert self.dynamic, "PQ requires dynamic to be set"
        logger.info(f"{prefix}")
        if prefix and "embed" in prefix:
            return UnquantizedEmbeddingMethod()

        if prefix:
            if "lm_head" in prefix:
                return UnquantizedLinearMethod()
            q_cls = self.layer_to_qconfig(prefix)
            if q_cls:
                return q_cls.get_quant_method(layer, prefix)
            else:
                return UnquantizedLinearMethod()
        return None


import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to the sharegpt/sonnet dataset. "
    "Or the huggingface dataset ID if using HF dataset.",
)
parser.add_argument(
    "--max-concurrency",
    type=int,
    default=1,
    help="Maximum number of concurrent requests. This can be used "
    "to help simulate an environment where a higher level component "
    "is enforcing a maximum number of concurrent requests. While the "
    "--request-rate argument controls the rate at which requests are "
    "initiated, this argument will control how many are actually allowed "
    "to execute at a time. This means that when used in combination, the "
    "actual request rate may be lower than specified with --request-rate, "
    "if the server is not processing requests fast enough to keep up.",
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of the model.",
)

parser.add_argument(
    "--quantization",
    type=str,
    default=None,
    help="Name of the model.",
)

parser.add_argument(
    "--use-llmpq",
    action="store_true",
)

# cpu_offload_gb
parser.add_argument(
    "--cpu-offload-gb",
    type=int,
    default=0,
    help="Number of GPUs to use for tensor parallelism. If not specified, "
    "the number of GPUs will be automatically determined based on the "
    "available GPUs.",
)

# --tensor-parallel-size
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Number of GPUs to use for tensor parallelism. If not specified, "
    "the number of GPUs will be automatically determined based on the "
    "available GPUs.",
)

parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.dataset_path
    model = args.model
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    prompts = [dp[0] for dp in data]
    max_tokens = max(dp[1] for dp in data)
    # SPDX-License-Identifier: Apache-2.0
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=max_tokens)
    tensor_parallel_size = args.tensor_parallel_size
    cpu_offload_gb = args.cpu_offload_gb
    dtype = args.dtype
    if args.use_llmpq:
        llm = LLM(
            model=model,
            quantization="llmpq",
            load_format="dummy",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            cpu_offload_gb=cpu_offload_gb,
        )  # try cpu offload
    else:
        if args.quantization:
            llm = LLM(
                model=model,
                load_format="dummy",
                quantization=args.quantization,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                cpu_offload_gb=cpu_offload_gb,
            )
        else:
            llm = LLM(
                model=model,
                load_format="dummy",
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                cpu_offload_gb=cpu_offload_gb,
            )
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     tkn_num = len(output.outputs[0].token_ids)
    # print(
    #     f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}"
    # )
