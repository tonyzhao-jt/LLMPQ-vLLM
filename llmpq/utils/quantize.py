import re
import os
from collections import defaultdict
from typing import Any, Dict, List, Type, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel

from llmpq.config import PQConfig
from llmpq.logger import init_logger

logger = init_logger(__name__)

QUANTIZATION_REGISTRY: Dict[str, Type["BaseQuantizer"]] = {}


def register_quantization_method(name: str):
    def decorator(cls: Type["BaseQuantizer"]):
        QUANTIZATION_REGISTRY[name] = cls
        return cls

    return decorator


def get_quantize_dynamic(
    model_id: str, pq_config: PQConfig, pattern: str = r"model\.layers\.(\d+)\."
):
    """
    Get quantization dynamic
    """
    ada_bits = list(map(str, pq_config.adaptive_qbits.split(",")))
    # remove the check
    # config = AutoConfig.from_pretrained(model_id)
    # ref_model = AutoModel.from_config(config)

    # logger.info("Dummy model loaded")
    # module_names = []
    # for name, module in ref_model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         module_names.append(name)

    # pattern = re.compile(pattern)

    # # Organize module names by layer index
    # layer_modules = defaultdict(list)
    # layer_modules["other"] = []
    # for module_name in module_names:
    #     match = pattern.search(module_name)
    #     if match:
    #         layer_index = int(match.group(1))  # Extract the layer index
    #         layer_modules[layer_index].append(module_name)
    #     else:
    #         # Handle modules outside the layers (e.g., "lm_head")
    #         layer_modules["other"].append(module_name)

    # assert (
    #     len(layer_modules) == pq_config.num_layers + 1 # add an other.
    # ), f"layer_modules {len(layer_modules)} not matched, {len(layer_modules)}"
    dynamic = {}
    # [bits, group_size, sym, desc_act, mse, pack_dtype]
    for layer_idx, qbits in enumerate(ada_bits):
        if qbits == 16:
            # skip
            dynamic[r"+:.*\." + f"{layer_idx}" + r"\..*"] = {}
        else:
            dynamic[r"+:.*\." + f"{layer_idx}" + r"\..*"] = {"bits": qbits}
    # handle lm head
    return dynamic


class BaseQuantizer:
    @staticmethod
    def quantize(model_id: str, quant_path: str, bits: int, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


# GPTQ
@register_quantization_method("gptq")
class GPTQQuantizer(BaseQuantizer):
    @staticmethod
    def quantize(model_id: str, quant_path: str, bits: int, **kwargs):
        if bits not in [2, 3, 4, 8]:
            raise ValueError("`bits` must be either 2, 3, 4, or 8.")

        from gptqmodel import GPTQModel, QuantizeConfig

        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        ).select(range(1024))["text"]

        quant_config = QuantizeConfig(bits=bits, group_size=128)

        model = GPTQModel.load(model_id, quant_config)
        model.quantize(calibration_dataset, batch_size=1)
        model.save(quant_path)

        print(
            f"Model quantized to {bits}-bit using GPTQ and saved to {quant_path}."  # noqa
        )

    @staticmethod
    def quantize_adaptive(
        model_id: str, quant_path: str, pq_config: PQConfig, **kwargs
    ):
        dynamic = get_quantize_dynamic(model_id, pq_config)
        from gptqmodel import GPTQModel, QuantizeConfig

        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        ).select(range(1024))["text"]

        quant_config = QuantizeConfig(bits=4, group_size=128, dynamic=dynamic)
        model = GPTQModel.load(model_id, quant_config)
        model.quantize(calibration_dataset, batch_size=2)
        model.save(quant_path)

        # ref
        # dynamic = {
        #     # `.*\.` matches the layers_node prefix
        #     # layer index start at 0
        #     # positive match: layer 19, gate module
        #     r"+:.*\.18\..*gate.*": {"bits": 4, "group_size": 32},
        #     # positgive match: layer 20, gate module (prefix defaults to positive if missing) # noqa
        #     r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},
        #     # negative match: skip layer 21, gate module
        #     r"-:.*\.20\..*gate.*": {},
        #     # negative match: skip all down modules for all layers
        #     r"-:.*down.*": {},
        # }

        print(
            f"Model quantized to {ada_bits} \n using GPTQ and saved to {quant_path}."  # noqa
        )  # noqa

    @staticmethod
    def get_model(quant_path: str):
        from gptqmodel import GPTQModel

        return GPTQModel.load(quant_path)

    @staticmethod
    def eval(
        model_id: str,
        eval_tasks: List[Any] = [],
        eval_pls_tasks: List[Any] = [],
    ):
        from gptqmodel import GPTQModel
        from gptqmodel.utils.eval import EVAL

        # tasks = tasks or [EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.EVALPLUS.HUMAN]
        # Use `lm-eval` as framework to evaluate the model
        lm_eval_results = GPTQModel.eval(
            model_id,
            framework=EVAL.LM_EVAL,
            tasks=eval_tasks,
            output_file="lm-eval_result.json",
        )
        # Use `evalplus` as framework to evaluate the model
        evalplus_results = GPTQModel.eval(
            model_id,
            framework=EVAL.EVALPLUS,
            tasks=eval_pls_tasks,
            output_file="evalplus_result.json",
        )
        return [lm_eval_results, evalplus_results]


# Bitsandbytes
@register_quantization_method("bitsandbytes")
class BitsAndBytesQuantizer(BaseQuantizer):
    @staticmethod
    def quantize(model_id: str, quant_path: str, bits: int, **kwargs):
        if bits not in [4, 8]:
            raise ValueError("`bits` must be either 4 or 8.")

        from transformers import BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )

        model.save_pretrained(quant_path)
        tokenizer.save_pretrained(quant_path)

        print(
            f"Model quantized to {bits}-bit using bitsandbytes and saved to {quant_path}."  # noqa
        )


# AWQ
@register_quantization_method("awq")
class AWQQuantizer(BaseQuantizer):
    @staticmethod
    def quantize(model_id: str, quant_path: str, bits: int, **kwargs):
        if bits not in [4]:
            raise ValueError("`bits` must be 4 for AWQ quantization.")

        from awq import AutoAWQForCausalLM

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM",
        }

        model = AutoAWQForCausalLM.from_pretrained(
            model_id, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )  # noqa

        model.quantize(tokenizer, quant_config=quant_config)

        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)

        print(
            f'Model quantized to {bits}-bit using AWQ and saved at "{quant_path}".'  # noqa
        )  # noqa


# smoothquant
@register_quantization_method("smoothquant")
class SmoothQuantQuantizer(BaseQuantizer):
    @staticmethod
    def quantize(
        model_id: str,
        quant_path: str,
        bits: int,
        dataset_path: Optional[str] = None,
        **kwargs,
    ):
        if bits not in [8, "8-tc"]:
            raise ValueError("`bits` must be 8 for Smoothquant.")

        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        NUM_CALIBRATION_SAMPLES = 512
        MAX_SEQUENCE_LENGTH = 2048

        # Load and preprocess the dataset
        if dataset_path is not None and os.path.exists(dataset_path):
            try:
                ds = load_dataset(dataset_path, split="test_sft")
            except Exception as e:
                print(f"dataset_path {dataset_path} not found, use default dataset")
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        else:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

        def preprocess(example):
            if tokenizer.chat_template:
                return {
                    "text": tokenizer.apply_chat_template(
                        example["messages"], tokenize=False
                    )
                }
            else:
                # 假设 example["messages"] 是一个字典，这里将其转换为字符串
                if isinstance(example["messages"], dict):
                    text = str(example["messages"])
                elif isinstance(example["messages"], list):
                    text = " ".join(map(str, example["messages"]))
                else:
                    text = str(example["messages"])
            return {"text": text}

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        # Configure the quantization algorithms
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        ]

        # Apply quantization
        oneshot(
            model=model_id,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=quant_path,
        )

        print(
            f'Model quantized to {bits}-bit using smoothquant and saved at "{quant_path}".'  # noqa
        )  # noqa


def quantize_model(
    method: str,
    model_id: str,
    quant_path: str,
    bits: int = 4,
    dataset_path: Optional[str] = None,
    **kwargs,
):
    """
    Quantize a model using the specified method.

    Args:
        method (str): The quantization method (e.g., "gptq" or "bitsandbytes").
        model_id (str): The model ID or path to the pre-trained model.
        quant_path (str): The path to save the quantized model.
        bits (int): The quantization precision.
    """
    if method not in QUANTIZATION_REGISTRY:
        raise ValueError(
            f"Unknown quantization method: {method}. Available methods: {list(QUANTIZATION_REGISTRY.keys())}"  # noqa
        )

    quantizer_class = QUANTIZATION_REGISTRY[method]
    quantizer_class.quantize(model_id, quant_path, bits, dataset_path=dataset_path)


def quantize_model_adaptive(
    model_id: str,
    quant_path: str,
    pq_config: PQConfig,
    dataset_path: Optional[str] = None,
):  # noqa
    """
    Quantize a model using the specified method.
    Args:
        method (str): The quantization method (e.g., "gptq" or "bitsandbytes").
        model_id (str): The model ID or path to the pre-trained model.
        quant_path (str): The path to save the quantized model.
        bits (int): The quantization precision.
    """
    method = pq_config.qmethod
    if method not in QUANTIZATION_REGISTRY:
        raise ValueError(
            f"Unknown quantization method: {method}. Available methods: {list(QUANTIZATION_REGISTRY.keys())}"  # noqa
        )

    quantizer_class = QUANTIZATION_REGISTRY[method]
    quantizer_class.quantize_adaptive(
        model_id, quant_path, pq_config, dataset_path=dataset_path
    )
