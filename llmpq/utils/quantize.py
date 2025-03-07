from typing import Callable, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义量化方法的注册表
QUANTIZATION_REGISTRY: Dict[str, Callable] = {}


# 注册装饰器
def register_quantization_method(name: str):
    def decorator(func: Callable):
        QUANTIZATION_REGISTRY[name] = func
        return func

    return decorator


# 注册 GPTQ 量化方法
@register_quantization_method("gptq")
def quantize_to_bit_gptq(model_id: str, quant_path: str, bits=4):
    """
    Quantize a model using GPTQ to 4-bit precision.

    Args:
        model_id (str): The model ID or path to the pre-trained model.
        quant_path (str): The path to save the quantized model.
        bits (int): The quantization precision (only 4 is supported for GPTQ).
    """
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
    model.quantize(calibration_dataset, batch_size=2)
    model.save(quant_path)

    print(
        f"Model quantized to {bits}-bit using GPTQ and saved to {quant_path}."
    )  # noqa


@register_quantization_method("bitsandbytes")
def quantize_to_bit_bitsandbytes(model_id: str, quant_path: str, bits=4):
    """
    Quantize a model using bitsandbytes to 4-bit or 8-bit precision.

    Args:
        model_id (str): The model ID or path to the pre-trained model.
        quant_path (str): The path to save the quantized model.
        bits (int): The quantization precision (4 or 8).
    """
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


@register_quantization_method("awq")
def quantize_to_bit_awq(model_id: str, quant_path: str, bits=4):
    """
    Quantize a model using AWQ to 4-bit precision.

    Args:
        model_id (str): The model ID or path to the pre-trained model.
        quant_path (str): The path to save the quantized model.
        bits (int): The quantization precision (only 4 is supported for AWQ).
    """
    if bits not in [4]:
        raise ValueError("`bits` must be 4 for AWQ quantization.")

    # require transformers==4.47.1 transformers now
    from awq import AutoAWQForCausalLM

    # 定义量化配置
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(
        model_id, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(
        f'Model quantized to {bits}-bit using AWQ and saved at "{quant_path}".'
    )  # noqa


def quantize_model(method: str, model_id: str, quant_path: str, bits=4):
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

    quantize_func = QUANTIZATION_REGISTRY[method]
    quantize_func(model_id, quant_path, bits)
