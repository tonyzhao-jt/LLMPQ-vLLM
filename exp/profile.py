# pip install -v gptqmodel --no-build-isolation
# pip install bitsandbytes
import os

import gzip
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List

import torch
from llmpq.dataset import DummyDataset
from llmpq.utils import get_device_name_by_torch  # noqa
from vllm import LLM, SamplingParams


def profile_model(
    model_id: str,
    model_shard_name: str,
    inputs: Dict[str, int],
    consider_bitwidth: Dict[str, List[int]],
    tp_size: int = 1,  # tp = 2 will hang the process.
    warmup: int = 5,
    repeat: int = 10,
    PROFILER_RAW: str = "tmp/vllm_profile",  # noqa
    PROFILER_PARSED: str = "tmp/vllm_profile_parsed",  # noqa
    work_dir: str = "/tmp/llmpq/work_dir",  # noqa
) -> Dict[str, Dict[str, float]]:
    # profile the model with given configs.
    from llmpq.profiler.utils import shard_model
    from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model

    model_id_wo_special = model_id.replace("/", "_")
    batch_size, prompt_length, output_tokens = (
        inputs["batch_size"],
        inputs["prompt_len"],
        inputs["output_tokens"],
    )
    prompts = DummyDataset(
        batch_size=batch_size, prompt_len=prompt_length
    ).gen_prompts()
    # Validate this model still works
    save_path = os.path.join(work_dir, f"{model_id_wo_special}-sharded")
    # make save path abs path
    save_path = os.path.abspath(save_path)

    shard_model(model_id, save_path, layer_num=2)

    method_bitwidth_model_path = defaultdict(dict)
    for method, bits in consider_bitwidth.items():
        if method == "noq":
            continue
        for _bits in bits:
            q_save_path = os.path.join(work_dir, f"qtmp-{model_id_wo_special}-{bit}bit")
            q_save_path = os.path.abspath(q_save_path)
            # check if has the file
            if not os.path.exists(q_save_path):
                quantize_model(method, save_path, q_save_path, bits=_bits)
            else:
                print(f"Found {q_save_path}, skip quantization")
            method_bitwidth_model_path[method][_bits] = q_save_path

    # setup tkn gen num
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        ignore_eos=True,
        max_tokens=output_tokens,
    )

    output_files = []
    gpu_memory_utilization = 0.5

    for qmethod, bitwidth_model_path in method_bitwidth_model_path.items():
        for bit, model_path in bitwidth_model_path.items():
            # llm = LLM(model=model_path, tensor_parallel_size=2, dtype=torch.float16, load_format="dummy") # noqa
            if qmethod == "noq":
                llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    dtype=torch.half,  # noqa
                    gpu_memory_utilization=gpu_memory_utilization,
                )  # noqa
            elif qmethod == "gptq":
                if bit == 3:
                    # now not support 3 bit
                    llm = LLM(
                        model=model_path,
                        tensor_parallel_size=2,
                        dtype=torch.half,  # noqa
                        quantization="gptq",
                        gpu_memory_utilization=gpu_memory_utilization,
                    )  # noqa
                else:
                    llm = LLM(
                        model=model_path,
                        tensor_parallel_size=tp_size,
                        dtype=torch.half,  # noqa
                        gpu_memory_utilization=gpu_memory_utilization,
                    )  # noqa
            elif qmethod == "bitsandbytes":
                llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    dtype=torch.half,
                    trust_remote_code=True,
                    quantization="bitsandbytes",
                    load_format="bitsandbytes",
                    gpu_memory_utilization=gpu_memory_utilization,
                )  # noqa
            elif qmethod == "awq":
                llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    quantization="AWQ",  # noqa
                    gpu_memory_utilization=gpu_memory_utilization,
                )  # noqa
            elif qmethod == "smoothquant":
                llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                )  # noqa
            else:
                raise ValueError(
                    f"Unknown quantization method: {qmethod}. Available methods: {list(QUANTIZATION_REGISTRY.keys())}"  # noqa
                )  # noqa

            for _ in range(warmup):
                outputs = llm.generate(prompts, sampling_params)

            llm.start_profile()
            for _ in range(repeat):
                outputs = llm.generate(prompts, sampling_params)  # noqa
            llm.stop_profile()

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")  # noqa

            time.sleep(10)
            del llm
            # find the first file under the profiler, rename it, move to PROFILER_PARSED # noqa
            # if Profile RAW not exists, create it
            if not os.path.exists(PROFILER_PARSED):
                os.makedirs(PROFILER_PARSED)
            for file in os.listdir(PROFILER_RAW):
                if file.endswith(".gz"):
                    new_file_name = f"{model_shard_name}-{qmethod}-{bit}-{tp_size}-pt.trace.json.gz"  # noqa
                    os.rename(
                        os.path.join(PROFILER_RAW, file),
                        os.path.join(
                            PROFILER_PARSED,
                            new_file_name,
                        ),  # noqa
                    )
                    # unzip it (.gz format)
                    with gzip.open(
                        os.path.join(PROFILER_PARSED, new_file_name), "rb"
                    ) as f_in:  # noqa
                        with open(
                            os.path.join(PROFILER_PARSED, new_file_name[:-3]),
                            "wb",  # noqa
                        ) as f_out:  # noqa
                            shutil.copyfileobj(f_in, f_out)
                            output_files.append(
                                os.path.join(
                                    PROFILER_PARSED, new_file_name[:-3]
                                )  # noqa
                            )
    return output_files


PROFILER_RAW = "/yourpath//llm_pq_v2/examples/tmp/vllm_profile"
PROFILER_PARSED = "/yourpath//llm_pq_v2/examples/tmp/vllm_profile_parsed"
REPEAT = 10
WARMUP = 5

os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILER_RAW
if __name__ == "__main__":
    # only profile 2 layers for quant and profiling.
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    model_shard_name = "Qwen2.5-14B-Instruct_sharded"
    consider_inputs = {
        "batch_size": 128,
        "prompt_len": 2048,
        "output_tokens": 10,
    }
    # consider_bitwidth = [4, 8, 16]
    consider_bitwidth = {
        "noq": [16],  # no quant
        "gptq": [4, 8],  # weight only
        # "awq": [4],  # weight only
        "smoothquant": [8],  # w8a8
    }

    output_files = profile_model(
        model_id,
        model_shard_name,
        consider_inputs,
        consider_bitwidth,
        warmup=WARMUP,
        repeat=REPEAT,
        PROFILER_RAW=PROFILER_RAW,
        PROFILER_PARSED=PROFILER_PARSED,
    )
