# pip install -v gptqmodel --no-build-isolation
# pip install bitsandbytes
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict

from llmpq.profiler import shard_model
from llmpq.utils import quantize_model, QUANTIZATION_REGISTRY

PROFILER_RAW = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile"
PROFILER_PARSED = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed"
os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILER_RAW
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_shard_name = "Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Validate this model still works
    save_path = f"tmp/llm_pq/{model_shard_name}-sharded"
    # make save path abs path
    save_path = os.path.abspath(save_path)

    shard_model(model_id, save_path)
    loaded_model = AutoModelForCausalLM.from_pretrained(save_path)

    # validate its
    # input_ids = tokenizer.encode("hello world", return_tensors="pt")
    # out_2 = loaded_model(input_ids)

    # consider_bitwidth = [3, 4, 8]
    consider_bitwidth = {
        # 'gptq': [4, 8],
        # 'bitsandbytes': [4, 8],
        'awq': [4, 8]
    }
    method_bitwidth_model_path = defaultdict(dict)
    for method, bits in consider_bitwidth.items():
        for _bits in bits:
            quant_path = f"tmp/llm_pq/{model_shard_name}-sharded-{method}-{_bits}"
            abs_path = os.path.abspath(quant_path)
            quantize_model(method, save_path, abs_path, bits=_bits)
            method_bitwidth_model_path[method][_bits] = abs_path

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # load it via VLLM
    # do profiling
    # collect the profiling result
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
    for qmethod, bitwidth_model_path in method_bitwidth_model_path.items():
        for bit, model_path in bitwidth_model_path.items():
            # llm = LLM(model=model_path, tensor_parallel_size=2, dtype=torch.float16, load_format="dummy") # noqa
            if qmethod == "gptq":
                llm = LLM(model=model_path, tensor_parallel_size=1, dtype=torch.float16)  # noqa
            elif qmethod == "bitsandbytes":
                llm = LLM(model=model_path, tensor_parallel_size=1, dtype=torch.bfloat16, trust_remote_code=True, 
                          quantization="bitsandbytes", load_format="bitsandbytes")  # noqa
            elif qmethod == 'awq':
                llm = LLM(model=model_path, tensor_parallel_size=1, quantization="AWQ")  # noqa
            else:
                raise ValueError(f"Unknown quantization method: {qmethod}. Available methods: {list(QUANTIZATION_REGISTRY.keys())}")  # noqa

            llm.start_profile()
            outputs = llm.generate(prompts, sampling_params)
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
                    os.rename(
                        os.path.join(PROFILER_RAW, file),
                        os.path.join(
                            PROFILER_PARSED,
                            f"{model_shard_name}-{qmethod}-{bit}-{file}",
                        ),  # noqa
                    )
