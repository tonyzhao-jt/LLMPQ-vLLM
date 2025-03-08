# pip install -v gptqmodel --no-build-isolation
# pip install bitsandbytes
import os
import time
import gzip
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from llmpq.dataset import AIMEDataset
from llmpq.profiler import shard_model
from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model
import shutil

PROFILER_RAW = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile"
PROFILER_PARSED = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed"
REPEAT = 10
WARMUP = 5
os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILER_RAW
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_shard_name = "Llama_3.2_1B_Instruct_sharded"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Validate this model still works
    save_path = f"tmp/llm_pq/{model_shard_name}"
    # make save path abs path
    save_path = os.path.abspath(save_path)

    # one embedding, two decoder layer, one processor(LM) # noqa 
    shard_model(model_id, save_path, layer_num = 2) 
    loaded_model = AutoModelForCausalLM.from_pretrained(save_path)

    # validate its
    # input_ids = tokenizer.encode("hello world", return_tensors="pt")
    # out_2 = loaded_model(input_ids)
    prompts = AIMEDataset().sample_n_prompts(10)
    tp_size = 1 # tp = 2 will hang the process. We give a rough estimation of tp cost now.

    # consider_bitwidth = [3, 4, 8]
    consider_bitwidth = {
        'gptq': [4, 8],
        # 'bitsandbytes': [4, 8],
        # "awq": [4]  # only support 4
    }
    method_bitwidth_model_path = defaultdict(dict)
    for method, bits in consider_bitwidth.items():
        for _bits in bits:
            quant_path = (
                f"tmp/llm_pq/{model_shard_name}-{method}-{_bits}"  # noqa
            )
            abs_path = os.path.abspath(quant_path)
            # check if has the file
            if not os.path.exists(abs_path):
                quantize_model(method, save_path, abs_path, bits=_bits)
            else:
                print(f"Found {abs_path}, skip quantization")
            method_bitwidth_model_path[method][_bits] = abs_path


    # load it via VLLM
    # do profiling
    # collect the profiling result
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
    for qmethod, bitwidth_model_path in method_bitwidth_model_path.items():
        for bit, model_path in bitwidth_model_path.items():
            # llm = LLM(model=model_path, tensor_parallel_size=2, dtype=torch.float16, load_format="dummy") # noqa
            if qmethod == "gptq":
                if bit == 3:
                    # now not support 3 bit
                    llm = LLM(
                        model=model_path,
                        tensor_parallel_size=2, 
                        dtype=torch.half,  # noqa
                        quantization='gptq',
                    )  # noqa
                else:
                    llm = LLM(
                        model=model_path,
                        tensor_parallel_size=tp_size,
                        dtype=torch.half,  # noqa
                    )  # noqa
            elif qmethod == "bitsandbytes":
                llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    dtype=torch.half,
                    trust_remote_code=True,
                    quantization="bitsandbytes",
                    load_format="bitsandbytes",
                )  # noqa
            elif qmethod == "awq":
                llm = LLM(
                    model=model_path, tensor_parallel_size=tp_size, quantization="AWQ"  # noqa
                )  # noqa
            else:
                raise ValueError(
                    f"Unknown quantization method: {qmethod}. Available methods: {list(QUANTIZATION_REGISTRY.keys())}"  # noqa
                )  # noqa

            for _ in range(WARMUP):
                outputs = llm.generate(prompts, sampling_params)

            llm.start_profile()
            for _ in range(REPEAT):
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
                    new_file_name = f"{model_shard_name}-{qmethod}-{bit}-{tp_size}-pt.trace.json.gz"  # noqa
                    os.rename(
                        os.path.join(PROFILER_RAW, file),
                        os.path.join(
                            PROFILER_PARSED,
                            new_file_name, 
                        ),  # noqa
                    )
                    # unzip it (.gz format)
                    with gzip.open(os.path.join(PROFILER_PARSED, new_file_name), 'rb') as f_in: # noqa
                        with open(os.path.join(PROFILER_PARSED, new_file_name[:-3]), 'wb') as f_out: # noqa
                            shutil.copyfileobj(f_in, f_out)
