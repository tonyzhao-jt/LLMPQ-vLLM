# pip install -v gptqmodel --no-build-isolation
# pip install bitsandbytes
import os
from llmpq.profiler import profile_model

PROFILER_RAW = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile"
PROFILER_PARSED = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed"
REPEAT = 10
WARMUP = 5

os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILER_RAW
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_shard_name = "Llama_3.2_1B_Instruct_sharded"
    batch_size = 8
    prompt_length = 128 
    output_tokens = 100

    consider_inputs = {
        'batch_size': 8,
        'prompt_len': 128,
        'output_tokens': 100,
    }
    # consider_bitwidth = [4, 8, 16]
    consider_bitwidth = {
        'noq': [16], # no quant
        'gptq': [4, 8],
        # 'bitsandbytes': [4, 8],
        # "awq": [4]  # only support 4
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

    


    
