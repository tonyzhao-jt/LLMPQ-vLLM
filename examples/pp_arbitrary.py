# run vllm in arbitrary partition 
# SPDX-License-Identifier: Apache-2.0
# https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_pipeline_partition.py
# https://github.com/vllm-project/vllm/tree/main/tests/distributed
from vllm import LLM, SamplingParams

'''
vllm serve gpt2 \
--tensor-parallel-size 4 \
--pipeline-parallel-size 2
'''

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# ray start --head 
llm = LLM(model="meta-llama/Llama-3.2-1B", 
          pipeline_parallel_size=4, 
          load_format='dummy', 
          enforce_eager=True, # try cpu offload
          distributed_executor_backend='ray') # try cpu offload

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tkn_num = len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}")