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

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", cpu_offload_gb=10) # try cpu offload
# llm = LLM(model="meta-llama/Llama-3.2-1B", load_format='dummy') # try dummy 
llm = LLM(model="meta-llama/Llama-3.2-1B", cpu_offload_gb=10) # try cpu offload
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tkn_num = len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}")