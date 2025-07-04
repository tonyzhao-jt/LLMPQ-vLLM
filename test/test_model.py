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

llm = LLM(
    model="/yourpath//llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-smoothquant-8",
    load_format="dummy",
)  # try cpu offload
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tkn_num = len(output.outputs[0].token_ids)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}"
    )
