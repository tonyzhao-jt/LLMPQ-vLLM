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

# 全包好了：对于任意模型
# indicator gen：
# 用 gen ind 生成 indicator, hess
# profiler:
# 先用下面这个简单测算不同 GPU 下的 Latency （可以加一个 TP 的维度，默认用 TP=node）
# https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override
# 打包成对应之前的 fit 格式
# 通信还是用之前的测法
# optimizer
# 用优化器解，解完后，得到计划直接用 vllm 跑 (核心是看怎么 PP 了，quant 这边直接 load 就行了)
# 这还涉及到 chunked prefill 啥的, 从效果上也是堆叠到上面
# PD 分离有点 contradict， KD 分离 本身增加了memory 使用，但 quant 一般永在 memory scarse 的情况下
# llm = LLM(model="meta-llama/Llama-3.2-1B", load_format='dummy') # try cpu offload
llm = LLM(model="RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8", load_format="dummy")
# 然后再用 vllm 直接跑就行了。。
# indicator 的脚本还是可以用之前的。

# 逻辑：用之前的脚本获取
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    tkn_num = len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}")