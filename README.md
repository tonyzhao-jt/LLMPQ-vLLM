# llm_pq_v2
LLM-PQ v2



# bug fix
```bash
    export LD_LIBRARY_PATH=/opt/conda/envs/llmpq/lib/python3.10//site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

# EXP models
3 sizes. Versus 2 hetero case each. 
# 8b 2 cases?
```
    meta-llama/Llama-3.1-8B # 8b
    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B # 32b
    meta-llama/Llama-3.1-70B-Instruct-evals # 70b
```
llama3.1-8b 

# Implementation Milestones
全包好了：对于任意模型
(1)indicator gen：https://github.com/tonyzhao-jt/LLM-PQ/tree/main/scripts/accuracyPPL
用 gen ind 生成 indicator, hess
(2)profiler: https://github.com/tonyzhao-jt/LLM-PQ/tree/main/scripts/profile
https://github.com/vllm-project/vllm/pull/11125/files
先用下面这个简单测算不同 GPU 下的 Latency （可以加一个 TP 的维度，默认用 TP=node）
https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override
打包成对应之前的 fit 格式. 通信还是用之前的测法 （可有可无了属于是）
（明天看一下怎么从 trace 拿数据，应该不复杂）
(4) 新的 cost model 结果可能要分析一下。包括 component 的时间。
(新的精度也可以用 vllm 直接跑)
(3) optimizer
用优化器解，解完后，得到计划直接用 vllm 跑 (dummy 就行了) (核心是看怎么 PP 了，quant 这边直接 load 就行了)
(4)讨论一下 chunked prefill 啥的, 从效果上也是堆叠到上面
PD 分离有点 contradict， KD 分离 本身增加了memory 使用，但 quant 一般永远在 memory scarse 的情况下
