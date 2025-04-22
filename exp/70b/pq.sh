export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="30,50"
export VLLM_PP_LAYER_PARTITION="24,56"
export VLLM_PP_LAYER_PARTITION="20,60"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PP_LAYER_PARTITION="40,40"
ray start --address='10.147.181.199:5678'

vllm serve /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-2-70B-ada \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 4  \
    --pipeline-parallel-size 2 \
    --dtype half