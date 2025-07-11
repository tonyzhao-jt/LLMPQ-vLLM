export CUDA_VISIBLE_DEVICES=0
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="32,32"


export CUDA_VISIBLE_DEVICES=0
ray start --address=
export VLLM_PP_LAYER_PARTITION="32,32"

vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --load-format dummy  \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 2 \
    --dtype half
