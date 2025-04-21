export CUDA_VISIBLE_DEVICES=0
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="24,40"

export CUDA_VISIBLE_DEVICES=0
ray start --address=10.147.194.32:5678


vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --load-format dummy  \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 2 \
    --dtype half