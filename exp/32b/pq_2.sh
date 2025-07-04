export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="14,15,15,20"

export CUDA_VISIBLE_DEVICES=0
ray start --address=

# python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model /yourpath//llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-q8-1

vllm serve /yourpath//llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-q8-1 \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 4 \
    --dtype half
