export CUDA_VISIBLE_DEVICES=0
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="24,40"
export VLLM_PP_LAYER_PARTITION="22,42"


export CUDA_VISIBLE_DEVICES=0
ray start --address=


python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model /yourpath//llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-q4-test

vllm serve /yourpath//llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-q4-test \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 2 \
    --dtype half
