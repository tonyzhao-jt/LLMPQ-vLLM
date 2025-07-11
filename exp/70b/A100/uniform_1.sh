export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="40,40"
export VLLM_PP_LAYER_PARTITION="10,10,10,10,10,10,10,10"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PP_LAYER_PARTITION="40,40"
ray start --address=
# export VLLM_PP_LAYER_PARTITION="40,40"


vllm serve /yourpath//llm_pq_v2/exp/70b/tmp/Llama-2-70B-4bit \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 8 \
    --dtype half

vllm serve 'meta-llama/Llama-2-70b-chat-hf'\
    --load-format dummy  \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --dtype half


python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model /yourpath//llm_pq_v2/exp/70b/tmp/Llama-2-70B-8bit
export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="40,40"
python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model /yourpath//llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-4bit
vllm serve /yourpath//llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-4bit \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2

vllm serve /yourpath//llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-4bit \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1

vllm serve /yourpath//llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-4bit \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4
