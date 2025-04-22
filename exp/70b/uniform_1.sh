export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="40,40"


export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PP_LAYER_PARTITION="40,40"
ray start --address='10.147.181.199:5678'
# export VLLM_PP_LAYER_PARTITION="40,40"

python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-70B-4bit
vllm serve /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-70B-4bit \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 4  \
    --pipeline-parallel-size 2 \
    --dtype half