export CUDA_VISIBLE_DEVICES=0
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="24,40"
export VLLM_PP_LAYER_PARTITION="22,42"

export CUDA_VISIBLE_DEVICES=0
ray start --address='10.128.97.213:5678'


python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model /opt/tiger/Saber/llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-ada-dummy-1

vllm serve /opt/tiger/Saber/llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-ada-dummy-1 \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 2 \
    --dtype half