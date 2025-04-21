# python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model /opt/tiger/Saber/llm_pq_v2/exp/14b/tmp/Qwen2.5-14B-Instruct-ada-dummy
export CUDA_VISIBLE_DEVICES=0,1
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="16,16,16" # 14 8bit + 2 16bit * 2 + 16 8-tc bit
export VLLM_PP_LAYER_PARTITION="15,16,17" # hybrid pack 2.
export VLLM_PP_LAYER_PARTITION="15,15,18"
export VLLM_PP_LAYER_PARTITION="15,16,17"
export VLLM_PP_LAYER_PARTITION="14,17,17"

export CUDA_VISIBLE_DEVICES=0
ray start --address='10.128.97.213:5678'

MODEL=/opt/tiger/Saber/llm_pq_v2/exp/14b/tmp/Qwen2.5-14B-Instruct-ada-dummy
vllm serve $MODEL \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 3 \
    --dtype half