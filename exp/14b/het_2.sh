#  python3 save_uniform_partition.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-cards 3 --bit 8

export CUDA_VISIBLE_DEVICES=0,1
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="16,17,15"
export VLLM_PP_LAYER_PARTITION="16,16,16"
export VLLM_PP_LAYER_PARTITION="15,15,18"
export VLLM_PP_LAYER_PARTITION="15,16,17"
export VLLM_PP_LAYER_PARTITION="14,17,17"

export CUDA_VISIBLE_DEVICES=0
ray start --address=''

MODEL=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8
vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 \
    --load-format dummy  \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 3 \
    --dtype half
