#  python3 save_uniform_partition.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-cards 3 --bit 8

export CUDA_VISIBLE_DEVICES=0,1
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="16,16,16"


export CUDA_VISIBLE_DEVICES=0
ray start --address=
export VLLM_PP_LAYER_PARTITION="16,16,16"

MODEL=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8
vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 \
    --load-format dummy  \
    --tensor-parallel-size 1  \
    --pipeline-parallel-size 3 \
    --dtype half
