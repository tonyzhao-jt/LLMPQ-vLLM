export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="40,40"


export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --address=10.147.194.32:5678
export VLLM_PP_LAYER_PARTITION="40,40"

# huihui-ai/Llama-3.3-70B-Instruct-abliterated-finetuned-GPTQ-Int8
# hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
# Sao10K/Llama-3.3-70B-Vulpecula-r1

vllm serve Sao10K/Llama-3.3-70B-Vulpecula-r1 \
    --load-format dummy  \
    --tensor-parallel-size 4  \
    --pipeline-parallel-size 2 \
    --dtype half