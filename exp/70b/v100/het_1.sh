export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="34,46"


export CUDA_VISIBLE_DEVICES=0,1
ray start --address='10.128.97.213:5678'
# export VLLM_PP_LAYER_PARTITION="40,40"


# huihui-ai/Llama-3.3-70B-Instruct-abliterated-finetuned-GPTQ-Int8
# hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
# Sao10K/Llama-3.3-70B-Vulpecula-r1

python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-2-70B-4bit

vllm serve /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-2-70B-4bit \
    --load-format dummy  \
    --tensor-parallel-size 4  \
    --pipeline-parallel-size 2 \
    --dtype half