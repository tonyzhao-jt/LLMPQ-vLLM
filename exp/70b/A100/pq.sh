export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --port 5678
export VLLM_PP_LAYER_PARTITION="42,38"

vllm serve /opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-ada \
    --load-format dummy  \
    --quantization llmpq \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 