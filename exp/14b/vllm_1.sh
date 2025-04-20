MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
MODEL=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl \
    --dtype half

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --load-format dummy \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 3 

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --tensor-parallel-size 4 \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl


# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --tensor-parallel-size 4 \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/mck.pkl