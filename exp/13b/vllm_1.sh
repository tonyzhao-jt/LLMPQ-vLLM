MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --tensor-parallel-size 4 \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/mck.pkl

python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --tensor-parallel-size 4 \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl
