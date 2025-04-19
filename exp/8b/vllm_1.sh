# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model unsloth/Meta-Llama-3.1-8B-Instruct \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl

python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/mck.pkl

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model unsloth/Meta-Llama-3.1-8B-Instruct \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl
