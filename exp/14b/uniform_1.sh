#  python3 save_uniform_partition.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-cards 3 --bit 8

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B # OOM
MODEL=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8
DTYPE='bfloat16'
DTYPE='half'
python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl \
    --dtype $DTYPE > benchmark_1_uniform_cnn.log 2>&1

python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl \
    --dtype $DTYPE > benchmark_1_uniform_loo.log 2>&1

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --tensor-parallel-size 4 \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/mck.pkl