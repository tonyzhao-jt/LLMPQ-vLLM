MODEL=/opt/tiger/Saber/llm_pq_v2/exp/13b/tmp/llama13b-int4
DTYPE='half'
# python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model $MODEL
python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl \
    --use-llmpq \
    --dtype $DTYPE > benchmark_1_pq_cnn.log 2>&1 

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl \
#     --dtype $DTYPE > benchmark_1_uniform_loo.log 2>&1
