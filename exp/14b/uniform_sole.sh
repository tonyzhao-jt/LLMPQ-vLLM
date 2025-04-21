MODEL=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
DTYPE='half'
python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl \
    --dtype $DTYPE > benchmark_1_uniform_cnn.log 2>&1