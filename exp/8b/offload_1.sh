export CUDA_VISIBLE_DEVICES=4
MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
# MODEL=study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int8
python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl \
    --cpu-offload-gb 50 \
    --dtype half > benchmark_1_offload_cnn.log 2>&1

python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
    --model $MODEL \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl \
    --cpu-offload-gb 50 \
    --dtype half > benchmark_1_offload_loo.log 2>&1

# python3 /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_single_card.py \
#     --model $MODEL \
#     --tensor-parallel-size 4 \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/mck.pkl