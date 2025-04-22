MODEL_ID=/opt/tiger/Saber/llm_pq_v2/exp/70b/tmp/Llama-2-70B-8bit
TYPE='het'
TYPE='uniform_tp'
# python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model $MODEL_ID

python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
    --backend vllm \
    --model $MODEL_ID \
    --dataset-name llmpq \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl  > benchmark_1_${TYPE}_cnn.log 2>&1 


# python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
#     --backend vllm \
#     --model $MODEL_ID \
#     --dataset-name llmpq \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl  > benchmark_2_$TYPE_loo.log 2>&1 