MODEL_ID=/opt/tiger/Saber/llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-ada-dummy-1
# python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model $MODEL_ID

# python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
#     --backend vllm \
#     --model $MODEL_ID \
#     --dataset-name llmpq \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl  > benchmark_1_pq_cnn.log 2>&1 


python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
    --backend vllm \
    --model $MODEL_ID \
    --dataset-name llmpq \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl  > benchmark_1_pq_loo.log 2>&1 