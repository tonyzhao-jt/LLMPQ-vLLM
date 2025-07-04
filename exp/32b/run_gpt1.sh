MODEL_ID=/yourpath//llm_pq_v2/exp/32b/tmp/Qwen2.5-32B-Instruct-q4-test
# python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model $MODEL_ID

# python /yourpath//llm_pq_v2/benchmarks/bench_serve.py \
#     --backend vllm \
#     --model $MODEL_ID \
#     --dataset-name llmpq \
#     --dataset-path /yourpath//llm_pq_v2/test/dataset/cnn.pkl  > benchmark_2_pq_cnn.log 2>&1


python /yourpath//llm_pq_v2/benchmarks/bench_serve.py \
    --backend vllm \
    --model $MODEL_ID \
    --dataset-name llmpq \
    --dataset-path /yourpath//llm_pq_v2/test/dataset/loo.pkl  > benchmark_2_pq_loo_test.log 2>&1
