MODEL_ID=/opt/tiger/Saber/llm_pq_v2/exp/14b/tmp/Qwen2.5-14B-Instruct-ada-dummy
# python3 /opt/tiger/Saber/llm_pq_v2/test/dataset/dataset_test.py --model $MODEL_ID

# python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
#     --backend vllm \
#     --model $MODEL_ID \
#     --dataset-name llmpq \
#     --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl


python /opt/tiger/Saber/llm_pq_v2/benchmarks/bench_serve.py \
    --backend vllm \
    --model $MODEL_ID \
    --dataset-name llmpq \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/loo.pkl