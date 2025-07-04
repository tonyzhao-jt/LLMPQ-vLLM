<<<<<<<< HEAD:exp/70b/A100/run_gpt1.sh
MODEL_ID=/yourpath//llm_pq_v2/exp/70b/tmp/Llama-3.3-70B-ada
========
MODEL_ID=/yourpath//llm_pq_v2/exp/70b/tmp/Llama-2-70B-ada
>>>>>>>> a675fb2abdb82fbb55ddda20a556c843f4a347d2:exp/70b/v100/run_gpt1.sh
TYPE='pq'
# python3 /yourpath//llm_pq_v2/test/dataset/dataset_test.py --model $MODEL_ID

python /yourpath//llm_pq_v2/benchmarks/bench_serve.py \
    --backend vllm \
    --model $MODEL_ID \
    --dataset-name llmpq \
    --dataset-path /yourpath//llm_pq_v2/test/dataset/cnn.pkl  > benchmark_1_${TYPE}_cnn.log 2>&1


# python /yourpath//llm_pq_v2/benchmarks/bench_serve.py \
#     --backend vllm \
#     --model $MODEL_ID \
#     --dataset-name llmpq \
#     --dataset-path /yourpath//llm_pq_v2/test/dataset/loo.pkl  > benchmark_2_pq_loo.log 2>&1
