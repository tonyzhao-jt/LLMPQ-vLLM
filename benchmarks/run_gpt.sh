# 4v dataset
# wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json

python bench_serve.py \
    --backend vllm \
    --model meta-llama/Llama-3.2-1B \
    --dataset-name sharegpt \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/benchmarks/sharegpt4v_instruct_gpt4-vision_cap100k.json \