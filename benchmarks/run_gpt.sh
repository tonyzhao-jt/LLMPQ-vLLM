# 4v dataset
# mkdir -p ./tmp && wget -P ./tmp https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
# mkdir -p ./tmp && wget -P ./tmp https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python bench_serve.py \
    --backend vllm \
    --model /opt/tiger/Saber/llm_pq_v2/test/tmp/Llama-3.2-1B-ada \
    --dataset-name llmpq \
    --dataset-path /opt/tiger/Saber/llm_pq_v2/test/dataset/cnn.pkl
