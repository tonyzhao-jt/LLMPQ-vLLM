# Benchmark
Copy from vLLM repo, perform the serving and throughput benchmarking.

# Test
install sharegpt
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O ShareGPT_V3_unfiltered_cleaned_split.json
```

```bash
python3 bench_single_card_registry.py --model unsloth/Meta-Llama-3.1-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```
