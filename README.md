# LLMPQ-vLLM
================

LLMPQ with vLLM backend. The backend acts as a hack to allow vLLM serving to be piplining heterogenous and precision (quantization) heterogenous （mixed precision）.

## Installation
---------------

```bash
pip install -e .
```

## Running
---------

Please ensure your vLLM version is `>=0.8.5` or use [vllm-PQ](https://github.com/tonyzhao-jt/vllm-PQ.git)

```bash
pip list | grep vllm
```

### Request
------------

Use `test/dataset` to generate dummy dataset (check it to modify dataset).

```bash
python3 dataset_test.py --model meta-llama/Llama-3.1-8B
```

You will see a `xxx.pkl` under the same folder. Use it as the request.

### Execute
-------------

Two possible ways to run the PQ:

1. Check `benchmarks/bench_single_card_registry.py` to run the quantization patch (single node).

```bash
python3 bench_single_card_registry.py --model meta-llama/Llama-3.1-8B --dataset-path /home/tonyzhao/local/LLMPQ-vLLM/test/dataset/cnn.pkl
```

2. Check files under `exp/` to check distributed inference scripts.
In that way, you need to use `vllm-PQ`, which embedes the hack into the vLLM source code.

```bash
git clone https://github.com/tonyzhao-jt/vllm-PQ.git
git checkout juntao/dev
cd vllm && pip install -e .
```

## Notice
--------

The optimization and profiler hasn't fully compatible to the vLLM backend. We provides final results enduer `exp/`, please modify it at will if you need to perform profiling and optimization calculation.

## Common vLLM Bugs
-------------------

```bash
export LD_LIBRARY_PATH=/opt/conda/envs/llmpq/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Citation
If you use LLM-PQ for your research, please cite our [paper](https://dl.acm.org/doi/10.1145/3627535.3638480):
```bibtex
@inproceedings{10.1145/3627535.3638480,
author = {Zhao, Juntao and Wan, Borui and Wu, Chuan and Peng, Yanghua and Lin, Haibin},
title = {POSTER: LLM-PQ:Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638480},
doi = {10.1145/3627535.3638480},
pages = {460–462},
keywords = {LM serving, heterogenous cluster, quantization},
series = {PPoPP '24}
}
```


