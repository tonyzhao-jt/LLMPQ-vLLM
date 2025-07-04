Here is a polished version of your README:

# LLMPQ-vLLM
================

LLMPQ with vLLM backend. The backend acts as a hack to allow vLLM serving to be piplining heterogenous and precision (quantization) heterogenous.

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
