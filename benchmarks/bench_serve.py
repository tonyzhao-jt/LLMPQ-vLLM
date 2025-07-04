# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import gc
import io
import json
import os
import random
import time
import warnings
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(len(token_ids) for token_ids in poem_token_ids) / len(
        poem_token_ids
    )

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [
        {
            "role": "user",
            "content": base_prompt,
        }
    ]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False
    )
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round((input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round((prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(
            prefix_lines + random.choices(poem_lines, k=num_lines_needed)
        )

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None)
        )

    return sampled_requests


def sample_vision_arena_requests(
    dataset,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:
    sampled_requests: List[Tuple[str, int, int, Dict[str, Collection[str]]]] = []
    for data in dataset:
        if len(sampled_requests) == num_requests:
            break

        prompt = data["turns"][0][0]["content"]

        prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_output_len is None:
            # Default max output len is set to 128
            print("--hf-output-len is not provided. Using default value 128.")
            fixed_output_len = 128

        prompt_len = len(prompt_token_ids)
        output_len = fixed_output_len

        assert isinstance(data["images"][0], Image), (
            "Input image format must be `PIL.Image.Image`, "
            f"given {type(data['image'])}."
        )
        image: Image = data["images"][0]
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format="JPEG")
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        mm_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_hf_requests(
    dataset_path: str,
    dataset_subset: Optional[str],
    dataset_split: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:

    # Special case for vision_arena dataset
    if dataset_path == "lmarena-ai/vision-arena-bench-v0.1" and dataset_subset is None:
        assert dataset_split == "train"
        dataset = load_dataset(
            dataset_path, name=dataset_subset, split=dataset_split, streaming=True
        )
        dataset = dataset.shuffle(seed=random_seed)
        return sample_vision_arena_requests(
            dataset, num_requests, tokenizer, fixed_output_len
        )

    dataset = load_dataset(
        dataset_path, name=dataset_subset, split=dataset_split, streaming=True
    )
    assert (
        "conversations" in dataset.features
    ), "HF Dataset must have 'conversations' column."
    filter_func = lambda x: len(x["conversations"]) >= 2
    filtered_dataset = dataset.shuffle(seed=random_seed).filter(filter_func)
    sampled_requests: List[Tuple[str, int, int, Dict[str, Collection[str]]]] = []
    for data in filtered_dataset:
        if len(sampled_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data["conversations"][1]["value"]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if fixed_output_len is None and (prompt_len < 4 or output_len < 4):
            # Prune too short sequences.
            continue
        if fixed_output_len is None and (
            prompt_len > 1024 or prompt_len + output_len > 2048
        ):
            # Prune too long sequences.
            continue

        if "image" in data and isinstance(data["image"], Image):
            image: Image = data["image"]
            image = image.convert("RGB")
            image_data = io.BytesIO()
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
            mm_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        elif "image" in data and isinstance(data["image"], str):
            if data["image"].startswith("http://") or data["image"].startswith(
                "file://"
            ):
                image_url = data["image"]
            else:
                image_url = f"file://{data['image']}"

            mm_content = {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        else:
            mm_content = None

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(
        0, tokenizer.vocab_size, size=prefix_len
    ).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(
            prefix_token_ids
            + [
                (offsets[i] + i + j) % tokenizer.vocab_size
                for j in range(input_lens[i])
            ]
        )

        input_requests.append(
            (prompt, int(prefix_len + input_lens[i]), int(output_lens[i]), None)
        )

    return input_requests


def sample_llmpq_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    input_requests = []
    for i, (prompt, output_len) in enumerate(data):
        prompt_len = len(tokenizer(prompt).input_ids)
        input_requests.append((prompt, prompt_len, output_len, None))

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert (
        burstiness > 0
    ), f"A positive burstiness factor is expected, but given {burstiness}."
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = input_requests[0]
    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend."
        )
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = request
        request_func_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2,
        )
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt, prompt_len, output_len, None)
                for prompt, prompt_formatted, prompt_len, output_len, _ in input_requests
            ]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt_formatted, prompt_len, output_len, None)
                for prompt, prompt_formatted, prompt_len, output_len, _ in input_requests
            ]

    elif args.dataset_name == "hf":
        input_requests = sample_hf_requests(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            random_seed=args.seed,
            fixed_output_len=args.hf_output_len,
        )

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "llmpq":
        input_requests = sample_llmpq_requests(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    goodput_config_dict = check_goodput_args(args)

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the " "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf", "llmpq"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).",
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral"],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        "always use the slow tokenizer. \n* "
        '"mistral" will always use the `mistral_common` tokenizer.',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the ``--model`` argument. ",
    )

    args = parser.parse_args()
    main(args)
