import gzip
import os
import shutil
import time
from typing import Dict,  Optional

from llmpq.dataset import DummyDataset
from llmpq.config import PQConfig
from llmpq.core import create_mix_precision_shards
from llmpq.utils import get_device_name_by_torch  # noqa
from vllm import LLM, SamplingParams

def profile_model(
    pq_config: PQConfig,
    inputs: Dict[str, int],
    tp_size: int = 1,  # tp = 2 will hang the process.
    warmup: int = 5,
    repeat: int = 10,
    PROFILER_RAW: Optional[str] = None,  # noqa
    PROFILER_PARSED: Optional[str] = None,  # noqa
    overwrite: bool = False,
    dtype: str = 'half'
) -> Dict[str, Dict[str, float]]:
    model_id = pq_config.model_id_or_path
    model_id_wo_special = model_id.replace("/", "_")
    work_dir = pq_config.work_dir

    device_name = get_device_name_by_torch()
    # set dir
    if PROFILER_RAW is None:
        PROFILER_RAW = os.path.join(work_dir, device_name, "vllm_profile")
        os.environ['VLLM_TORCH_PROFILER_DIR'] = PROFILER_RAW

    if PROFILER_PARSED is None:
        PROFILER_PARSED = os.path.join(work_dir, device_name, "vllm_profile_parsed")

    batch_size, prompt_length, output_tokens = (
        inputs["batch_size"],
        inputs["prompt_len"],
        inputs["output_tokens"],
    )
    prompts = DummyDataset(
        batch_size=batch_size, prompt_len=prompt_length
    ).gen_prompts()

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        ignore_eos=True,
        max_tokens=output_tokens,
    )

    bitwidth_model_shard_paths = create_mix_precision_shards(
        pq_config, overwrite, candidate_bitwidth=set(pq_config.AVAILABLE_BITS)
    )


    output_files = []
    gpu_memory_utilization = 0.8

    LLM_params = {
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        'enable_prefix_caching': False,
        "enable_chunked_prefill": False,
        'max_model_len': 2048,
    }

    for bit, model_path in bitwidth_model_shard_paths.items():
        LLM_params['model'] = model_path
        llm = LLM(
            **LLM_params,
        )  # noqa
        for _ in range(warmup):
            outputs = llm.generate(prompts, sampling_params)

        llm.start_profile()
        for _ in range(repeat):
            outputs = llm.generate(prompts, sampling_params)  # noqa
        llm.stop_profile()

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")  # noqa

        time.sleep(10)
        del llm
        # find the first file under the profiler, rename it, move to PROFILER_PARSED # noqa
        # if Profile RAW not exists, create it
        if not os.path.exists(PROFILER_PARSED):
            os.makedirs(PROFILER_PARSED)
        for file in os.listdir(PROFILER_RAW):
            if file.endswith(".gz"):
                new_file_name = f"{model_id_wo_special}-{bit}-{tp_size}-pt.trace.json.gz"  # noqa
                os.rename(
                    os.path.join(PROFILER_RAW, file),
                    os.path.join(
                        PROFILER_PARSED,
                        new_file_name,
                    ),  # noqa
                )
                # unzip it (.gz format)
                with gzip.open(
                    os.path.join(PROFILER_PARSED, new_file_name), "rb"
                ) as f_in:  # noqa
                    with open(
                        os.path.join(PROFILER_PARSED, new_file_name[:-3]),
                        "wb",  # noqa
                    ) as f_out:  # noqa
                        shutil.copyfileobj(f_in, f_out)
                        output_files.append(
                            os.path.join(
                                PROFILER_PARSED, new_file_name[:-3]
                            )  # noqa
                        )
    return output_files


if __name__ == "__main__":
    device_name = get_device_name_by_torch()
    PROFILER_RAW = f"./tmp/{device_name}/vllm_profile"
    PROFILER_PARSED = f"./tmp/{device_name}/vllm_profile_parsed"
    REPEAT = 10
    WARMUP = 2


    # only profile 2 layers for quant and profiling.
    for bs in [2, 4, 8]:
        PROFILER_RAW_BATCH = f"{PROFILER_RAW}/{bs}"
        PROFILER_PARSED_BATCH = f"{PROFILER_PARSED}/{bs}"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILER_RAW_BATCH
        consider_inputs = {
            "batch_size": bs,
            "prompt_len": 512,
            "output_tokens": 10,
        }
        pq_config = PQConfig(
            model_id_or_path="Qwen/Qwen2.5-32B",
            partition_config="2,6,6,2",
            AVAILABLE_BITS=[
                3, 4, 8, 16
            ],
            pipeline_parallel_size=4,
            random_bits=True,
            adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
            num_layers=16,
        )
        output_files = profile_model(
            pq_config,
            consider_inputs,
            warmup=WARMUP,
            repeat=REPEAT,
            PROFILER_RAW=PROFILER_RAW_BATCH,
            PROFILER_PARSED=PROFILER_PARSED_BATCH,
        )