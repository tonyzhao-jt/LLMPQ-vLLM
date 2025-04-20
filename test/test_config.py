import json
import os
from copy import deepcopy

from llmpq.config import PQConfig
from llmpq.utils import (QUANTIZATION_REGISTRY, get_quantize_dynamic,
                         quantize_model_adaptive, save_ckpt_dummy)
from safetensors.torch import load_file, save_file

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        qmethod="gptq",
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
