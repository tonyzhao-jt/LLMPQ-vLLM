import json
import os
from copy import deepcopy

from llmpq.config import PQConfig
from llmpq.utils import (
    QUANTIZATION_REGISTRY,
    get_quantize_dynamic,
    quantize_model_adaptive,
    save_ckpt_dummy,
)
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

import argparse

parser = argparse.ArgumentParser()
# model
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of the model.",
)

parser.add_argument(
    "--num-cards",
    type=int,
    required=True,
    help="Number of cards.",
)

parser.add_argument(
    "--bit",
    type=int,
    required=True,
    help="Number of bits.",
)

if __name__ == "__main__":
    model_id = parser.parse_args().model
    num_cards = parser.parse_args().num_cards
    bit = parser.parse_args().bit
    config = AutoConfig.from_pretrained(model_id)
    num_layers = config.num_hidden_layers
    partition = [num_layers // num_cards] * num_cards
    partition_config = ",".join([str(x) for x in partition])
    pipeline_parallel_size = num_cards
    adaptive_qbits = [bit] * num_layers
    adaptive_qbits = ",".join([str(x) for x in adaptive_qbits])
    pq_config = PQConfig(
        model_id_or_path=model_id,
        partition_config=partition_config,
        pipeline_parallel_size=pipeline_parallel_size,
        adaptive_qbits=adaptive_qbits,
        num_layers=num_layers,
    )

    pq_config.save("./tmp/execute")
