# run vllm in arbitrary partition
# SPDX-License-Identifier: Apache-2.0
# https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_pipeline_partition.py
# https://github.com/vllm-project/vllm/tree/main/tests/distributed
import os

from transformers import AutoConfig
from vllm import SamplingParams

from llmpq.optimizer import is_partition_config_valid

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# use autoconfig to get the layer number of the model
model_id = "meta-llama/Llama-3.2-1B"
config = AutoConfig.from_pretrained(model_id)
num_layers = config.num_hidden_layers
partition_config = "2,6,6,2"
pipeline_parallel_size = 4
is_partition_config_valid(
    (partition_config, pipeline_parallel_size), num_layers
)  # noqa
os.environ["VLLM_PP_LAYER_PARTITION"] = partition_config

"""
    vllm serve meta-llama/Llama-3.2-1B \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --load-format dummy
"""
