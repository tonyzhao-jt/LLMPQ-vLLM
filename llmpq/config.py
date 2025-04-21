import json
import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Any 

logger = getLogger(__name__)

def _default_bits_factory() -> List[Any]:
    from .utils import get_device_capacity
    major, _ = get_device_capacity()
    if major > 7:
        return [4, 8, '8-tc', 16]
    else:
        return [4, 8, 16]

def _default_bits_wo_info_factory() -> List[Any]:
    return [4, 8, 16]

@dataclass
class PQConfig:
    model_id_or_path: str = ""  # e.g. "meta-llama/Llama-3.2-1B"
    partition_config: str = ""  # e.g. "2,6,6,2"
    pipeline_parallel_size: int = 0  # e.g. 4
    qmethod: str = ("",)  # e.g. gptq
    adaptive_qbits: str = ""  # e.g. "4,4,8,8,8"
    num_layers: int = 16
    prepost_bit: int = 8
    # mixed-precision setup
    random_bits: bool = False # assign random bits
    bit_4_q_method: str = 'gptq'
    bit_8_q_method: str = 'gptq'
    bit_8_q_tc_method: str = 'smoothquant'
    # ref model path
    ref_4_qmodel_path: str = None
    ref_8_qmodel_path: str = None
    ref_8_tc_qmodel_path: str = None
    ref_16_model_path: str = None
    # working dir
    work_dir: str = '/tmp/llmpq/work_dir'
    # v1: algo related
    gamma: float = 0.5 # expected generated tokens
    theta: float = 0.1 # control the concern for accuracy
    MEM_UNIT: str = 'MB'
    AVAILABLE_BITS: List[Any] = field(
        default_factory=_default_bits_factory
    )
    AVAILABLE_BITS_WO_INFO: List[Any] = field(
        default_factory=_default_bits_wo_info_factory
    )
    CUDA_CONTEXT_MEM: float = 430 + 1500 # 430MB cuda context allocation + 1.5 G Torch Temp Allocation
                              # conforms to the huggingface's script, which reduce by 2GB
    RATIO_AVOID_OOM: float = 0.95 # 95% of the memory is used to avoid OOM
    SLO_RATE: float = 1.5 # times of fp16 inference time to be SLO.
    
    # dump and load function (in json)
    def dump(self):
        return {
            "partition_config": self.partition_config,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "adaptive_qbits": self.adaptive_qbits,
        }

    def exec_scripts(self, port: int = 5678, run_dummy: bool = False):
        """
        dump to the execution scripts
        """
        from llmpq.utils import is_partition_config_valid
        assert is_partition_config_valid(
            (self.partition_config, self.pipeline_parallel_size),
            self.num_layers,  # noqa
        )

        qbits_num = len(self.adaptive_qbits.split(","))
        assert (
            qbits_num == self.num_layers
        ), f"qbits {self.adaptive_qbits} number {qbits_num} not matched"

        ray_scripts = f"""
<remove me: head>
ray start --head --port {port}
export VLLM_PP_LAYER_PARTITION="{self.partition_config}"

<remove me: worker>
ray start --address='ray://<address_above>:{port}'
export VLLM_PP_LAYER_PARTITION="{self.partition_config}"
        """.strip()

        exec_scripts = f"""
vllm serve <your_quant_path> \\
    --tensor-parallel-size 4 \\
    --pipeline-parallel-size 2 \\
{"--load-format dummy" if run_dummy else ""}
        """.strip()

        return ray_scripts, exec_scripts

    def save(self, folder: str):
        # create folder if not exsits
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, "pq_config.json")
        with open(file_path, "w") as f:
            json.dump(self.dump(), f)
        exec_scripts = self.exec_scripts()
        with open(os.path.join(folder, "ray_scripts.sh"), "w") as f:
            f.write(exec_scripts[0])
        with open(os.path.join(folder, "exec_scripts.sh"), "w") as f:
            f.write(exec_scripts[1])

        logger.info(
            f"Config saved to {file_path} \n"
            f"Ray scripts saved to {os.path.join(folder, 'ray_scripts.sh')} \n"  # noqa
            f"Exec scripts saved to {os.path.join(folder, 'exec_scripts.sh')} \n"  # noqa
        )

    def load(self, file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        self.partition_config = config["partition_config"]
        self.pipeline_parallel_size = config["pipeline_parallel_size"]
        self.adaptive_qbits = config["adaptive_qbits"]


@dataclass
class GenerationConfig:
    global_bz: int
    micro_bz: int
    # prompt length, generated sequence length
    s: int
    n: int 

# init one 
gen_config = GenerationConfig(global_bz=16, micro_bz=4, s=512, n=100)