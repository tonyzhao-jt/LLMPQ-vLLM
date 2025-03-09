import json
import os
from dataclasses import dataclass
from logging import getLogger

from .optimizer import is_partition_config_valid

logger = getLogger(__name__)


@dataclass
class PQConfig:
    model_id_or_path: str = ""  # e.g. "meta-llama/Llama-3.2-1B"
    partition_config: str = ""  # e.g. "2,6,6,2"
    pipeline_parallel_size: int = 0  # e.g. 4
    qmethod: str = ("",)  # e.g. gptq
    adaptive_qbits: str = ""  # e.g. "4,4,8,8,8"
    num_layers: int = 16

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
vllm serve meta-llama/Llama-3.2-1B \\
    --tensor-parallel-size 4 \\
    --pipeline-parallel-size 2 \\
{"--load-format dummy" if run_dummy else ""}
        """.strip()

        return ray_scripts, exec_scripts

    def save(self, folder: str):
        # create folder if not exsits
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, "config.json")
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
