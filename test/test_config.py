# take the partition config from user side and launch the scripts
# or take a config file (json file and do launch)
import os

from llmpq.config import PQConfig
from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model_adaptive

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        qmethod="gptq",
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
    quant_path = "./tmp/Llama-3.2-1B-Instruct-gptqmodel-4bit-dynamic"
    if os.path.exists(quant_path):
        # load the model from the path.
        model = QUANTIZATION_REGISTRY[pq_config.qmethod].get_model(quant_path)
    else:
        # perform adaptive quantization based on that.
        quantize_model_adaptive(
            pq_config.model_id_or_path, quant_path, pq_config
        )  # noqa
    pq_config.save("./tmp")
