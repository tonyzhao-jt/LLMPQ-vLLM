# take the partition config from user side and launch the scripts
# or take a config file (json file and do launch)
import os
import json 
from llmpq.config import PQConfig
from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model_adaptive, get_quantize_dynamic

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        qmethod="gptq",
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
    # get the dynamic 
    pattern:str=r"layers\.(\d+)\."
    dynamic = get_quantize_dynamic(pq_config.model_id_or_path, pq_config, pattern=pattern)
    print(dynamic)
    # dump the dymaic to tmp
    with open("./tmp/dynamic.json", "w") as f:
        json.dump(dynamic, f)
    quant_path = "./tmp/Llama-3.2-1B-Instruct-adaptive"
    if os.path.exists(quant_path):
        # load the model from the path.
        model = QUANTIZATION_REGISTRY[pq_config.qmethod].get_model(quant_path)
    else:
        # perform adaptive quantization based on that.
        quantize_model_adaptive(
            pq_config.model_id_or_path, quant_path, pq_config
        )  # noqa
    pq_config.save("./tmp")
