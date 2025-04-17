from typing import List, Optional

from transformers import PretrainedConfig

from llmpq.costmodel.mem import create_mem_estimator
from llmpq.utils import get_h1_h2_from_config
from llmpq.costmodel.comm import CommCostModel
from llmpq.costmodel.lat import LatCostModel


def init_cost_model(
    config: PretrainedConfig,
    gb_size: int,
    micro_bz: int,
    prompt_len: int,
    output_token_num: int,
    device_names: List[str],
    device_cnts: List[int],
    comm_cost_model_folder: Optional[str] = None,
    cost_model_store_path: str = '/tmp/llmpq_costmodel'
):
    # init the cost model with configs
    h1, h2 = get_h1_h2_from_config(config)
    num_hidden_layers = config.num_hidden_layers
    model_mem_estimator = create_mem_estimator(h1, h2, gb_size, prompt_len, output_token_num, config)

    # we assume each pod posses homogenous devices
    assert len(device_names) > 0, "device_names cannot be empty"
    assert len(device_names) == len(
        device_cnts
    ), "device_names and device_cnts must have the same length"
    single_card = device_cnts[0] == 1
    comm_cost_model = CommCostModel(
        comm_cost_model_folder=comm_cost_model_folder, single_card=single_card
    )
    lat_cost_model = LatCostModel(
        device_names, cost_model_store_path=cost_model_store_path
    )
    lat_cost_model.register_hyper_params(micro_bz, prompt_len, output_token_num, h1, h2)
    return model_mem_estimator, comm_cost_model, lat_cost_model, num_hidden_layers


if __name__ == "__main__":
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    init_cost_model(config,
                    gb_size=8,
                    micro_bz=1,
                    prompt_len=1024,
                    output_token_num=1024,
                    device_names=["NVIDIA_A100"],
                    device_cnts=[1]
                    )
