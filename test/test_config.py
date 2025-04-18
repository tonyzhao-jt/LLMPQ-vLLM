# take the partition config from user side and launch the scripts
# or take a config file (json file and do launch)
import json
import os
from copy import deepcopy

from llmpq.config import PQConfig
from llmpq.utils import (QUANTIZATION_REGISTRY, get_quantize_dynamic,
                         quantize_model_adaptive, save_ckpt_dummy)

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        qmethod="gptq",
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )

    # combine two configs to get the final config.
    unquantized_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded/model.safetensors"
    smooth_quant_8bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-smoothquant-8/model.safetensors"
    gptq_4bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded-gptq-4/model.safetensors"
    # save_ckpt_dummy(pq_config.model_id_or_path, "./tmp/Llama-3.2-1B-Instruct-adaptive")

    from safetensors.torch import load_file, save_file

    new_states = {}
    smoothquant_tensors = load_file(smooth_quant_8bit_tensors)
    gptq_tensors = load_file(gptq_4bit_tensors)
    unquantized_states = load_file(unquantized_tensors)
    # lm head and embedding
    lm_head_weight = smoothquant_tensors["lm_head.weight"]
    model_embed_weight = smoothquant_tensors["model.embed_tokens.weight"]
    model_norm_weight = smoothquant_tensors["model.norm.weight"]

    unquantized_layer_states = {}
    for k, v in unquantized_states.items():
        if "layers.0" in k:
            unquantized_layer_states[k] = v

    smoothquant_layer_states = {}
    for k, v in smoothquant_tensors.items():
        if "layers.0" in k:
            smoothquant_layer_states[k] = v

    gptqquant_layer_states = {}
    for k, v in gptq_tensors.items():
        if "layers.0" in k:
            gptqquant_layer_states[k] = v

    # craft
    new_states["lm_head.weight"] = lm_head_weight
    new_states["model.embed_tokens.weight"] = model_embed_weight
    new_states["model.norm.weight"] = model_norm_weight
    # craft layers
    bitwidths = pq_config.adaptive_qbits.split(",")
    for layer_idx, bit in enumerate(bitwidths):
        print(layer_idx)
        select_layer_states = unquantized_layer_states
        if bit == 8:
            # use smoothquant
            select_layer_states = smoothquant_layer_states
        elif bit == 4:
            # use gptq
            select_layer_states = gptqquant_layer_states

        for k, v in select_layer_states.items():
            new_k = k.replace("layers.0", f"layers.{layer_idx}")
            new_states[new_k] = deepcopy(v)

    # save the new states
    save_file(new_states, "./tmp/Llama-3.2-1B-Instruct-adaptive/model.safetensors")
    # # 加载 safetensors 格式的检查点
    # checkpoint_path = "./tmp/Llama-3.2-1B-Instruct-adaptive/model.safetensors"
    # state_dict = load_file(checkpoint_path)

    # # 打印原始的键名
    # print("原始的键名:")
    # for key in state_dict.keys():
    #     print(key)

    # # get the dynamic
    # pattern:str=r"layers\.(\d+)\."
    # dynamic = get_quantize_dynamic(pq_config.model_id_or_path, pq_config, pattern=pattern)
    # print(dynamic)
    # # dump the dymaic to tmp
    # with open("./tmp/dynamic.json", "w") as f:
    #     json.dump(dynamic, f)
    # quant_path = "./tmp/Llama-3.2-1B-Instruct-adaptive"
    # if os.path.exists(quant_path):
    #     # load the model from the path.
    #     model = QUANTIZATION_REGISTRY[pq_config.qmethod].get_model(quant_path)
    # else:
    #     # perform adaptive quantization based on that.
    #     quantize_model_adaptive(
    #         pq_config.model_id_or_path, quant_path, pq_config
    #     )  # noqa
    # pq_config.save("./tmp")
