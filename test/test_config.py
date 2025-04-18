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

    # combine two configs to get the final config.
    unquantized_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded/model.safetensors"
    smooth_quant_8bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-smoothquant-8/model.safetensors"
    gptq_4bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded-gptq-4/model.safetensors"
    # save_ckpt_dummy(pq_config.model_id_or_path, "./tmp/Llama-3.2-1B-Instruct-adaptive")

    # 检查文件是否存在
    for file_path in [unquantized_tensors, smooth_quant_8bit_tensors, gptq_4bit_tensors]:
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，请检查路径。")
            break
    else:
        new_states = {}
        try:
            smoothquant_tensors = load_file(smooth_quant_8bit_tensors)
            gptq_tensors = load_file(gptq_4bit_tensors)
            unquantized_states = load_file(unquantized_tensors)
        except Exception as e:
            print(f"加载文件时出错: {e}")
        else:
            # lm head and embedding
            lm_head_weight = smoothquant_tensors["lm_head.weight"]
            model_embed_weight = smoothquant_tensors["model.embed_tokens.weight"]
            model_norm_weight = smoothquant_tensors["model.norm.weight"]

            unquantized_layer_states = {k: v for k, v in unquantized_states.items() if "layers.0" in k}
            smoothquant_layer_states = {k: v for k, v in smoothquant_tensors.items() if "layers.0" in k}
            gptqquant_layer_states = {k: v for k, v in gptq_tensors.items() if "layers.0" in k}

            # modify smooth quant xxx_scale
            keys = list(smoothquant_layer_states.keys())
            for k in keys:
                if 'scale' in k:
                    smoothquant_layer_states[k] = smoothquant_layer_states[k][0]

            # craft
            new_states["lm_head.weight"] = lm_head_weight
            new_states["model.embed_tokens.weight"] = model_embed_weight
            new_states["model.norm.weight"] = model_norm_weight

            # craft layers
            bitwidths = pq_config.adaptive_qbits.split(",")
            for layer_idx, bit in enumerate(bitwidths):
                if bit == '8':
                    select_layer_states = smoothquant_layer_states
                elif bit == '4':
                    select_layer_states = gptqquant_layer_states
                else:
                    select_layer_states = unquantized_layer_states

                for k, v in select_layer_states.items():
                    new_k = k.replace("layers.0", f"layers.{layer_idx}")
                    new_states[new_k] = deepcopy(v)

            # save the new states
            save_dir = "./tmp/Llama-3.2-1B-Instruct-adaptive"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            try:
                save_file(new_states, os.path.join(save_dir, "model.safetensors"))
                print("新的 safetensors 文件已保存。")
            except Exception as e:
                print(f"保存文件时出错: {e}")

            # 加载 safetensors 格式的检查点
            checkpoint_path = os.path.join(save_dir, "model.safetensors")
            try:
                state_dict = load_file(checkpoint_path)
                # 打印原始的键名
                print("原始的键名:")
                for key in state_dict.keys():
                    print(key)
            except Exception as e:
                print(f"加载保存的文件时出错: {e}")

            # get the dynamic
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