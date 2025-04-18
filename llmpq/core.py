from .config import PQConfig
from llmpq.logger import init_logger
import os 
import json
from safetensors.torch import load_file, save_file
from copy import deepcopy

logger = init_logger(__name__)
def create_ada_model(
        pq_config: PQConfig, 
        save_dir: str, 
        pattern: str=r"layers\.(\d+)\.",
        overwrite: bool=False,
    ):
    from llmpq.profiler import shard_model
    from llmpq.utils import get_quantize_dynamic
    from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model
    model_id = pq_config.model_id_or_path
    bitwidths = set(pq_config.adaptive_qbits.split(","))
    bit_4_q_method = pq_config.bit_4_q_method
    bit_8_q_method = pq_config.bit_8_q_method
    bits_method = {
        4: bit_4_q_method,
        8: bit_8_q_method,
        16: None,
    }
    assert bit_4_q_method in QUANTIZATION_REGISTRY
    assert bit_8_q_method in QUANTIZATION_REGISTRY
    work_dir = pq_config.work_dir
    # check if exists, if not create one
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    save_path_dict = {}
    for bit, method in bits_method.items():
        logger.info(f"quantize {bit}bit with {method}")
        # shard the model to 2 layers and quantize it
        save_path = os.path.join(work_dir, f"tmp-{bit}bit")
        q_save_path = os.path.join(work_dir, f"qtmp-{bit}bit")
        if os.path.exists(q_save_path) and not overwrite:
            logger.info(f"q_save_path {q_save_path} exists, skip")
        else:
            shard_model(model_id, save_path, layer_num=2)
        if bit != 16:
            if os.path.exists(q_save_path):
                logger.info(f"q_save_path {q_save_path} exists, skip")
            else:
                quantize_model(method, save_path, q_save_path, bits=bit)
            save_path_dict[bit] = q_save_path
        else:
            save_path_dict[bit] = save_path
    
    # temporarily works like that, change later.
    unquantized_tensors = os.join(save_path_dict[16], "model.safetensors")
    smooth_quant_8bit_tensors = os.join(save_path_dict[8], "model.safetensors")
    gptq_4bit_tensors = os.join(save_path_dict[4], "model.safetensors")

    # check all files exists or not
    if not os.path.exists(unquantized_tensors):
        raise ValueError(f"unquantized_tensors {unquantized_tensors} not exists")
    if not os.path.exists(smooth_quant_8bit_tensors):
        raise ValueError(f"smooth_quant_8bit_tensors {smooth_quant_8bit_tensors} not exists")
    if not os.path.exists(gptq_4bit_tensors):
        raise ValueError(f"gptq_4bit_tensors {gptq_4bit_tensors} not exists")

    # combine two configs to get the final config.
    # unquantized_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded/model.safetensors"
    # smooth_quant_8bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-smoothquant-8/model.safetensors"
    # gptq_4bit_tensors = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/NVIDIA_A100-SXM4-40GB/Llama_3.2_1B_Instruct_sharded-gptq-4/model.safetensors"
    # save_ckpt_dummy(pq_config.model_id_or_path, "./tmp/Llama-3.2-1B-Instruct-adaptive")

    logger.info(f"Generate CKPT for {model_id}, bits: {pq_config.adaptive_qbits}")
    for file_path in [unquantized_tensors, smooth_quant_8bit_tensors, gptq_4bit_tensors]:
        if not os.path.exists(file_path):
            logger.info(f"file {file_path} not exists")
            break
    else:
        new_states = {}
        try:
            smoothquant_tensors = load_file(smooth_quant_8bit_tensors)
            gptq_tensors = load_file(gptq_4bit_tensors)
            unquantized_states = load_file(unquantized_tensors)
        except Exception as e:
            raise e
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
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            try:
                save_file(new_states, os.path.join(save_dir, "model.safetensors"))
                logger.info(f"new states saved to {save_dir}")
            except Exception as e:
                raise e

            checkpoint_path = os.path.join(save_dir, "model.safetensors")
            try:
                state_dict = load_file(checkpoint_path)
                logger.info("original keys:")
                for key in state_dict.keys():
                    logger.info(key)
            except Exception as e:
                raise e
    
    logger.info("Generate QConfig")
    dynamic = get_quantize_dynamic(pq_config.model_id_or_path, pq_config, pattern=pattern)
    quantization_config = {
        'quant_method': 'llmpq',
        'dynamic': dynamic,
    }
    # dump the dymaic to tmp
    qconfig_path = os.join(save_dir, "quantization_config.json")
    with open(qconfig_path, "w") as f:
        json.dump(quantization_config, f)
    logger.info(f"Generate Done, quantization_config {quantization_config}")