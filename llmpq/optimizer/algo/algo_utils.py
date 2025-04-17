import itertools
import os

import gurobipy as gp
import numpy as np
import pulp

from llmpq.costmodel.mem import (estimate_all_layer_mem,
                                 get_mem_with_layer_bit_pair)
from llmpq.utils import assign_uniform_bit
from llmpq.utils.v1.device import get_single_device_mem_constraints
from llmpq.logger import init_logger

logger = init_logger(__name__)

SIGNAL_BASE = 1234
NOT_AVAILABLE = SIGNAL_BASE + 1
FP16_ENOUGH = SIGNAL_BASE + 2

def get_final_strat_file_name(model_id: str, device_info):
    file_name = f"sols_" + f"{model_id}" + "_" + device_info + ".pkl"
    return file_name


def ilp_env(timeLimit=None):
    # check whether file exists under
    path = "/opt/gurobi/"
    if not os.path.exists(path):
        raise Exception("Gurobi is not installed")
    env = gp.Env(empty=True)
    env.setParam("WLSACCESSID", "1b28dca7-337e-4811-b346-01087e09cd64")
    env.setParam("WLSSECRET", "629520bd-a114-45d7-b828-bfc5235c198d")
    env.setParam("LICENSEID", 965996)
    if timeLimit is not None:
        env.setParam("TimeLimit", timeLimit)
    env.start()


def create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance):
    args = {"msg": verbose_ilp, "timeLimit": ilp_time_limit, "MIPGap": ilp_tolerance}
    if ilp_tolerance is None:
        args.pop("MIPGap")
    if ilp_time_limit is None:
        args.pop("timeLimit")
    solver = pulp.GUROBI(**args)
    return solver


def estimate_min_max_mem(estimator, layers, max_bit=16, min_bit=2):
    bit_map = {}
    assign_uniform_bit(layers, max_bit, bit_map)
    max_mem = estimate_all_layer_mem(estimator, layers, bit_map)
    assign_uniform_bit(layers, min_bit, bit_map)
    min_mem = estimate_all_layer_mem(estimator, layers, bit_map)
    return max_mem, min_mem


# layer device bit
def interpret_ilp_result_i_j_b(ilp_result, BITs):
    device_layer_dict = {}
    layer_to_bit_map = {}
    for layer, (device_rank, bit_idx) in ilp_result.items():
        bit_pair = BITs[bit_idx]
        if device_rank not in device_layer_dict:
            device_layer_dict[device_rank] = [layer]
            layer_to_bit_map[device_rank] = [bit_pair]
        else:
            device_layer_dict[device_rank].append(layer)
            layer_to_bit_map[device_rank].append(bit_pair)

    partition_result = {}
    start = 0
    # sort device_layer_dict by device_rank
    device_layer_dict = {k: device_layer_dict[k] for k in sorted(device_layer_dict)}
    # sort the partition among layers.
    for device_rank, layers in device_layer_dict.items():
        partition_result[device_rank] = [start, start + len(layers)]
        start += len(layers)
    # generate bitwidth mapping
    bit_assignment_result = {}
    for device_rank, (layer_start, layer_end) in partition_result.items():
        bit_pairs = layer_to_bit_map[device_rank]
        bit_pair_idx = 0
        for layer in range(layer_start, layer_end):
            attn_layer = layer * 2
            ffn_layer = layer * 2 + 1
            bit_pair = bit_pairs[bit_pair_idx]
            attn_bit, ffn_bit = bit_pair
            # map
            bit_assignment_result[attn_layer] = attn_bit
            bit_assignment_result[ffn_layer] = ffn_bit
            bit_pair_idx += 1

    return {
        "partition_result": partition_result,
        "bit_assignment": bit_assignment_result,
    }


def get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size):
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = (
        np.tile(mem_bits_vector, (group_L, 1)) * group_size
    )  # repeat the mem_bits_vector for group_L times
    M = np.ceil(M).astype(int)  # ceil
    return M


def get_device_topo_available_mem_with_order(
    current_D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=1
):
    M_d = np.array(
        [
            get_single_device_mem_constraints(device_name)
            for d_rank, device_name in current_D.items()
        ]
    )
    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit="MB")[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(
        prefill_bz, bz_decode_max, unit="MB"
    )[0]
    temp_later_decode = model_mem_estimator.calculate_temp_tensor_size_next_i(
        unit="MB"
    )[0]
    M_d[0] -= post_pre_mem
    if len(M_d) > 1:
        M_d[1:] -= temp_later_decode * time_mult_times
    M_d[0] -= max(temp_tensor_mem, temp_later_decode * time_mult_times)
    return M_d


def get_combinations(input_list, num_objects):
    """
    This function takes a list and a number of objects to choose from the list.
    It returns all possible combinations of the list for the given number of objects.
    """
    return list(itertools.combinations(input_list, num_objects))


def force_zero_3d(lat, z, prob):
    lat_shape = lat.shape
    for i in range(lat_shape[0]):
        for j in range(lat_shape[1]):
            for b in range(lat_shape[2]):
                if lat[i][j][b] >= float("inf"):
                    prob += z[(i, j, b)] == 0


def force_zero_2d(lat, z, prob):
    lat_shape = lat.shape
    for i in range(lat_shape[0]):
        for j in range(lat_shape[1]):
            if lat[i][j] >= float("inf"):
                prob += z[(i, j)] == 0


def force_zero(lat, z, prob):
    lat_shape = lat.shape
    if len(lat_shape) == 2:
        force_zero_2d(lat, z, prob)
    elif len(lat_shape) == 3:
        force_zero_3d(lat, z, prob)


def set_root_folder() -> setattr:
    ROOT_DIR = os.environ.get("ROOT_DIR", None)
    # set to the tmp folder under cwd
    if ROOT_DIR is None:
        ROOT_DIR = os.path.join(os.getcwd(), "tmp")
        os.makedirs(ROOT_DIR, exist_ok=True)
        os.environ["ROOT_DIR"] = ROOT_DIR
        logger.info(f"ROOT_DIR is set to {ROOT_DIR}")
    # check
    assert ROOT_DIR is not None, "ROOT_DIR is not set"
    return ROOT_DIR