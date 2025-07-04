import itertools

# import logger
import logging
import math
import os

# default libs
import pickle
from collections import defaultdict

import numpy as np

# setup ilp configs
import pulp

from llmpq.config import PQConfig, gen_config
from llmpq.costmodel.comm.utils import get_comm_cost, get_comm_payload_size
from llmpq.costmodel.core import init_cost_model
from llmpq.costmodel.lat import get_latency_with_layer_device_bit_pair
from llmpq.optimizer.algo.algo_utils import (
    NOT_AVAILABLE,
    create_ilp_solver,
    force_zero,
    get_device_topo_available_mem_with_order,
    get_M_with_bitwidth_pair,
    interpret_ilp_result_i_j_b,
    set_root_folder,
)
from llmpq.profiler.indicator.v1 import assign_omega_constant  # noqa
from llmpq.profiler.indicator.v1 import assign_omega_uniform
from llmpq.utils.bits_pair import get_available_bits_pair
from llmpq.utils.v1.device import create_device_mesh_and_mem, get_device_info
from llmpq.utils.v1.miscs import (
    decouple_result_group,
    get_default_decode_bz,
    get_factors,
    partition_a_into_b_bins,
)

logger = logging.getLogger(__name__)

# default libs
# setup ilp configs
import pulp

unit = PQConfig.MEM_UNIT
slo_rate = PQConfig.SLO_RATE

import math


def solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, bz_pack):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    l_prefill, l_decode = l
    comm_prefill, comm_decode = comm
    global_bz, prefill_bz, bz_decode_max = bz_pack
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts(
        "z",
        [(i, j, b) for i in range(L) for j in range(N) for b in range(B)],
        cat=pulp.LpBinary,
    )
    y = pulp.LpVariable.dicts(
        "y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary
    )
    x = pulp.LpVariable.dicts(
        "x", [(i, j) for i in range(L) for j in range(N)], cat=pulp.LpBinary
    )

    T_prefill_j = pulp.LpVariable.dicts(
        "T_prefill_j", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous
    )
    T_decode_j = pulp.LpVariable.dicts(
        "T_decode_j", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous
    )
    T_prefill = pulp.LpVariable("T_prefill", lowBound=0, cat=pulp.LpContinuous)
    T_decode = pulp.LpVariable("T_decode", lowBound=0, cat=pulp.LpContinuous)

    # from alpa
    # cost = stage_sum + (micro_bs - 1) * lowest_stage_cost
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)
    prefill_stage_sum = pulp.lpSum([T_prefill_j[j] for j in range(N)])
    decode_stage_sum = pulp.lpSum([T_decode_j[j] for j in range(N)])
    prob += (
        prefill_stage_sum
        + (prefill_micro_bs_num - 1) * T_prefill
        + decode_stage_sum
        + (decode_micro_bs_num - 1) * T_decode * (mu_n - 1)
        + theta
        * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])
    )

    force_zero(l_prefill, z, prob)
    # Define the constraints
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for i in range(L):
        for j in range(N):
            prob += pulp.lpSum([z[(i, j, b)] for b in range(B)]) == x[(i, j)]

    for i in range(L):
        prob += pulp.lpSum([x[(i, j)] for j in range(N)]) == 1
    for i in range(L):
        prob += pulp.lpSum([y[(i, b)] for b in range(B)]) == 1
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1

    for j in range(N):
        prob += (
            pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)])
            <= M_d[j]
        )
        prob += (
            pulp.lpSum(
                [z[(i, j, b)] * l_prefill[i][j][b] for i in range(L) for b in range(B)]
            )
            <= T_prefill_j[j]
        )
        prob += (
            pulp.lpSum(
                [z[(i, j, b)] * l_decode[i][j][b] for i in range(L) for b in range(B)]
            )
            <= T_decode_j[j]
        )

        prob += T_prefill_j[j] >= comm_prefill[j]
        prob += T_decode_j[j] >= comm_decode[j]
        prob += T_prefill >= T_prefill_j[j]
        prob += T_decode >= T_decode_j[j]

    prob += x[(0, 0)] == 1  # the first layer must lie in the first device
    prob += x[(L - 1, N - 1)] == 1
    if N > 1:
        for i in range(1, L):
            for j in range(N - 1):
                for k in range(j + 1, N):
                    prob += x[(i, j)] + x[(i - 1, k)] <= 1

    solver = create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance)
    status = prob.solve(solver)

    if status == pulp.LpStatusOptimal:
        # Print the solution status
        # print("llm_pq Result Found")
        # store the optimal solution
        result = {}
        # print z variable result
        for i in range(L):
            for j in range(N):
                for b in range(B):
                    if z[(i, j, b)].varValue > 0:
                        # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                        result[i] = (j, b)
        # min([v.varValue for k, v in z.items()])
        # sum omega
        # print("omega = {}".format(pulp.value(pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)]))))
        # print latency T_sum
        T_sum = (
            pulp.value(prefill_stage_sum)
            + (prefill_micro_bs_num - 1) * T_prefill.varValue
            + pulp.value(decode_stage_sum)
            + (decode_micro_bs_num - 1) * T_decode.varValue * (mu_n - 1)
        )
        print("T_sum = {}".format(T_sum))
        print("Objective = {}".format(pulp.value(prob.objective)))
        return result, pulp.value(prob.objective)
    else:
        return NOT_AVAILABLE, 1e10


def prepare_for_ilp(
    num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack
):
    time_mult_times = PQConfig.TIME_MULT_TIMES
    model_mem_estimator, comm_cost_model, lat_cost_model = cost_model_pack
    global_bz, prefill_bz, bz_decode_max = bz_pack

    L = num_hidden_layers  # in partition, regard as a whole
    N = len(current_D)  # number of devices
    assert (
        L % group_size == 0
    ), f"L should be divisible by group_size, but L={L}, group_size={group_size}"
    group_L = L // group_size

    BITs = get_available_bits_pair(available_bits)
    """
        Constraint related
    """
    # get memory requirement of each layer
    M = get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size)
    M_d = get_device_topo_available_mem_with_order(
        current_D,
        model_mem_estimator,
        prefill_bz,
        bz_decode_max,
        time_mult_times=time_mult_times,
    )

    # latency table for prefill and decode
    # latency
    l_prefill = np.zeros((group_L, N, len(BITs)))
    l_decode = np.zeros((group_L, N, len(BITs)))

    for i in range(group_L):
        l_prefill[i, :, :] = (
            get_latency_with_layer_device_bit_pair(
                current_D,
                BITs,
                lat_cost_model,
                prefill_bz,
                s,
                0,
                use_profiler_prediction=use_profiler_prediction,
            )
            * group_size
        )
        l_decode[i, :, :] = (
            get_latency_with_layer_device_bit_pair(
                current_D,
                BITs,
                lat_cost_model,
                bz_decode_max,
                1,
                s + int(mu_n / 2),
                use_profiler_prediction=use_profiler_prediction,
            )
            * group_size
        )
    # preposet
    if len(current_D) > 1:
        first_device_name = current_D[0]
        # prefill
        prefill_prepost_cost = lat_cost_model.fetch_prepost_lat(
            first_device_name, 0, prefill_bz, s
        ) * math.ceil(global_bz / prefill_bz)
        # decode
        # print(bz_decode_max, s + int(mu_n / 2))
        decode_prepost_cost = lat_cost_model.fetch_prepost_lat(
            first_device_name, 1, bz_decode_max, s + int(mu_n / 2)
        )
        # add to l_prefill and l_decode
        l_prefill[:, 0, :] += prefill_prepost_cost
        l_decode[:, 0, :] += decode_prepost_cost
    # omega
    omega = assign_omega_constant(group_L, BITs)
    # omega = assign_omega_uniform(group_L, BITs)
    if omega_file is not None:
        # open and load with pickle
        with open(omega_file, "rb") as f:
            omega_loaded = pickle.load(f)
        # check whether the shape is matched, as raise error
        # all_BITs = get_available_bits_pair(PQConfig.AVAILABLE_BITS)
        # BITs_idx = [all_BITs.index(bit_pair) for bit_pair in BITs]
        # omega_loaded = omega_loaded[:, BITs_idx]
        if omega_loaded.shape[0] != group_L and omega_loaded.shape[0] == L:
            new_omega_loaded = np.zeros((group_L, omega_loaded.shape[1]))
            for i in range(group_L):
                new_omega_loaded[i] = np.mean(
                    omega_loaded[i * group_size : (i + 1) * group_size], axis=0
                )
            omega_loaded = new_omega_loaded

        if omega_loaded.shape != omega.shape:
            print(omega_loaded.shape, omega.shape)
            raise ValueError("omega shape mismatched")
        omega = omega_loaded

    # comm
    comm_prefill_payload, comm_decode_payload = get_comm_payload_size(
        lat_cost_model, s, prefill_bz, bz_decode_max, comm_multiplier
    )
    comm_prefill = get_comm_cost(current_D, comm_cost_model, comm_prefill_payload)
    comm_decode = get_comm_cost(current_D, comm_cost_model, comm_decode_payload)

    # print('----- communication cost ---- ')
    # print(comm_prefill, comm_decode)
    return (
        group_L,
        N,
        BITs,
        M_d,
        M,
        (l_prefill, l_decode),
        omega,
        (comm_prefill, comm_decode),
    )


# algo 2


def rename_device_name(D):
    ref_D = {}
    cnt = defaultdict(int)
    for rank, device_name in D.items():
        cnt[device_name] += 1
        ref_D[device_name + str(cnt[device_name])] = rank
    return ref_D


def reset_device_rank_index(D, current_D):
    # D is the previous rank index
    ref_D = rename_device_name(D)
    ref_current_D = rename_device_name(current_D)
    # get mappings
    maps = {}
    for ref_current_device_name, ref_current_rank in ref_current_D.items():
        original_rank = ref_D[ref_current_device_name]
        maps[ref_current_rank] = original_rank
    return maps


# for debug
def check_performance(lat, result):
    # check lat
    for layer_idx, (device, bit_idx) in result.items():
        cur_lat = lat[(layer_idx, device, bit_idx)]
        print(
            "layer {}, device {}, bit {}, lat {}".format(
                layer_idx, device, bit_idx, cur_lat
            )
        )


# Algo1
def solve_ilp_for_best(T, current_D, cost_model_pack, bz_pack):
    print("Try", bz_pack)
    num_hidden_layers = len(T) // 2
    SLO_lat = None
    L, N, BITs, M_d, M, lat, omega, comm = prepare_for_ilp(
        num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack
    )
    result, obj_value = solve_ilp_pulp(
        L, N, BITs, M, M_d, lat, omega, comm, theta, bz_pack
    )
    # result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
    # check_performance(lat[0], result)
    if result != NOT_AVAILABLE:
        result = decouple_result_group(group_size, result)
        result = interpret_ilp_result_i_j_b(result, BITs)
    ilp_res = {}
    ilp_res["obj"] = obj_value
    ilp_res["plan"] = result
    return ilp_res


# Algo2
# enumerate all devices combinations
# enumerata all hybrid micro-batch combinations
# micro-batch candidates


def enumerate_best_result(args):
    model_mem_estimator, comm_cost_model, lat_cost_model, T = args.init_pack
    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    best_plan = {}
    best_plan["obj"] = float("inf")

    num_device_all = sum(device_numbers)
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
    # candidate_prefill_bzs = [i for i in range(1, bz_decode_max + 1)]
    candidate_prefill_bzs = get_factors(bz_decode_max)
    # device order candidates
    device_name_with_its_number = list(zip(device_names, device_numbers))
    permutations = itertools.permutations(device_name_with_its_number)
    # experiment setups
    if args.force_fixed_D:
        # dont change device order
        permutations = [device_name_with_its_number]
    elif args.force_reverse_D:
        permutations = [device_name_with_its_number[::-1]]

    for perm in permutations:
        new_device_names, new_device_numbers = zip(*perm)
        for prefill_bz in candidate_prefill_bzs:
            current_strategy = (
                prefill_bz,
                bz_decode_max,
                new_device_names,
                new_device_numbers,
            )
            # reset the mapping of communication cost model
            current_D = create_device_mesh_and_mem(
                new_device_names, new_device_numbers
            )[0]
            maps = reset_device_rank_index(D, current_D)
            comm_cost_model.set_device_rank_map(maps)
            # test
            bz_pack = (global_bz, prefill_bz, bz_decode_max)
            res = solve_ilp_for_best(T, current_D, cost_model_pack, bz_pack)
            print(res["obj"], best_plan["obj"])
            if res["obj"] < best_plan["obj"]:
                print("Better Plan Generated")
                best_plan = {
                    "prefill_bz": prefill_bz,
                    "bz_decode_max": bz_decode_max,
                    "bz_decode_bss": strat,
                    "device_names": new_device_names,
                    "device_numbers": new_device_numbers,
                    "plan": res["plan"],
                    "obj": res["obj"],
                    "D": current_D,
                    "maps": maps,
                    "name": "llm_pq",
                }
            comm_cost_model.clear_device_rank_map()
    # log the best plan
    print("best plan: ", best_plan)
    # save the result
    # file_name = f'llm_pq_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan


"""
    Initiailization
"""
# global variables
global_bz, micro_bz = None, None
s, n = None, None
model_size, device_info = None, None
use_profiler_prediction = False
D = None
device_numbers, device_names = None, None
available_bits = None
config = None
theta = None
mu_n = None
ROOT_DIR = set_root_folder()
cost_model_store_path = f"{ROOT_DIR}/scripts/cost_model_store"
comm_cost_model_dir = f"{ROOT_DIR}/scripts/comm_cost_model"
lat_profile_result_path = f"{ROOT_DIR}/scripts/lat_profiled_result"
omega_file = None
group_size = 1
ilp_seed = 0
ilp_time_limit = 20
ilp_tolerance = None
verbose_ilp = False
comm_multiplier = 1


def main(args):
    global global_bz, micro_bz, s, n
    global model_size, device_info
    global use_profiler_prediction
    global D
    global device_numbers, device_names
    global available_bits
    global config
    global theta
    global mu_n
    global cost_model_store_path, comm_cost_model_dir
    global lat_profile_result_path
    global omega_file
    global group_size
    global ilp_seed, ilp_time_limit, ilp_tolerance, verbose_ilp
    global comm_multiplier
    # global variables

    omega_file = args.omega_file
    ilp_seed = args.ilp_seed
    group_size = args.group_size
    ilp_tolerance = args.ilp_tolerance
    ilp_time_limit = args.ilp_time_limit
    ilp_tolerance = args.ilp_tolerance
    verbose_ilp = args.debug
    if ilp_tolerance is not None:
        pulp.LpSolverDefault.eps = ilp_tolerance
    comm_multiplier = args.comm_multiplier

    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    model_size = args.model_size  # '66b'
    device_names = (
        args.device_names
    )  # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers  # [2, 3]
    # generation configs
    config = args.config

    gamma = PQConfig.gamma  # expected generated tokens
    theta = PQConfig.theta  # control the concern for accuracy
    mu_n = int(gamma * n)
    available_bits = (
        PQConfig.AVAILABLE_BITS
    )  # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = (
        args.use_profiler_prediction
    )  # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f"{args.comm_cost_model_dir}/{device_info}"
    cost_model_store_path = None  # initialize the cost model

    if args.init_pack is None:
        model_mem_estimator, comm_cost_model, lat_cost_model, T = init_cost_model(
            config,
            global_bz,
            micro_bz,
            s,
            n,
            device_names,
            device_numbers,
            comm_cost_model_dir,
            cost_model_store_path,
        )
        args.init_pack = (model_mem_estimator, comm_cost_model, lat_cost_model, T)
        lat_cost_model.update_profiled_result(args.lat_profile_dir)
        lat_cost_model.update_profiled_prepost_result(args.lat_prepost_profile_dir)
        if args.fit:
            lat_cost_model.fit_regression_cost_model()
        else:
            if not args.use_profiler_prediction:
                lat_cost_model.load_regression_cost_model()

    lat_profile_result_path = args.lat_profile_dir
    return enumerate_best_result(args)


# def llm_pq_main():
#     ilp_env()
#     args = common_argparser()
#     best_plan = main(args)
#     print(best_plan)

# if __name__ == '__main__':
#     from .utils import get_final_strat_file_name
#     from .entry import convert_to_llm_pq_result2partitions
#     ilp_env()
#     args = common_argparser()
#     best_plan = main(args)
#     # model size
#     sols = {
#         'llm_pq': convert_to_llm_pq_result2partitions(best_plan)
#     }
#     model_name = args.model_name
#     folder = args.store_folder
#     file_name = get_final_strat_file_name(model_name, model_size, device_info)
#     save_with_pickle(sols, file_name, folder)
#     logger.info(f'All plans saved to {file_name} in {folder}')
