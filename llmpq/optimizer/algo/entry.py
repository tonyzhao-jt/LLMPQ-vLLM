import copy
import os
import time

# globals
from llmpq.config import PQConfig, gen_config
# device
from llmpq.costmodel.core import init_cost_model
from llmpq.costmodel.lat import run_simu
from llmpq.costmodel.mem import check_memory_budget
# logger
from llmpq.logger import assert_log, init_logger
# methods
from llmpq.optimizer.algo.adabits import main as adaptive_bits_main
# arg parser
from llmpq.optimizer.algo.algo_utils import (NOT_AVAILABLE,
                                             get_final_strat_file_name,
                                             set_root_folder)
from llmpq.optimizer.algo.argparser import common_argparser
from llmpq.optimizer.algo.interpreter import \
    convert_to_llm_pq_result2partitions
from llmpq.optimizer.algo.llm_pq_h import main as llm_pq_h_main
from llmpq.optimizer.algo.llm_pq_main import main as llm_pq_main
from llmpq.optimizer.algo.pipeedge_ilp import main as pipeedge_ilp_main
from llmpq.utils.save import save_with_pickle
from llmpq.utils.v1.device import get_device_info

# from llmpq.optimizer.algo.uniform import main as uniform_main # 这个现在可能不用了, 直接跑vllm


logger = init_logger(__name__)


# for debug
def check_minimum_bit_of_sols(sol):
    bit_assignment = sol["plan"]["bit_assignment"]
    minimum_bit = 16
    for k, v in bit_assignment.items():
        if type(v) is str:
            v = 8
        if v < minimum_bit:
            minimum_bit = v
    logger.info(f"minimum_bit: {minimum_bit}")


def log_result(result, name):
    logger.info(f"{name} result: Minimax Lat {result}")


def algo_main():
    # check whether exists /opt/gurobi/ and file under file
    assert_log(
        os.path.exists("/opt/gurobi/"),
        "Please install gurobi and put the license file under /opt/gurobi/",
    )

    args = common_argparser()
    # device info
    # modelname and size
    model_id = args.model_id
    device_names = (
        args.device_names
    )  # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers  # [2, 3]
    device_info = get_device_info(device_names, device_numbers)
    args.device_info = device_info  # use to store device info

    # run simulation
    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    device_names = (
        args.device_names
    )  # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers  # [2, 3]
    gamma = PQConfig.gamma  # expected generated tokens
    mu_n = int(gamma * n)
    # generation configs
    config = args.config
    comm_cost_model_dir = f"{args.comm_cost_model_dir}/{device_info}"
    root_folder = set_root_folder()
    cost_model_store_path = os.path.join(
        root_folder, "cost_model", device_info
    )
    (
        model_mem_estimator,
        comm_cost_model,
        lat_cost_model,
        T,
    ) = init_cost_model(
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

    # get solutions
    sol_adabits = adaptive_bits_main(args)
    # check how long llm_pq takes
    start = time.time()
    if args.llm_pq_efficient:
        # sol_llm_pq = llm_pq_ef_main(args)
        sol_llm_pq = llm_pq_h_main(args)
    else:
        sol_llm_pq = llm_pq_main(args)
    end = time.time()
    llmpq_cost = end - start
    logger.info(f"llm_pq takes {llmpq_cost} seconds")
    no_info_bits = copy.deepcopy(PQConfig.AVAILABLE_BITS)[::-1]

    # find first solution that is valid
    # uniform_sols = {}
    # for bit in no_info_bits:
    #     logger.info(f"Try uniform bit: {bit}")
    #     args.uniform_bit = bit
    #     sol_uniform = uniform_main(args)
    #     if sol_uniform["plan"] != NOT_AVAILABLE:
    #         logger.info(f"Uniform solution found, use bit: {bit}")
    #         uniform_sols[bit] = sol_uniform
    #         break

    # same to pipeedge
    for bit in no_info_bits:
        logger.info(f"Try pipeedge bit: {bit}")
        args.pe_bit = bit
        sol_pipeedge = pipeedge_ilp_main(args)
        if sol_pipeedge["plan"] != NOT_AVAILABLE:
            logger.info(f"PipeEdge solution found, use bit: {bit}")
            break
    # solution packs
    sols = {}
    sols["adabits"] = sol_adabits
    sols["llm_pq"] = sol_llm_pq
    sols["pipeedge"] = sol_pipeedge
    # sols['pipeedge_adaptive'] = sol_pipeedge_adaptive
    # for bit, sol in uniform_sols.items():
    #     sols[f"uniform"] = sol
    for sol_name, sol in sols.items():
        logger.info(f"start to run {sol_name}")
        if sol["plan"] == NOT_AVAILABLE:
            logger.info(f"no plan for {sol_name}")
            continue
        check_memory_budget(sol, model_mem_estimator, name=sol_name)
        convert_to_llm_pq_result2partitions(sol)
        result = run_simu(
            gen_config,
            sol,
            lat_cost_model,
            comm_cost_model,
            args.use_profiler_prediction,
            mu_n=mu_n,
            comm_multiplier=args.comm_multiplier,
        )

        log_result(result, sol_name)

    for sol_name, sol in sols.items():
        logger.info(f"Minimum bit of {sol_name}")
        check_minimum_bit_of_sols(sol)

    # device info
    # pipedge device
    D_original = sol_pipeedge["D"]
    D_llm_pq = sol_llm_pq["D"]
    # check whether same, same key value
    for k, v in D_original.items():
        if D_llm_pq[k] != v:
            logger.info("llmpq D not same")
            logger.info(D_llm_pq)
            break

    logger.info("Running Time: {end-start}")

    sols["mu_n"] = mu_n
    sols["n"] = n
    sols["gloabl_bz"] = global_bz
    sols["prompt_length"] = s
    sols["model_id"] = model_id
    # store the solution
    # with device_names and model id
    file_name = get_final_strat_file_name(model_id, device_info)
    if args.fname_suffix is not None:
        # insert before .pkl
        file_name = file_name[:-4] + args.fname_suffix + ".pkl"
    folder = args.store_folder
    save_with_pickle(sols, file_name, folder)
    logger.info(f"All plans saved to {file_name} in {folder}")


if __name__ == "__main__":
    algo_main()
