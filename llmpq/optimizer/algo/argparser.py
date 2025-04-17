import argparse
import os
import numpy as np
from transformers import AutoConfig 

from llmpq.config import PQConfig, gen_config
from llmpq.optimizer.algo.algo_utils import set_root_folder
from llmpq.utils.v1.device import get_device_info
from llmpq.logger import assert_log, init_logger

from llmpq.utils.misc import parse_model_id

logger = init_logger(__name__)
ROOT_DIR = set_root_folder() # act as work directory
def verbose_device_info(device_names, device_numbers, device_info):
    print(f"device_names {device_names}")
    print(f"device_numbers {device_numbers}")
    print(f"device_info {device_info}")

def common_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m")
    parser.add_argument("--device_names", nargs="+", type=str, required=True)
    parser.add_argument("--device_numbers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--SLO-aware", action="store_true", help="add slo into constraints"
    )
    parser.add_argument("--omega_file", type=str, default=None)
    # parser.add_argument("--omega-constant", action="store_true", help="use constant omega")
    parser.add_argument(
        "--use_profiler_prediction", action="store_true", help="use profiler prediction"
    )
    parser.add_argument(
        "--comm_cost_model_dir",
        type=str,
        default=f"{ROOT_DIR}/profile/comm_cost_model/",
    )
    parser.add_argument(
        "--lat_profile_dir",
        type=str,
        default=f"{ROOT_DIR}/profile/lat_profiled_result",
    )
    parser.add_argument(
        "--lat_prepost_profile_dir",
        type=str,
        default=f"{ROOT_DIR}/profile/lat_prepost_profiled_result",
    )
    parser.add_argument(
        "--store_folder", type=str, default=f"{ROOT_DIR}/part_strategy"
    )
    # ilp control
    # different seed result in different performance
    parser.add_argument("--ilp_seed", type=int, default=42)
    parser.add_argument(
        "--group_size", type=int, default=1
    )  # when search space is too large, need to group
    parser.add_argument("--ilp_tolerance", type=float, default=None)
    parser.add_argument("--ilp_time_limit", type=int, default=None)
    parser.add_argument("--adapp_group_size", type=int, default=1)
    # algo control
    parser.add_argument("--pe_bit", type=int, default=8)
    parser.add_argument("--uniform_bit", type=int, default=8)
    parser.add_argument(
        "--adabits_tc", action="store_true", help="use adabit-tc"
    )  # case when all device support tc
    parser.add_argument("--init_pack", default=None)
    parser.add_argument("--uniform-hybrid", default=True)
    # llm_pq-efficient
    parser.add_argument(
        "--llm_pq-efficient", action="store_true", help="use llm_pq-efficient"
    )
    # experiment setup control
    parser.add_argument("--s", type=int, default=512)  # prompt legnth
    parser.add_argument("--n", type=int, default=100)  # max_tokens
    parser.add_argument("--global_bz", type=int, default=16)  # global batch size
    parser.add_argument("--theta", type=float, default=0.0001)  # concern for accuracy
    parser.add_argument(
        "--gamma", type=float, default=1
    )  # expected token numbers (x max) to generate
    parser.add_argument(
        "--comm_multiplier", type=float, default=1
    )  # multiply communication when not only the hidden space is passed.
    parser.add_argument("--time_mult_times", type=float, default=1)
    # D related
    parser.add_argument("--force-fixed-D", action="store_true", help="force fixed D")
    parser.add_argument(
        "--force-reverse-D", action="store_true", help="force reverse D"
    )
    # for debug and fit
    parser.add_argument("--fit", action="store_true", help="fit cost model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fname", type=str, default=None)
    # choose from 'adabits' 'adaqpipe' 'pipeedge' 'uniform'
    parser.add_argument(
        "--test_method", type=str, default="adabits", help="test method"
    )
    # storage control
    parser.add_argument(
        "--fname-suffix", type=str, default=None
    )  # add suffix to generated solution plan
    args = parser.parse_args()

    # temporary memory control
    PQConfig.TIME_MULT_TIMES = args.time_mult_times

    # modelname and size
    model_id = args.model_id 
    config = AutoConfig.from_pretrained(model_id)
    args.model_id = parse_model_id(args.model_id)
    args.config = config

    # set configs
    gen_config.global_bz = args.global_bz
    gen_config.s = args.s
    gen_config.n = args.n
    PQConfig.gamma = args.gamma
    PQConfig.theta = args.theta

    # checks
    device_names = (
        args.device_names
    )  # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers  # [2, 3]
    device_info = get_device_info(device_names, device_numbers)
    args.device_info = device_info
    assert len(device_names) == len(
        device_numbers
    ), f"device_names and device_numbers \
          should have the same length {device_names} {device_numbers}"

    if args.debug:
        verbose_device_info(args.device_names, args.device_numbers, device_info)

    # check omega file valid if exits
    if args.omega_file is not None:
        assert_log(
            os.path.exists(args.omega_file),
            f"omega file {args.omega_file} does not exist",
        )
        assert_log(
            model_id in args.omega_file,
            f"omega file {args.omega_file} does not contain model_id {model_id}",
        )

    return args
