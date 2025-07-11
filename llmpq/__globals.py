"""
To compatible with some v1's feature.
May deprecated later.
"""

MEM_UNIT = "MB"
CUDA_CONTEXT_MEM = (
    430 + 1500
)  # 430MB cuda context allocation + 1.5 G Torch Temp Allocation
# conforms to the huggingface's script, which reduce by 2GB
RATIO_AVOID_OOM = 0.95  # 95% of the memory is used to avoid OOM
TIME_MULT_TIMES = 1  # whether consider the temporary memory
SLO_RATE = 1.5  # times of fp16 inference time to be SLO.
# global setup for available bits
# all supported bits
# 2,3,4,8: GPTQ weight only quantization
# 8:tc: cutlass int8
# 8:tc-li: bitsandbytes int8
# 16: default, torch
SUPPORTED_BITS = [2, 3, 4, 8, "8:tc", "8:tc-li", 16]
# Author: Here, we remove 2-bit since its performance is quite bad in accuracy.
AVAILABLE_BITS = [3, 4, "8:tc-li", 16]  # actualy use bits
# AVAILABLE_BITS=[2, 3, 4, 8, '8:tc', '8:tc-li', 16] # cutlass failed to perform correct when dealing with multiple inference.
AVAILABLE_BITS_WO_INFO = [3, 4, "8:tc-li", 16]
gamma = 0.5  # expected generated tokens
theta = 0.1  # control the concern for accuracy

# TODO: for the time beding, the pipeline group is implemented by dist_pipeline_group. Later we can move it here.
# communication groups setups.
__TENSOR__MODEL__PARALLEL__GROUP__ = None
__TP__LOCAL__WORLD__SIZE__ = None
__TP__LOCAL__RANK__ = None
__TP__GROUP__RANKS__ = None
__GLOBAL__RANK__ = None


__PIPELINE__MODEL__PARALLEL__GROUP__ = None
__DEVICE__INDEX__ = None  # the device rank
__STAGE__ID__ = None  # the stage rank

__ROW__FIRST__RANK__ = None

# RPC
__CURRENT__SHARDED__MODEL__ = None
__USE__NCCL__ = False
