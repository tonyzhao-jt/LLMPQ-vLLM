"""
Some function that I don't want to pay attention on what's going on
"""


def decouple_result_group(group_size, plan):
    # result: {i: (j, b)}, layer i place on device j withb
    # when i is a group, i is the first layer in the group
    new_plan = {}
    for i, (j, b) in plan.items():
        for k in range(group_size):
            new_plan[i * group_size + k] = (j, b)  # set bit like that
    return new_plan


def partition_a_into_b_bins(a, b):
    remainders = a % b
    ideal_allocation = a // b
    allocation = []
    for i in range(b):
        allocation.append(ideal_allocation)
    for i in range(remainders):
        allocation[i] += 1
    return allocation


def get_default_decode_bz(global_bz, num_device_all):
    bz_decode_max = max(partition_a_into_b_bins(global_bz, num_device_all))
    return bz_decode_max


def get_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors
