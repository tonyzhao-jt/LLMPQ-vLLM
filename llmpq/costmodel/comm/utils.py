import numpy as np


def get_comm_payload_size(cost_model, s, prefill_bz, bz_decode_max, comm_multiplier):
    comm_size_prefill = (
        cost_model.h1 * s * prefill_bz * 2 / 1024 / 1024 * comm_multiplier
    )
    comm_size_decode = (
        cost_model.h1 * 1 * bz_decode_max * 2 / 1024 / 1024 * comm_multiplier
    )
    return comm_size_prefill, comm_size_decode


def get_comm_cost(current_D, comm_cost_model, comm_size):
    device_length = len(current_D)
    comm = np.zeros(device_length)
    for idx in range(device_length):
        comm[idx] = comm_cost_model.predict_comm_time(
            idx, (idx + 1) % device_length, comm_size
        )
    return comm
