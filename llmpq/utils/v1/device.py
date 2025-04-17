
'''
    v1
'''
from llmpq.config import PQConfig
def get_device_info(device_names, device_numbers):
    device_info = [f'{device_name}_{device_numbers[idx]}' for idx, device_name in enumerate(device_names)]
    device_info = '_'.join(device_info)
    return device_info

def create_device_mesh_grid(device_mesh):
    # based on the device_mesh, create D
    D = {}
    for start_rank, single_node_device in device_mesh.items():
        num_devices, device_type = single_node_device
        for i in range(num_devices):
            D[start_rank + i] = device_type
    return D

def get_device_mem_offline(device_name, unit='MB'):
    # check through torch.cuda.get_device_properties(0).total_memory / 1024**2
    device_name = device_name.upper()
    mem_table = {
        "A100-SXM4-40GB": 39.4 * 1024, # not exactly = 40 * 1024
        'TESLA_T4': 14.76 * 1024,
        'TESLA_V100-SXM2-32GB': 31.74 * 1024,
        'TESLA_V100-SXM2-16GB': 14.76 * 1024,
        'NVIDIA_A100-SXM4-40GB': 39.4 * 1024,
        'A100-SXM-80GB': 79.35 * 1024,
        'TESLA_P100-PCIE-12GB': 11.91 * 1024,
        'NVIDIA_A100-SXM4-80GB': 79 * 1024,
    }
    if device_name in mem_table:
        mem = mem_table[device_name]
    else:
        raise ValueError("Unknown device name: {}".format(device_name))
    if unit == 'GB':
        mem = mem / 1024
    return mem

def get_single_device_mem_constraints(device_name):
    device_name = device_name.upper()
    offline_device_mem = get_device_mem_offline(device_name, unit=PQConfig.MEM_UNIT)
    device_mem = PQConfig.RATIO_AVOID_OOM * offline_device_mem - PQConfig.CUDA_CONTEXT_MEM
    return device_mem


def get_device_mesh_overall_mem_constraints(D):
    overall_mem = 0
    for device_rank, device_name in D.items():
        device_mem = get_single_device_mem_constraints(device_name)
        overall_mem += device_mem
    return overall_mem

def create_device_mesh_and_mem(device_names, device_numbers):
    device_rank = []
    start_rank = 0
    for i in range(len(device_numbers)):
        device_rank.append(start_rank)
        start_rank += device_numbers[i]

    # create device mesh
    device_mesh = {device_rank[i]: [device_numbers[i], device_names[i]] for i in range(len(device_numbers))}

    D = create_device_mesh_grid(device_mesh)
    max_device_mem = get_device_mesh_overall_mem_constraints(D)
    return D, max_device_mem