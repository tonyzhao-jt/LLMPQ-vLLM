import os
import pickle
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed):
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def save_with_pickle(
    result: Any, file_name: str = None, folder_path: str = None  # noqa
):
    assert file_name is not None, "file_name should not be None"
    # if folder path not exists, create one
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    result_path = os.path.join(folder_path, file_name)
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Result saved to {result_path}")


def get_device_name_by_torch():
    import torch

    gpu_name = ""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"found {gpu_count} GPUs:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            # assume to be homogeneous
            print(f"GPU: {gpu_name}")
            break
    else:
        print("NO GPU found")

    # replace the " " and other
    gpu_name = gpu_name.replace(" ", "_")
    return gpu_name

def get_device_capacity():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
    else:
        raise ValueError("No GPU found")
    return major, minor

def parse_model_id(model_id: str):
    # remove "/" and other special characters
    model_id = model_id.replace("/", "_")
    model_id = model_id.replace("-", "_")
    model_id = model_id.replace(' ', '_')
    return model_id
