import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistConfig:
    local_rank: int
    rank: int
    group_rank: int
    world_size: int
    ngpus: int

    def __init__(self, local_rank, rank, group_rank, world_size, ngpus):
        self.local_rank = local_rank
        self.rank = rank
        self.group_rank = group_rank
        self.world_size = world_size
        self.ngpus = ngpus


def create_device_mesh(rank, local_rank, world_size):
    backend = dist.get_backend()
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    # dist.init_process_group(backend='gloo', init_method='env://') # no need to initialize

    if backend == "gloo":
        node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int64)
        node_info_list = [
            torch.zeros(len(node_info), dtype=torch.int64) for _ in range(world_size)
        ]
    else:
        device = torch.device("cuda:{}".format(local_rank))
        node_info = torch.tensor(
            [node_first_rank, local_rank], dtype=torch.int32, device=device
        )
        node_info_list = [
            torch.zeros(len(node_info), dtype=torch.int32, device=device)
            for _ in range(world_size)
        ]
    dist.all_gather(node_info_list, node_info)
    # dist.destroy_process_group()
    # print("Process group closed")
    # based on the first node, create a mesh with ranks has the same first rank on the row
    # and ranks has the same local rank on the column
    device_mesh = {}
    for i in range(world_size):
        first_rank, local_rank = node_info_list[i].tolist()
        if first_rank not in device_mesh:
            device_mesh[first_rank] = []
        device_mesh[first_rank].append(local_rank + first_rank)
    return device_mesh


def create_device_mesh_nccl(rank, local_rank, world_size):
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )

    device = torch.device("cuda", local_rank)
    node_info = torch.tensor(
        [node_first_rank, local_rank], dtype=torch.int64, device=device
    )
    node_info_list = [
        torch.zeros(len(node_info), dtype=torch.int64, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(node_info_list, node_info)
    # dist.destroy_process_group()
    # print("Process group closed")
    # based on the first node, create a mesh with ranks has the same first rank on the row
    # and ranks has the same local rank on the column
    device_mesh = {}
    for i in range(world_size):
        first_rank, local_rank = node_info_list[i].tolist()
        if first_rank not in device_mesh:
            device_mesh[first_rank] = []
        device_mesh[first_rank].append(local_rank + first_rank)
    return device_mesh


def init_env_gloo():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    # neighbor ranks
    torch.cuda.set_device(local_rank)
    hard_device_mesh = create_device_mesh(rank, local_rank, world_size)
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    return dist_cfg, hard_device_mesh


def init_env():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    # neighbor ranks
    torch.cuda.set_device(local_rank)
    hard_device_mesh = create_device_mesh_nccl(rank, local_rank, world_size)
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    return dist_cfg, hard_device_mesh
