"""
v1 comm cost model with alpha beta.
"""

import os
import pickle
import numpy as np


class CommCostModel:
    def __init__(self, comm_cost_model_folder: str, single_card: bool = False) -> None:
        self.cost_model = {}
        self.single_card = single_card
        if not single_card and comm_cost_model_folder is not None:
            assert os.path.exists(
                comm_cost_model_folder
            ), f"Folder {comm_cost_model_folder} does not exist."
            for file in os.listdir(comm_cost_model_folder):
                file_path = os.path.join(comm_cost_model_folder, file)
                if "cost_model" in file and file.endswith(".pkl"):
                    with open(file_path, "rb") as f:
                        self.cost_model.update(pickle.load(f))
        self.rank_map = (
            {}
        )  # in some case, we need to change the device order, then the rank of the device will change

    def set_device_rank_map(self, map):
        self.rank_map = map.copy()

    def clear_device_rank_map(self):
        self.rank_map = {}

    def print_model_available_keys(self):
        print(self.cost_model.keys())

    def predict_comm_time(self, start_rank, end_rank, data_size):
        if self.single_card:
            return 0  # single card, no inter communication, very small
        if start_rank == end_rank:
            return 0
        if start_rank in self.rank_map:
            start_rank = self.rank_map[start_rank]
        if end_rank in self.rank_map:
            end_rank = self.rank_map[end_rank]

        key = f"{start_rank}_{end_rank}"
        if key not in self.cost_model:
            key = f"{end_rank}_{start_rank}"
        if key not in self.cost_model:
            raise ValueError(f"Cannot find cost model for {key}")

        model = self.cost_model[key]
        poly = np.poly1d(model)
        cost = poly(data_size)
        return cost
