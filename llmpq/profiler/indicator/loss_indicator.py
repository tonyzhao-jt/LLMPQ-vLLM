import os
from typing import Dict

import pandas as pd

from .base_indicator import Indicator


class LossIndicator(Indicator):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path: str = model_path
        self.df: pd.DataFrame = None
        self.group_df: pd.DataFrame = None
        self.parse()

    def parse(self) -> None:
        """Parse the quant_log.csv file and extract relevant data."""
        quant_res_path = os.path.join(self.model_path, "quant_log.csv")
        if not os.path.exists(quant_res_path):
            print("quant_log.csv not found")
            exit(1)  # Use a non-zero exit code to indicate an error
        self.df = pd.read_csv(quant_res_path)
        # Example of df structure:
        #    layer            module      loss  damp   time
        # 0      0  self_attn.k_proj   1.13409  0.01  1.022
        self.group_df = self.df[["layer", "module", "loss"]]
        # rename loss as ind
        self.group_df = self.group_df.rename(columns={"loss": "ind"})

    def layer_wise(self) -> Dict[int, float]:
        """Calculate the total loss per layer."""
        return self.group_df.groupby("layer")["ind"].sum().to_dict()

    def module_wise(self) -> Dict[str, float]:
        """Calculate the total loss per module."""
        return self.group_df.groupby("module")["ind"].sum().to_dict()
