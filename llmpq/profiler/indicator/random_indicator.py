import random
from typing import Dict

import pandas as pd
from transformers import AutoConfig

from .base_indicator import Indicator


class RandomIndicator(Indicator):
    def __init__(
        self,
        model_path: str,
        bit: int,
        r_min: float = 4e5,
        r_max: float = 4e6,
    ) -> None:
        super().__init__()
        self.model_path: str = model_path
        config = AutoConfig.from_pretrained(model_path)
        self.layer_num = config.num_hidden_layers
        self.df: pd.DataFrame = None
        self.group_df: pd.DataFrame = None
        self.r_min = r_min
        self.r_max = r_max
        self.bit = bit
        self.parse()

    def parse(self) -> None:
        """Based on the layer number to generate random indicators."""
        assert self.layer_num > 1, "Layer number must be greater than 1"

        def create_random_indicator():
            x = self.r_min + (self.r_max - self.r_min) * random.random()
            indicator = x / (2 ** (self.bit - 1) - 1)
            return indicator

        self.group_df = pd.DataFrame(
            {
                "layer": [i for i in range(self.layer_num)],
                "ind": [
                    create_random_indicator() for _ in range(self.layer_num)
                ],  # noqa
            }
        )

    def layer_wise(self) -> Dict[int, float]:
        """Calculate the total loss per layer."""
        return self.group_df.groupby("layer")["ind"].sum().to_dict()

    def module_wise(self) -> Dict[str, float]:
        assert False, "RandomIndicator now does not support module indicator"
        """Calculate the total loss per module."""
        return self.group_df.groupby("module")["ind"].sum().to_dict()
