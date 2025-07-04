import json
from collections import defaultdict
from typing import Dict, List

import pandas as pd


class Indicator:
    def __init__(self) -> None:
        self.df: pd.DataFrame = None
        self.support_module_wise = False

    def parse(self) -> None:
        pass

    def layer_wise(self) -> Dict[int, float]:
        pass

    def module_wise(self) -> Dict[str, float]:
        pass


class MixPrecisionIndicatorContainer:
    def __init__(self, name: str = "mix_ind") -> None:
        self.name = name
        self.cont: Dict[int, Indicator] = {}

    def add(self, precision: int, indicator: Indicator) -> None:
        self.cont[precision] = indicator

    def layer_wise(self) -> Dict[int, List[float]]:
        layer_wise_list: Dict[int, List[float]] = defaultdict(list)
        for _precision, indicator in self.cont.items():
            layer_loss = indicator.layer_wise()
            for layer, loss in layer_loss.items():
                layer_wise_list[layer].append(loss)
        return layer_wise_list

    def module_wise(self) -> Dict[str, List[float]]:
        module_wise_list: Dict[str, List[float]] = defaultdict(list)
        for _precision, indicator in self.cont.items():
            module_loss = indicator.module_wise()
            for module, loss in module_loss.items():
                module_wise_list[module].append(loss)
        return module_wise_list

    def store(self, file_path: str) -> None:
        data = {
            "name": self.name,
            "cont": {
                str(precision): {
                    "df": (
                        indicator.df.to_dict(orient="split")
                        if indicator.df is not None
                        else None
                    ),
                    "support_module_wise": indicator.support_module_wise,
                }
                for precision, indicator in self.cont.items()
            },
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> "MixPrecisionIndicatorContainer":
        with open(file_path, "r") as f:
            data = json.load(f)

        container = cls(data["name"])

        for precision_str, indicator_data in data["cont"].items():
            indicator = Indicator()

            if indicator_data["df"] is not None:
                indicator.df = pd.DataFrame.from_dict(
                    indicator_data["df"], orient="split"
                )

            indicator.support_module_wise = indicator_data[
                "support_module_wise"
            ]  # noqa

            container.add(int(precision_str), indicator)

        return container
