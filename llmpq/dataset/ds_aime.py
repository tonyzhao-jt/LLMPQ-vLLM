from typing import Dict, List, Optional

from .dataset_base import BaseDataset


class AIME2025Dataset(BaseDataset):
    def __init__(self):
        super().__init__(["opencompass/AIME2025", "AIME2025-I"])

    def sample(self, length: Optional[int] = None) -> List[Dict]:
        data = self.ds["test"]  # Assuming the main data is in 'train'
        if length is None or length >= len(data):
            return data
        return data.shuffle(seed=42).select(range(length))

    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        prompts = []
        for sample in sampled_data:
            problem = sample["question"]
            answer = sample["answer"]
            prompt = f"Problem: {problem}\n" f"Answer: {answer}"
            prompts.append(prompt)
        return prompts


class AIMEDataset(BaseDataset):
    def __init__(self):
        super().__init__(["Maxwell-Jia/AIME_2024"])

    def sample(self, length: Optional[int] = None) -> List[Dict]:
        data = self.ds["train"]  # Assuming the main data is in 'train'
        if length is None or length >= len(data):
            return data
        return data.shuffle(seed=42).select(range(length))

    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        prompts = []
        for sample in sampled_data:
            problem = sample["Problem"]
            solution = sample["Solution"]
            answer = sample["Answer"]
            prompt = (
                f"Problem: {problem}\n"
                f"Solution: {solution}\n"
                f"Answer: {answer}"  # noqa
            )
            prompts.append(prompt)
        return prompts
