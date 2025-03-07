from .dataset_base import BaseDataset
from typing import List, Dict

class AIMEDataset(BaseDataset):
    def __init__(self):
        super().__init__("Maxwell-Jia/AIME_2024")

    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        """
        Construct prompts for the AIME dataset.

        Args:
            sampled_data (List[Dict]): A list of sampled data points.

        Returns:
            List[str]: A list of constructed prompts.
        """
        prompts = []
        for sample in sampled_data:
            problem = sample['Problem']
            solution = sample['Solution']
            answer = sample['Answer']
            prompt = (
                f"Problem: {problem}\n"
                f"Solution: {solution}\n"
                f"Answer: {answer}"
            )
            prompts.append(prompt)
        return prompts
