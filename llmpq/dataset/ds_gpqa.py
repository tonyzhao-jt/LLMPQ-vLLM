from typing import Dict, List

from .dataset_base import BaseDataset


class GPQADataset(BaseDataset):
    def __init__(self):
        super().__init__(["Idavidrein/gpqa"])

    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        """
        Construct prompts for the GPQA dataset.

        Args:
            sampled_data (List[Dict]): A list of sampled data points.

        Returns:
            List[str]: A list of constructed prompts.
        """
        prompts = []
        for sample in sampled_data:
            question = sample["Question"]
            answer = sample["Answer"]
            prompt = f"Question: {question}\n" f"Answer: {answer}"
            prompts.append(prompt)
        return prompts
