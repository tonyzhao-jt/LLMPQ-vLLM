from .dataset_base import BaseDataset
from typing import List, Dict
class GPQADataset(BaseDataset):
    def __init__(self):
        super().__init__("path/to/gpqa_dataset")  # Replace with actual GPQA dataset path

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
            question = sample['Question']
            answer = sample['Answer']
            prompt = (
                f"Question: {question}\n"
                f"Answer: {answer}"
            )
            prompts.append(prompt)
        return prompts