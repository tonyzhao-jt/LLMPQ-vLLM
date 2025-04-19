from typing import Dict, List, Optional, Tuple

from .dataset_base import BaseDataset


class CNNDataset(BaseDataset):
    def __init__(self, data_files:str = None, tokenizer=None):
        super().__init__(["abisee/cnn_dailymail"], data_files=data_files, tokenizer=tokenizer)

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
            question = sample["article"]
            prompt = f"Summarize the article: {question}\n"
            prompts.append(prompt)
        return prompts
    
    def construct_output(self, sampled_data: List[Dict]) -> List[str]:
        """
        Construct prompts for the GPQA dataset.
        Args:
            sampled_data (List[Dict]): A list of sampled data points.
        Returns:
            List[str]: A list of constructed prompts.
        """
        outputs = []
        for sample in sampled_data:
            highlights = sample["highlights"]
            outputs.append(highlights)
        return outputs

    
    def sample_n_serving_prompt(self, n: int) -> List[Tuple[str, int, Optional[int]]]:
        """
        NOTE the function is not correct for the moment.
        As tokenizer is needed.
        """
        assert self.tokenizer is not None
        sampled_data = self.sample(length=n)
        serving_prompts = []
        # tuple 
        for sample in sampled_data:
            highlights = sample["highlights"]
            article = sample["article"]
            prompt = f"Summarize the article: {article}\n"
            output_len = len(self.tokenizer(highlights).input_ids)
            serving_prompts.append((prompt, output_len))
        
        return serving_prompts
