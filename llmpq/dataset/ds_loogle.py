from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from .dataset_base import BaseDataset

class LooGLEDataset(BaseDataset):
    def __init__(self, data_files:str = None, tokenizer=None):
        ds_path = "bigai-nlco/LooGLE"
        self.tokenizer = tokenizer
        if data_files is not None:
            self.ds = load_dataset(ds_path, "longdep_qa", data_files=data_files)
        else:
            self.ds = load_dataset(ds_path, "longdep_qa")

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
            context = sample["context"]
            question = sample["question"]
            prompt = f"Context: {context} Question: {question}"
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
        prompts = []
        for sample in sampled_data:
            answer = sample["answer"]
            prompts.append(answer)
        return prompts


    def sample_n_serving_prompt(self, n: int) -> List[Tuple[str, int, Optional[int]]]:
        """
        NOTE the function is not correct for the moment.
        As tokenizer is needed.
        """
        sampled_data = self.sample(length=n)
        serving_prompts = []
        # tuple 
        for sample in sampled_data:
            # dict_keys(['id', 'doc_id', 'task', 'context', 'question', 'answer', 'evidence', 'title'])
            context = sample["context"]
            question = sample["question"]
            answer = sample["answer"]
            prompt = f"Context: {context} Question: {question}"
            output_len = len(self.tokenizer(answer).input_ids)
            serving_prompts.append((prompt, output_len))
        
        return serving_prompts
