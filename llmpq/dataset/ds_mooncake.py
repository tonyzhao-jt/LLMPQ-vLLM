from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from .dataset_base import BaseDataset

class MoonCakeDataset(BaseDataset):
    def __init__(self, data_files:str = None):
        ds_path = "mooncake"
        if data_files is not None:
            self.ds = load_dataset('json', data_files=data_files)
        else:
            raise NotImplementedError("MoonCakeDataset is not implemented yet.")
    
    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        prompts = []
        for sample in sampled_data:
            input_length = sample["input_length"]
            prompt = f"a" * input_length
            prompts.append(prompt)
        return prompts
    
    def construct_output(self, sampled_data: List[Dict]) -> List[str]:
        outputs = []
        for sample in sampled_data:
            output_length = sample["output_length"]
            output = f"b" * output_length
            outputs.append(output)
        return outputs

    def sample_n_serving_prompt(self, n: int) -> List[Tuple[str, int, Optional[int]]]:
        """
        NOTE the function is not correct for the moment.
        As tokenizer is needed.
        """
        sampled_data = self.sample(length=n)
        serving_prompts = []
        # tuple 
        for sample in sampled_data:
            # import pdb; pdb.set_trace()
            # dict_keys(['timestamp', 'input_length', 'output_length', 'hash_ids'])
            input_length = sample["input_length"]
            output_length = sample["output_length"]
            prompt = f"a" * input_length
            serving_prompts.append((prompt, input_length, output_length))
        
        return serving_prompts
