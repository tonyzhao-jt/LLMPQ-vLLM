from datasets import load_dataset
import numpy as np 
from typing import Optional, List, Dict, Type

class BaseDataset:
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset by loading it from Hugging Face.

        Args:
            dataset_path (str): The Hugging Face dataset path.
        """
        self.ds = load_dataset(dataset_path)

    def sample(self, length: Optional[int] = None) -> List[Dict]:
        """
        Sample a specified number of data points from the dataset. If no length is provided, return the entire dataset.

        Args:
            length (Optional[int]): The number of samples to retrieve. If None, return all data.

        Returns:
            List[Dict]: A list of sampled data points, where each point is a dictionary.
        """
        data = self.ds['train']  # Assuming the main data is in 'train'
        if length is None or length >= len(data):
            return data
        return data.shuffle(seed=42).select(range(length))

    def construct_prompt(self, sampled_data: List[Dict]) -> List[str]:
        """
        Construct prompts for language models using the sampled data. This method should be overridden by subclasses.

        Args:
            sampled_data (List[Dict]): A list of sampled data points.

        Returns:
            List[str]: A list of constructed prompts.
        """
        raise NotImplementedError("Subclasses must implement this method to define their prompt template.")

    def sample_n_prompts(self, n: int) -> List[str]:
        """
        Sample n prompts from the dataset.

        Args:
            n (int): The number of prompts to sample.

        Returns:
            List[str]: A list of sampled prompts.
        """
        sampled_data = self.sample(length=n)
        return self.construct_prompt(sampled_data)
    

    def distribution(self, sample_n: Optional[int] = None) -> Dict:
        """
        Get the prompt length distribution of the dataset.
        Returns:
            Dict: A dictionary containing the distribution of the dataset.
        """
        if sample_n is None:
            sample_n = len(self.ds['train'])
        sampled_data = self.sample(length=sample_n)
        prompts = self.construct_prompt(sampled_data)
        prompt_lengths = [len(prompt) for prompt in prompts]
        # percentile
        percentiles = [50, 75, 90, 95, 99]
        distribution = {f"{p}th percentile": np.percentile(prompt_lengths, p) for p in percentiles}
        return distribution
        