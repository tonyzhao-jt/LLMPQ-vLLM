from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from .dataset_base import BaseDataset

from llmpq.logger import init_logger

logger = init_logger(__name__)


class LooGLEDataset(BaseDataset):
    def __init__(self, data_files: str = None, tokenizer=None):
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

    def sample_n_serving_prompt(
        self, n: int, max_seq_len: Optional[int] = None
    ) -> List[Tuple[str, int, Optional[int]]]:
        """
        NOTE the function is not correct for the moment.
        As tokenizer is needed.
        """
        assert self.tokenizer is not None
        serving_prompts = []
        attempt = 0
        max_attempts = 10

        while len(serving_prompts) < n and attempt < max_attempts:
            sampled_data = self.sample(length=n)
            for sample in sampled_data:
                context = sample["context"]
                question = sample["question"]
                answer = sample["answer"]
                prompt = f"Context: {context} Question: {question}"
                prompt_len = len(self.tokenizer(prompt).input_ids)
                output_len = len(self.tokenizer(answer).input_ids)
                if max_seq_len is not None:
                    if prompt_len + output_len > max_seq_len:
                        continue
                serving_prompts.append((prompt, output_len))
                logger.info(f"len(serving_prompts) {len(serving_prompts)}")
                if len(serving_prompts) >= n:
                    break
            attempt += 1
            logger.info(f"attempt {attempt}")

        assert len(serving_prompts) == n, f"len(serving_prompts) {len(serving_prompts)}"
        return serving_prompts
