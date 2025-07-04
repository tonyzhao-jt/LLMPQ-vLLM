from typing import Dict, List, Optional, Tuple

from .dataset_base import BaseDataset
from llmpq.logger import init_logger

logger = init_logger(__name__)


class CNNDataset(BaseDataset):
    def __init__(self, data_files: str = None, tokenizer=None):
        super().__init__(
            ["abisee/cnn_dailymail", "1.0.0"], data_files=data_files, tokenizer=tokenizer
        )

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
        max_attempts = 10  # 最大尝试次数，避免无限循环

        while len(serving_prompts) < n and attempt < max_attempts:
            sampled_data = self.sample(length=n)
            for sample in sampled_data:
                highlights = sample["highlights"]
                article = sample["article"]
                prompt = f"Summarize the article: {article}\n"
                prompt_len = len(self.tokenizer(prompt).input_ids)
                output_len = len(self.tokenizer(highlights).input_ids)
                if max_seq_len is not None:
                    if prompt_len + output_len > max_seq_len:
                        continue
                serving_prompts.append((prompt, output_len))
                logger.info(f"prompt_len: {prompt_len}, output_len: {output_len}")
                if len(serving_prompts) >= n:
                    break
            attempt += 1

        assert len(serving_prompts) == n, f"len(serving_prompts) {len(serving_prompts)}"
        return serving_prompts
