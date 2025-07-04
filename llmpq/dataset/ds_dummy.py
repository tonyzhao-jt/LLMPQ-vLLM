from typing import Dict, List, Optional

from .dataset_base import BaseDataset


class DummyDataset(BaseDataset):
    def __init__(self, batch_size: int, prompt_len: int):
        self.batch_size = batch_size
        self.prompt_len = prompt_len

    def gen_prompts(self) -> List[str]:
        # generate prompt with length of prompt_len and batch_size
        prompts = []
        for i in range(self.batch_size):
            prompts.append("a" * self.prompt_len)
        return prompts
