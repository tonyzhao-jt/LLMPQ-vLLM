# """
#     Collect the stats to gen indicators.
# """

# import pickle
# from collections import defaultdict
# from time import perf_counter

# import numpy as np
# import torch
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# from llmpq.profiler.indicator import get_loaders

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# model_cfg = AutoConfig.from_pretrained(model_id)
# num_layers = model_cfg.num_hidden_layers
# c4_loader = get_loaders("c4", model=model_id)
# # import pdb
# # pdb.set_trace()

# # create torch dataloader
# from torch.utils.data import DataLoader


# def tokenize_function(examples):
#     return tokenizer(examples["text"])
